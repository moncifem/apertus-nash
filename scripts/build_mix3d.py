"""Rebuild the mix3d SFT data (stage 1 and stage 2) from its public sources.

The third-party rows are not redistributed. data/mix3d/manifest.jsonl.gz lists every row with its metadata, a key (hash of the normalised
problem) and a content hash. The project's own rows are taken from this repo's datasets: data/synthetic/sft.jsonl (the synthetic
reasoning set) and data/nash/sft.jsonl. The third-party rows are rebuilt: this script streams the same Hugging Face files the original
build read, applies the same extraction, keeps the rows whose key and content hash match the manifest, and writes the two training files
in their training order:

    python scripts/build_mix3d.py                 # -> data/mix3d/stage1.jsonl, data/mix3d/stage2.jsonl
    python scripts/build_mix3d.py --only cohere   # one source family at a time (cohere, cascade2, nemotron)

Needs: pyarrow, huggingface_hub. Some sources are gated on the Hub (accept their terms first) and the Cohere set is CC-BY-NC-4.0.
The sources are read at their current `main` revision; a row whose text changed upstream fails its content hash and is reported, never
silently replaced. 198 rows came from small third-party sets that this script does not fetch; they are listed in the manifest as
`text: unavailable` and left out (0.2% of the data).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "mix3d"
HF = "https://huggingface.co/datasets"
CASCADE = f"{HF}/nvidia/Nemotron-Cascade-2-SFT-Data/resolve/main"
IFCHAT = f"{HF}/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3/resolve/main/data"
CP = f"{HF}/nvidia/Nemotron-SFT-Competitive-Programming-v2/resolve/main/data"
COHERE = "CohereLabs/tiny-aya-l2-thinker-multilingual-reasoning"

# source name in the manifest -> (url, bytes read from the start of the file in the original build, row format)
STREAMS = {
    "cascade2_chat": (f"{CASCADE}/chat/chat_part_1.jsonl", 2500, "cascade"),
    "cascade2_math": (f"{CASCADE}/math/math_notool.jsonl", 3000, "cascade"),
    "cascade2_proof": (f"{CASCADE}/math/math_proof.jsonl", 1500, "cascade"),
    "cascade2_science": (f"{CASCADE}/science/science.jsonl", 2500, "cascade"),
    "nemotron_chat3": (f"{IFCHAT}/chat.jsonl", 2500, "generic"),
    "nemotron_ifchat3_if": (f"{IFCHAT}/instruction_following.jsonl", 1500, "generic"),
    "nemotron_math_v3": (f"{HF}/nvidia/Nemotron-SFT-Math-v3/resolve/main/data/train.jsonl", 3000, "generic"),
    "nemotron_cp2_py": (f"{CP}/competitive_programming_python_00.jsonl", 3000, "generic"),
    "nemotron_competitive_programming_python_01": (f"{CP}/competitive_programming_python_01.jsonl", 6000, "generic"),
    "nemotron_competitive_programming_cpp_00": (f"{CP}/competitive_programming_cpp_00.jsonl", 6000, "generic"),
    "nemotron_competitive_programming_cpp_01": (f"{CP}/competitive_programming_cpp_01.jsonl", 6000, "generic"),
}
FIELDS = ("id", "split", "level", "domain", "problem", "history", "thinking", "reference_answer", "answer_style", "prompt_style",
          "language", "source", "order", "approx_tokens")


# ---------------------------------------------------------------- the original extraction, unchanged
def norm_key(problem: str) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", problem.lower()).strip().encode()).hexdigest()


def content_hash(problem: str, thinking: str, answer: str) -> str:
    return hashlib.sha256(json.dumps([problem, thinking, answer], ensure_ascii=False).encode()).hexdigest()[:20]


def unbox(t: str) -> str:
    """Replace every \\boxed{...} in the thinking by its content (balanced braces)."""
    out, i = [], 0
    while True:
        j = t.find("\\boxed{", i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k, d = j + 7, 1
        while k < len(t) and d > 0:
            d += (t[k] == "{") - (t[k] == "}")
            k += 1
        if d != 0:
            out.append(t[j:])
            break
        out.append(t[j + 7:k - 1])
        i = k
    return "".join(out)


def final_text(problem: str, thinking: str, answer: str) -> tuple[str, str, str]:
    return problem.strip(), unbox(thinking).strip(), answer.strip()


def from_cascade(r: dict):
    """Nemotron-Cascade-2 rows: one user turn, one assistant turn holding <think>...</think> then the answer."""
    ms = r.get("messages") or []
    users = [m for m in ms if m["role"] == "user"]
    asst = [m for m in ms if m["role"] == "assistant"]
    if len(users) != 1 or len(asst) != 1:
        return None
    m = re.match(r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*$", asst[0]["content"] or "", re.S)
    if not m or not m.group(2) or len(m.group(1)) < 200:
        return None
    return (users[0]["content"] or "").strip(), m.group(1), m.group(2).strip()


def from_generic(r: dict):
    """Nemotron SFT rows: one user turn, one final assistant turn with `reasoning_content` (or `reasoning`) and `content`."""
    ms = r.get("messages") or []
    users = [m for m in ms if m["role"] == "user"]
    asst = [m for m in ms if m["role"] == "assistant"]
    if len(users) != 1 or len(asst) != 1 or ms[-1]["role"] != "assistant":
        return None
    a = asst[0]
    th = (a.get("reasoning_content") or a.get("reasoning") or "").strip()
    ans = (a.get("content") or "").strip()
    if len(th) < 200 or not ans:
        return None
    return (users[0]["content"] or "").strip(), th, ans


# ---------------------------------------------------------------- matching
class Wanted:
    """Manifest rows still missing their text, looked up by key."""

    def __init__(self, rows):
        self.by_key = collections.defaultdict(list)
        for r in rows:
            self.by_key[r["key"]].append(r)
        self.left = len(rows)
        self.text: dict[str, tuple] = {}

    def offer(self, problem, thinking, answer) -> None:
        cands = self.by_key.get(norm_key(problem))
        if not cands:
            return
        p, t, a = final_text(problem, thinking, answer)
        h = content_hash(p, t, a)
        for r in cands:
            if r["id"] not in self.text and r["sha"] == h:
                self.text[r["id"]] = (p, t, a)
                self.left -= 1


def stream_lines(url: str, max_mb: int):
    headers = {"Range": f"bytes=0-{max_mb * 1_000_000}"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as resp:
        for raw in resp:
            yield raw


def fill_stream(name: str, rows: list, local: Path | None = None) -> dict:
    url, max_mb, fmt = STREAMS[name]
    want = Wanted(rows)
    adapt = from_cascade if fmt == "cascade" else from_generic
    lines = open(local, "rb") if local else stream_lines(url, max_mb)
    for raw in lines:
        try:
            r = json.loads(raw)
        except ValueError:
            continue                                           # the last line of a byte-range read is cut
        x = adapt(r)
        if x:
            want.offer(*x)
        if want.left == 0:
            break
    return want.text


def fill_cohere(rows: list, local: Path | None = None) -> dict:
    import pyarrow.parquet as pq
    if local is None:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(COHERE, repo_type="dataset", allow_patterns=["data/*/train.parquet"]))
    want = Wanted(rows)
    for f in sorted(local.glob("data/*/train.parquet")):
        for r in pq.read_table(f).to_pylist():
            want.offer(r["question"], r["thinking"], r["output"])                                   # translated rows
            want.offer(r["original_question"], r["original_thinking"], r["original_output"])       # rows kept in English (source ...:en)
        if want.left == 0:
            break
    return want.text


# ---------------------------------------------------------------- the project's own rows
OWN_FILES = {"synthetic": ROOT / "data" / "synthetic" / "sft.jsonl", "nash": ROOT / "data" / "nash" / "sft.jsonl"}


def repo_text(manifest: list) -> dict:
    """Text of the rows that come from this repo's datasets (manifest `text: repo`, pointing at `ref`), checked against the content hash.
    One Nash row was edited after mix3d was built; the manifest keeps the text it was trained with (`text: inline`)."""
    rows = {name: {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))} for name, path in OWN_FILES.items()}
    text = {}
    for m in manifest:
        if m["text"] == "inline":
            text[m["id"]] = (m["problem"], m["thinking"], m["reference_answer"])
        elif m["text"] == "repo":
            r = rows[m["ref"]["file"]][m["ref"]["id"]]
            t = r["thinking"] if isinstance(r["thinking"], str) else "\n".join(r["thinking"])
            for v in ((r["problem"], t, r["reference_answer"]), (r["problem"].strip(), t.strip(), r["reference_answer"].strip())):
                if content_hash(*v) == m["sha"]:
                    text[m["id"]] = v
                    break
            else:
                print(f"warning: {m['ref']['file']} row {m['ref']['id']} no longer matches mix3d row {m['id']}; left out")
    return text


# ---------------------------------------------------------------- main
def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_stage(path: Path, manifest: list, text: dict, stage: str) -> int:
    rows = sorted((r for r in manifest if r.get(stage)), key=lambda r: r[stage]["order"])
    n = 0
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            if r["id"] not in text:
                continue
            p, t, a = text[r["id"]]
            row = dict(id=r["id"], split=r[stage]["split"], level=None, domain=r["domain"], problem=p, history=[], thinking=t,
                       reference_answer=a, answer_style=r["answer_style"], prompt_style="raw", language=r["language"],
                       source=r["source"], order=r[stage]["order"], approx_tokens=r["approx_tokens"])
            w.write(json.dumps({k: row[k] for k in FIELDS}, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=["cohere", "cascade2", "nemotron"], help="fetch one source family only")
    ap.add_argument("--out", type=Path, default=MIX)
    args = ap.parse_args()

    manifest = read_gz(MIX / "manifest.jsonl.gz")
    text = repo_text(manifest)
    print(f"{'own datasets (data/synthetic, data/nash)':45s} {len(text):6d} / {sum(r['text'] in ('repo', 'inline') for r in manifest):6d} rows")
    groups = collections.defaultdict(list)
    for r in manifest:
        if r["text"] == "hub":
            groups[r["source"].split(":")[0]].append(r)

    for name, rows in sorted(groups.items()):
        family = "cohere" if name.startswith("cohere") else name.split("_")[0]
        if args.only and family != args.only:
            continue
        got = fill_cohere(rows) if family == "cohere" else fill_stream(name, rows)
        text.update(got)
        print(f"{name:45s} {len(got):6d} / {len(rows):6d} rows", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for stage in ("stage1", "stage2"):
        total = sum(1 for r in manifest if r.get(stage))
        n = write_stage(args.out / f"{stage}.jsonl", manifest, text, stage)
        print(f"{stage}: wrote {n} of {total} rows -> {args.out / (stage + '.jsonl')}")
    missing = [r for r in manifest if r["id"] not in text]
    if missing:
        print(f"missing text for {len(missing)} rows:", dict(collections.Counter(r['source'].split(':')[0] for r in missing).most_common()))


if __name__ == "__main__":
    sys.exit(main())
