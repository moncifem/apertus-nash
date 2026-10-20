"""Dataset preparation: GSM8K + filtered NuminaMath-1.5 + the generated curriculum,
dedup, decontamination, splits.

Reference solutions are used only to derive gold answers; they never enter the policy prompt.

The `curriculum` source reads `out/curriculum.jsonl`, the product of the pipeline in
`dataset/SPEC.md`. That dataset is ordered for progressive learning, so the reader
preserves `order` and only keeps rows whose answer can be checked mechanically
(`answer_check.type` in {numeric, exact, symbolic, multi}) - a rubric-only row has no
verifiable reward and would contribute nothing to GRPO.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("rlaif.data")


@dataclass
class Problem:
    id: str
    source: str
    source_revision: str
    split: str
    problem: str
    gold: str


def norm_hash(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    t = re.sub(r"[^\w\s]", "", t)
    return hashlib.sha256(t.encode()).hexdigest()


def _gsm8k(revision: str):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", revision=revision)
    out = {}
    for split in ("train", "test"):
        rows = []
        for i, r in enumerate(ds[split]):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append(Problem(f"gsm8k:{split}:{i}", "openai/gsm8k", revision, split, r["question"].strip(), gold))
        out[split] = rows
    return out


def _numina_row_ok(r: dict) -> bool:
    ans = (r.get("answer") or "").strip()
    if not ans or len(ans) > 48:
        return False
    if ans.lower() in {"proof", "notfound", "none", "n/a", "not found"}:
        return False
    if r.get("question_type") not in (None, "math-word-problem"):
        return False  # drop MCQ / proof / true-false
    for k in ("problem_is_valid", "solution_is_valid"):
        if k in r and str(r[k]) != "Yes":
            return False
    if r.get("synthetic") is True:
        return False
    if any(x in ans for x in ("\\text", ";", "\\quad", "or ")):
        return False  # units / prose / multiple answers: unsupported answer types
    prob = r.get("problem") or ""
    return 0 < len(prob) <= 3000 and "prove" not in prob.lower()


def _numina(revision: str, limit: int, seed: int, verifier):
    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train", revision=revision, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    cand = []
    for i, r in enumerate(ds):
        if _numina_row_ok(r):
            cand.append(Problem(f"numina:{r.get('source', 'x')}:{i}", "AI-MO/NuminaMath-1.5", revision,
                                "train", r["problem"].strip(), r["answer"].strip()))
        if len(cand) >= int(limit * 1.3):
            break
    ok = verifier.gold_ok([p.gold for p in cand])
    kept = [p for p, k in zip(cand, ok) if k][:limit]
    log.info("numina: %d candidates, %d with parseable gold, kept %d", len(cand), sum(ok), len(kept))
    return kept


def _curriculum(dc, seed: int, verifier):
    """Load out/curriculum.jsonl (dataset/SPEC.md product) as Problem rows.

    Keeps rows whose answer is mechanically checkable and whose level is in range,
    preserves the file's `order`, and keeps the row's own split by default so the
    Section 11 family-wise split is honoured.
    """
    path = Path(dc.curriculum_path)
    if not path.is_absolute():
        # configs live at the repo root-relative path `out/curriculum.jsonl`
        path = Path(__file__).resolve().parents[2] / dc.curriculum_path
    if not path.exists():
        raise FileNotFoundError(
            f"curriculum source requested but {path} does not exist; "
            f"run the SPEC.md pipeline (dataset/SPEC.md Section 7) first")

    out: dict[str, list[Problem]] = {"train": [], "val": [], "test": []}
    stats = {"read": 0, "kept": 0, "dropped_level": 0, "dropped_type": 0,
             "dropped_check": 0, "dropped_bad_gold": 0}
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("order", 0))

    for r in rows:
        stats["read"] += 1
        if not (dc.curriculum_min_level <= int(r.get("level", -1)) <= dc.curriculum_max_level):
            stats["dropped_level"] += 1
            continue
        if r.get("answer_type") not in dc.curriculum_answer_types:
            stats["dropped_type"] += 1
            continue
        if ((r.get("answer_check") or {}).get("type") or "none") not in dc.curriculum_check_types:
            stats["dropped_check"] += 1
            continue
        gold = (r.get("reference_answer") or "").strip()
        if not gold or len(gold) > 64:
            stats["dropped_bad_gold"] += 1
            continue
        split = r.get("split") if dc.curriculum_use_row_split else "train"
        if split not in ("train", "val", "test"):
            split = "train"
        out[split].append(Problem(
            id=str(r.get("id", f"curriculum:{stats['read']}")),
            source="curriculum",
            source_revision=f"SPEC.md-v1.0-order-{r.get('order', 0)}",
            split=split,
            problem=(r.get("problem") or "").strip(),
            gold=gold,
        ))
        stats["kept"] += 1

    # Anything not marked val/test falls back to train.
    place = {p.id: s for s, ps in out.items() for p in ps}
    for p in list(out["val"]) + list(out["test"]):
        if p.split not in ("val", "test"):
            out["train"].append(p)
    out["val"] = [p for p in out["val"] if p.split == "val"]
    out["test"] = [p for p in out["test"] if p.split == "test"]
    _ = place

    # Re-verify the gold strings the reward will compare against.
    allp = out["train"] + out["val"] + out["test"]
    if allp:
        ok = verifier.gold_ok([p.gold for p in allp])
        good = {"train": [], "val": [], "test": []}
        dropped = 0
        for p, k in zip(allp, ok):
            if k:
                good[p.split].append(p)
            else:
                dropped += 1
        out = good
        stats["dropped_bad_gold"] += dropped
        stats["kept"] = len(allp) - dropped

    log.info("curriculum: %s", stats)
    return out


def _rewritten(dc) -> dict[str, list[Problem]]:
    """RL prompts from the SFT rewrite rows. gold = the DeepSeek-written reference reply (the judge's ground truth); the row's own split is kept."""
    out = {"train": [], "val": [], "test": []}
    for l in open(dc.rewritten_path):
        r = json.loads(l)
        if dc.rewritten_single_turn_only and r.get("history"):
            continue
        split = r.get("split", "train")
        split = split if split in out else "train"
        out[split].append(Problem(r["id"], "sft_rewritten_rows", "local", split, r["problem"].strip(), (r.get("reference_answer") or "").strip()))
    return out


def _nash(dc, verifier) -> dict[str, list[Problem]]:
    from .structured import encode_gold, policy_problem
    if dc.answer_format != "plain" or dc.reasoning_format != "prose":
        raise ValueError("Nash requires answer_format=plain and reasoning_format=prose")
    path = Path(dc.nash_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    raw = path.read_bytes()
    revision = hashlib.sha256(raw).hexdigest()
    out = {"train": [], "val": [], "test": []}
    ids = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in ids or row["split"] not in out:
            raise ValueError("Invalid Nash ID or split")
        ids.add(row["id"])
        gold = encode_gold(row["expected_answer"],row.get("symbol_assumptions"))
        out[row["split"]].append(Problem(row["id"], "nash", revision,
                                         row["split"], policy_problem(row), gold))
    all_rows = [p for split in out.values() for p in split]
    valid = verifier.gold_ok([p.gold for p in all_rows])
    if len(valid) != len(all_rows) or not all(valid):
        raise ValueError("Nash reference outcomes failed validation")
    return out


def prepare_data(dc, seed: int, verifier) -> dict[str, list[Problem]]:
    if "nash" in dc.sources:
        if dc.sources != ["nash"]:
            raise ValueError("Nash currently requires sources=[nash]")
        return _nash(dc, verifier)
    if dc.sources == ["rewritten"]:
        return _rewritten(dc)
    cache = Path(dc.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({**dc.model_dump(), "seed": seed}, sort_keys=True).encode()).hexdigest()[:16]
    manifest = cache / "data-manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("key") == key:
        return {s: [Problem(**json.loads(l)) for l in (cache / f"{s}.jsonl").read_text().splitlines()]
                for s in ("train", "val", "test")}

    # GSM8K's test split is the contamination reference for every source, so load it
    # whenever any source is in play.
    gsm = _gsm8k(dc.gsm8k_revision)
    test = gsm["test"]
    pool = []
    if "gsm8k" in dc.sources:
        pool += gsm["train"][: dc.gsm8k_max_examples] if dc.gsm8k_max_examples else gsm["train"]
    if "numina" in dc.sources:
        pool += _numina(dc.numina_revision, dc.numina_max_examples, seed, verifier)

    # The generated curriculum keeps its own family-wise split (SPEC.md Section 11).
    curr = (_curriculum(dc, seed, verifier) if "curriculum" in dc.sources
            else {"train": [], "val": [], "test": []})
    pool += curr["train"]

    test_hashes = {norm_hash(p.problem) for p in test}
    seen, train_pool, removed_contam, removed_dup = set(), [], 0, 0
    for p in pool:
        h = norm_hash(p.problem)
        if h in test_hashes:
            removed_contam += 1
            continue
        if h in seen:
            removed_dup += 1
            continue
        seen.add(h)
        train_pool.append(p)

    # Held-out curriculum rows are decontaminated and deduped too, but never re-split:
    # their family grouping must stay intact.
    hedged: dict[str, list[Problem]] = {"val": [], "test": []}
    for split in ("val", "test"):
        for p in curr[split]:
            h = norm_hash(p.problem)
            if h in test_hashes or h in seen:
                removed_contam += 1
                continue
            seen.add(h)
            hedged[split].append(p)

    rng = random.Random(seed)
    rng.shuffle(train_pool)
    val = train_pool[: dc.val_size]
    train = train_pool[dc.val_size:]
    for p in val:
        p.split = "val"
    val += hedged["val"]
    test = test + hedged["test"]
    log.info("curriculum rows kept: train=%d val=%d test=%d",
             len(curr["train"]), len(hedged["val"]), len(hedged["test"]))

    for name, rows in (("train", train), ("val", val), ("test", test)):
        (cache / f"{name}.jsonl").write_text("\n".join(json.dumps(asdict(p)) for p in rows) + "\n")
    manifest.write_text(json.dumps({
        "key": key, "config": dc.model_dump(), "seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "removed_exact_overlap_with_gsm8k_test": removed_contam, "removed_duplicates": removed_dup,
        "note": "Only exact normalized-text overlap is removed. Pretraining contamination is unknown.",
    }, indent=2))
    log.info("data: train=%d val=%d test=%d (dropped %d overlap, %d dup)",
             len(train), len(val), len(test), removed_contam, removed_dup)
    return {"train": train, "val": val, "test": test}
