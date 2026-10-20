"""Curriculum rows -> SFT examples. Pure Python plus a tokenizer; no torch.

A row's training target is exactly what the RL policy is asked to emit:

    <think_open> thinking lines joined by newlines <think_close> \\boxed{answer} <end>

and every target is parsed with the same parser that scores RL rollouts (`rlaif.parsing`), so SFT and RL
agree on the format. The prompt is rendered with the same chat-template call as `train.render_prompt`.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


class SkipRow(Exception):
    """A row that cannot become a clean SFT example; the message is the reason."""


@dataclass
class SFTExample:
    id: str
    order: int
    level: int | None
    domain: str | None
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str
    answer_style: str = "boxed"       # "boxed" (math, RL-compatible) or "plain" (general questions, no \\boxed)
    prompt_variant: str = "default"   # concise / standard / detailed / raw / default (the single shared wrapper)
    reasoning_format: str = "numbered"

    @property
    def n_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)


# ------------------------------------------------------------------------------------ loading
def load_rows(path: str | Path, splits: list[str] | None = None, limit: int = 0) -> list[dict]:
    """Rows in curriculum order: sorted by `order` when every row has it, file order otherwise."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from e
    if splits:
        keep = set(splits)
        rows = [r for r in rows if r.get("split", "train") in keep]
    if rows and all(isinstance(r.get("order"), int) for r in rows):
        rows.sort(key=lambda r: r["order"])
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------------------------ text building
_ANSWER_PREFIX = re.compile(r"^\s*answer\s*[:\-]\s*", re.IGNORECASE)


def answer_to_boxed_text(answer: str) -> str:
    """Reference answer -> the text that goes inside \\boxed{...}."""
    a = _ANSWER_PREFIX.sub("", str(answer)).strip()
    # \boxed{} is already math mode: drop the $ delimiters but keep escaped dollar signs (\$).
    a = a.replace("\\$", "\x00").replace("$", "").replace("\x00", "\\$").strip()
    a = re.sub(r"\s*\n\s*", " ", a)
    if not a:
        raise SkipRow("empty reference_answer")
    if "\\boxed" in a:
        raise SkipRow("reference_answer already contains \\boxed")
    depth = 0
    for ch in a:
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            break
    if depth != 0:
        raise SkipRow("unbalanced braces in reference_answer")
    return a


ANSWER_STYLES = ("boxed", "plain", "free")


def answer_style_of(row: dict) -> str:
    style = row.get("answer_style") or "boxed"
    if style not in ANSWER_STYLES:
        raise SkipRow(f"unknown answer_style {style!r} (expected one of {ANSWER_STYLES})")
    return style


def plain_answer_text(answer: str, markers: list[str]) -> str:
    """Reference answer -> the plain text that follows the thinking span (no \\boxed, newlines kept)."""
    a = _ANSWER_PREFIX.sub("", str(answer)).strip()
    if not a:
        raise SkipRow("empty reference_answer")
    if "\\boxed" in a:
        raise SkipRow("plain reference_answer contains \\boxed")
    for m in markers:
        if m in a:
            raise SkipRow(f"reference_answer contains the marker {m!r}")
    return a


def free_answer_text(answer: str, markers: list[str]) -> str:
    """Reference answer -> the reply exactly as written (explanation, optional \\boxed{...} anywhere, newlines kept)."""
    a = str(answer).strip()
    if not a:
        raise SkipRow("empty reference_answer")
    for m in markers:
        if m in a:
            raise SkipRow(f"reference_answer contains the marker {m!r}")
    return a


def thinking_to_text(thinking, markers: list[str]) -> str:
    if isinstance(thinking, str):
        text = thinking.strip()
        if not text:
            raise SkipRow("empty thinking text")
    elif isinstance(thinking, list) and thinking and all(isinstance(s, str) for s in thinking):
        lines = [s.strip() for s in thinking]
        if any(not s for s in lines):
            raise SkipRow("empty thinking line")
        text = "\n".join(lines)
    else:
        raise SkipRow("thinking is not non-empty text or a non-empty list of strings")
    for m in markers:
        if m in text:
            raise SkipRow(f"thinking contains the marker {m!r}")
    if "\\boxed" in text:
        raise SkipRow("thinking contains \\boxed")
    return text


def build_completion(row: dict, think_open: str, think_close: str, end_token: str) -> str:
    markers = [think_open, think_close, end_token]
    # no_thinking rows (chat, plain code answers) get an EMPTY think block: <|inner_prefix|><|inner_suffix|>reply
    think = "" if row.get("no_thinking") else thinking_to_text(row.get("thinking"), markers)
    if answer_style_of(row) == "plain":
        return f"{think_open}{think}{think_close}{plain_answer_text(row.get('reference_answer', ''), markers)}{end_token}"
    if answer_style_of(row) == "free":
        return f"{think_open}{think}{think_close}{free_answer_text(row.get('reference_answer', ''), markers)}{end_token}"
    ans = answer_to_boxed_text(row.get("reference_answer", ""))
    return f"{think_open}{think}{think_close}\\boxed{{{ans}}}{end_token}"


def render_prompt(tok, template: str, problem: str, thinking: bool, system: str | None = None,
                  history: list[dict] | None = None) -> str:
    """Identical to train.render_prompt (kept separate so SFT does not import the RL stack).

    `system` replaces the chat template's built-in system message (which says "Apertus 1.5 Omni");
    None keeps the template default. `history` = earlier user/assistant turns of a multi-turn chat (system turns are ignored:
    the configured system prompt always wins)."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or []) if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": template.replace("{problem}", problem)})
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


# ------------------------------------------------------------------------------------ prompt variants
VARIANT_NAMES = ("concise", "standard", "detailed")
# Every wrapper must keep the two things the parser and the rewards depend on.
_VARIANT_MUST_HAVE = {"{problem}": "{problem}", "\\boxed{answer}": "\\boxed{answer}",
                      "numbered steps": None}      # None = regex check below: any form of "number(ed)"
_NUMBERED = re.compile(r"number", re.IGNORECASE)


def load_variants(path: str | Path) -> dict[str, list[str]]:
    """{variant: [paraphrased wrapper, ...]}. Each wrapper asks for numbered steps and one \\boxed{answer}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for name, wrappers in data.items():
        if name.startswith("_"):
            continue
        if name not in VARIANT_NAMES:
            raise ValueError(f"{path}: unknown variant {name!r} (expected one of {VARIANT_NAMES})")
        if not isinstance(wrappers, list) or not wrappers:
            raise ValueError(f"{path}: variant {name!r} needs a non-empty list of wrappers")
        for i, w in enumerate(wrappers):
            miss = [k for k, lit in _VARIANT_MUST_HAVE.items()
                    if (lit not in w if lit else not _NUMBERED.search(w))]
            if miss:
                raise ValueError(f"{path}: {name}[{i}] is missing {miss}")
        out[name] = list(wrappers)
    return out


def pick_template(row: dict, default: str, variants: dict[str, list[str]] | None) -> str:
    """Wrapper for one row: bare question for raw rows, else a deterministic paraphrase of its variant."""
    if row.get("prompt_style") == "raw" or row.get("prompt_variant") == "raw":
        return "{problem}"
    name = row.get("prompt_variant")
    if variants and name in variants:
        pool = variants[name]
        h = int.from_bytes(hashlib.sha256(str(row.get("id", "")).encode()).digest()[:4], "big")
        return pool[h % len(pool)]
    return default


def _encode(tok, text: str) -> list[int]:
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:  # fake tokenizers in tests
        return list(tok.encode(text))


# ------------------------------------------------------------------------------------ dataset
def build_dataset(tok, rows: list[dict], template: str, *, thinking: bool, think_open: str,
                  think_close: str, end_token: str, end_id: int | None, max_tokens: int,
                  variants: dict[str, list[str]] | None = None, system: str | None = None,
                  max_target_id: int | None = None):
    """Returns (examples, skipped) where skipped is a list of (id, reason). Order is preserved."""
    examples, skipped = [], []
    for k, row in enumerate(rows):
        rid = str(row.get("id", f"row{k}"))
        try:
            problem = row.get("problem")
            if not isinstance(problem, str) or not problem.strip():
                raise SkipRow("missing problem")
            completion = build_completion(row, think_open, think_close, end_token)
            # raw rows get the bare user message; the others a wrapper paraphrase chosen by `prompt_variant`
            tmpl = pick_template(row, template, variants)
            p_ids = _encode(tok, render_prompt(tok, tmpl, problem.strip(), thinking, system, row.get("history")))
            c_ids = _encode(tok, completion)
            if end_id is not None and c_ids[-1] != end_id:
                raise SkipRow("completion does not end with the end token after tokenization")
            if max_target_id is not None and max(c_ids) >= max_target_id:
                # the fp32 head only has `max_target_id` rows: a larger target id is an out-of-range gather (CUDA assert)
                raise SkipRow(f"target token id {max(c_ids)} is outside the output head ({max_target_id} rows)")
            if len(p_ids) + len(c_ids) > max_tokens:
                raise SkipRow(f"too long: {len(p_ids) + len(c_ids)} > {max_tokens} tokens")
            examples.append(SFTExample(rid, int(row.get("order", k + 1)), row.get("level"), row.get("domain"),
                                       p_ids, c_ids, completion, answer_style_of(row),
                                       "raw" if tmpl == "{problem}" else (row.get("prompt_variant") or "default"),
                                       "prose" if isinstance(row.get("thinking"), str) else "numbered"))
        except SkipRow as e:
            skipped.append((rid, str(e)))
    return examples, skipped


def _pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def dataset_stats(examples: list[SFTExample]) -> dict:
    if not examples:
        return {"n": 0}
    p = [len(e.prompt_ids) for e in examples]
    c = [len(e.completion_ids) for e in examples]
    return {
        "n": len(examples),
        "prompt_tokens": {"mean": round(statistics.fmean(p), 1), "p50": _pct(p, .5), "p90": _pct(p, .9), "max": max(p)},
        "completion_tokens": {"mean": round(statistics.fmean(c), 1), "p50": _pct(c, .5), "p90": _pct(c, .9), "max": max(c)},
        "trainable_tokens": sum(c),
        "plain_answer_rows": sum(e.answer_style == "plain" for e in examples),
        "by_prompt_variant": dict(Counter(e.prompt_variant for e in examples)),
        "completion_share_of_tokens": round(sum(c) / (sum(c) + sum(p)), 3),
        "by_level": dict(sorted(Counter(e.level for e in examples).items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "by_domain": dict(Counter(e.domain for e in examples).most_common()),
    }


def check_with_rl_parser(tok, examples: list[SFTExample], think_open: str, think_close: str, max_steps: int = 48) -> dict:
    """Parse every target with the RL parser. The share that is hard-valid should be 100%."""
    from .parsing import parse_completion

    n, hard, strict, steps_ok = 0, 0, 0, 0
    errors: Counter = Counter()
    bad: list[str] = []
    for e in examples:
        p = parse_completion(e.completion_ids, tok, think_open=think_open, think_close=think_close,
                             opened_in_prompt=False, finish_reason="stop", max_steps=max_steps,
                             require_box=e.answer_style == "boxed", reasoning_format=e.reasoning_format)
        n += 1
        hard += p.hard_valid
        strict += p.format_valid
        steps_ok += p.spans_ok
        errors.update(p.errors)
        if not p.hard_valid and len(bad) < 5:
            bad.append(f"{e.id}: {p.errors}")
    return {"n": n, "hard_valid": hard, "format_valid": strict, "steps_ok": steps_ok,
            "errors": dict(errors), "first_hard_failures": bad}
