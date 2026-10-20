"""Parse an Apertus completion (token IDs) into thinking steps + final answer (boxed by default, plain text on request).

Works on token IDs plus decoded character offsets, so every step maps to an exact half-open
span of completion tokens. Pure Python: no torch / transformers import needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

# A step starts at a line beginning with "1." / "1)" / "1:" / "Step 1:" / "**1.**" followed by whitespace.
# "1.5 is ..." does NOT start a step (no whitespace right after the delimiter).
STEP_RE = re.compile(
    r"^[ \t]*(?:\*\*)?(?:step[ \t]*)?(\d{1,3})[.):](?:\*\*)?(?=[ \t]|$)",
    re.IGNORECASE | re.MULTILINE,
)
SPECIAL_RE = re.compile(r"<\|[^<>|\s]{1,64}\|>")


class TokenizerLike(Protocol):
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


# Errors that make a trace unusable for the outcome reward (thinking span / answer extraction).
# Everything else (numbering, preamble, step count) is step *structure*: it only gates the
# step judge and earns a small bonus, because Apertus thinks in free-form prose by default.
HARD_ERRORS = frozenset({
    "think_open_count", "think_close_count", "nested_think_open", "no_thinking", "close_before_open",
    "no_boxed_answer", "multiple_boxed_answers", "empty_answer", "no_answer_line",
})

# Free-form replies end with one machine-readable line:  **Answer:** <result>   (answer_mode="anchor")
ANCHOR_RE = re.compile(r"^[ \t]*\*\*Answer:\*\*[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)


def repetition_fraction(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that already occurred earlier in `text` (degenerate loops -> ~1)."""
    words = text.split()
    if len(words) < n + 8:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@dataclass
class Step:
    id: int
    text: str
    char_start: int
    char_end: int
    tok_start: int = -1
    tok_end: int = -1  # exclusive


@dataclass
class ParsedTrace:
    format_valid: bool
    errors: list[str]
    truncated: bool
    thinking_text: str = ""
    answer_text: str = ""
    final_answer: str | None = None
    steps: list[Step] = field(default_factory=list)
    thinking_tokens: int = 0
    spans_ok: bool = False
    rep_frac: float = 0.0

    @property
    def hard_valid(self) -> bool:
        return not any(e in HARD_ERRORS for e in self.errors)


class _PrefixLen:
    """len(decode(ids[:k])) with memoisation; used to map characters back to tokens."""

    def __init__(self, tok: TokenizerLike, ids: Sequence[int]):
        self.tok, self.ids, self.cache = tok, list(ids), {0: 0}

    def __call__(self, k: int) -> int:
        if k not in self.cache:
            self.cache[k] = len(self.tok.decode(self.ids[:k], skip_special_tokens=False))
        return self.cache[k]

    def token_at(self, char_pos: int) -> int:
        """Index of the token whose decoded text covers char_pos."""
        lo, hi = 0, len(self.ids) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self(mid + 1) > char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo


def extract_boxed(text: str) -> list[str]:
    """All top-level \\boxed{...} contents with balanced braces."""
    out: list[str] = []
    i = 0
    while True:
        k = text.find("\\boxed", i)
        if k < 0:
            return out
        j = k + len("\\boxed")
        while j < len(text) and text[j] in " \t":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = j
            continue
        depth, start = 0, j + 1
        end = -1
        for p in range(j, len(text)):
            c = text[p]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = p
                    break
        if end < 0:  # unbalanced -> not an answer
            return out
        out.append(text[start:end])
        i = end + 1


def parse_completion(
    completion_ids: Sequence[int],
    tokenizer: TokenizerLike,
    *,
    think_open: str,
    think_close: str,
    opened_in_prompt: bool,
    finish_reason: str,
    max_steps: int,
    require_box: bool = True,
    answer_mode: str = "boxed",
    reasoning_format: str = "numbered",
) -> ParsedTrace:
    if reasoning_format not in ("numbered", "prose"):
        raise ValueError(f"Unknown reasoning format: {reasoning_format}")
    ids = list(completion_ids)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    truncated = finish_reason == "length"
    errors: list[str] = []

    # --- locate thinking span -------------------------------------------------------------
    if opened_in_prompt:
        t_start = 0
        close_at = text.find(think_close)
        head = text if close_at < 0 else text[:close_at]
        if think_open in head:
            errors.append("nested_think_open")
    else:
        n_open = text.count(think_open)
        if n_open != 1:
            errors.append("think_open_count")
        t_start = text.find(think_open) + len(think_open) if n_open >= 1 else -1

    n_close = text.count(think_close)
    if n_close != 1:
        errors.append("think_close_count")
    if t_start < 0 or n_close == 0:
        return ParsedTrace(False, errors or ["no_thinking"], truncated)
    t_end = text.find(think_close, t_start)
    if t_end < 0:
        errors.append("close_before_open")
        return ParsedTrace(False, errors, truncated)

    thinking = text[t_start:t_end]
    if think_open in thinking and "nested_think_open" not in errors:
        errors.append("nested_think_open")
    answer = SPECIAL_RE.sub("", text[t_end + len(think_close):]).strip()

    # --- final answer ---------------------------------------------------------------------
    # require_box=True  (math / RL rows): exactly one \\boxed{} after the thinking span.
    # require_box=False (general questions, e.g. identity): the whole text after the thinking span is the answer.
    final: str | None = None
    if answer_mode == "anchor":
        # the LAST '**Answer:** x' line of the reply is the result (surrounding $ and ** markers are stripped)
        found = ANCHOR_RE.findall(answer)
        if not found:
            errors.append("no_answer_line")
        else:
            final = found[-1].strip().strip("*").strip().strip("$").strip() or None
            if final is None:
                errors.append("empty_answer")
    elif require_box:
        boxes = extract_boxed(answer)
        if not boxes:
            errors.append("no_boxed_answer")
        elif len(boxes) > 1:
            errors.append("multiple_boxed_answers")
        else:
            final = boxes[0].strip() or None
            if final is None:
                errors.append("empty_answer")
    else:
        final = answer or None
        if final is None:
            errors.append("empty_answer")

    # --- reasoning units: numbered legacy traces or unnumbered prose -----------------------
    steps: list[Step] = []
    matches = list(STEP_RE.finditer(thinking))
    if reasoning_format == "prose":
        if matches:
            errors.append("numbered_prose")
        paragraphs = list(re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", thinking))
        if not paragraphs:
            errors.append("no_steps")
        if len(paragraphs) > max_steps:
            errors.append("too_many_steps")
        for k, paragraph in enumerate(paragraphs, 1):
            steps.append(Step(k, paragraph.group().strip(),
                              t_start + paragraph.start(), t_start + paragraph.end()))
    elif not matches:
        errors.append("no_steps")
    else:
        # Prose before step 1 is allowed: its tokens simply get no process weight.
        if len(matches) > max_steps:
            errors.append("too_many_steps")
        for k, m in enumerate(matches):
            s = m.start()
            e = matches[k + 1].start() if k + 1 < len(matches) else len(thinking)
            body = thinking[m.end():e].strip()
            num = int(m.group(1))
            if num != k + 1:
                errors.append("bad_numbering")
                break
            if not body or not re.search(r"\w", body):
                errors.append("empty_step")
                break
            steps.append(Step(num, body, t_start + s, t_start + e))

    # --- character -> token mapping --------------------------------------------------------
    pl = _PrefixLen(tokenizer, ids)
    tok_think_start = pl.token_at(t_start) if t_start < len(text) else len(ids)
    tok_close = pl.token_at(t_end)
    thinking_tokens = max(0, tok_close - tok_think_start)

    spans_ok = bool(steps) and "bad_numbering" not in errors and "empty_step" not in errors
    if spans_ok:
        starts = [pl.token_at(st.char_start) for st in steps]
        ends = starts[1:] + [tok_close]
        for st, a, b in zip(steps, starts, ends):
            st.tok_start, st.tok_end = a, b
            if b <= a:
                spans_ok = False

    return ParsedTrace(
        format_valid=not errors,
        errors=errors,
        truncated=truncated,
        thinking_text=thinking,
        answer_text=answer,
        final_answer=final,
        steps=steps,
        thinking_tokens=thinking_tokens,
        spans_ok=spans_ok,
        rep_frac=repetition_fraction(thinking),
    )
