"""Outcome rewards, conciseness reward, group advantages, and per-token process targets.

Pure Python so it can be unit-tested without a GPU.
"""
from __future__ import annotations

from typing import Sequence


def ramp(update: int, start: int, n: int, weight: float) -> float:
    """0 before `start`, then linear to `weight` over `n` updates."""
    if update < start or weight == 0:
        return 0.0
    return weight * min(1.0, (update - start + 1) / max(1, n))


def kimi_length_rewards(lengths: Sequence[int], correct: Sequence[bool]) -> list[float]:
    """Group-relative conciseness signal (Kimi k1.5): shortest -> +0.5, longest -> -0.5.

    Correct answers get the full value (so shorter correct reasoning is preferred);
    incorrect answers only ever receive the negative part (long wrong answers are penalised,
    short wrong answers are never rewarded).
    """
    if not lengths:
        return []
    lo, hi = min(lengths), max(lengths)
    if hi == lo:
        return [0.0] * len(lengths)
    out = []
    for L, c in zip(lengths, correct):
        lam = 0.5 - (L - lo) / (hi - lo)
        out.append(lam if c else min(0.0, lam))
    return out


def repetition_scale(frac: float, lo: float = 0.2, hi: float = 0.6) -> float:
    """0 below `lo` repeated-4-gram fraction, linearly up to 1 at `hi` (natural re-checking stays free)."""
    return min(1.0, max(0.0, (frac - lo) / (hi - lo)))


def outcome_rewards(
    correct: Sequence[bool],
    format_valid: Sequence[bool],
    truncated: Sequence[bool],
    thinking_tokens: Sequence[int],
    *,
    format_penalty: float,
    truncation_penalty: float,
    lam_len: float,
    step_ok: Sequence[bool] | None = None,
    rep_frac: Sequence[float] | None = None,
    step_bonus: float = 0.0,
    rep_penalty: float = 0.0,
) -> tuple[list[float], list[float]]:
    """r_i = c_i - pen_format - pen_trunc + step_bonus*[steps ok] - rep_penalty*rep_i + lam_len * length_term_i.

    `format_valid` here means *hard* validity (thinking span + single boxed answer).

    The length term is computed only among completed, well-formed members of the group so that
    thinking-token counts are comparable.
    """
    n = len(correct)
    eligible = [i for i in range(n) if format_valid[i] and not truncated[i]]
    lterm = [0.0] * n
    for i, v in zip(eligible, kimi_length_rewards([thinking_tokens[i] for i in eligible],
                                                  [correct[i] for i in eligible])):
        lterm[i] = v
    rewards = []
    for i in range(n):
        r = 1.0 if correct[i] else 0.0
        if truncated[i]:
            r -= truncation_penalty
        elif not format_valid[i]:
            r -= format_penalty
        if step_bonus and step_ok is not None and step_ok[i] and format_valid[i] and not truncated[i]:
            r += step_bonus
        if rep_penalty and rep_frac is not None:
            r -= rep_penalty * repetition_scale(rep_frac[i])
        r += lam_len * lterm[i]
        rewards.append(r)
    return rewards, lterm


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Dr. GRPO: mean-centred, NO std scaling."""
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def process_token_targets(
    spans: Sequence[tuple[int, int]], qs: Sequence[float], n_tokens: int
) -> tuple[list[float], list[float]]:
    """Per-token step reward q_t and weight w_t such that

        sum_t w_t * S(rho_t, q_t) == sum_j mean_{t in step j} S(rho_t, q_j) / n_steps

    i.e. each step's tokens are averaged, then steps are averaged. Adding/splitting steps cannot
    increase the total weight (sum_t w_t == 1 for any non-empty trace). Tokens outside the
    thinking steps (answer, EOS) get weight 0.
    """
    q = [0.0] * n_tokens
    w = [0.0] * n_tokens
    n_steps = len(spans)
    if n_steps == 0:
        return q, w
    for (s, e), qq in zip(spans, qs):
        if not (0 <= s < e <= n_tokens):
            raise ValueError(f"bad step span {(s, e)} for {n_tokens} tokens")
        per = 1.0 / ((e - s) * n_steps)
        for t in range(s, e):
            q[t] = qq
            w[t] = per
    return q, w
