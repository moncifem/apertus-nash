"""Programmatic 'method signatures' for the novelty bonus (no LLM, so it cannot be talked into anything).

A solution's signature is a MinHash sketch of the content-word 3-grams of its thinking: numbers, single-letter variables and filler
words are dropped, so two solutions that use the same method on different numbers look alike while a different technique does not.
novelty = 1 - (largest similarity to the other solutions of the group) scaled by 1 - (largest similarity to the archive of earlier winners).
The bonus is only paid to verified or on-track candidates (reward_v2), so noise or gibberish cannot earn it.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

N_HASH = 64
_P = (1 << 61) - 1
_A = [(0x9E3779B97F4A7C15 * (i + 1)) % _P | 1 for i in range(N_HASH)]
_B = [(0xC2B2AE3D27D4EB4F * (i + 7)) % _P for i in range(N_HASH)]
STOP = frozenset("the and for that this with from are was were then than there their which have has had not but can will would should could "
                 "need first next now also just into onto over under what when where while because since thus hence therefore note recall "
                 "answer check step use using used get got let say so we it its is be of to in on at by as an or if".split())


def words(text: str) -> list[str]:
    t = re.sub(r"\d+(\.\d+)?", " ", (text or "").lower())
    return [w for w in re.findall(r"[a-zα-ω][a-zα-ω'\-]{2,}", t) if w not in STOP]


def signature(text: str, k: int = 3) -> tuple[int, ...] | None:
    w = words(text)
    if len(w) < k + 4:
        return None
    sh = {zlib.crc32(" ".join(w[i:i + k]).encode()) for i in range(len(w) - k + 1)}
    return tuple(min((a * h + b) % _P for h in sh) for a, b in zip(_A, _B))


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b)) / N_HASH


def group_novelty(sigs: list) -> list[float | None]:
    """For each signature: 1 - max similarity to the others (None when it has no signature; 0.5 when it is alone)."""
    out = []
    for i, s in enumerate(sigs):
        if s is None:
            out.append(None)
            continue
        others = [similarity(s, t) for j, t in enumerate(sigs) if j != i and t is not None]
        out.append(1.0 - max(others) if others else 0.5)
    return out


class Archive:
    """Signatures of earlier verified solutions per domain (bounded). A method that was already rewarded stops being novel."""

    def __init__(self, path: str | None = None, cap: int = 1500):
        self.path, self.cap, self.data = path, cap, {}
        if path and Path(path).exists():
            try:
                self.data = {d: [tuple(s) for s in v] for d, v in json.loads(Path(path).read_text()).items()}
            except Exception:  # noqa: BLE001
                self.data = {}

    def novelty(self, domain: str, sig) -> float:
        if sig is None:
            return 1.0
        sims = [similarity(sig, s) for s in self.data.get(domain, [])]
        return 1.0 - max(sims) if sims else 1.0

    def add(self, domain: str, sig) -> None:
        if sig is None:
            return
        lst = self.data.setdefault(domain, [])
        lst.append(sig)
        if len(lst) > self.cap:
            del lst[: len(lst) - self.cap]

    def save(self) -> None:
        if self.path:
            Path(self.path).write_text(json.dumps({d: [list(s) for s in v] for d, v in self.data.items()}))
