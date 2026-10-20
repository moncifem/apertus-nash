"""Final-answer verification with Math-Verify in spawned, time-bounded worker processes.

Statuses:
  ok                 -> verification ran; bool is the verdict
  gold_unparseable   -> data failure (whole group is excluded)
  timeout / error:*  -> operational failure (whole group is excluded)
Generated text is only parsed by Math-Verify's LaTeX/expression grammar; nothing is eval'ed.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

OPERATIONAL_PREFIXES = ("timeout", "error", "gold_unparseable")


def _verify_one(gold: str, pred: str) -> tuple[str, bool]:
    try:
        from .structured import GOLD_PREFIX, check
        if gold.startswith(GOLD_PREFIX):
            return check(gold, pred)
        from math_verify import parse, verify

        g = parse("\\boxed{" + gold + "}")
        if not g:
            return ("gold_unparseable", False)
        p = parse("\\boxed{" + pred + "}")
        if not p:
            return ("ok", False)  # model output the parser rejects counts as incorrect
        return ("ok", bool(verify(g, p)))
    except Exception as e:  # noqa: BLE001
        return ("error:" + type(e).__name__, False)


def _gold_ok(gold: str) -> bool:
    try:
        from .structured import GOLD_PREFIX, decode_gold
        if gold.startswith(GOLD_PREFIX):
            decode_gold(gold)
            return True
        from math_verify import parse

        return bool(parse("\\boxed{" + gold + "}"))
    except Exception:  # noqa: BLE001
        return False


def is_operational_failure(status: str) -> bool:
    return status.startswith(OPERATIONAL_PREFIXES)


class Verifier:
    def __init__(self, workers: int = 4, timeout_s: float = 10.0):
        self.workers, self.timeout_s = workers, timeout_s
        self._new_pool()

    def _new_pool(self) -> None:
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))

    def _run(self, fn, args_list):
        futs = [self.pool.submit(fn, *a) for a in args_list]
        out, broken = [], False
        for f in futs:
            try:
                out.append(f.result(timeout=self.timeout_s))
            except FutureTimeout:
                out.append(None)
                broken = True
            except Exception:  # noqa: BLE001  (e.g. BrokenProcessPool)
                out.append(None)
                broken = True
        if broken:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self._new_pool()
        return out

    def check(self, pairs: list[tuple[str, str]]) -> list[tuple[str, bool]]:
        if not pairs:
            return []
        res = self._run(_verify_one, pairs)
        return [r if r is not None else ("timeout", False) for r in res]

    def gold_ok(self, golds: list[str]) -> list[bool]:
        res = self._run(_gold_ok, [(g,) for g in golds])
        return [bool(r) for r in res]

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
