"""Trainer-side client of reward service v3 (reward_service/reward.py): the trainer sends raw groups and applies the finished rewards.

All scoring (correctness, format, thinking judge, bonus, penalties, weights) lives in the service container and can be edited while training runs.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from collections import Counter

log = logging.getLogger("rlaif.reward")


class RewardClient:
    def __init__(self, url: str, timeout_s: float = 900.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        self.stats: Counter = Counter()
        self._cost = 0.0
        self.version = None

    def health(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url + "/reward/health", timeout=5) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            return None

    def cost_usd(self) -> float:
        return self._cost

    def score_groups(self, jobs: list[dict]) -> list[dict | None]:
        """jobs: {problem, gold, domain, cands:[{thinking, reply, truncated, hard_valid, ntok}]} -> per group {"rewards", "parts"} or None."""
        if not jobs:
            return []
        req = urllib.request.Request(self.url + "/reward", json.dumps({"groups": jobs}).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                d = json.load(r)
        except Exception as e:  # noqa: BLE001
            self.stats["unreachable"] += 1
            log.error("reward service call failed: %r", e)
            return [None] * len(jobs)
        self.version = d.get("version")
        u = d.get("usage", {})
        self._cost += float(u.get("cost_usd", 0.0))
        self.stats["requests"] += 1
        self.stats["groups"] += len(jobs)
        for k in ("calls", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens"):
            self.stats[k] += int(u.get(k, 0))
        for k, v in (d.get("stats") or {}).items():
            self.stats[k] += int(v)
        res = list(d.get("results") or [])
        res = [r if (r and len(r.get("rewards", [])) == len(j["cands"])) else None for r, j in zip(res, jobs)]
        self.stats["failed_groups"] += sum(r is None for r in res)
        return res + [None] * (len(jobs) - len(res))


def apply_rewards(groups, results) -> int:
    """Write the service's rewards and parts onto the trajectories (groups[i] is a list of Traj). Groups without a result get NaN rewards
    (the caller drops them). Returns the number of usable groups."""
    usable = 0
    for g, res in zip(groups, results):
        if res is None:
            for t in g:
                t.reward = float("nan")
            continue
        usable += 1
        for t, r, p in zip(g, res["rewards"], res["parts"]):
            t.reward = float(r)
            t.correct = bool(p.get("correct"))
            t.verify_status = "v3"
            t.reward_parts = p
    return usable
