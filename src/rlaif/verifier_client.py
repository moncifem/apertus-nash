"""Trainer-side client of the verifier agent service (reward_service/): the trainer only sends batches of prompt groups and gets verdicts back.

The agent, its prompt and the code sandbox live in their own containers and can be edited while training runs.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

log = logging.getLogger("rlaif.verifier")


@dataclass
class CandVerdict:
    answer: str = ""
    matched: bool | None = None            # trusted: recomputed by the service from numbers printed by sandbox code
    equivalent: str | None = None          # agent-judged label for non-numeric answers: yes|no|partial|unclear
    checks_ok: int = 0
    checks_n: int = 0
    near_miss: bool = False                # numeric answer within 5% but outside its tolerance
    lhs: float | None = None               # the numbers the service compared (for audits)
    rhs: float | None = None

    @property
    def progress(self) -> float | None:
        return self.checks_ok / self.checks_n if self.checks_n >= 2 else None


@dataclass
class GroupVerdict:
    cands: list = field(default_factory=list)      # CandVerdict per candidate, in the order the group was sent
    steps: int = 0


class RemoteVerifier:
    def __init__(self, url: str, timeout_s: float = 1800.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        self.stats: Counter = Counter()
        self._cost = 0.0
        self.version = None

    def health(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            return None

    def cost_usd(self) -> float:
        return self._cost

    def verify_groups(self, jobs: list[dict]) -> list[GroupVerdict | None]:
        """jobs: dicts with problem, gold_reply, gold_answer, cands=[{thinking, reply, finish}]. A group the service could not verify is None."""
        if not jobs:
            return []
        req = urllib.request.Request(self.url + "/verify", json.dumps({"groups": jobs}).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                d = json.load(r)
        except Exception as e:  # noqa: BLE001
            self.stats["unreachable"] += 1
            log.error("verifier service call failed: %r", e)
            return [None] * len(jobs)
        self.version = d.get("version")
        u = d.get("usage", {})
        self._cost += float(u.get("cost_usd", 0.0))
        self.stats["requests"] += 1
        self.stats["groups"] += len(jobs)
        for k in ("calls", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens", "errors"):
            self.stats[k] += int(u.get(k, 0))
        for k, v in (d.get("stats") or {}).items():
            self.stats[k] += int(v)
        out = []
        for res in d.get("results", []):
            if res is None:
                self.stats["failed_groups"] += 1
                out.append(None)
            else:
                out.append(GroupVerdict([CandVerdict(**{k: v for k, v in c.items() if k in CandVerdict.__dataclass_fields__}) for c in res["cands"]], res.get("steps", 0)))
        return out + [None] * (len(jobs) - len(out))
