"""DeepSeek as the reward function: one integer score in [-100, 100] per rollout (prompts/judge/deepseek_judge_rl.md), reward = score / 100.

The key is read from a file (a docker secret), never from the config or the logs. A failed call is retried; if it still fails the score is
None and the trainer drops that prompt group instead of guessing a reward.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .judge_items import items_to_reward, parse_items

log = logging.getLogger("rlaif.deepseek")
INT_RE = re.compile(r"-?\d+")


def split_completion(text: str, think_open: str, think_close: str) -> tuple[str, str]:
    """(thinking, reply) from a decoded completion; special tokens are stripped, nothing else is changed."""
    body = text.replace("<|assistant_end|>", "")
    if think_close in body:
        th, reply = body.split(think_close, 1)
    else:
        th, reply = body, ""
    return th.replace(think_open, "").strip(), reply.strip()


def render_input(problem: str, reference: str | None, thinking: str, reply: str, finish: str) -> str:
    return ("PROBLEM:\n" + problem + "\n\nGROUND TRUTH (reference solution, treat as correct unless clearly impossible): " + (reference or "(none)")
            + "\n\nCANDIDATE THINKING:\n" + (thinking or "(empty)") + "\n\nCANDIDATE REPLY:\n" + (reply or "(empty)")
            + "\n\nFINISH: " + finish + "\n\nReturn only the integer score.")


def clip_middle(text: str, n: int) -> str:
    """Keep the head (2/3) and the tail (1/3) of an over-long text, marking the cut."""
    if n <= 0 or len(text) <= n:
        return text
    h = n * 2 // 3
    return text[:h] + f"\n[... {len(text) - n} characters omitted ...]\n" + text[-(n - h):]


def render_group(problem: str, reference: str | None, cands: list[dict], max_thinking_chars: int = 0) -> str:
    out = "PROBLEM:\n" + problem + "\n\nREFERENCE: " + (reference or "(none)") + "\n\nCANDIDATES:\n"
    for i, c in enumerate(cands, 1):
        out += f"[{i}] FINISH: {c['finish']}\nTHINKING:\n" + (clip_middle(c["thinking"], max_thinking_chars) or "(empty)") + "\nREPLY:\n" + (c["reply"] or "(empty)") + "\n\n"
    return out + "Return only the lines."


def parse_score(text: str | None) -> float | None:
    """Integer in [-100, 100] -> reward in [-1, 1]; anything else is a failed call (None)."""
    if not text:
        return None
    m = INT_RE.search(text.strip().split()[0].strip("`*") if text.strip() else "")
    if not m:
        return None
    v = int(m.group(0))
    return max(-100, min(100, v)) / 100.0


class DeepSeekScorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key = Path(cfg.deepseek_key_path).read_text().strip()
        self.system = Path(cfg.deepseek_prompt_path).read_text()
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    def _call(self, user: str, max_tokens: int | None = None, validator=None) -> str | None:
        c = self.cfg
        validator = validator or (lambda t: parse_score(t) is not None)
        body = {"model": c.deepseek_model, "temperature": 0, "max_tokens": max_tokens or c.deepseek_max_tokens, "reasoning_effort": c.deepseek_effort,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(c.deepseek_base_url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        for attempt in range(c.retries):
            try:
                with urllib.request.urlopen(req, timeout=c.timeout_s) as r:
                    d = json.load(r)
                with self._lock:
                    u = d.get("usage", {})
                    self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.stats["completion_tokens"] += u.get("completion_tokens", 0)
                    self.stats["calls"] += 1
                    self.stats["cache_hit_tokens"] += u.get("prompt_cache_hit_tokens", 0)
                    self.stats["cache_miss_tokens"] += u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - u.get("prompt_cache_hit_tokens", 0))
                txt = d["choices"][0]["message"].get("content")
                if txt and validator(txt):
                    return txt
                with self._lock:
                    self.stats["unparseable"] += 1
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as e:
                with self._lock:
                    self.stats["errors"] += 1
                log.warning("deepseek call failed (attempt %d): %r", attempt + 1, e)
            time.sleep(2 ** attempt)
        return None

    def cost_usd(self) -> float:
        """Estimated spend of this process at the configured (peak, i.e. conservative) prices, per 1M tokens."""
        c, st = self.cfg, self.stats
        return (st["cache_hit_tokens"] * c.price_hit + st["cache_miss_tokens"] * c.price_miss + st["completion_tokens"] * c.price_out) / 1e6

    def balance_usd(self) -> float | None:
        """Live account balance from the API (None if the call fails)."""
        req = urllib.request.Request(self.cfg.deepseek_base_url.rstrip("/") + "/user/balance", headers={"Authorization": "Bearer " + self.key})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return float(json.load(r)["balance_infos"][0]["total_balance"])
        except Exception:  # noqa: BLE001
            return None

    def _call_group(self, user: str, n: int) -> list | None:
        """One chat call for n candidates; asks again once if any line is missing its core fields."""
        for _ in range(2):
            txt = self._call(user, max_tokens=70 * n + 40, validator=lambda t: all(i is not None for i in parse_items(t, n)))
            if txt is not None:
                return parse_items(txt, n)
        return None

    def score_groups(self, groups: list[dict]):
        """groups: dicts with problem, reference and cands=[{thinking, reply, finish}]. ONE call per group (the problem, reference and
        instructions are sent once, not once per sample). Returns (rewards, items): per group a list of rewards in [-1, 1] or None."""
        def one(g):
            user = render_group(g["problem"], g.get("reference"), g["cands"], self.cfg.max_thinking_chars)
            items = self._call_group(user, len(g["cands"]))
            if items is None:
                return [None] * len(g["cands"]), [None] * len(g["cands"])
            return [items_to_reward(i) for i in items], items
        with ThreadPoolExecutor(self.cfg.max_concurrency) as ex:
            res = list(ex.map(one, groups))
        return [r for r, _ in res], [i for _, i in res]

    def score_batch(self, jobs: list[dict]) -> list[float | None]:
        """jobs: dicts with problem, reference, thinking, reply, finish. Returns rewards in [-1, 1] (None = failed call)."""
        users = [render_input(j["problem"], j.get("reference"), j["thinking"], j["reply"], j["finish"]) for j in jobs]
        with ThreadPoolExecutor(self.cfg.max_concurrency) as ex:
            outs = list(ex.map(self._call, users))
        return [parse_score(o) for o in outs]
