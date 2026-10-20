"""Frozen step judge (Qwen3-4B-Instruct-2507 served by vLLM, OpenAI-compatible API).

Prefix-only requests: problem + previous steps + ONE target step. The judge never sees later
steps, the final answer, the gold answer, or correctness. Verdicts are schema-validated and
cached in SQLite (WAL). Connection failures after retries raise JudgeUnavailable (training
pauses; scores are never fabricated). Malformed outputs after retries leave that step's
verdict as None (zero process signal) and are counted; too many in a batch also pauses.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1


class StepVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: int
    validity: Literal["valid", "invalid", "uncertain"]
    relevance: Literal["useful", "irrelevant", "uncertain"]
    redundant: bool
    depends_on_error: bool
    evidence: str = Field(max_length=200)


VERDICT_SCHEMA = StepVerdict.model_json_schema()


class JudgeUnavailable(RuntimeError):
    pass


def verdict_to_q(v: StepVerdict | None, rc) -> float:
    """Deterministic verdict -> step reward. Invalidity takes precedence over style."""
    if v is None:
        return rc.q_uncertain
    if v.validity == "invalid":
        return rc.q_invalid           # new local error
    if v.validity == "uncertain" or v.depends_on_error:
        return rc.q_uncertain         # abstain / pure propagation of an earlier error
    if v.redundant or v.relevance == "irrelevant":
        return rc.q_redundant         # valid but wasteful -> conciseness pressure
    if v.relevance == "uncertain":
        return rc.q_uncertain
    return rc.q_valid


def _neutralize(s: str) -> str:
    # Untrusted text must not be able to close our XML-ish delimiters.
    return s.replace("</", "<\\/").replace("<step", "<\\step").replace("<problem", "<\\problem")


def build_messages(rubric: str, problem: str, previous: list[str], target_id: int, target: str) -> list[dict]:
    prev = "\n".join(f"<step id=\"{i + 1}\">{_neutralize(t)}</step>" for i, t in enumerate(previous)) or "(none)"
    user = (
        f"<problem>\n{_neutralize(problem)}\n</problem>\n"
        f"<previous_steps>\n{prev}\n</previous_steps>\n"
        f"<target_step id=\"{target_id}\">\n{_neutralize(target)}\n</target_step>\n\n"
        f"Return the JSON verdict for target step {target_id} only."
    )
    return [{"role": "system", "content": rubric}, {"role": "user", "content": user}]


def _strip_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


class StepJudge:
    def __init__(self, jc, rc, rubric_text: str, tokenizer, model_sha: str):
        self.cfg, self.rc, self.rubric, self.tok = jc, rc, rubric_text, tokenizer
        self.rubric_sha = hashlib.sha256(rubric_text.encode()).hexdigest()
        self.model_sha = model_sha
        self.db = sqlite3.connect(jc.cache_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, verdict TEXT, created REAL)")
        self.db.commit()
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ readiness
    def wait_ready(self) -> None:
        deadline = time.time() + self.cfg.ready_timeout_s
        last = None
        while time.time() < deadline:
            try:
                r = httpx.get(self.cfg.base_url.rstrip("/") + "/models", timeout=10)
                if r.status_code == 200 and any(m.get("id") == self.cfg.served_model_name
                                                for m in r.json().get("data", [])):
                    return
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as e:
                last = repr(e)
            time.sleep(5)
        raise JudgeUnavailable(f"judge not ready at {self.cfg.base_url}: {last}")

    # ------------------------------------------------------------------ cache
    def _key(self, problem: str, previous: list[str], target_id: int, target: str) -> str:
        payload = json.dumps({
            "schema": SCHEMA_VERSION, "model": self.cfg.hf_id, "model_sha": self.model_sha,
            "rubric": self.rubric_sha, "temp": self.cfg.temperature, "max_out": self.cfg.max_output_tokens,
            "problem": problem, "previous": previous, "id": target_id, "target": target,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get(self, key: str) -> StepVerdict | None:
        row = self.db.execute("SELECT verdict FROM verdicts WHERE key=?", (key,)).fetchone()
        return StepVerdict.model_validate_json(row[0]) if row else None

    def _put(self, key: str, v: StepVerdict) -> None:
        self.db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?)", (key, v.model_dump_json(), time.time()))

    # ------------------------------------------------------------------ public API
    def judge_batch(self, items: list[tuple[str, list[str]]]) -> list[list[StepVerdict | None]]:
        """items: [(problem_text, [step_text, ...]), ...] -> verdicts aligned with steps."""
        before = Counter(self.stats)
        res = asyncio.run(self._judge_batch(items))
        self.db.commit()
        n = sum(len(s) for _, s in items)
        bad = self.stats["malformed"] - before["malformed"]
        if n and bad / n > self.cfg.max_malformed_fraction:
            raise JudgeUnavailable(f"{bad}/{n} judge responses malformed after retries")
        return res

    async def _judge_batch(self, items):
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        results = [[None] * len(steps) for _, steps in items]
        async with httpx.AsyncClient(base_url=self.cfg.base_url, timeout=self.cfg.timeout_s) as client:
            index, coros = [], []
            for i, (problem, steps) in enumerate(items):
                for j in range(len(steps)):
                    index.append((i, j))
                    coros.append(self._judge_step(client, sem, problem, steps[:j], j + 1, steps[j]))
            outs = await asyncio.gather(*coros)
        for (i, j), v in zip(index, outs):
            results[i][j] = v
        return results

    async def _judge_step(self, client, sem, problem, previous, target_id, target):
        key = self._key(problem, previous, target_id, target)
        hit = self._get(key)
        if hit is not None:
            self.stats["cache_hit"] += 1
            return hit
        messages = build_messages(self.rubric, problem, previous, target_id, target)
        prompt_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_in = len(self.tok(prompt_text, add_special_tokens=False).input_ids)
        if n_in + self.cfg.max_output_tokens > self.cfg.max_context_tokens:
            self.stats["context_overflow"] += 1  # never truncate premises; step gets no signal
            return None
        body = {
            "model": self.cfg.served_model_name,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if self.cfg.use_json_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "step_verdict", "schema": VERDICT_SCHEMA, "strict": True}}
        connected, last_err = False, None
        for attempt in range(self.cfg.retries + 1):
            if attempt:
                await asyncio.sleep(min(30.0, 2.0 ** attempt))
            async with sem:
                t0 = time.time()
                try:
                    r = await client.post("/chat/completions", json=body)
                except httpx.HTTPError as e:
                    last_err = repr(e)
                    self.stats["http_error"] += 1
                    continue
                self.stats["latency_ms_sum"] += int((time.time() - t0) * 1000)
            if r.status_code >= 500 or r.status_code == 429:
                last_err = f"HTTP {r.status_code}"
                continue
            if r.status_code >= 400:
                raise JudgeUnavailable(f"judge rejected request: HTTP {r.status_code}: {r.text[:300]}")
            connected = True
            self.stats["calls"] += 1
            try:
                content = r.json()["choices"][0]["message"]["content"] or ""
                v = StepVerdict.model_validate_json(_strip_fence(content))
            except (ValidationError, KeyError, IndexError, ValueError):
                self.stats["retry_malformed"] += 1
                continue
            if v.step_id != target_id:
                self.stats["retry_wrong_id"] += 1
                continue
            self._put(key, v)
            self.stats[f"validity_{v.validity}"] += 1
            return v
        if not connected:
            raise JudgeUnavailable(f"judge unreachable after {self.cfg.retries + 1} attempts: {last_err}")
        self.stats["malformed"] += 1
        return None
