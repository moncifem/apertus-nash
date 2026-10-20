"""Reward service v3 (hot-reloaded by server.py): THE AGENT DECIDES THE REWARD.

There are no reward weights, penalties or formulas in this file. The rubric (what counts as correct, the thinking score T, the bonus, the numbers
for correct / wrong / truncated / attack answers) is written in reward_prompt.md. A DeepSeek agent reads one prompt group (the problem, the verified
reference, all candidates and a few FACTS this module measures exactly: the exact grader's result check, the repetition numbers of the thinking,
the reply defects) and submits one {correct, format, T, bonus, reward, why} per candidate with a single `finish` call. It writes no code.
This module only
  * measures the facts and builds the agent's input, then runs the (budgeted, repaired when malformed) call,
  * validates the submission (every candidate once, numbers finite and inside reward_min..reward_max from reward_config.json),
  * returns the agent's numbers unchanged. A group the agent cannot score is returned as null: the trainer drops it, a reward is never guessed.
For auditing only (never used in the reward) the exact structured-answer grader's verdict is attached to each candidate as `exact_grader`.
"""
from __future__ import annotations

import collections
import json
import math
import re
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "finish", "description": "Submit the final scores of all candidates.",
     "parameters": {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {
         "i": {"type": "integer"}, "correct": {"type": "boolean"}, "format": {"type": "boolean"}, "T": {"type": "number"},
         "bonus": {"type": "number"}, "reward": {"type": "number"}, "why": {"type": "string"}}, "required": ["i", "reward"]}}},
                    "required": ["scores"]}}},
]


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cost_usd(cfg: dict, usage: dict) -> float:
    return (usage.get("cache_hit_tokens", 0) * cfg["price_hit"] + usage.get("cache_miss_tokens", 0) * cfg["price_miss"]
            + usage.get("completion_tokens", 0) * cfg["price_out"]) / 1e6


def chat(cfg: dict, key: str, messages: list, usage: dict):
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": cfg["max_tokens_per_turn"], "reasoning_effort": cfg["reasoning_effort"],
            "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
    for attempt in range(cfg["retries"]):
        try:
            d = _post(cfg["base_url"].rstrip("/") + "/chat/completions", body, {"Authorization": "Bearer " + key}, cfg["timeout_s"])
            u = d.get("usage", {})
            hit = u.get("prompt_cache_hit_tokens", 0)
            usage["calls"] = usage.get("calls", 0) + 1
            usage["completion_tokens"] = usage.get("completion_tokens", 0) + u.get("completion_tokens", 0)
            usage["cache_hit_tokens"] = usage.get("cache_hit_tokens", 0) + hit
            usage["cache_miss_tokens"] = usage.get("cache_miss_tokens", 0) + u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - hit)
            return d["choices"][0]["message"], u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        except Exception as e:  # noqa: BLE001
            usage["errors"] = usage.get("errors", 0) + 1
            usage["last_error"] = f"{type(e).__name__}: {e}"[:200]
            time.sleep(min(8, 2 ** attempt))
    return None


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n * 2 // 3] + f"\n[... {len(s) - n} characters omitted ...]\n" + s[-(n - n * 2 // 3):]


def _fence(text: str) -> str:
    return (text or "").replace("<candidate", "<_candidate").replace("</candidate", "</_candidate")


def repetition(thinking: str):
    """(r, m): r = larger of the share of lines/sentences that repeat an earlier one and the share of 8-word windows that already appeared;
    m = the largest number of lines/sentences that open with the same two words. Units under 4 words are ignored."""
    text = thinking or ""
    units = [u for u in re.split(r"\n+", text) if u.strip()]
    if len(units) < 4:
        units = re.split(r"(?<=[.!?])\s+", text)
    norm = [" ".join(u.lower().split()) for u in units]
    norm = [u for u in norm if len(u.split()) >= 4]
    if not norm:
        return 0.0, 1
    seen, dup = set(), 0
    for u in norm:
        if u in seen:
            dup += 1
        seen.add(u)
    words = " ".join(text.lower().split()).split()
    gram = 0.0
    if len(words) >= 8:
        grams = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
        seen_g, d = set(), 0
        for g in grams:
            if g in seen_g:
                d += 1
            seen_g.add(g)
        gram = d / len(grams)
    opens = collections.Counter(" ".join(u.split()[:2]) for u in norm)
    return round(max(dup / len(norm), gram), 2), max(opens.values())


def reply_facts(reply: str) -> dict:
    reply = reply or ""
    cut = reply.split("Result JSON")[0] if "Result JSON" in reply else reply.split("{")[0]
    return {"marker": reply.count("<|inner_suffix|>") + reply.count("<|inner_prefix|>"), "blocks": reply.count("Result JSON"), "prose_words": len(cut.split())}


def exact_results(job: dict) -> list:
    """The exact structured-answer grader per candidate: {"status": match|mismatch|no_json, "wrong": [...]} or None (not graded / unavailable / too slow)."""
    import subprocess
    import sys
    from pathlib import Path
    n = len(job["cands"])
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).with_name("exact_cli.py"))], capture_output=True, text=True, timeout=40,
                           input=json.dumps({"gold": job.get("gold") or "", "replies": [None if c.get("truncated") else (c.get("reply") or "") for c in job["cands"]]}))
        out = json.loads(p.stdout.strip().splitlines()[-1])
        return out if isinstance(out, list) and len(out) == n else [None] * n
    except Exception:  # noqa: BLE001
        return [None] * n


def facts_line(c: dict, ex) -> str:
    if c.get("truncated"):
        return "FACTS: truncated"
    if ex is None:
        check = "UNKNOWN (compare the reply's result with the REFERENCE yourself)"
    elif ex["status"] == "match":
        check = "MATCH"
    elif ex["status"] == "mismatch":
        check = "MISMATCH (wrong fields: " + ", ".join(ex["wrong"]) + ")"
    else:
        check = "NO_JSON (no result object found: compare the result stated in the reply with the REFERENCE yourself)"
    r, m = repetition(c.get("thinking") or "")
    rf = reply_facts(c.get("reply") or "")
    return f"FACTS: result_check={check}; thinking_repetition r={r} m={m}; reply marker_count={rf['marker']} answer_blocks={rf['blocks']} prose_words={rf['prose_words']}"


def first_message(cfg: dict, job: dict, exact=None) -> str:
    out = ("PROBLEM:\n" + _clip(job["problem"], cfg["problem_chars"]) + "\n\nREFERENCE (verified result):\n" + _clip(reference_text(job.get("gold")), cfg["reference_chars"])
           + f"\n\n{len(job['cands'])} CANDIDATES:\n")
    exact = exact or [None] * len(job["cands"])
    for i, (c, ex) in enumerate(zip(job["cands"], exact), 1):
        fin = "length" if c.get("truncated") else "stop"
        out += (f"\n<candidate {i} finish={fin}>\n" + facts_line(c, ex) + "\nTHINKING:\n" + _fence(_clip(c.get("thinking") or "(empty)", cfg["thinking_chars"]))
                + "\nREPLY:\n" + _fence(_clip(c.get("reply") or "(empty)", cfg["reply_chars"])) + f"\n</candidate {i}>\n")
    return out


def reference_text(gold) -> str:
    """The verified result shown to the agent: the reference JSON for structured-answer rows, the raw reference text otherwise."""
    gold = gold or ""
    try:
        import structured_check
        if gold.startswith(structured_check.GOLD_PREFIX):
            expected, assumptions = structured_check.gold_spec(gold)
            return json.dumps(expected, ensure_ascii=False) + (f"\n(symbol assumptions: {json.dumps(assumptions)})" if assumptions else "")
    except Exception:  # noqa: BLE001
        pass
    return gold


def validate_scores(cfg: dict, scores, n: int):
    """The agent's submission -> n dicts, or an error string. Only structure and the allowed range are checked, never the value."""
    if not isinstance(scores, list):
        return "scores must be a list with one object per candidate"
    out: list = [None] * n
    for s in scores:
        try:
            i = int(s["i"]) - 1
            r = float(s["reward"])
        except Exception:  # noqa: BLE001
            return "every score needs an integer i and a numeric reward"
        if not 0 <= i < n or out[i] is not None:
            return f"candidate index {s.get('i')} is out of range or repeated (valid: 1..{n}, each once)"
        if not math.isfinite(r) or not cfg["reward_min"] <= r <= cfg["reward_max"]:
            return f"reward {s['reward']} for candidate {i + 1} is outside [{cfg['reward_min']}, {cfg['reward_max']}]"
        T = s.get("T")
        out[i] = {"reward": r, "correct": bool(s.get("correct")), "format": bool(s.get("format")),
                  "T": float(T) if isinstance(T, (int, float)) and math.isfinite(T) else None,
                  "bonus": float(s["bonus"]) if isinstance(s.get("bonus"), (int, float)) and math.isfinite(s["bonus"]) else 0.0,
                  "why": str(s.get("why", ""))[:240]}
    if any(o is None for o in out):
        return f"missing scores for candidates {[k + 1 for k, o in enumerate(out) if o is None]}"
    return out


def exact_grader(job: dict) -> list:
    """Diagnostics view of exact_results: True / False / None per candidate (the audit script reads it)."""
    return [None if e is None else e["status"] == "match" for e in exact_results(job)]


def score_group(cfg: dict, key: str, prompt: str, job: dict, usage: dict, stats: dict, chat_fn=None):
    """job: {problem, gold, domain, cands:[{thinking, reply, truncated, hard_valid, ntok}]} -> {"rewards": [...], "parts": [...]} or None."""
    chat_fn = chat_fn or (lambda msgs: chat(cfg, key, msgs, usage))
    n = len(job["cands"])
    exact = exact_results(job)
    head = [{"role": "system", "content": prompt}, {"role": "user", "content": first_message(cfg, job, exact)}]
    exchanges: list = []
    used, repairs, result = 0, 0, None
    for step in range(cfg["max_steps"] + cfg["max_repairs"]):
        window = [m for ex in exchanges[-cfg["window_exchanges"]:] for m in ex]
        while window and window[0].get("role") == "tool":
            window = window[1:]
        msgs = head + window
        if step >= cfg["max_steps"] - 1 or used > 0.7 * cfg["budget_tokens_per_group"]:
            msgs = msgs + [{"role": "user", "content": "Budget nearly spent: call finish NOW with the scores of all candidates."}]
        res = chat_fn(msgs)
        if res is None:
            stats["chat_failed"] = stats.get("chat_failed", 0) + 1
            return None
        m, tok = res
        used += tok
        calls = m.get("tool_calls") or []
        asst = {"role": "assistant", "content": (m.get("content") or "")[:300]}
        if calls:
            asst["tool_calls"] = calls
        ex, finished = [asst], False
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "finish":
                v = validate_scores(cfg, args.get("scores"), n)
                if isinstance(v, list):
                    result, out, finished = v, "accepted", True
                else:
                    stats["bad_finish"] = stats.get("bad_finish", 0) + 1
                    repairs += 1
                    out = "REJECTED: " + v + ". Call finish again with a corrected list."
            else:
                out = "unknown tool"
            ex.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        exchanges.append(ex)
        if finished:
            break
        if not calls:
            exchanges.append([{"role": "user", "content": "Call the finish tool now with the scores of all candidates."}])
        if used > cfg["budget_tokens_per_group"] or repairs > cfg["max_repairs"]:
            break
    stats["tokens_used"] = stats.get("tokens_used", 0) + used
    if result is None:
        stats["no_finish"] = stats.get("no_finish", 0) + 1
        return None
    parts = [dict(p, truncated=bool(c.get("truncated")), exact_grader=(None if e is None else e["status"] == "match")) for p, c, e in zip(result, job["cands"], exact)]
    return {"rewards": [p["reward"] for p in result], "parts": parts}
