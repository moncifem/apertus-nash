"""Reward v2: verifier-first, judge as a gated bonus, plus a bonus for NEW ideas that are verified or demonstrably on track.

Per candidate i of a group (all numbers are starting values, set in RewardV2Cfg):

  v_i   verified correctness in [0, 1]       exact match of the **Answer:** line with the reference (cheap, positive-only)
                                             or the agent's code-verified match (numbers printed by a sandbox re-run) = 1.0,
                                             agent-judged equivalence for non-numeric answers = 0.7 (partial 0.35)
  p_i   on-track progress in [0, 1]          fraction of the candidate's own intermediate claims that the agent verified by computation
  J_i   judge quality in [0, 1]              (itemised DeepSeek reward + 1) / 2
  n_i   novelty in [0, 1]                    method signature distance to the group's other solutions and to the archive of past winners

  R_ver = w_fmt*[answer line] + w_v*v + w_near*[numeric near miss] + w_track*p*[v < 0.7]      (unverifiable rows: w_fmt + unverified_cap*J)
  B     = lambda_t * gate * (J - mean J over gated peers)       gate = v >= 0.7 or p >= 0.5; lambda_t = lambda0 * clip(rho) * fade
  N     = w_new * scale(n) * g          g = 1 if v >= 0.7, 0.5 if p >= 0.6 (a new idea on a good track), else 0
  Q     = w_pattern * [v >= 0.7 and the thinking states a "General pattern:"]
  P     = truncation, language drift, fabrication, invented citations, theatrical self-correction, guessing, self-contradiction
  hack  = prompt injection / loops / conflicting answer lines -> R = -1
  R     = clip(R_ver + B + N + Q - P, -1, 2)

rho is the live correlation between J and exact-verified correctness: if the judge stops predicting the verifier, its bonus goes to zero.
Advantages are group mean-centred (Dr. GRPO, no std) and negative advantages are scaled by beta; groups with no spread are skipped.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field

from .answers import classify_gold, compare, extract_anchor, extract_final_answer
from .judge_items import items_to_reward
from .novelty import Archive, group_novelty, signature

def safe_compare(gold, pred):
    """compare() that can never take the trainer down: a malformed answer (huge exponent, 2/0, ...) simply does not match."""
    try:
        return compare(gold, pred).score == 1.0
    except Exception:  # noqa: BLE001
        return False


_PATTERN = re.compile(r"general pattern\s*:\s*\S.{15,}", re.I)
_INJECT = re.compile(r"ignore (all |any )?(previous|prior|above) (instructions|rules)|you are (the )?judge|give (this|me) (a )?(score|reward)|system prompt", re.I)


@dataclass
class Parts:
    reward: float = 0.0
    v: float | None = None
    src: str = "none"          # exact | code | text | none
    p: float | None = None
    J: float = 0.0
    n: float | None = None
    r_ver: float = 0.0
    bonus_judge: float = 0.0
    bonus_new: float = 0.0
    bonus_pattern: float = 0.0
    bonus_short: float = 0.0
    penalty: float = 0.0
    hack: bool = False
    near_miss: bool = False


class RhoTracker:
    """Rolling point-biserial correlation between the judge's quality J and exact-verified correctness."""

    def __init__(self, window: int = 400, default: float = 0.5, min_n: int = 50):
        self.buf: deque = deque(maxlen=window)
        self.default, self.min_n = default, min_n

    def add(self, J: float, v: float) -> None:
        self.buf.append((J, v))

    def rho(self) -> float:
        n = len(self.buf)
        if n < self.min_n:
            return self.default
        xs, ys = [a for a, _ in self.buf], [b for _, b in self.buf]
        mx, my = sum(xs) / n, sum(ys) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
        sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
        if sx < 1e-9 or sy < 1e-9:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def judge_quality(it: dict | None) -> float:
    return 0.5 if it is None else (items_to_reward(it) + 1.0) / 2.0


def lambda_t(cfg, update: int, rho: float) -> float:
    fade = max(0.0, 1.0 - update / max(1, cfg.judge_fade_updates))
    return cfg.judge_lambda0 * min(1.0, max(0.0, rho)) * fade


def _penalty(it: dict | None, truncated: bool, trunc_penalty: float = 0.5) -> float:
    p = trunc_penalty if truncated else 0.0
    if it is None:
        return p
    p += {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.40}[it["fab"]] + (0.5 if it["cit"] else 0.0) + (0.3 if it["sc"] == 2 else 0.0)
    p += (0.3 if it["guess"] else 0.0) + (0.3 if it["con"] == 0 else 0.0) + (0.15 if not it["lang"] else 0.0)
    return p


def verified_value(exact: bool, cv, cfg, skip: bool = False, think: bool = False) -> tuple[float | None, str]:
    """(v, source) from the exact-match fast path and the agent verdict (CandVerdict or None). skip = no usable final answer (truncated or
    missing **Answer:** line while the reference has one): it is wrong by construction and never costs an agent call."""
    if exact:
        return 1.0, "exact"
    if think:
        return cfg.think_credit, "think"
    if skip:
        return 0.0, "skip"
    if cv is None:
        return None, "none"
    if cv.matched is True:
        return 1.0, "code"
    if cv.matched is False:
        return 0.0, "code"
    return {"yes": (cfg.w_text_equiv, "text"), "partial": (cfg.w_text_equiv / 2, "text"), "no": (0.0, "text")}.get(cv.equivalent or "", (None, "none"))


def group_rewards(cfg, cands: list[dict], update: int, rho: float, archive: Archive | None, domain: str) -> list[Parts]:
    """cands: dicts with thinking, reply, truncated, items (judge dict or None), exact (bool), cv (CandVerdict or None)."""
    n = len(cands)
    parts = [Parts() for _ in range(n)]
    sigs = [signature(c["thinking"]) for c in cands]
    gnov = group_novelty(sigs)
    for i, (c, P) in enumerate(zip(cands, parts)):
        it, cv = c.get("items"), c.get("cv")
        P.J = judge_quality(it)
        v, P.src = verified_value(c.get("exact", False), cv, cfg, c.get("skip", False), c.get("think", False))
        if v is not None and it is not None and (it["luck"] or it["guess"]):
            v *= 0.5                                            # a lucky or guessed hit is not a demonstrated solution
        P.v = v
        P.p = cv.progress if cv is not None and (v is None or v < cfg.gate_v) and not c.get("mc") else None   # a wrong option letter is not "partly right"
        P.near_miss = bool(cv and cv.near_miss and v == 0.0)
        has_answer = (extract_final_answer(c["reply"]) if cfg.accept_boxed else extract_anchor(c["reply"])) is not None
        fmt = cfg.w_fmt * float(has_answer and (it is None or it["fmt"] == 2))
        if v is None:
            P.r_ver = fmt + cfg.unverified_cap * P.J
        else:
            P.r_ver = fmt + cfg.w_verified * v + (cfg.w_near if P.near_miss else 0.0) + (cfg.w_track * P.p if (P.p is not None and v < cfg.gate_v) else 0.0)
        anchors = {a.strip().lower() for a in re.findall(r"^\s*\*\*Answer:?\*\*:?\s*(.+?)\s*$", c["reply"], re.M)}
        P.hack = bool(it and (it["inj"] or it["loop"])) or len(anchors) > 1 or bool(_INJECT.search(c["reply"] + c["thinking"]))
        P.penalty = _penalty(it, c.get("truncated", False), cfg.trunc_penalty)
    gated = [P.v is not None and P.v >= cfg.gate_v or (P.p is not None and P.p >= cfg.gate_p) for P in parts]
    lam = lambda_t(cfg, update, rho)
    gj = [P.J for P, g in zip(parts, gated) if g]
    mj = sum(gj) / len(gj) if gj else 0.0
    for i, (c, P) in enumerate(zip(cands, parts)):
        if gated[i] and len(gj) > 1:
            P.bonus_judge = lam * (P.J - mj)
        g = 1.0 if (P.v is not None and P.v >= cfg.gate_v) else (0.5 if (P.p is not None and P.p >= cfg.gate_new_p) else 0.0)
        if g and gnov[i] is not None:
            P.n = gnov[i] * (archive.novelty(domain, sigs[i]) if archive else 1.0)
            P.bonus_new = cfg.w_new * g * max(0.0, (P.n - cfg.new_floor) / (1.0 - cfg.new_floor))
        if P.v is not None and P.v >= cfg.gate_v and _PATTERN.search(c["thinking"]):
            P.bonus_pattern = cfg.w_pattern
        P.reward = -1.0 if P.hack else max(-1.0, min(cfg.reward_max, P.r_ver + P.bonus_judge + P.bonus_new + P.bonus_pattern + P.bonus_short - P.penalty))
    ver = [i for i, P in enumerate(parts) if P.v is not None and P.v >= cfg.gate_v and not P.hack and cands[i].get("ntok")]
    if len(ver) >= 2:                                           # efficiency, only among verified answers of the same prompt: shortest +w_short, longest +0
        lo, hi = min(cands[i]["ntok"] for i in ver), max(cands[i]["ntok"] for i in ver)
        if hi > lo:
            for i in ver:
                parts[i].bonus_short = cfg.w_short * (hi - cands[i]["ntok"]) / (hi - lo)
                parts[i].reward = max(-1.0, min(cfg.reward_max, parts[i].reward + parts[i].bonus_short))
    if archive is not None:                                     # past winners make the same method less novel next time
        for P, s in zip(parts, sigs):
            if P.v is not None and P.v >= cfg.gate_v and not P.hack:
                archive.add(domain, s)
    return parts


def advantages(rewards: list[float], beta: float = 1.0) -> list[float]:
    m = sum(rewards) / len(rewards)
    return [(r - m) * (beta if r < m else 1.0) for r in rewards]


class RewardV2:
    """Glue: exact fast path -> one ReAct agent session per group (only when something is unconfirmed) -> formulas above."""

    def __init__(self, cfg, agent, archive: Archive | None = None):
        self.cfg, self.agent, self.archive = cfg, agent, archive
        self.rho_tracker = RhoTracker(cfg.rho_window, cfg.rho_default)
        self.stats: Counter = Counter()

    def score(self, groups: list[dict], update: int):
        """groups: dicts with problem, gold_reply, domain, cands=[{thinking, reply, finish, truncated, items}].
        Returns a list (per group) of Parts lists, or None for a group the agent could not verify (the trainer drops it, never guesses)."""
        cfg = self.cfg
        gold = [extract_anchor(g.get("gold_reply")) for g in groups]
        exact_flags, skip_flags, need = [], [], []
        for gi, g in enumerate(groups):
            ex, sk = [], []
            for c in g["cands"]:
                a = extract_final_answer(c["reply"]) if cfg.accept_boxed else extract_anchor(c["reply"])
                ex.append(bool(gold[gi] and a and not c.get("truncated") and safe_compare(gold[gi], a)))
                sk.append(bool(c.get("truncated") or (gold[gi] and a is None)))
                th = False                                       # a reply-less candidate whose thinking ends in the exact reference answer
                if gold[gi] and cfg.think_credit > 0 and not c.get("truncated") and not c["reply"].strip() and not ex[-1]:
                    ta = extract_final_answer(c["thinking"][-600:])
                    th = bool(ta and safe_compare(gold[gi], ta))
                c["_think"] = th
            exact_flags.append(ex)
            skip_flags.append(sk)
            if not all(e or k or c["_think"] for e, k, c in zip(ex, sk, g["cands"])):
                need.append(gi)
        # only groups with an unconfirmed candidate go to the agent service, and only those candidates (exact matches and answer-less ones are settled)
        over_budget = self.agent is not None and self.agent.cost_usd() >= cfg.agent_max_cost_usd
        sub: dict[int, list[int]] = {gi: [k for k in range(len(groups[gi]["cands"])) if not (exact_flags[gi][k] or skip_flags[gi][k] or groups[gi]["cands"][k]["_think"])] for gi in need}
        verdicts: dict[int, object] = {}
        if need and self.agent is not None and not over_budget:
            res = self.agent.verify_groups([{"problem": groups[gi]["problem"], "gold_reply": groups[gi].get("gold_reply"), "gold_answer": gold[gi],
                                             "cands": [groups[gi]["cands"][k] for k in sub[gi]]} for gi in need])
            verdicts = dict(zip(need, res))
        if over_budget:
            self.stats["agent_budget_hit"] += 1
        out = []
        rho = self.rho_tracker.rho()
        for gi, g in enumerate(groups):
            gv = verdicts.get(gi)
            if gi in need and gv is None:
                self.stats["groups_unverified"] += 1
                out.append(None)
                continue
            where = {k: j for j, k in enumerate(sub.get(gi, []))}
            cands = [dict(c, mc=(classify_gold(gold[gi]) == "mc"), exact=exact_flags[gi][k], skip=skip_flags[gi][k], think=c["_think"], cv=(gv.cands[where[k]] if (gv is not None and k in where) else None))
                     for k, c in enumerate(g["cands"])]
            parts = group_rewards(cfg, cands, update, rho, self.archive, g.get("domain", "general"))
            for P, c in zip(parts, cands):
                if P.src in ("exact", "code") and P.v is not None and c.get("items") is not None:
                    self.rho_tracker.add(P.J, 1.0 if P.v >= 1.0 else 0.0)
                self.stats["cands"] += 1
                self.stats["verified"] += int(P.v is not None and P.v >= cfg.gate_v)
                self.stats["exact"] += int(P.src == "exact")
                self.stats["skipped_no_answer"] += int(P.src == "skip")
                self.stats["code_verified"] += int(P.src == "code" and (P.v or 0) >= cfg.gate_v)
                self.stats["on_track"] += int(P.p is not None and P.p >= cfg.gate_new_p)
                self.stats["novel_paid"] += int(P.bonus_new > 0)
                self.stats["hack"] += int(P.hack)
            out.append(parts)
        self.stats["groups_agent"] += len(need)
        return out
