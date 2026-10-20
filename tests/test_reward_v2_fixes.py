import dataclasses

import pytest

from rlaif.answers import extract_final_answer
from rlaif.config import RewardV2Cfg
from rlaif.judge_items import DEFAULTS
from rlaif.novelty import Archive
from rlaif.reward_v2 import RewardV2, group_rewards


@pytest.mark.parametrize("text,exp", [
    ("blah\n\n**Answer:** 25", "25"),
    ("steps...\n\\[ \\boxed{[2\\sqrt{5}, \\infty)} \\]", "[2\\sqrt{5}, \\infty)"),
    ("so\n**Final Answer**\n\\boxed{7}", "7"),
    ("so\n**Final Answer:** 12.\n", "12"),
    ("Final Answer\n42", "42"),
    ("**Answer:** B\nalso \\boxed{C}", "B"),            # the explicit answer line wins
    ("no answer here", None), ("", None), (None, None),
])
def test_extract_final_answer(text, exp):
    assert extract_final_answer(text) == exp


class FakeAgent:
    def __init__(self):
        self.sent = []

    def cost_usd(self):
        return 0.0

    def verify_groups(self, jobs):
        from rlaif.verifier_client import CandVerdict, GroupVerdict
        self.sent = jobs
        return [GroupVerdict([CandVerdict(matched=False) for _ in j["cands"]], 1) for j in jobs]


def item(**kw):
    it = dict(DEFAULTS); it.update(ans=4, val=4, eff=3, prop=2, fmt=2); it.update(kw); return it


def cand(reply, thinking="Some reasonable method to reach the answer step by step carefully today.", ntok=500, truncated=False, **kw):
    return {"thinking": thinking, "reply": reply, "truncated": truncated, "items": item(**kw), "finish": "stop", "ntok": ntok}


def run(cands, cfg=None, gold="**Answer:** 25"):
    ag = FakeAgent()
    v2 = RewardV2(cfg or RewardV2Cfg(), ag, Archive(None))
    out = v2.score([{"problem": "p", "gold_reply": gold, "domain": "m", "cands": cands}], 0)
    return out[0], ag


def test_boxed_and_final_answer_replies_are_verified_without_the_agent():
    parts, ag = run([cand("Solved.\n\n\\[\\boxed{25}\\]"), cand("**Final Answer**\n25"), cand("\n\n**Answer:** 25")])
    assert [p.src for p in parts] == ["exact"] * 3 and ag.sent == []
    off = RewardV2Cfg(accept_boxed=False)
    parts, _ = run([cand("Solved.\n\n\\[\\boxed{25}\\]")], off)
    assert parts[0].src == "skip" and parts[0].v == 0.0           # the old behaviour is still reachable


def test_replyless_candidate_with_the_right_conclusion_gets_partial_credit():
    right = cand("", thinking="We set up the sums, subtract consecutive equations and find the period.\n**Answer:** 25")
    wrong = cand("", thinking="We set up the sums and then wander without ever concluding anything useful at all.")
    parts, ag = run([right, wrong, cand("\n\n**Answer:** 25")])
    assert parts[0].src == "think" and parts[0].v == 0.4 and parts[1].src == "skip" and parts[1].v == 0.0
    assert parts[2].reward > parts[0].reward > parts[1].reward and ag.sent == []
    assert parts[0].bonus_judge == parts[0].bonus_new == 0.0       # below the gate: no judge or novelty bonus
    parts, _ = run([right], RewardV2Cfg(think_credit=0.0))
    assert parts[0].src == "skip"


def test_shorter_verified_answer_gets_the_efficiency_bonus():
    parts, _ = run([cand("\n\n**Answer:** 25", ntok=300), cand("\n\n**Answer:** 25", ntok=900),
                    cand("\n\n**Answer:** 25", ntok=600), cand("\n\n**Answer:** 9", ntok=100)])
    assert abs(parts[0].bonus_short - 0.10) < 1e-9 and parts[1].bonus_short == 0.0 and abs(parts[2].bonus_short - 0.05) < 1e-9
    assert parts[3].bonus_short == 0.0                            # a wrong answer is never rewarded for being short
    assert parts[0].reward > parts[2].reward > parts[1].reward
    one, _ = run([cand("\n\n**Answer:** 25", ntok=300), cand("\n\n**Answer:** 9", ntok=900)])
    assert one[0].bonus_short == 0.0                              # needs at least two verified answers to compare
    cfg = RewardV2Cfg(w_short=0.0)
    parts, _ = run([cand("\n\n**Answer:** 25", ntok=300), cand("\n\n**Answer:** 25", ntok=900)], cfg)
    assert parts[0].bonus_short == 0.0


def test_truncation_penalty_is_configurable_and_parts_are_serialisable():
    cs = [cand("", truncated=True), cand("\n\n**Answer:** 25")]
    base = group_rewards(RewardV2Cfg(), [dict(c, exact=False, skip=c["truncated"], cv=None) for c in cs], 0, 0.5, None, "m")
    strong = group_rewards(RewardV2Cfg(trunc_penalty=0.8), [dict(c, exact=False, skip=c["truncated"], cv=None) for c in cs], 0, 0.5, None, "m")
    assert strong[0].reward < base[0].reward
    d = dataclasses.asdict(base[0])
    assert {"reward", "v", "src", "p", "J", "n", "r_ver", "bonus_judge", "bonus_new", "bonus_pattern", "bonus_short", "penalty", "hack"} <= set(d)


def test_malformed_answers_cannot_crash_scoring():
    from rlaif.answers import compare, parse_quantity
    from rlaif.reward_v2 import safe_compare
    for bad in ["1e999999999", "3 x 10^99999999", "3 × 10^400", "2/0", "9" * 500, "10^(1000000)"]:
        assert parse_quantity(bad) is None or parse_quantity(bad)[0] == parse_quantity(bad)[0]
        assert compare("140", bad).score == 0.0
        assert safe_compare("140", bad) is False
    assert safe_compare("140", "140") is True
