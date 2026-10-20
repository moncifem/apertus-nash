from rlaif.judge_items import FIELDS, items_to_reward, parse_items

BASE = dict(ans=4, val=4, luck=0, con=2, fab=0, cit=0, chk=2, sc=0, eff=3, prop=2, lang=1, fmt=2, trunc=0, inj=0, loop=0, guess=0)

def r(**kw): return items_to_reward(BASE | kw)

def test_clean_answer_scores_top():
    assert r() >= 0.95

def test_ordering_of_failure_modes():
    clean, rambling, fake, lucky, wrong = r(), r(eff=1, prop=1), r(sc=2), r(luck=1), r(ans=0, val=1)
    assert clean > rambling > lucky > wrong
    assert clean > fake > lucky

def test_real_correction_never_beats_no_correction_by_much_and_fake_is_capped():
    assert r(sc=1, chk=2, ans=4) <= r(sc=0) + 0.06          # a correction is not a goal: at most a small bonus
    assert r(sc=2) <= 0.55 and r(sc=2) < r(sc=0) - 0.3        # theatrical correction: capped and clearly worse than none
    assert r(sc=0) >= r(sc=1, chk=0)                           # no bonus without a real check

def test_catastrophic_flags():
    assert r(inj=1) == -1.0 and r(loop=1) == -1.0
    assert r(cit=1) < 0.3 and r(fab=3, ans=0) < 0.0 and r(trunc=1, ans=0, val=1, chk=0) < 0.15

def test_wrong_answer_is_capped_and_wrong_language_costs():
    assert r(ans=0) <= 0.25 and r(lang=0) < r()

def test_parse_items_is_tolerant():
    txt = "1: ans=4 val=4 luck=0 con=2 inj=1\n2: ans=3 val=2\n3: garbage\n4: val=2"
    its = parse_items(txt, 4)
    assert its[0]["inj"] == 1 and its[0]["fab"] == 0 and its[1]["con"] == 2     # missing minor fields take neutral defaults
    assert its[2] is None and its[3] is None                                    # core fields missing -> unusable
    assert set(its[0]) == set(FIELDS)

def test_parse_items_clamps_ranges():
    it = parse_items("1: ans=9 val=-3", 1)[0]
    assert it["ans"] == 4 and it["val"] == 0


def test_spend_guard_cost_model():
    from collections import Counter
    from types import SimpleNamespace
    from rlaif.deepseek_judge import DeepSeekScorer
    s = DeepSeekScorer.__new__(DeepSeekScorer)
    s.cfg = SimpleNamespace(price_hit=0.006, price_miss=0.30, price_out=1.20)
    s.stats = Counter(cache_hit_tokens=1_000_000, cache_miss_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(s.cost_usd() - (0.006 + 0.30 + 1.20)) < 1e-9
