from types import SimpleNamespace

import pytest

from rlaif.judge import StepVerdict, verdict_to_q
from rlaif.rewards import group_advantages, kimi_length_rewards, outcome_rewards, process_token_targets, ramp

RC = SimpleNamespace(q_invalid=-1.0, q_uncertain=0.0, q_redundant=-0.5, q_valid=1.0)


def v(**kw):
    base = dict(step_id=1, validity="valid", relevance="useful", redundant=False, depends_on_error=False, evidence="x")
    base.update(kw)
    return StepVerdict(**base)


def test_verdict_mapping():
    assert verdict_to_q(v(), RC) == 1.0
    assert verdict_to_q(v(validity="invalid"), RC) == -1.0
    assert verdict_to_q(v(validity="invalid", redundant=True), RC) == -1.0  # invalidity wins
    assert verdict_to_q(v(depends_on_error=True), RC) == 0.0               # propagation not re-punished
    assert verdict_to_q(v(validity="uncertain"), RC) == 0.0
    assert verdict_to_q(v(redundant=True), RC) == -0.5
    assert verdict_to_q(v(relevance="irrelevant"), RC) == -0.5
    assert verdict_to_q(None, RC) == 0.0


def test_kimi_length_prefers_short_correct():
    r = kimi_length_rewards([100, 200, 300], [True, True, False])
    assert r[0] == 0.5 and r[1] == 0.0 and r[2] == -0.5
    assert kimi_length_rewards([100, 300], [False, True]) == [0.0, -0.5]  # short wrong never rewarded


def test_outcome_and_advantages():
    rewards, _ = outcome_rewards([True, True, False, False], [True, True, False, True], [False, False, False, True],
                                 [100, 300, 50, 4000], format_penalty=0.2, truncation_penalty=0.5, lam_len=0.2)
    assert rewards[0] > rewards[1] > rewards[2] > rewards[3]
    adv = group_advantages(rewards)
    assert abs(sum(adv)) < 1e-9


def test_all_wrong_group_zero_outcome_advantage():
    rewards, _ = outcome_rewards([False] * 4, [True] * 4, [False] * 4, [10, 20, 30, 40],
                                 format_penalty=0.2, truncation_penalty=0.5, lam_len=0.0)
    assert group_advantages(rewards) == [0.0] * 4


def test_process_weights_cannot_grow_with_more_steps():
    q1, w1 = process_token_targets([(0, 30)], [1.0], 40)
    q3, w3 = process_token_targets([(0, 10), (10, 20), (20, 30)], [1.0, 1.0, 1.0], 40)
    tot = lambda q, w: sum(a * b for a, b in zip(q, w))
    assert tot(q1, w1) == pytest.approx(1.0) and tot(q3, w3) == pytest.approx(1.0)  # splitting gains nothing
    q4, w4 = process_token_targets([(0, 10), (10, 20), (20, 30), (30, 35)], [1.0, 1.0, 1.0, -0.5], 40)
    assert tot(q4, w4) < 1.0  # adding a redundant step lowers it
    assert all(x == 0 for x in w1[30:])  # answer tokens get no process weight


def test_bad_span_rejected():
    with pytest.raises(ValueError):
        process_token_targets([(5, 5)], [1.0], 10)


def test_ramp():
    assert ramp(9, 10, 5, 0.2) == 0.0
    assert ramp(10, 10, 5, 0.2) == pytest.approx(0.04)
    assert ramp(100, 10, 5, 0.2) == pytest.approx(0.2)


def test_step_bonus_and_repetition_penalty():
    base, _ = outcome_rewards([True, True], [True, True], [False, False], [100, 100],
                              format_penalty=0.2, truncation_penalty=0.5, lam_len=0.0)
    r, _ = outcome_rewards([True, True], [True, True], [False, False], [100, 100],
                           format_penalty=0.2, truncation_penalty=0.5, lam_len=0.0,
                           step_ok=[True, False], rep_frac=[0.0, 0.9], step_bonus=0.1, rep_penalty=0.2)
    assert r[0] == pytest.approx(base[0] + 0.1)
    assert r[1] == pytest.approx(base[1] - 0.2)  # fully looping trace loses the whole penalty
