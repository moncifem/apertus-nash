import pytest

pytest.importorskip("math_verify")

from rlaif.verify import Verifier, is_operational_failure  # noqa: E402


def test_verifier_cases():
    v = Verifier(workers=2, timeout_s=30)
    try:
        res = v.check([("3", "3"), ("1/2", r"\frac{1}{2}"), ("0.5", "1/2"), ("3", "4"), ("18", "18.0"),
                       ("3", "not an answer at all ???")])
        assert [ok for _, ok in res] == [True, True, True, False, True, False]
        assert not any(is_operational_failure(s) for s, _ in res)
        assert v.gold_ok(["42", r"\frac{3}{4}"]) == [True, True]
    finally:
        v.close()
