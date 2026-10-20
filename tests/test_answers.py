import pytest

from rlaif.answers import classify_gold, compare, extract_anchor


def score(g, p):
    return compare(g, p).score


@pytest.mark.parametrize("g,p,exp", [
    ("42", "42.0", 1.0), ("42", "43", 0.25), ("35", "36", 0.25), ("3.2 × 10^5 m/s", "320000 m/s", 1.0),
    ("1500 m", "1.5 km", 1.0), ("2.5 atm", "253312 Pa", 1.0), ("12 J", "12 kJ", 0.0), ("37 °C", "310.15 K", 1.0),
    ("75%", "0.75", 1.0), ("0.75", "75%", 1.0), ("1/3", "0.3333", 1.0), ("-3", "−3", 1.0), ("6.02e23", "6.02 × 10^23", 1.0),
    ("B", "(b)", 1.0), ("B", "C", 0.0), ("H2O", "H₂O", 1.0), ("Na+, Cl-", "Cl-, Na+", 1.0), ("H2O", "H3O", 0.0),
    ("9.8 m/s^2", "9.81 m/s^2", 1.0), ("5", "five", 0.0),
])
def test_compare(g, p, exp):
    assert score(g, p) == exp


def test_uncheckable_and_symbolic_are_never_called_wrong():
    assert compare("The enzyme is inhibited by the allosteric binding of the product at a distal site", "x").status == "uncheckable"
    assert compare("ATP synthase and proton gradient", "ATP synthase").status == "uncheckable"      # a phrase: left to the agent
    v = compare("(x-2)(y-2)=4", "x")
    assert v.status == "needs_symbolic" and v.score is None


def test_extract_anchor():
    assert extract_anchor("blah\n\n**Answer:** $x = 4$.") == "x = 4"
    assert extract_anchor("**Answer:** a\n**Answer:** b") == "b"
    assert extract_anchor("no anchor") is None and extract_anchor(None) is None


def test_classify():
    assert classify_gold("B") == "mc" and classify_gold("3.5 m") == "numeric" and classify_gold("H2O") == "text"
