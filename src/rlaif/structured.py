"""Structured outcomes for the existing time-bounded local verifier.

Only submitted outcome fields are graded. Prose proofs are not certified and
candidate Python is never executed. Reference verification stays in publication.
"""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

GOLD_PREFIX = "STRUCTURED_V1:"
RESULT_MARKER = "Result JSON:"
MAX_JSON = 30000


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(text):
    if not isinstance(text, str) or len(text) > MAX_JSON:
        raise ValueError("Structured answer size limit")
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


@lru_cache(maxsize=1)
def _grader():
    path = Path(__file__).resolve().parents[2] / "reward_service/grade_answers.py"
    spec = importlib.util.spec_from_file_location("exact_outcomes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_gold(expected, assumptions=None):
    spec={"kind":"structured", "expected":expected,
          "symbol_assumptions":assumptions or {}}
    return GOLD_PREFIX+json.dumps(spec,ensure_ascii=False)


def _gold_spec(gold):
    expected = _json(gold[len(GOLD_PREFIX):])
    assumptions={}
    if (isinstance(expected,dict) and set(expected)=={"kind","expected","symbol_assumptions"}
            and expected["kind"]=="structured"):
        assumptions=expected["symbol_assumptions"]
        expected=expected["expected"]
    if not isinstance(assumptions,dict):
        raise ValueError("Invalid trusted symbol assumptions")
    if not isinstance(expected, dict) or not expected or not _grader().grade(expected, expected,assumptions)["correct"]:
        raise ValueError("Invalid Nash gold")
    return expected,assumptions


def decode_gold(gold):
    return _gold_spec(gold)[0]


def check(gold, reply):
    """Return existing verifier statuses; malformed candidates are incorrect."""
    try:
        expected,assumptions = _gold_spec(gold)
    except (ValueError, TypeError, RecursionError):
        return "gold_unparseable", False
    try:
        if not isinstance(reply, str) or len(reply) > MAX_JSON:
            return "ok", False
        # Optional proof precedes one explicit result; never scrape a JSON object
        # out of arbitrary prose or the thinking span.
        lines = reply.splitlines()
        markers = [i for i, line in enumerate(lines) if line.strip() == RESULT_MARKER]
        if markers:
            if len(markers) != 1:
                return "ok", False
            payload = "\n".join(lines[markers[0] + 1:]).strip()
        else:
            payload = reply.strip()
        candidate = _json(payload)
        return "ok", bool(_grader().grade(candidate, expected,assumptions)["correct"])
    except (ValueError, TypeError, RecursionError):
        return "ok", False


def answer_shape(value):
    """Describe field names/types/shapes without leaking reference values."""
    if isinstance(value, dict):
        return {key: answer_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [answer_shape(item) for item in value]
    return "boolean" if isinstance(value, bool) else "exact expression"


def policy_problem(row):
    schema = json.dumps(answer_shape(row["expected_answer"]), ensure_ascii=False)
    return row["problem"].strip() + (
        "\n\nGive a short proof explanation in the language of the problem. "
        "Then put Result JSON: on its own line, followed by one JSON object "
        "with exactly the field names and array shapes below. Replace each "
        "placeholder with your result. Use JSON booleans for boolean fields; "
        "use exact numbers or quoted arithmetic expressions for other fields. "
        "Allowed expressions use arithmetic, small powers, sqrt and positive "
        "constant cbrt. Do not include Python or extra text after the object. "
        "The shape below contains no reference values.\n" + schema)


def validate_training_config(cfg):
    """Prevent accidental judge-only rewards or a context longer than 4K."""
    if "nash" not in cfg.data.sources:
        return
    if cfg.reward_v2.enabled:
        raise ValueError("Nash does not use reward_v2; use the reward service (reward_v3) or the local exact verifier")
    if cfg.judge.backend == "deepseek" and not cfg.reward_v3.enabled:
        raise ValueError("Nash with the DeepSeek backend needs reward_v3.enabled (the reward service grades the exact outcome itself)")
    if cfg.train.max_prompt_tokens + cfg.train.max_completion_tokens > 4096:
        raise ValueError("Nash prompt plus completion budgets must not exceed 4096")

