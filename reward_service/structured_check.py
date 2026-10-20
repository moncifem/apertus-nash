"""Structured outcomes, standalone copy for the reward service (mirrors src/rlaif/structured.py: same rules, same grader file).
Never executes candidate code. The reply is: optional explanation, one `Result JSON:` line, one JSON object."""
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
    spec = importlib.util.spec_from_file_location("exact_outcomes", Path(__file__).resolve().parent / "grade_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_spec(gold):
    expected = _json(gold[len(GOLD_PREFIX):])
    assumptions = {}
    if isinstance(expected, dict) and set(expected) == {"kind", "expected", "symbol_assumptions"} and expected["kind"] == "structured":
        assumptions, expected = expected["symbol_assumptions"], expected["expected"]
    if not isinstance(assumptions, dict) or not isinstance(expected, dict) or not expected:
        raise ValueError("Invalid Nash gold")
    return expected, assumptions


def _payload(reply):
    lines = reply.splitlines()
    markers = [i for i, line in enumerate(lines) if line.strip() == RESULT_MARKER]
    if markers:
        if len(markers) != 1:
            raise ValueError("several Result JSON markers")
        return "\n".join(lines[markers[0] + 1:]).strip()
    return reply.strip()


def shape(value):
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(v) for v in value]
    return "boolean" if isinstance(value, bool) else "value"


def check(gold, reply):
    """(status, correct). status 'gold_unparseable' means the job itself is bad (the group is dropped, never guessed)."""
    try:
        expected, assumptions = gold_spec(gold)
    except (ValueError, TypeError, RecursionError):
        return "gold_unparseable", False
    try:
        if not isinstance(reply, str) or len(reply) > MAX_JSON:
            return "ok", False
        return "ok", bool(_grader().grade(_json(_payload(reply)), expected, assumptions)["correct"])
    except Exception:  # noqa: BLE001  (a malformed candidate is simply incorrect)
        return "ok", False


def format_ok(gold, reply):
    """True when the reply ends in exactly one Result JSON object with exactly the requested field names and shapes (right or wrong values)."""
    try:
        expected, _ = gold_spec(gold)
        return shape(_json(_payload(reply))) == shape(expected)
    except Exception:  # noqa: BLE001
        return False
