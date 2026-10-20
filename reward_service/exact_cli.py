"""The exact structured-answer grader in its own process, so a pathological expression can only cost a timeout, never stall the service.
stdin: {"gold": str, "replies": [str|null, ...]}
stdout: [{"status": "match"|"mismatch"|"no_json", "wrong": [top-level fields that differ]} | null, ...]   (null = not graded: no structured gold, or truncated)
The result is read after the LAST `Result JSON:` line (or the first JSON object in the reply when there is no marker); text after the object is ignored."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structured_check  # noqa: E402


def payload(reply):
    lines = reply.splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == structured_check.RESULT_MARKER]
    text = "\n".join(lines[marks[-1] + 1:]) if marks else reply
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder(object_pairs_hook=structured_check._unique_object).raw_decode(text[i:])
    except Exception:  # noqa: BLE001
        return None
    return obj


def grade_one(expected, assumptions, reply):
    obj = payload(reply)
    if not isinstance(obj, dict):
        return {"status": "no_json", "wrong": []}
    g = structured_check._grader()
    wrong = []
    for k, v in expected.items():
        try:
            ok = k in obj and bool(g.equivalent(obj[k], v, assumptions))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            wrong.append(k)
    wrong += [k for k in obj if k not in expected]          # extra fields count as wrong
    return {"status": "mismatch" if wrong else "match", "wrong": wrong[:8]}


job = json.load(sys.stdin)
gold = job["gold"]
if not gold.startswith(structured_check.GOLD_PREFIX):
    print(json.dumps([None] * len(job["replies"])))
else:
    try:
        expected, assumptions = structured_check.gold_spec(gold)
        print(json.dumps([None if r is None else grade_one(expected, assumptions, r) for r in job["replies"]]))
    except Exception:  # noqa: BLE001
        print(json.dumps([None] * len(job["replies"])))
