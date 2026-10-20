"""Itemised DeepSeek judging: the judge writes one short line of small integers per candidate (one per criterion), and this module turns them
into a reward in [-1, 1] with the V2 weights, penalties and caps. prompts/judge/deepseek_judge_items.md defines the fields."""
import re


FIELDS = ("ans", "val", "luck", "con", "fab", "cit", "chk", "sc", "eff", "prop", "lang", "fmt", "trunc", "inj", "loop", "guess")
RANGES = dict(ans=4, val=4, luck=1, con=2, fab=3, cit=1, chk=2, sc=2, eff=3, prop=2, lang=1, fmt=2, trunc=1, inj=1, loop=1, guess=1)
DEFAULTS = dict(ans=0, val=0, luck=0, con=2, fab=0, cit=0, chk=1, sc=0, eff=2, prop=2, lang=1, fmt=1, trunc=0, inj=0, loop=0, guess=0)
_LINE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.)-]\s*(.*)$")


def parse_items(text: str, n: int):
    """'1: ans=4 val=3 ...' lines -> list of n dicts (None where a line is missing or incomplete)."""
    out: list = [None] * n
    for ln in (text or "").splitlines():
        m = _LINE.match(ln)
        if not m or not (1 <= int(m.group(1)) <= n):
            continue
        kv = dict((k, int(v)) for k, v in re.findall(r"(\w+)\s*=\s*(-?\d+)", m.group(2)))
        if "ans" in kv and "val" in kv:     # the two core fields are required; missing minor ones take neutral defaults
            out[int(m.group(1)) - 1] = {f: max(0, min(RANGES[f], kv.get(f, DEFAULTS[f]))) for f in FIELDS}
    return out


def items_to_reward(it: dict) -> float:
    """V2 reward model: weighted base, bonuses, penalties, hard caps. Returns a value in [-1, 1]."""
    if it["inj"] or it["loop"]:
        return -1.0
    C, D, F = it["ans"] / 4, it["val"] / 4, it["con"] / 2
    G, V, U = 1 - it["fab"] / 3, it["chk"] / 2, 0.0 if it["guess"] else 1.0
    E = (it["eff"] / 3 + it["prop"] / 2) / 2
    if it["luck"]:
        C, D = min(C, 0.5), min(D, 0.25)
    r = 0.30 * C + 0.25 * D + 0.15 * F + 0.12 * G + 0.08 * V + 0.05 * U + 0.05 * E
    if it["sc"] == 1 and it["ans"] >= 3 and not it["luck"] and it["chk"] == 2:
        r += 0.05
    pen = {0: 0.0, 1: 0.08, 2: 0.20, 3: 0.60}[it["fab"]]
    pen += {0: 0.40, 1: 0.05, 2: 0.0}[it["con"]] + {3: 0.0, 2: 0.05, 1: 0.25, 0: 0.35}[it["eff"]]
    pen += {2: 0.0, 1: 0.05, 0: 0.30}[it["prop"]] + (0.45 if it["sc"] == 2 else 0.0) + (0.80 if it["cit"] else 0.0)
    pen += (0.15 if not it["lang"] else 0.0) + (0.40 if it["trunc"] else 0.0) + (0.05 if it["fmt"] == 0 else 0.0) + (0.45 if it["guess"] else 0.0)
    r -= pen
    caps = [1.0]
    if it["ans"] <= 1: caps.append(0.25)
    if it["luck"]: caps.append(0.35)
    if it["fab"] == 3: caps.append(0.20)
    if it["con"] == 0: caps.append(0.25)
    if it["guess"]: caps.append(0.15)
    if it["sc"] == 2: caps.append(0.55)      # a theatrical correction can never score as a clean answer
    return max(-1.0, min(r, min(caps)))
