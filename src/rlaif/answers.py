"""Domain-aware final-answer comparison: the trusted part of reward v2 (no LLM involved).

The gold is the last `**Answer:** x` line of the reference reply. Its kind decides how a candidate answer is compared:
  mc        a choice letter                       -> letter equality
  numeric   number (+ optional unit, %, 10^k)     -> relative tolerance after unit conversion
  text      short word/formula/list               -> normalised equality (chemical formulas, comma-separated sets)
  symbolic  an expression                         -> Math-Verify (the caller batches these through rlaif.verify.Verifier)
  long      anything else (explanations)          -> not machine-comparable: status "uncheckable"
compare() returns a Verdict; score is 1.0 equivalent, 0.25 numeric near miss, 0.0 wrong, None when the gold cannot be checked.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

_ANCHOR = re.compile(r"^\s*\*\*Answer:?\*\*:?\s*(.+?)\s*$", re.M)
_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻−–", "01234567890123456789+--" + "-")


def extract_anchor(text: str | None) -> str | None:
    """Content of the LAST `**Answer:** ...` line, without trailing punctuation/math delimiters."""
    if not text:
        return None
    found = _ANCHOR.findall(text)
    if not found:
        return None
    a = found[-1].strip()
    for _ in range(3):                                   # peel trailing punctuation and math delimiters in any order
        a = a.rstrip(" .;").strip()
        a = re.sub(r"^\$+|\$+$", "", a).strip()
        a = re.sub(r"^\\\(|\\\)$", "", a).strip()
    return a or None


def _last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(text) and d:
        d += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[i + 7:j - 1].strip() if d == 0 else None


_FINAL = re.compile(r"^\s*(?:\*\*)?final answer(?:\*\*)?\s*[:：]?\s*(?:\*\*)?\s*(.*?)\s*$", re.I)


def _clean_answer(a: str) -> str | None:
    for _ in range(3):
        a = a.strip().rstrip(" .;").strip()
        a = re.sub(r"^\$+|\$+$", "", a).strip()
        a = re.sub(r"^\\\[|\\\]$|^\\\(|\\\)$", "", a).strip()
    b = _last_boxed(a)
    return (b if b else a) or None


def extract_final_answer(text: str | None) -> str | None:
    """The answer a reply commits to: the last `**Answer:**` line, else the last \\boxed{...}, else the content after a `Final Answer` heading
    (same line or the next non-empty line). Used to read the candidates' answers, so a correct boxed answer is not thrown away."""
    if not text:
        return None
    a = extract_anchor(text)
    if a:
        return a
    b = _last_boxed(text)
    if b:
        return _clean_answer(b)
    lines = text.splitlines()
    for k in range(len(lines) - 1, -1, -1):
        m = _FINAL.match(lines[k])
        if m:
            content = m.group(1)
            if not content:
                content = next((x.strip() for x in lines[k + 1:] if x.strip()), "")
            return _clean_answer(content) if content else None
    return None


# ---------------------------------------------------------------- units
_PREFIX = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "u": 1e-6, "μ": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "c": 1e-2, "d": 1e-1}
_BASE = {  # unit -> (dimension, factor to the SI base of that dimension); prefixes are applied on top where allowed
    "m": ("length", 1.0), "s": ("time", 1.0), "g": ("mass", 1e-3), "mol": ("amount", 1.0), "K": ("temp", 1.0),
    "J": ("energy", 1.0), "N": ("force", 1.0), "Pa": ("pressure", 1.0), "W": ("power", 1.0), "V": ("voltage", 1.0),
    "A": ("current", 1.0), "C": ("charge", 1.0), "Hz": ("freq", 1.0), "L": ("volume", 1e-3), "l": ("volume", 1e-3),
    "eV": ("energy", 1.602176634e-19), "cal": ("energy", 4.184), "M": ("conc", 1000.0), "ohm": ("resistance", 1.0), "Ω": ("resistance", 1.0),
}
_FIXED = {  # units that must not be read as prefix+base
    "min": ("time", 60.0), "h": ("time", 3600.0), "hr": ("time", 3600.0), "day": ("time", 86400.0), "d": ("time", 86400.0), "yr": ("time", 3.15576e7),
    "atm": ("pressure", 101325.0), "bar": ("pressure", 1e5), "mmHg": ("pressure", 133.322), "torr": ("pressure", 133.322), "mbar": ("pressure", 100.0),
    "kcal": ("energy", 4184.0), "kWh": ("energy", 3.6e6), "u": ("mass", 1.66053907e-27), "Da": ("mass", 1.66053907e-27), "amu": ("mass", 1.66053907e-27),
    "Å": ("length", 1e-10), "angstrom": ("length", 1e-10), "in": ("length", 0.0254), "ft": ("length", 0.3048), "mi": ("length", 1609.344),
    "lb": ("mass", 0.45359237), "kg": ("mass", 1.0), "mL": ("volume", 1e-6), "ml": ("volume", 1e-6), "cm3": ("volume", 1e-6), "m3": ("volume", 1.0),
    "mM": ("conc", 1.0), "mmol": ("amount", 1e-3), "rad": ("angle", 1.0), "deg": ("angle", math.pi / 180), "°": ("angle", math.pi / 180),
    "%": ("percent", 0.01), "rpm": ("rpm", 1.0),
}


def _unit_info(u: str):
    u = (u or "").strip().replace(" ", "").replace("·", "").replace("^", "")
    u = u.translate(_SUBS).replace("μ", "u").replace("µ", "u")
    if not u:
        return None
    if u in _FIXED:
        return _FIXED[u]
    if u in _BASE:
        return _BASE[u]
    if len(u) > 1 and u[0] in _PREFIX and u[1:] in _BASE and u[1:] not in ("l",):
        dim, f = _BASE[u[1:]]
        return (dim, f * _PREFIX[u[0]])
    return None


def _norm_unit(u: str) -> str:
    return re.sub(r"[\s·*^]", "", (u or "")).translate(_SUBS).replace("μ", "u").replace("µ", "u").lower()


def _to_float(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if len(s) > 60:
        return None
    try:
        if "/" in s and re.fullmatch(r"-?\d+(\.\d+)?/\d+(\.\d+)?", s):
            a, b = s.split("/")
            return float(a) / float(b)
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


_NUM = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_QTY = re.compile(
    rf"^\s*(?:≈|~|approximately\s+)?\s*\$?\s*(?P<v>{_NUM}(?:/\d+)?)\s*(?:(?:[x×*]|\\times|\\cdot)\s*10\s*\^?\s*\{{?(?P<e>[-+]?\d+)\}}?)?\s*(?P<u>.*?)\s*$")


def parse_quantity(s: str | None):
    """'3.2 × 10^5 m/s' -> (320000.0, 'm/s', sig_digits); None when s is not a plain quantity."""
    if not s:
        return None
    t = s.translate(_SUBS).replace("\\,", "").replace("\\ ", " ").replace("$", "").replace("\\%", "%").replace("\\text{", "").replace("}", "")
    t = re.sub(r"\\mathrm\{|\\mathbf\{|\\", "", t) if "\\times" not in t and "\\cdot" not in t else t.replace("}", "")
    m = _QTY.match(t)
    if not m:
        return None
    v = _to_float(m.group("v"))
    if v is None:
        return None
    if m.group("e"):
        if len(m.group("e").lstrip("+-")) > 3:          # 10^99999999 and the like: not a plain quantity (and int->float overflows)
            return None
        try:
            v *= 10.0 ** int(m.group("e"))
        except OverflowError:
            return None
        if not math.isfinite(v):
            return None
    unit = (m.group("u") or "").strip().rstrip(".")
    if unit and re.search(r"[=+\-*/]\s*\d|\d\s*[+=]", unit):    # an expression, not a unit
        return None
    if len(unit) > 24 or len(unit.split()) > 3:
        return None
    digits = re.sub(r"[^0-9]", "", m.group("v").split("e")[0].split("E")[0]).lstrip("0")
    exact_int = not re.search(r"[.eE/]", m.group("v")) and not m.group("e")
    return v, unit, (0 if exact_int else max(1, len(digits)))


# ---------------------------------------------------------------- classification + comparison
_MC = re.compile(r"^\(?([A-Ja-j])\)?[.):]?$|^(?:option|choice|answer)\s*\(?([A-Ja-j])\)?$", re.I)


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_SUBS).casefold()
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), s)
    s = re.sub(r"[\s_{}$\\*`\"'.]", "", s)
    return s


def classify_gold(gold: str | None) -> str:
    if not gold:
        return "long"
    g = gold.strip()
    if _MC.match(g):
        return "mc"
    if parse_quantity(g):
        return "numeric"
    if len(g) <= 60 and len(g.split()) <= 8 and not re.search(r"[=\\^]|\\frac|\\sqrt", g) and not re.search(r"\d\s*[+\-*/]\s*\w", g):
        return "text"
    if len(g) <= 160 and re.search(r"[=\\^()/+\-*]|\\frac|\\sqrt|\d", g):
        return "symbolic"
    return "long"


@dataclass
class Verdict:
    kind: str                 # mc | numeric | text | symbolic | long
    status: str               # ok | uncheckable | no_answer | needs_symbolic
    score: float | None       # 1.0 | 0.25 | 0.0 | None
    detail: str = ""


def _kelvin(v: float, u: str):
    n = _norm_unit(u)
    if n in ("°c", "degc", "celsius", "oc", "℃"):
        return v + 273.15, "K"
    if n in ("°f", "degf", "fahrenheit", "℉"):
        return (v - 32) * 5 / 9 + 273.15, "K"
    return v, u


def compare_quantities(g, p, tol_rel: float, near_rel: float) -> tuple[float, str]:
    gv, gu, gsig = g
    pv, pu, _ = p
    gv, gu = _kelvin(gv, gu)
    pv, pu = _kelvin(pv, pu)
    gi, pi = _unit_info(gu), _unit_info(pu)
    note = ""
    if gu and pu and _norm_unit(gu) != _norm_unit(pu):
        if gi and pi and gi[0] == pi[0]:
            pv = pv * pi[1] / gi[1]
            note = f"converted {pu}->{gu}"
        elif gi and pi and gi[0] != pi[0] and "percent" not in (gi[0], pi[0]):
            return 0.0, f"incompatible units {pu} vs {gu}"
        elif not (gi and pi):
            return 0.0, f"unit mismatch {pu} vs {gu}"
    elif gu and not pu and gi and gi[0] == "percent":
        pv = pv * 100 if abs(pv) <= 1.5 < abs(gv) else pv
    elif pu and not gu and pi and pi[0] == "percent" and abs(gv) <= 1.5 < abs(pv):
        pv = pv * 0.01
    # exact integers (counts, 42, 320000) must match exactly; a decimal gold tolerates the rounding of its own printed digits
    tol = 1e-6 if gsig == 0 else min(max(tol_rel, 0.5 * 10 ** (1 - gsig)), 0.05)
    err = abs(pv - gv) / max(abs(gv), 1e-12) if gv != 0 else abs(pv - gv)
    if err <= tol or math.isclose(pv, gv, rel_tol=tol, abs_tol=1e-9):
        return 1.0, note + f" err={err:.2g}"
    if err <= near_rel:
        return 0.25, note + f" near err={err:.2g}"
    return 0.0, note + f" err={err:.2g}"


def compare(gold: str | None, pred: str | None, tol_rel: float = 0.01, near_rel: float = 0.05) -> Verdict:
    kind = classify_gold(gold)
    if kind == "long":
        return Verdict(kind, "uncheckable", None, "gold is not machine-comparable")
    if not pred or not pred.strip():
        return Verdict(kind, "no_answer", 0.0)
    if kind == "mc":
        gm, pm = _MC.match(gold.strip()), _MC.match(pred.strip())
        if not pm:
            return Verdict(kind, "ok", 0.0, "prediction is not a choice letter")
        gl = (gm.group(1) or gm.group(2)).upper()
        pl = (pm.group(1) or pm.group(2)).upper()
        return Verdict(kind, "ok", 1.0 if gl == pl else 0.0)
    if kind == "numeric":
        g, p = parse_quantity(gold), parse_quantity(pred)
        if not p:
            return Verdict(kind, "ok", 0.0, "prediction is not a number")
        s, note = compare_quantities(g, p, tol_rel, near_rel)
        return Verdict(kind, "ok", s, note)
    if kind == "text":
        gs, ps = _norm_text(gold), _norm_text(pred)
        if gs == ps:
            return Verdict(kind, "ok", 1.0)
        gparts = [x for x in re.split(r"[,;]", gold) if x.strip()]
        if len(gparts) > 1 and all(len(x.split()) <= 3 for x in gparts):          # a list of short items: compare as a set
            same = {_norm_text(x) for x in gparts} == {_norm_text(x) for x in re.split(r"[,;]", pred) if x.strip()}
            return Verdict(kind, "ok", 1.0 if same else 0.0, "set compare")
        if len(gold.split()) >= 3:                       # a phrase: a different wording is not a wrong answer, so it is left to the agent
            return Verdict(kind, "uncheckable", None, "free-text phrase")
        return Verdict(kind, "ok", 0.0)
    return Verdict("symbolic", "needs_symbolic", None)          # caller resolves with Math-Verify (rlaif.verify.Verifier)
