import json
from pathlib import Path

import pytest

from rlaif.config import Config
from rlaif.data import _rewritten
from rlaif.deepseek_judge import parse_score, render_input, split_completion

def test_parse_score_scale_and_failures():
    assert parse_score("85") == 0.85 and parse_score("-100") == -1.0 and parse_score("250") == 1.0 and parse_score("**40**") == 0.4
    assert parse_score("") is None and parse_score("no idea") is None and parse_score(None) is None

def test_split_completion_strips_special_tokens():
    th, reply = split_completion("<|inner_prefix|>Key idea: x\n1. y<|inner_suffix|>The answer is **3**.<|assistant_end|>", "<|inner_prefix|>", "<|inner_suffix|>")
    assert th == "Key idea: x\n1. y" and reply == "The answer is **3**."
    assert split_completion("<|inner_prefix|>cut off", "<|inner_prefix|>", "<|inner_suffix|>")[1] == ""

def test_render_input_has_all_parts():
    t = render_input("P", "REF", "TH", "RP", "length")
    assert "PROBLEM:\nP" in t and "REF" in t and "TH" in t and "RP" in t and "FINISH: length" in t and t.endswith("integer score.")

def test_rewritten_loader_keeps_split_and_skips_history(tmp_path):
    rows = [dict(id="a", split="train", problem="Q1", reference_answer="R1", history=[]),
            dict(id="b", split="val", problem="Q2", reference_answer="R2", history=[]),
            dict(id="c", split="train", problem="Q3", reference_answer="R3", history=[{"role": "user", "content": "x"}])]
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows))
    c = Config(); c.data.rewritten_path = str(p)
    d = _rewritten(c.data)
    assert [x.id for x in d["train"]] == ["a"] and [x.id for x in d["val"]] == ["b"] and d["train"][0].gold == "R1"

@pytest.mark.parametrize("name", ["4_rl_v3.yaml", "5_rl_v4.yaml"])
def test_rl_config_validates(name):
    from pathlib import Path
    from rlaif.config import load_config
    from rlaif.structured import validate_training_config
    root = Path(__file__).resolve().parents[1]
    c = load_config(str(root / "configs" / name))
    assert c.judge.backend == "deepseek" and c.reward_v3.enabled and not c.reward_v2.enabled and c.data.sources == ["nash"]
    assert root.joinpath(c.data.policy_prompt_path).exists() and root.joinpath(c.data.nash_path).exists()
    validate_training_config(c)


def test_judge_input_is_clipped_in_the_middle_when_asked():
    from rlaif.deepseek_judge import clip_middle, render_group
    long = "A" * 3000 + "MIDDLE" + "Z" * 3000
    c = clip_middle(long, 900)
    assert len(c) < 1000 and c.startswith("A") and c.endswith("Z") and "omitted" in c and "MIDDLE" not in c
    assert clip_middle("short", 900) == "short" and clip_middle(long, 0) == long
    cands = [{"thinking": long, "reply": "r", "finish": "stop"}]
    assert len(render_group("p", "g", cands, 900)) < len(render_group("p", "g", cands)) / 4
