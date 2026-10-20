import json
import math

import pytest

from conftest import CLOSE, END, OPEN, CharTok

from rlaif.parsing import parse_completion
from rlaif.sft_data import (SkipRow, SFTExample, answer_to_boxed_text, build_completion, build_dataset,
                           dataset_stats, load_rows, render_prompt)

tok = CharTok()
TEMPLATE = "Solve.\n\nProblem:\n{problem}"
END_ID = CharTok.SPECIAL[END]


class FakeChatTok(CharTok):
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        sys_ = "".join(f"<s:{m['content']}>" for m in msgs if m["role"] == "system")
        user = next(m for m in msgs if m["role"] == "user")["content"]
        return sys_ + "<u>" + user + "</u><a>"


ROW = {
    "id": "math.self_similarity.0001", "order": 3, "split": "train", "level": 2, "domain": "math",
    "problem": "Evaluate the nested radical.",
    "thinking": ["Key idea: x = sqrt(6 + x); prove the limit exists first.",
                 "1. a_1 = sqrt(6) < 3 and a_n < 3 implies a_(n+1) < 3.",
                 "2. The sequence increases and is bounded, so it converges to L.",
                 "3. L^2 = L + 6 gives L = 3 since L > 0.",
                 "Answer: 3"],
    "reference_answer": "$3$",
}


def build(rows, max_tokens=10_000):
    return build_dataset(FakeChatTok(), rows, TEMPLATE, thinking=True, think_open=OPEN, think_close=CLOSE,
                         end_token=END, end_id=END_ID, max_tokens=max_tokens)


def test_raw_prompt_style_skips_the_task_wrapper():
    raw = {**ROW, "id": "identity.name.0001", "prompt_style": "raw", "problem": "Who are you?"}
    (ex_raw,), _ = build([raw])
    (ex_wrapped,), _ = build([{**raw, "prompt_style": None}])
    assert tok.decode(ex_raw.prompt_ids) == "<u>Who are you?</u><a>"
    assert "Solve." in tok.decode(ex_wrapped.prompt_ids)
    assert ex_raw.completion_ids == ex_wrapped.completion_ids


PLAIN = {**ROW, "id": "identity.who.0001", "answer_style": "plain", "prompt_style": "raw", "problem": "Who are you?",
         "thinking": ["Key idea: state the name.", "1. I am Apertus Euler.", "Answer: I am Apertus Euler."],
         "reference_answer": "I am Apertus Euler, a math and science model."}


def test_plain_answer_has_no_box_and_ends_with_the_end_token():
    text = build_completion(PLAIN, OPEN, CLOSE, END)
    assert text.endswith(CLOSE + "I am Apertus Euler, a math and science model." + END)
    assert "\\boxed" not in text
    (ex,), skipped = build([PLAIN])
    assert not skipped and ex.answer_style == "plain" and ex.completion_ids[-1] == END_ID


def test_continuous_thinking_string_survives_dataset_rendering():
    thought = "Before computing, I factor x^2-1=(x-1)*(x+1). General pattern: preserve the domain."
    row = {**PLAIN, "thinking": thought}
    (example,), skipped = build([row])
    assert not skipped
    assert tok.decode(example.completion_ids) == OPEN + thought + CLOSE + row["reference_answer"] + END


@pytest.mark.parametrize("thought", [" ", "Use " + CLOSE, "see \\boxed{2}"])
def test_continuous_thinking_cannot_bypass_content_guards(thought):
    _, skipped = build([{**PLAIN, "thinking": thought}])
    assert skipped


def test_plain_target_passes_the_parser_only_without_require_box():
    (ex,), _ = build([PLAIN])
    kw = dict(think_open=OPEN, think_close=CLOSE, opened_in_prompt=False, finish_reason="stop", max_steps=48)
    ok = parse_completion(ex.completion_ids, tok, require_box=False, **kw)
    assert ok.hard_valid and ok.final_answer == "I am Apertus Euler, a math and science model."
    strict = parse_completion(ex.completion_ids, tok, **kw)           # math / RL default still demands a box
    assert not strict.hard_valid and "no_boxed_answer" in strict.errors


def test_check_with_rl_parser_uses_each_rows_own_style():
    from rlaif.sft_data import check_with_rl_parser
    exs, _ = build([ROW, PLAIN])
    rep = check_with_rl_parser(FakeChatTok(), exs, OPEN, CLOSE)
    assert rep["n"] == 2 and rep["hard_valid"] == 2 and dataset_stats(exs)["plain_answer_rows"] == 1


@pytest.mark.parametrize("mutate, reason", [
    ({"answer_style": "markdown"}, "unknown answer_style"),
    ({"reference_answer": "see \\boxed{3}"}, "contains \\boxed"),
    ({"reference_answer": "   "}, "empty reference_answer"),
    ({"reference_answer": "done" + END}, "marker"),
])
def test_bad_plain_rows_are_skipped_with_a_reason(mutate, reason):
    _, skipped = build([{**PLAIN, **mutate}])
    assert skipped and reason in skipped[0][1]


VARIANTS = {"concise": ["C0 {problem} \\boxed{answer} numbered", "C1 {problem} \\boxed{answer} numbered"],
            "standard": ["S0 {problem} \\boxed{answer} numbered"], "detailed": ["D0 {problem} \\boxed{answer} numbered"]}


def build_v(rows, **kw):
    return build_dataset(FakeChatTok(), rows, TEMPLATE, thinking=True, think_open=OPEN, think_close=CLOSE,
                         end_token=END, end_id=END_ID, max_tokens=10_000, **kw)


def test_pick_template_raw_variant_and_default():
    from rlaif.sft_data import pick_template
    assert pick_template({**ROW, "prompt_style": "raw"}, TEMPLATE, VARIANTS) == "{problem}"
    assert pick_template({**ROW, "prompt_variant": "raw"}, TEMPLATE, VARIANTS) == "{problem}"
    assert pick_template({**ROW, "prompt_variant": "detailed"}, TEMPLATE, VARIANTS) == VARIANTS["detailed"][0]
    assert pick_template(ROW, TEMPLATE, VARIANTS) == TEMPLATE                      # no variant -> shared wrapper
    assert pick_template({**ROW, "prompt_variant": "standard"}, TEMPLATE, None) == TEMPLATE
    row = {**ROW, "prompt_variant": "concise"}
    assert pick_template(row, TEMPLATE, VARIANTS) == pick_template(row, TEMPLATE, VARIANTS)   # deterministic per id
    seen = {pick_template({**row, "id": f"r{i}"}, TEMPLATE, VARIANTS) for i in range(40)}
    assert seen == set(VARIANTS["concise"])                                                    # both paraphrases are used


def test_variants_change_the_prompt_but_not_the_target():
    (a,), _ = build_v([{**ROW, "prompt_variant": "concise"}], variants=VARIANTS)
    (b,), _ = build_v([{**ROW, "prompt_variant": "detailed"}], variants=VARIANTS)
    assert a.completion_ids == b.completion_ids and a.prompt_ids != b.prompt_ids
    assert (a.prompt_variant, b.prompt_variant) == ("concise", "detailed")
    assert dataset_stats([a, b])["by_prompt_variant"] == {"concise": 1, "detailed": 1}


def test_system_prompt_replaces_the_default_and_is_not_trained_on():
    (ex,), _ = build_v([ROW], system="You are Euler.")
    text = tok.decode(ex.prompt_ids)
    assert text.startswith("<s:You are Euler.>") and ex.completion_ids[-1] == END_ID
    (plain,), _ = build_v([ROW])
    assert not tok.decode(plain.prompt_ids).startswith("<s:")


def test_load_variants_validates(tmp_path):
    from rlaif.sft_data import load_variants
    good = tmp_path / "v.json"; good.write_text(json.dumps({"_doc": "x", **VARIANTS}))
    assert set(load_variants(good)) == set(VARIANTS)
    for bad in ({"concise": ["no placeholders here"]}, {"fast": ["{problem} \\boxed{answer} numbered"]}, {"concise": []}):
        p = tmp_path / "bad.json"; p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_variants(p)


def test_answer_to_boxed():
    assert answer_to_boxed_text("$3$") == "3"
    assert answer_to_boxed_text("Answer: $\\frac12$") == "\\frac12"
    assert answer_to_boxed_text("the set $\\{-1,0,1\\}$") == "the set \\{-1,0,1\\}"  # no $ inside \\boxed
    assert answer_to_boxed_text("$(5/2, 3)$ weighted by 2, 1, 1") == "(5/2, 3) weighted by 2, 1, 1"
    assert answer_to_boxed_text("it costs \\$300") == "it costs \\$300"            # escaped dollars survive
    assert answer_to_boxed_text("a) 2\nb) 3") == "a) 2 b) 3"
    for bad in ["", "   ", "\\boxed{3}", "x^{2"]:
        with pytest.raises(SkipRow):
            answer_to_boxed_text(bad)


def test_completion_shape_and_single_boxed():
    c = build_completion(ROW, OPEN, CLOSE, END)
    assert c.startswith(OPEN) and c.endswith(END)
    assert c.count(OPEN) == c.count(CLOSE) == c.count(END) == 1
    assert c.count("\\boxed") == 1 and c.split(CLOSE)[1] == "\\boxed{3}" + END
    assert "\n" in c and "\n\n" not in c


def test_target_is_hard_valid_for_the_rl_parser():
    ex, skipped = build([ROW])
    assert not skipped
    p = parse_completion(ex[0].completion_ids, tok, think_open=OPEN, think_close=CLOSE,
                         opened_in_prompt=False, finish_reason="stop", max_steps=48)
    assert p.hard_valid, p.errors
    assert p.final_answer == "3"
    assert p.spans_ok and len(p.steps) == 3  # the "Key idea" line is tolerated prose before step 1


def test_prompt_and_completion_are_separate_and_end_token_last():
    ex, _ = build([ROW])
    e = ex[0]
    assert tok.decode(e.prompt_ids).startswith("<u>Solve.") and tok.decode(e.prompt_ids).endswith("<a>")
    assert e.completion_ids[0] == CharTok.SPECIAL[OPEN] and e.completion_ids[-1] == END_ID
    assert e.n_tokens == len(e.prompt_ids) + len(e.completion_ids)


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r.update(thinking=[]), "thinking"),
    (lambda r: r.update(thinking=["ok", ""]), "empty thinking line"),
    (lambda r: r.update(thinking=["uses " + CLOSE]), "marker"),
    (lambda r: r.update(thinking=["see \\boxed{2}"]), "boxed"),
    (lambda r: r.update(reference_answer=""), "empty"),
    (lambda r: r.update(problem=" "), "problem"),
])
def test_bad_rows_are_skipped_with_a_reason(mutate, reason):
    row = json.loads(json.dumps(ROW))
    mutate(row)
    ex, skipped = build([row])
    assert not ex and len(skipped) == 1 and reason in skipped[0][1]


def test_too_long_rows_are_skipped():
    ex, skipped = build([ROW], max_tokens=50)
    assert not ex and "too long" in skipped[0][1]


def test_load_rows_orders_filters_and_limits(tmp_path):
    rows = [dict(ROW, id="c", order=3, split="val"), dict(ROW, id="a", order=1), dict(ROW, id="b", order=2)]
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert [r["id"] for r in load_rows(f)] == ["a", "b", "c"]  # sorted by order, blank line ignored
    assert [r["id"] for r in load_rows(f, ["train"])] == ["a", "b"]
    assert [r["id"] for r in load_rows(f, ["val"])] == ["c"]
    assert [r["id"] for r in load_rows(f, ["train", "val"], limit=2)] == ["a", "b"]
    no_order = [dict(r) for r in rows]
    for r in no_order:
        r.pop("order")
    g = tmp_path / "d.jsonl"
    g.write_text("\n".join(json.dumps(r) for r in no_order))
    assert [r["id"] for r in load_rows(g)] == ["c", "a", "b"]  # file order kept when `order` is missing
    (tmp_path / "bad.jsonl").write_text("{not json}\n")
    with pytest.raises(ValueError):
        load_rows(tmp_path / "bad.jsonl")


def test_stats_counts():
    ex, _ = build([ROW, dict(ROW, id="x", level=0, domain="phys")])
    s = dataset_stats(ex)
    assert s["n"] == 2 and s["by_level"] == {0: 1, 2: 1} and s["by_domain"] == {"math": 1, "phys": 1}
    assert s["trainable_tokens"] == sum(len(e.completion_ids) for e in ex)


# ---- pure training helpers
sft = pytest.importorskip("rlaif.sft")


def test_lr_schedule_shape():
    base, total, warm = 1e-4, 100, 5
    lrs = [sft.lr_at(s, total, warm, base, 0.1) for s in range(total)]
    assert lrs[0] == pytest.approx(base / warm) and lrs[warm - 1] == pytest.approx(base)
    assert lrs[warm] == pytest.approx(base)                      # cosine starts at the peak
    assert all(a >= b - 1e-15 for a, b in zip(lrs[warm:], lrs[warm + 1:]))
    assert lrs[-1] == pytest.approx(base * 0.1, rel=0.02)


def test_batches_keep_curriculum_order_and_cover_everything():
    b = sft.make_batches(10, 4, 2, 0, 1)
    assert [len(x[1]) for x in b] == [4, 4, 2, 4, 4, 2]
    for ep in (0, 1):
        flat = [i for e, ix in b if e == ep for i in ix]
        assert flat == list(range(10))                           # strict order, every epoch


def test_shuffle_window_is_local_and_deterministic():
    a = sft.make_batches(20, 5, 1, 5, 7)
    b = sft.make_batches(20, 5, 1, 5, 7)
    assert a == b
    flat = [i for _, ix in a for i in ix]
    assert sorted(flat) == list(range(20))
    for w in range(0, 20, 5):                                    # indices never leave their window
        assert sorted(flat[w:w + 5]) == list(range(w, w + 5))
    assert flat != list(range(20))


@pytest.mark.parametrize("name", ["1_sft_mix3d_stage1.yaml", "2_sft_mix3d_stage2.yaml", "3_sft_nash.yaml"])
def test_shipped_config_is_valid(name):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = sft.load_sft_config(str(root / "configs" / name))
    assert cfg.train.batch_examples == 16 and cfg.model.lora_rank == 64 and cfg.model.lora_alpha == 128
    assert root.joinpath(cfg.data.prompt_path).exists()
    with pytest.raises(Exception):
        sft.SFTConfig(bogus=1)


def test_history_turns_are_rendered_and_system_turns_dropped():
    from rlaif.sft_data import render_prompt
    hist = [{"role": "system", "content": "OLD SYSTEM"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    seen = {}

    class T:
        def apply_chat_template(self, msgs, **kw):
            seen["msgs"] = msgs
            return "|".join(f"{m['role']}:{m['content']}" for m in msgs)

    out = render_prompt(T(), "{problem}", "q2", True, system="SYS", history=hist)
    assert out == "system:SYS|user:q1|assistant:a1|user:q2"          # old system turn dropped, configured one first


def test_plain_reply_keeps_markdown_and_newlines_in_the_target():
    reply = "The result is **4**.\n\n- one\n- two\n\n**Answer:** 4"
    row = {**ROW, "id": "rw-1", "answer_style": "plain", "prompt_style": "raw", "reference_answer": reply}
    text = build_completion(row, OPEN, CLOSE, END)
    assert text.endswith(CLOSE + reply + END) and "\\boxed" not in text


def test_no_thinking_rows_get_an_empty_think_block():
    row = {**ROW, "id": "chat-1", "no_thinking": True, "thinking": [], "answer_style": "plain", "prompt_style": "raw",
           "reference_answer": "Hello! How can I help?"}
    text = build_completion(row, OPEN, CLOSE, END)
    assert text == f"{OPEN}{CLOSE}Hello! How can I help?{END}"
    (ex,), skipped = build([row])
    assert not skipped
    p = parse_completion(ex.completion_ids, tok, think_open=OPEN, think_close=CLOSE, opened_in_prompt=False,
                         finish_reason="stop", max_steps=48, require_box=False)
    assert p.hard_valid and p.final_answer == "Hello! How can I help?" and p.thinking_text == ""


def test_free_style_keeps_boxed_inside_explanation():
    reply = "The limit is 3, since the quadratic terms cancel.\n\nHence \\boxed{3}."
    row = {**PLAIN, "id": "free-1", "answer_style": "free", "reference_answer": reply}
    text = build_completion(row, OPEN, CLOSE, END)
    assert text.endswith(CLOSE + reply + END)
    (ex,), skipped = build([row])
    assert not skipped and ex.answer_style == "free" and ex.completion_ids[-1] == END_ID


@pytest.mark.parametrize("bad", ["", "x " + END + " y"])
def test_free_style_rejects_marker_and_empty(bad):
    row = {**PLAIN, "id": "free-2", "answer_style": "free", "reference_answer": bad}
    ex, skipped = build([row])
    assert skipped and not ex


def test_target_ids_outside_the_output_head_are_skipped():
    ex, sk = build_dataset(FakeChatTok(), [dict(ROW, id="a")], TEMPLATE, thinking=True, think_open=OPEN, think_close=CLOSE,
                           end_token=END, end_id=END_ID, max_tokens=10_000, max_target_id=10)
    assert not ex and "outside the output head" in sk[0][1]
    ex, sk = build_dataset(FakeChatTok(), [dict(ROW, id="a")], TEMPLATE, thinking=True, think_open=OPEN, think_close=CLOSE,
                           end_token=END, end_id=END_ID, max_tokens=10_000, max_target_id=10**9)
    assert len(ex) == 1
