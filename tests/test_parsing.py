from conftest import CLOSE, END, OPEN, CharTok

from rlaif.parsing import extract_boxed, parse_completion

tok = CharTok()


def test_prose_paragraphs_have_exact_token_spans_without_numbering():
    thought = "Before computing, x^2-1=(x-1)*(x+1).\n\nTo check, substitute x=1."
    ids = tok.encode(OPEN + thought + CLOSE + "The answer is 1." + END)
    result = parse_completion(ids, tok, think_open=OPEN, think_close=CLOSE,
                              opened_in_prompt=False, finish_reason="stop", max_steps=48,
                              require_box=False, reasoning_format="prose")
    assert result.format_valid and result.spans_ok, result.errors
    assert len(result.steps) == 2
    assert result.steps[0].tok_end == result.steps[1].tok_start
    assert tok.decode(ids[result.steps[0].tok_start:result.steps[0].tok_end]).startswith("Before computing")
    assert result.final_answer == "The answer is 1."


def test_prose_mode_accepts_one_block_and_rejects_numbered_steps():
    for thought, valid in [("I factor x^2-1.", True), ("1. I factor x^2-1.", False)]:
        result = parse_completion(tok.encode(OPEN+thought+CLOSE+"1"+END), tok,
                                  think_open=OPEN, think_close=CLOSE, opened_in_prompt=False,
                                  finish_reason="stop", max_steps=48, require_box=False,
                                  reasoning_format="prose")
        assert result.format_valid == valid


def parse(text, opened=True, finish="stop", max_steps=48):
    return parse_completion(tok.encode(text), tok, think_open=OPEN, think_close=CLOSE,
                            opened_in_prompt=opened, finish_reason=finish, max_steps=max_steps)


def test_valid_trace_and_exact_spans():
    text = "\n1. Divide by 2: x = 6/2 = 3.\n   (2 is nonzero)\n2. So x = 3.\n" + CLOSE + "\\boxed{3}" + END
    p = parse(text)
    assert p.format_valid, p.errors
    assert p.final_answer == "3"
    assert [s.id for s in p.steps] == [1, 2]
    assert "nonzero" in p.steps[0].text  # continuation line belongs to step 1
    ids = tok.encode(text)
    for s in p.steps:
        assert tok.decode(ids[s.tok_start:s.tok_end]).strip().startswith(f"{s.id}.")
    assert p.steps[0].tok_end == p.steps[1].tok_start
    assert ids[p.steps[-1].tok_end] == CharTok.SPECIAL[CLOSE]  # answer tokens excluded
    assert p.spans_ok


def test_model_emits_open_marker():
    p = parse(OPEN + "1. 2+2=4\n" + CLOSE + "\\boxed{4}", opened=False)
    assert p.format_valid and p.final_answer == "4"


def test_decimal_line_is_not_a_step():
    p = parse("1. Price is\n1.5 dollars each, so 3 cost 4.5\n" + CLOSE + "\\boxed{4.5}")
    assert p.format_valid and len(p.steps) == 1


def test_format_errors():
    assert "bad_numbering" in parse("1. a=1\n3. b=2\n" + CLOSE + "\\boxed{1}").errors
    assert "multiple_boxed_answers" in parse("1. a=1\n" + CLOSE + "\\boxed{1} or \\boxed{2}").errors
    assert "think_close_count" in parse("1. a=1\n\\boxed{1}", finish="length").errors
    p = parse("Let me think.\n1. a=1\n" + CLOSE + "\\boxed{1}")
    assert p.format_valid and p.spans_ok  # prose before step 1 is tolerated
    assert "nested_think_open" in parse("1. a " + OPEN + "\n" + CLOSE + "\\boxed{1}").errors
    assert "empty_step" in parse("1. a=1\n2. \n" + CLOSE + "\\boxed{1}").errors
    assert "too_many_steps" in parse("".join(f"{i}. x\n" for i in range(1, 6)) + CLOSE + "\\boxed{1}", max_steps=3).errors


def test_boxed_in_reasoning_is_ignored():
    p = parse("1. Guess \\boxed{7}\n2. Correct is 8\n" + CLOSE + "\\boxed{8}")
    assert p.final_answer == "8"


def test_extract_boxed_nested():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == [r"\frac{1}{2}"]


def test_truncated_flag():
    p = parse("1. a=1\n2. b", finish="length")
    assert p.truncated and not p.format_valid


def test_hard_vs_soft_errors():
    p = parse("Just free-form reasoning, no numbered steps.\n" + CLOSE + "\\boxed{7}")
    assert "no_steps" in p.errors and not p.format_valid
    assert p.hard_valid and not p.spans_ok  # usable for outcome reward, not for the step judge
    assert not parse("1. a=1\n" + CLOSE + "no answer").hard_valid
    assert not parse("1. a=1\n" + CLOSE + "\\boxed{1} \\boxed{2}").hard_valid


def test_repetition_fraction():
    from rlaif.parsing import repetition_fraction
    assert repetition_fraction("the quick brown fox jumps over the lazy dog today and tomorrow again") == 0.0
    assert repetition_fraction("wait let me check again " * 40) > 0.9


def test_anchor_mode_reads_the_last_answer_line_and_strips_markup():
    from conftest import CLOSE, OPEN, CharTok
    tok = CharTok()
    kw = dict(think_open=OPEN, think_close=CLOSE, opened_in_prompt=False, finish_reason="stop", max_steps=48, answer_mode="anchor")

    def run(reply):
        text = f"{OPEN}Question: q\nKey idea: k\n1. step\n2. done{CLOSE}{reply}"
        return parse_completion([tok.encode(c)[0] if False else ord(c) for c in text], tok, **kw)

    ok = run("The result is **4**.\n\n**Answer:** $\\dfrac{1}{3}$")
    assert ok.final_answer == "\\dfrac{1}{3}" and ok.hard_valid
    assert run("**Answer:** 1\nlater text\n**Answer:** **7**").final_answer == "7"          # last anchor wins
    missing = run("It is 4, trust me.")
    assert missing.final_answer is None and "no_answer_line" in missing.errors and not missing.hard_valid
    assert "no_boxed_answer" not in missing.errors                                          # boxed answers are not required in this mode
