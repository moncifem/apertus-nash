from rlaif.novelty import Archive, group_novelty, signature, similarity

A = "Multiply by 2ab and factor so that integer solutions become divisor pairs. Then (a-2)(b-2)=4 lists the divisor pairs of four, giving the solutions."
B = "Multiply by 2ab and factor so that integer solutions become divisor pairs. Then (a-3)(b-3)=9 lists the divisor pairs of nine giving solutions."
C = "Assume a is at most b by symmetry, bound a between three and four using the inequality, then solve for b directly in each remaining case."


def test_same_method_different_numbers_is_similar_and_other_method_is_not():
    a, b, c = signature(A), signature(B), signature(C)
    assert similarity(a, b) > 0.3 > similarity(a, c)
    nov = group_novelty([a, b, c])
    assert nov[2] > nov[0] and nov[2] > nov[1]


def test_short_text_has_no_signature_and_archive_reduces_novelty(tmp_path):
    assert signature("too short") is None and group_novelty([None, signature(A)])[0] is None
    ar = Archive(str(tmp_path / "a.json"))
    s = signature(A)
    assert ar.novelty("math", s) == 1.0
    ar.add("math", s); ar.save()
    assert Archive(str(tmp_path / "a.json")).novelty("math", s) == 0.0
    assert ar.novelty("physics", s) == 1.0
