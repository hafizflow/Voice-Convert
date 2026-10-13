from qari.match.align import local_align, similarity, skeleton


def test_similarity_bounds_and_alef_tolerance():
    assert similarity("الله", "الله") == 1.0
    assert similarity("", "ب") == 0.0
    assert similarity("ذلك", "ذالك") == 0.75
    assert similarity(skeleton("ذالك"), skeleton("ذلك")) == 1.0   # skeletons agree


def test_skeleton_strips_alef_but_keeps_alef_only_word():
    assert skeleton("الرحمان") == "لرحمن"
    assert skeleton("ا") == "ا"


def test_exact_substring_aligns_with_all_matches():
    a = local_align(["ب", "ج", "د"], ["و", "ب", "ج", "د", "ه"])
    assert (a.r_start, a.r_end, a.q_start, a.q_end) == (1, 3, 0, 2)
    assert [o.kind for o in a.ops] == ["match"] * 3


def test_missing_word_inside_the_range_is_reported():
    a = local_align(["كتب", "علكم", "قتل"], ["كتب", "عليكم", "الصيام", "قتل"][:2] + ["xxxx", "قتل"])
    kinds = [o.kind for o in a.ops]
    assert "missing" in kinds or "substitution" in kinds


def test_extra_query_word_inside_the_range_is_reported():
    a = local_align(["كتب", "زززز", "عليكم", "الصيام"], ["كتب", "عليكم", "الصيام"])
    assert [o.kind for o in a.ops].count("extra") == 1


def test_no_alignment_for_unrelated_words():
    assert local_align(["قططط"], ["جملل"]) is None
