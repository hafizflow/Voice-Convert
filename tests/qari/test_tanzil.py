from pathlib import Path

import pytest

from qari.text.tanzil import build_words, check_complete, parse_tanzil

FIXTURE = Path(__file__).parent / "fixtures" / "tanzil_sample.txt"


@pytest.fixture(scope="module")
def sample_words():
    return build_words(parse_tanzil(FIXTURE.read_text(encoding="utf-8")))


def test_comments_and_blank_lines_are_skipped():
    rows = parse_tanzil("# c\n\n1|1|بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ\n#x\n")
    assert rows == [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")]


def _ayah(words, s, a):
    return [w for w in words if (w.surah, w.ayah) == (s, a)]


def test_fatiha_basmala_is_ayah_one_and_kept(sample_words):
    assert [w.pos for w in _ayah(sample_words, 1, 1)] == [1, 2, 3, 4]


def test_basmala_prefix_stripped_from_other_surahs(sample_words):
    assert [w.norm for w in _ayah(sample_words, 2, 1)] == ["الم"]


def test_surah_9_has_no_prefix_to_strip(sample_words):
    assert _ayah(sample_words, 9, 1)[0].norm == "براءه"


def test_surah_95_basmala_spelled_with_shadda_is_still_detected(sample_words):
    first = _ayah(sample_words, 95, 1)[0]
    assert (first.norm, first.pos) == ("والتين", 1)


def test_gids_are_contiguous_and_uthmani_is_verbatim(sample_words):
    assert [w.gid for w in sample_words] == list(range(len(sample_words)))
    raw = FIXTURE.read_text(encoding="utf-8")
    assert all(w.uthmani in raw for w in sample_words)


def test_missing_basmala_prefix_is_an_error():
    rows = [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"), (2, 1, "الٓمٓ ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ")]
    with pytest.raises(ValueError, match="2:1"):
        build_words(rows)


def test_ayah_gap_is_an_error():
    rows = [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"), (1, 3, "x")]
    with pytest.raises(ValueError, match="gap"):
        build_words(rows)


def test_crlf_and_bom_are_tolerated():
    text = "﻿# c\r\n1|1|بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ\r\n"
    assert parse_tanzil(text) == [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")]


def test_word_that_normalizes_to_nothing_is_an_error():
    rows = [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"), (1, 2, "َ")]
    with pytest.raises(ValueError, match="normalizes to nothing"):
        build_words(rows)


def test_check_complete_rejects_missing_surahs():
    # the sample has surahs 1, 2, 9, 95, 97 only
    words = build_words(parse_tanzil(FIXTURE.read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="114"):
        check_complete(words)


def test_check_complete_rejects_truncated_surah():
    rows = parse_tanzil(FIXTURE.read_text(encoding="utf-8"))
    rows = [r for r in rows if r[:2] != (1, 7)]
    with pytest.raises(ValueError):
        check_complete(build_words(rows))


def test_ayah_that_is_only_the_basmala_is_an_error():
    rows = [(1, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"), (2, 1, "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")]
    with pytest.raises(ValueError, match="2:1"):
        build_words(rows)
