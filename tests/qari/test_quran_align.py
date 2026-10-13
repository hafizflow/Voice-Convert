from qari.align.quran_align import INTERPOLATED, OK, SPLIT, convert, convert_ayah
from qari.text.index import WordIndex


def T(w):
    return (w.start_ms, w.end_ms, w.flag)


def test_clean_ayah_maps_one_to_one():
    a = convert_ayah([[0, 1, 30, 600], [1, 2, 610, 900], [2, 3, 910, 2000]], ["aa", "bb", "cc"])
    assert [T(w) for w in a.words] == [(30, 600, OK), (610, 900, OK), (910, 2000, OK)]
    assert not a.suspect


def test_merged_segment_is_split_by_character_weight():
    a = convert_ayah([[0, 2, 0, 900], [2, 3, 910, 1500]], ["ab", "abcd", "x"])
    assert [T(w) for w in a.words] == [(0, 300, SPLIT), (300, 900, SPLIT), (910, 1500, OK)]


def test_missing_word_is_interpolated_between_neighbours():
    a = convert_ayah([[0, 1, 0, 500], [2, 3, 700, 1000]], ["aa", "bb", "cc"])
    assert T(a.words[1]) == (500, 700, INTERPOLATED)


def test_missing_run_is_split_by_weight():
    a = convert_ayah([[0, 1, 0, 100], [3, 4, 400, 500]], ["a", "a", "aaa", "b"])
    assert [T(w) for w in a.words[1:3]] == [(100, 175, INTERPOLATED), (175, 400, INTERPOLATED)]


def test_non_positive_duration_segment_is_treated_as_missing():
    a = convert_ayah([[0, 1, 0, 500], [1, 2, 890, 890], [2, 3, 900, 1200]], ["aa", "bb", "cc"])
    assert T(a.words[1]) == (500, 900, INTERPOLATED)
    b = convert_ayah([[0, 1, 0, 500], [1, 2, 10800, 10790], [2, 3, 10800, 11000]], ["aa", "bb", "cc"])
    assert b.words[1].flag == INTERPOLATED and b.words[1].end_ms >= b.words[1].start_ms


def test_trailing_missing_word_has_open_end():
    a = convert_ayah([[0, 1, 0, 500]], ["aa", "bb"])
    assert T(a.words[1]) == (500, None, INTERPOLATED)


def test_leading_missing_word_starts_at_zero():
    a = convert_ayah([[1, 2, 400, 800]], ["aa", "bb"])
    assert T(a.words[0]) == (0, 400, INTERPOLATED)


def test_out_of_range_index_marks_ayah_suspect_and_is_ignored():
    a = convert_ayah([[0, 1, 0, 500], [1, 2, 600, 900]], ["aa"])
    assert a.suspect and len(a.words) == 1 and T(a.words[0]) == (0, 500, OK)


def test_convert_uses_index_word_counts():
    idx = WordIndex.from_file()
    n = idx.ayah_range(112, 1)[1] - idx.ayah_range(112, 1)[0] + 1
    segs = [[i, i + 1, i * 100, i * 100 + 90] for i in range(n)]
    out = convert([{"surah": 112, "ayah": 1, "segments": segs, "stats": {}}], idx)
    assert len(out[(112, 1)].words) == n and not out[(112, 1)].suspect
