from qari.align.quran_align import INTERPOLATED, OK, SPLIT, SUSPECT, convert, convert_ayah
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


def test_trailing_missing_word_shares_the_last_span_instead_of_starting_after_it():
    # 50:34 / 2:181: the last segment already contains the unlocated words
    a = convert_ayah([[0, 1, 0, 500]], ["aa", "bb"])
    assert [T(w) for w in a.words] == [(0, 250, SPLIT), (250, 500, INTERPOLATED)]


def test_two_trailing_missing_words_get_distinct_ordered_slices():
    a = convert_ayah([[0, 3, 0, 900]], ["aaa", "bb", "cccc", "d", "ee"])
    ws = [T(w) for w in a.words]
    assert ws[3][0] >= ws[2][1] and ws[4][0] >= ws[3][1] and ws[4][1] == 900
    assert all(w[1] is not None and w[1] > w[0] for w in ws)


def test_missing_word_between_touching_neighbours_is_cut_from_the_denser_one():
    # 40 ms gap is too small to hold a word: it lives inside the neighbour with more ms per character
    a = convert_ayah([[0, 1, 0, 100], [2, 3, 110, 1110]], ["aaaa", "bbbb", "cccc"])
    assert T(a.words[2])[2] == SPLIT and T(a.words[1])[2] == INTERPOLATED
    assert a.words[1].start_ms >= 110 and a.words[1].end_ms <= a.words[2].start_ms + 600
    assert a.words[1].end_ms == a.words[2].start_ms and a.words[2].end_ms == 1110


def test_leading_missing_word_starts_at_zero():
    a = convert_ayah([[1, 2, 400, 800]], ["aa", "bb"])
    assert T(a.words[0]) == (0, 400, INTERPOLATED)


def test_source_with_more_words_than_tanzil_is_wholly_suspect():
    # 10:1: source counts 6 words, Tanzil 5. Per-word times cannot be trusted at all.
    a = convert_ayah([[0, 1, 400, 7200], [2, 3, 7210, 7480], [4, 5, 7490, 8700], [5, 6, 8710, 11670]],
                     ["a", "bb", "ccc", "dddd", "eeeee"])
    assert a.suspect and all(w.flag == SUSPECT and w.end_ms is None for w in a.words)
    assert a.words[0].start_ms == 400


def test_overlapping_index_ranges_are_suspect():
    a = convert_ayah([[0, 2, 0, 1000], [1, 2, 5000, 5100], [2, 3, 5200, 5300]], ["a", "b", "c"])
    assert a.suspect


def test_time_going_backwards_never_yields_start_after_end():
    a = convert_ayah([[0, 1, 1000, 1500], [1, 2, 200, 400], [2, 3, 1600, 1800],
                      [3, 4, 1900, 2000], [4, 5, 2100, 2200]], ["a"] * 5)
    prev_end = 0
    for w in a.words:
        assert w.end_ms >= w.start_ms >= prev_end
        prev_end = w.end_ms


def test_negative_times_are_clamped_or_rejected():
    a = convert_ayah([[0, 1, -10, 500], [1, 2, 510, 900]], ["a", "b"])
    assert a.words[0].start_ms == 0
    b = convert_ayah([[0, 1, -3000, 500], [1, 2, 510, 900]], ["a", "b"])
    assert all(w.start_ms >= 0 for w in b.words)


def test_ten_ms_overlap_is_clamped_not_rejected():
    a = convert_ayah([[0, 1, 0, 7440], [1, 2, 7420, 11750]], ["aa", "bb"])
    assert T(a.words[1]) == (7440, 11750, OK)


def test_convert_uses_index_word_counts():
    idx = WordIndex.from_file()
    n = idx.ayah_range(112, 1)[1] - idx.ayah_range(112, 1)[0] + 1
    segs = [[i, i + 1, i * 100, i * 100 + 90] for i in range(n)]
    out = convert([{"surah": 112, "ayah": 1, "segments": segs, "stats": {}}], idx)
    assert len(out[(112, 1)].words) == n and not out[(112, 1)].suspect
