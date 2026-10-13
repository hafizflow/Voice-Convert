import json

import pytest

from qari.align.quran_align import AyahTimings, WordTiming, convert
from qari.align.store import Segment, TimestampStore
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex


@pytest.fixture(scope="module")
def idx():
    return WordIndex.from_file()


@pytest.fixture(scope="module")
def timings(idx):
    """Every ayah gets words at 100 ms steps, 90 ms long."""
    entries = []
    for s, a in sorted({(w.surah, w.ayah) for w in idx.words}):
        f, l = idx.ayah_range(s, a)
        entries.append({"surah": s, "ayah": a, "stats": {},
                        "segments": [[i, i + 1, i * 100, i * 100 + 90] for i in range(l - f + 1)]})
    return convert(entries, idx)


@pytest.fixture()
def store(idx, timings):
    return TimestampStore.from_timings(load_reciter("afasy"), idx, timings, source="test")


def test_range_inside_one_ayah_is_one_segment(store, idx):
    g = idx.gid(2, 255, 3)
    segs = store.segments(g, g + 4)
    assert segs == [Segment("https://everyayah.com/data/Alafasy_128kbps/002255.mp3", 200, 690, 2, 255, 3, 7, "exact", False)]


def test_range_crossing_ayahs_and_surahs_yields_one_segment_per_ayah(store, idx):
    last_of_fatiha = idx.surah_range(1)[1]
    segs = store.segments(last_of_fatiha - 1, last_of_fatiha + 2)  # 1:7 last 2 words + 2:1 first 2? 2:1 has 1 word
    assert [(s.surah, s.ayah) for s in segs] == [(1, 7), (2, 1), (2, 2)]
    assert segs[0].audio_url.endswith("001007.mp3")


def test_approximate_words_make_the_segment_approx(idx, timings):
    t = dict(timings)
    words = list(t[(112, 1)].words)
    words[1] = WordTiming(words[1].start_ms, words[1].end_ms, 2)
    t[(112, 1)] = AyahTimings(words, False)
    s = TimestampStore.from_timings(load_reciter("afasy"), idx, t, source="test")
    g = idx.gid(112, 1, 1)
    assert s.segments(g, g)[0].quality == "exact"
    assert s.segments(g, g + 2)[0].quality == "approx"


def test_suspect_ayah_plays_whole_but_refuses_partial_ranges(idx, timings):
    t = dict(timings)
    n = len(t[(10, 1)].words)
    t[(10, 1)] = AyahTimings([WordTiming(400, None, 3)] * n, True)
    s = TimestampStore.from_timings(load_reciter("afasy"), idx, t, source="test")
    first, last = idx.ayah_range(10, 1)
    whole = s.segments(first, last)[0]
    assert (whole.start_ms, whole.end_ms, whole.quality) == (400, None, "exact")
    part = s.segments(first + 2, first + 2)[0]
    assert part.quality == "unreliable"


def test_segments_rejects_bad_gid_ranges(store, idx):
    for a, b in [(-1, -1), (10, 5), (0, len(idx))]:
        with pytest.raises(ValueError):
            store.segments(a, b)


def test_load_rejects_file_for_another_reciter(store, tmp_path):
    p = tmp_path / "afasy.json"
    store.write(p)
    other = load_reciter("afasy")
    other = type(other)("husary", other.display_name, other.audio_url_template, other.timestamps_source,
                        other.timestamps_file, other.license)
    with pytest.raises(ValueError, match="husary"):
        TimestampStore.load(p, other, store.index)


def test_basmala_segment_uses_1_1_audio(store, idx):
    b = store.basmala_segment()
    assert b.audio_url.endswith("001001.mp3") and (b.surah, b.ayah, b.first_pos, b.last_pos) == (1, 1, 1, 4)
    assert (b.start_ms, b.end_ms) == (0, 390) and b.quality == "exact"


def test_write_and_load_roundtrip(store, idx, tmp_path):
    p = tmp_path / "afasy.json"
    store.write(p)
    again = TimestampStore.load(p, load_reciter("afasy"), idx)
    g = idx.gid(112, 1, 2)
    assert again.segments(g, g) == store.segments(g, g)
    assert again.is_suspect(2, 2) is False


def test_load_rejects_ayah_with_wrong_word_count(store, idx, tmp_path):
    p = tmp_path / "bad.json"
    store.write(p)
    d = json.loads(p.read_text())
    d["ayahs"]["112:1"].pop()
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="112:1"):
        TimestampStore.load(p, load_reciter("afasy"), idx)


def test_load_rejects_missing_ayah(store, idx, tmp_path):
    p = tmp_path / "bad.json"
    store.write(p)
    d = json.loads(p.read_text())
    del d["ayahs"]["114:6"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="114:6"):
        TimestampStore.load(p, load_reciter("afasy"), idx)


def test_suspect_ayahs_are_reported(idx, timings):
    t = dict(timings)
    t[(10, 1)] = type(t[(10, 1)])(t[(10, 1)].words, True)
    s = TimestampStore.from_timings(load_reciter("afasy"), idx, t, source="test")
    assert s.is_suspect(10, 1) and not s.is_suspect(10, 2)


def test_range_ending_on_last_word_is_marked_and_tiny_words_are_approx(idx, timings):
    t = dict(timings)
    words = list(t[(112, 1)].words)
    words[0] = WordTiming(0, 20, 0)          # source says 20 ms: not a hearable word
    t[(112, 1)] = AyahTimings(words, False)
    s = TimestampStore.from_timings(load_reciter("afasy"), idx, t, source="test")
    first, last = idx.ayah_range(112, 1)
    assert s.segments(first, first)[0].quality == "approx"
    assert s.segments(first, first)[0].ends_ayah is False
    assert s.segments(first + 1, last)[0].ends_ayah is True
