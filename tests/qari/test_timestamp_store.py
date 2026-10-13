import json

import pytest

from qari.align.quran_align import convert
from qari.align.store import Segment, TimestampStore
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex


@pytest.fixture(scope="module")
def idx():
    return WordIndex.from_file()


@pytest.fixture(scope="module")
def timings(idx):
    """Every ayah gets words at 100 ms steps, 90 ms long; 2:2's last word is open-ended."""
    entries = []
    for s, a in sorted({(w.surah, w.ayah) for w in idx.words}):
        f, l = idx.ayah_range(s, a)
        entries.append({"surah": s, "ayah": a, "stats": {},
                        "segments": [[i, i + 1, i * 100, i * 100 + 90] for i in range(l - f + 1)]})
    t = convert(entries, idx)
    t[(2, 2)].words[-1] = type(t[(2, 2)].words[-1])(t[(2, 2)].words[-1].start_ms, None, 2)
    return t


@pytest.fixture()
def store(idx, timings):
    return TimestampStore.from_timings(load_reciter("afasy"), idx, timings, source="test")


def test_range_inside_one_ayah_is_one_segment(store, idx):
    g = idx.gid(2, 255, 3)
    segs = store.segments(g, g + 4)
    assert segs == [Segment("https://everyayah.com/data/Alafasy_128kbps/002255.mp3", 200, 690, 2, 255, 3, 7)]


def test_range_crossing_ayahs_and_surahs_yields_one_segment_per_ayah(store, idx):
    last_of_fatiha = idx.surah_range(1)[1]
    segs = store.segments(last_of_fatiha - 1, last_of_fatiha + 2)  # 1:7 last 2 words + 2:1 first 2? 2:1 has 1 word
    assert [(s.surah, s.ayah) for s in segs] == [(1, 7), (2, 1), (2, 2)]
    assert segs[0].audio_url.endswith("001007.mp3")


def test_open_ended_word_passes_through_as_none(store, idx):
    first, last = idx.ayah_range(2, 2)
    seg = store.segments(first, last)[0]
    assert seg.end_ms is None and seg.start_ms == 0


def test_basmala_segment_uses_1_1_audio(store, idx):
    b = store.basmala_segment()
    assert b.audio_url.endswith("001001.mp3") and (b.surah, b.ayah, b.first_pos, b.last_pos) == (1, 1, 1, 4)
    assert (b.start_ms, b.end_ms) == (0, 390)


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
