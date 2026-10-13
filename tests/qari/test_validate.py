import numpy as np
import soundfile as sf

from qari.align.quran_align import AyahTimings, WordTiming, convert
from qari.align.store import TimestampStore
from qari.align.validate import cut_clip, sample_clips, stats
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex

IDX = WordIndex.from_file()


def make_store(mutate=None):
    entries = []
    for s, a in sorted({(w.surah, w.ayah) for w in IDX.words}):
        f, l = IDX.ayah_range(s, a)
        entries.append({"surah": s, "ayah": a, "stats": {},
                        "segments": [[i, i + 1, i * 100, i * 100 + 90] for i in range(l - f + 1)]})
    t = convert(entries, IDX)
    if mutate:
        mutate(t)
    return TimestampStore.from_timings(load_reciter("afasy"), IDX, t, source="test")


def test_clean_store_has_no_defects():
    s = stats(make_store())
    assert s["words"] == len(IDX)
    assert (s["non_positive"], s["overlaps"], s["open_ended"], s["suspect_ayahs"]) == (0, 0, 0, 0)
    assert s["gaps_over_500ms"] == 0 and s["flags"] == {"ok": len(IDX), "split": 0, "interpolated": 0, "suspect": 0}


def test_stats_find_injected_defects():
    def mutate(t):
        w = t[(112, 1)].words
        w[1] = WordTiming(w[1].start_ms, w[1].start_ms, 0)         # zero duration
        w[2] = WordTiming(w[1].end_ms - 5, w[2].end_ms, 0)        # overlaps previous word
        w[3] = WordTiming(5000, None, 2)                          # open ended + 4 s gap
        t[(1, 2)] = AyahTimings(t[(1, 2)].words, True)
    s = stats(make_store(mutate))
    assert s["non_positive"] == 1 and s["overlaps"] == 1 and s["open_ended"] == 1
    assert s["gaps_over_500ms"] == 1 and s["suspect_ayahs"] == 1 and s["flags"]["interpolated"] == 1


def test_cut_clip_slices_by_milliseconds_with_padding():
    sr = 1000
    audio = np.arange(5000, dtype=np.float32)
    assert cut_clip(audio, sr, 1000, 2000).tolist() == list(range(1000, 2000))
    assert len(cut_clip(audio, sr, 1000, 2000, pad_ms=100)) == 1200
    assert len(cut_clip(audio, sr, 4900, None)) == 100            # open end: to end of file
    assert len(cut_clip(audio, sr, 0, 500, pad_ms=100)) == 600     # padding clamps at file start


def test_sample_clips_writes_one_wav_per_word(tmp_path):
    store = make_store()
    sr = 8000
    fake = lambda url: (np.zeros(sr * 2, dtype=np.float32), sr)    # 2 s of silence per ayah
    written = sample_clips(store, n_ayahs=3, seed=1, out_dir=tmp_path, fetch=fake)
    assert written and all(p.suffix == ".wav" for p in written)
    data, rate = sf.read(written[0])
    assert rate == sr and len(data) > 0
    assert sample_clips(store, 3, 1, tmp_path / "again", fake) == [
        tmp_path / "again" / p.name for p in written]               # deterministic for a seed


def test_check_durations_reports_tail_after_last_word():
    from qari.align.validate import check_durations
    sr = 1000
    fake = lambda url: (np.zeros(5000, dtype=np.float32), sr)
    rows = check_durations(make_store(), 2, 3, fake)
    assert len(rows) == 2 and all(r["audio_ms"] == 5000 for r in rows)
    assert all(r["tail_ms"] == 5000 - r["last_end_ms"] for r in rows)
