"""Validate a built timestamp file against the real audio.

    python -m tools.validate_timestamps --reciter afasy --durations 40 --clips 5
Fetches a few ayah files from the reciter's origin (nothing is stored except the clips you ask for).
"""
import argparse
import json
import sys
from pathlib import Path

from qari.align.store import TimestampStore
from qari.align.validate import check_durations, sample_clips, stats
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reciter", default="afasy")
    ap.add_argument("--durations", type=int, default=0, help="N random ayahs: compare last word end to file length")
    ap.add_argument("--clips", type=int, default=0, help="N random ayahs: cut every word to a wav for listening")
    ap.add_argument("--pad-ms", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clips-dir", type=Path, default=ROOT / "work" / "qari_clips")
    ap.add_argument("--list-defects", action="store_true")
    args = ap.parse_args(argv)

    reciter = load_reciter(args.reciter)
    idx = WordIndex.from_file()
    store = TimestampStore.load(ROOT / "data" / "timestamps" / reciter.timestamps_file, reciter, idx)
    print(json.dumps(stats(store), indent=2))
    if args.list_defects:
        for g in range(len(idx)):
            w, t = idx[g], store.word(g)
            if t.end_ms is None or t.end_ms <= t.start_ms or (g and idx[g - 1].ayah == w.ayah and t.start_ms < (store.word(g - 1).end_ms or 0)):
                print(f"defect {w.surah}:{w.ayah} word {w.pos} {t}")
    if args.durations:
        rows = check_durations(store, args.durations, args.seed)
        tails = sorted(r["tail_ms"] for r in rows if r["tail_ms"] is not None)
        print(f"durations: n={len(rows)} tail_ms min={tails[0]} median={tails[len(tails)//2]} max={tails[-1]}")
        for r in rows:
            if r["tail_ms"] is None or r["tail_ms"] < 0 or r["tail_ms"] > 1500:
                print("  check", r)
    if args.clips:
        files = sample_clips(store, args.clips, args.seed, args.clips_dir, pad_ms=args.pad_ms)
        print(f"wrote {len(files)} clips to {args.clips_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
