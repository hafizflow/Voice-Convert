"""Build data/timestamps/<reciter>.json from the pinned quran-align release (CC BY 4.0).

    python -m tools.build_timestamps --reciter afasy
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import httpx

from qari.align.quran_align import convert
from qari.align.store import TimestampStore
from qari.align.validate import stats
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex

ZIP_URL = "https://github.com/cpfair/quran-align/releases/download/release-2016-11-24/quran-align-data-2016-11-24.zip"
ZIP_SHA256 = "5eeb045d8a7895208c94d2d7ec243567f8f550728835411527c4ffa1e789c9b7"
MEMBERS = {"afasy": "Alafasy_128kbps.json"}
ROOT = Path(__file__).resolve().parents[1]


def fetch_zip(cache: Path) -> Path:
    path = cache / "quran-align-data-2016-11-24.zip"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(httpx.get(ZIP_URL, timeout=120, follow_redirects=True).raise_for_status().content)
        tmp.replace(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != ZIP_SHA256:
        raise SystemExit(f"{path}: sha256 {got} != pinned {ZIP_SHA256}; delete it and re-run, "
                         "or the release asset changed and the pin needs a deliberate update")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reciter", default="afasy")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "timestamps")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    args = ap.parse_args(argv)

    reciter = load_reciter(args.reciter)
    if reciter.timestamps_source != "quran-align" or args.reciter not in MEMBERS:
        raise SystemExit(f"{args.reciter}: no quran-align source configured")
    idx = WordIndex.from_file()
    entries = json.loads(zipfile.ZipFile(fetch_zip(args.cache)).read(MEMBERS[args.reciter]))
    store = TimestampStore.from_timings(reciter, idx, convert(entries, idx),
                                        source=f"quran-align release-2016-11-24 {MEMBERS[args.reciter]}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / reciter.timestamps_file
    store.write(path)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps(stats(store), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
