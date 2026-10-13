"""Install a trained Applio/RVC model into the voice registry.

    python -m tools.install_voice --name qari012 --display-name "Qari 012" \
        --f0-source /path/to/training.wav --consent-source "..."
"""
import argparse
import re
import shutil
from pathlib import Path

import yaml

APPLIO_LOGS = Path(__file__).resolve().parent.parent / "third_party" / "applio" / "logs"
_WEIGHTS = re.compile(r"^(?P<name>.+)_(?P<epoch>\d+)e_(?P<steps>\d+)s\.pth$")


def latest_weights(logs_dir: Path, name: str) -> Path:
    best, best_epoch = None, -1
    for p in logs_dir.glob("*.pth"):
        m = _WEIGHTS.match(p.name)
        if m and m["name"] == name and int(m["epoch"]) > best_epoch:
            best, best_epoch = p, int(m["epoch"])
    if best is None:
        raise FileNotFoundError(f"no exported weights like {name}_<N>e_<M>s.pth in {logs_dir}")
    return best


def median_f0(path: str, seconds: float = 90.0) -> float:
    import librosa

    y, sr = librosa.load(path, sr=16000, mono=True, duration=seconds)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
    f0 = f0[voiced & ~(f0 != f0)]
    if len(f0) == 0:
        raise ValueError("no voiced frames found; cannot estimate F0")
    return float(sorted(f0)[len(f0) // 2])


def install_voice(voice_id: str, display_name: str, logs_dir: Path, voices_dir: Path,
                  f0_median_hz: float, consent_source: str) -> Path:
    if not consent_source.strip():
        raise ValueError("consent_source is required")
    weights = latest_weights(logs_dir, voice_id)
    index = next(iter(sorted(logs_dir.glob("*.index"))), None)
    dest = voices_dir / voice_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dest / "model.pth")
    meta = {"name": display_name, "f0_median_hz": round(f0_median_hz, 1),
            "consent_source": consent_source.strip(), "model": "model.pth"}
    if index is not None:
        shutil.copy2(index, dest / "added.index")
        meta["index"] = "added.index"
    (dest / "voice.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Applio model name; also the voice id")
    p.add_argument("--display-name", required=True)
    p.add_argument("--f0-source", required=True, help="a wav/mp3 of the target voice for F0 estimation")
    p.add_argument("--consent-source", required=True)
    p.add_argument("--logs", default=str(APPLIO_LOGS / "{name}"))
    p.add_argument("--voices", default="voices")
    a = p.parse_args()
    f0 = median_f0(a.f0_source)
    dest = install_voice(a.name, a.display_name, Path(a.logs.format(name=a.name)), Path(a.voices),
                         f0, a.consent_source)
    print(f"installed {dest} (median F0 {f0:.0f} Hz)")


if __name__ == "__main__":
    main()
