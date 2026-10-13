import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    f0_median_hz: float
    consent_source: str
    model_path: Path | None
    index_path: Path | None


def _load_one(d: Path) -> Voice:
    meta = yaml.safe_load((d / "voice.yaml").read_text())
    if not isinstance(meta, dict):
        raise ValueError("voice.yaml is not a mapping")
    consent = meta.get("consent_source")
    if not isinstance(consent, str) or not consent.strip():
        raise ValueError("consent_source is required")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    f0 = float(meta["f0_median_hz"])
    if f0 <= 0:
        raise ValueError("f0_median_hz must be positive")
    model = _optional_file(d, meta.get("model"))
    index = _optional_file(d, meta.get("index"))
    return Voice(d.name, name.strip(), f0, consent.strip(), model, index)


def _optional_file(d: Path, rel) -> Path | None:
    if rel is None:
        return None
    p = (d / str(rel)).resolve()
    if d.resolve() not in p.parents or not p.is_file():
        raise ValueError(f"declared file not found inside voice dir: {rel}")
    return d / str(rel)


def load_registry(root: Path) -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not root.is_dir():
        return voices
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            voices[d.name] = _load_one(d)
        except Exception as e:  # one bad voice must not stop the server
            log.warning("skipping voice %s: %s", d.name, e)
    return voices
