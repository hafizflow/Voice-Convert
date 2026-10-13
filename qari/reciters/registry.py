"""Reciter registry: one YAML file per reciter in registry/. Adding a Qari needs no code change."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_DIR = Path(__file__).parent / "registry"
_REQUIRED = ("id", "display_name", "audio_url_template", "timestamps_source", "timestamps_file", "license")
_REQUIRED_LICENSE = ("audio", "audio_status", "timestamps")


@dataclass(frozen=True)
class Reciter:
    id: str
    display_name: str
    audio_url_template: str     # format fields: {surah:03d} {ayah:03d}
    timestamps_source: str      # which align/ source builds the timestamp file
    timestamps_file: str        # path relative to data/timestamps/
    license: dict[str, str]     # audio, audio_status, timestamps: free text, read before launch

    def audio_url(self, surah: int, ayah: int) -> str:
        return self.audio_url_template.format(surah=surah, ayah=ayah)


AUDIO_STATUSES = frozenset({"confirmed", "unconfirmed", "denied"})
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load_reciter(reciter_id: str, registry_dir: Path = REGISTRY_DIR) -> Reciter:
    if not _SAFE_ID.match(reciter_id):
        raise KeyError(f"unknown reciter: {reciter_id!r}")
    path = Path(registry_dir) / f"{reciter_id}.yaml"
    if not path.exists():
        raise KeyError(f"unknown reciter: {reciter_id}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    lic = raw.get("license") or {}
    missing += [f"license.{k}" for k in _REQUIRED_LICENSE if k not in lic]
    if missing:
        raise ValueError(f"reciter {reciter_id}: missing {', '.join(missing)}")
    problems = []
    if raw["id"] != reciter_id:
        problems.append(f"id {raw['id']!r} does not match file name")
    if "{surah" not in raw["audio_url_template"] or "{ayah" not in raw["audio_url_template"]:
        problems.append("audio_url_template needs {surah} and {ayah}")
    if lic["audio_status"] not in AUDIO_STATUSES:
        problems.append(f"license.audio_status must be one of {sorted(AUDIO_STATUSES)}")
    if not all(isinstance(lic[k], str) and lic[k].strip() for k in _REQUIRED_LICENSE):
        problems.append("license values must be non-empty strings")
    tf = Path(raw["timestamps_file"])
    if tf.is_absolute() or ".." in tf.parts:
        problems.append("timestamps_file must be a plain relative path")
    if problems:
        raise ValueError(f"reciter {reciter_id}: " + "; ".join(problems))
    return Reciter(**{k: raw[k] for k in _REQUIRED})


def list_reciters(registry_dir: Path = REGISTRY_DIR) -> list[str]:
    return sorted(p.stem for p in Path(registry_dir).glob("*.yaml"))
