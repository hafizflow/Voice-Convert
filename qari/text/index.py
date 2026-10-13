from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TANZIL = Path(__file__).resolve().parents[2] / "data" / "tanzil" / "quran-uthmani.txt"


@dataclass(frozen=True)
class Word:
    gid: int        # position in the whole mushaf, 0-based, contiguous
    surah: int
    ayah: int
    pos: int        # 1-based within the ayah
    uthmani: str    # exact Tanzil token, the only text ever displayed
    norm: str       # matching key from normalize(); never displayed


class WordIndex:
    def __init__(self, words: tuple[Word, ...]):
        self.words = words
        self._ayah: dict[tuple[int, int], list[int]] = {}
        self._surah: dict[int, list[int]] = {}
        for w in words:
            a = self._ayah.setdefault((w.surah, w.ayah), [w.gid, w.gid])
            a[1] = w.gid
            s = self._surah.setdefault(w.surah, [w.gid, w.gid])
            s[1] = w.gid
        self._pos = {(w.surah, w.ayah, w.pos): w.gid for w in words}

    @classmethod
    def from_file(cls, path: Path = DEFAULT_TANZIL) -> "WordIndex":
        from qari.text.tanzil import build_words, check_complete, parse_tanzil
        words = build_words(parse_tanzil(Path(path).read_text(encoding="utf-8")))
        check_complete(words)
        return cls(words)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, gid: int) -> Word:
        return self.words[gid]

    def gid(self, surah: int, ayah: int, pos: int) -> int:
        return self._pos[(surah, ayah, pos)]

    def ayah_range(self, surah: int, ayah: int) -> tuple[int, int]:
        first, last = self._ayah[(surah, ayah)]
        return first, last

    def surah_range(self, surah: int) -> tuple[int, int]:
        first, last = self._surah[surah]
        return first, last

    def text(self, start_gid: int, end_gid: int) -> str:
        """Tanzil Uthmani text for the inclusive gid range."""
        return " ".join(w.uthmani for w in self.words[start_gid:end_gid + 1])
