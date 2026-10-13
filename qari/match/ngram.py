"""Inverted index of word n-grams over the global word stream; votes for alignment diagonals."""
from __future__ import annotations

from collections import Counter, defaultdict

from qari.match.align import skeleton
from qari.text.index import WordIndex


class NgramIndex:
    def __init__(self, index: WordIndex, max_postings: int = 400):
        self.keys = [skeleton(w.norm) for w in index.words]
        self._vocab: dict[str, int] = {}
        self._tok = [self._vocab.setdefault(k, len(self._vocab)) for k in self.keys]
        self._grams: dict[int, dict[tuple[int, ...], list[int]]] = {}
        for n in (2, 3):
            g: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for p in range(len(self._tok) - n + 1):
                g[tuple(self._tok[p:p + n])].append(p)
            self._grams[n] = {k: v for k, v in g.items() if len(v) <= max_postings}

    def tokens(self, words: list[str]) -> list[int]:
        return [self._vocab.get(w, -1) for w in words]

    def diagonals(self, q_tokens: list[int], n: int) -> Counter:
        """votes[d] = number of query n-grams whose occurrence implies query word 0 sits at gid d."""
        votes: Counter = Counter()
        table = self._grams[n]
        for i in range(len(q_tokens) - n + 1):
            key = tuple(q_tokens[i:i + n])
            if -1 in key:
                continue
            for p in table.get(key, ()):
                votes[p - i] += 1
        return votes

    def top_diagonals(self, votes: Counter, top_n: int, smooth: int = 2, radius: int = 3,
                      min_fraction: float = 0.0) -> list[int]:
        if not votes:
            return []
        smoothed = {d: sum(votes.get(d + k, 0) for k in range(-smooth, smooth + 1)) for d in votes}
        floor = min_fraction * max(smoothed.values())
        chosen: list[int] = []
        for d, v in sorted(smoothed.items(), key=lambda kv: (-kv[1], kv[0])):
            if v < floor:
                break
            if all(abs(d - c) > radius for c in chosen):
                chosen.append(d)
                if len(chosen) == top_n:
                    break
        return chosen
