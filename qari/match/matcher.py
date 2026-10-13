"""Locate recognized words in the Quran: candidates -> local alignment -> ambiguity -> result.

Order of work per call: normalize -> strip isti'adha/Basmala -> n-gram diagonal voting (or surah-start
anchoring after a Basmala) -> Smith-Waterman per candidate -> non-overlapping candidates ranked ->
ambiguity / session decision -> confidence. All thresholds live in MatcherConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from qari.match.align import Alignment, Op, local_align, similarity, skeleton
from qari.match.ngram import NgramIndex
from qari.match.strip import strip_basmala, strip_istiadha
from qari.text.index import WordIndex
from qari.text.normalize import normalize_words


@dataclass(frozen=True)
class MatcherConfig:
    min_words: int = 3                 # below this: never auto-play, return candidates
    min_words_anchored: int = 2        # after a leading Basmala the surah start is known
    top_n: int = 25                    # diagonals tried (ties on common phrases need room)
    slack: int = 5                     # extra Quran words either side of the query in the window
    gap: float = -0.7
    ambiguity_margin: float = 0.05     # normalized score within which another place counts as a tie
    min_score: float = 0.45            # normalized score below which nothing is returned
    session_window: int = 150          # words: max distance for the session prior to break a tie
    max_alternatives: int = 8
    edge_extend: int = 2               # unaligned leading/trailing query words adopted as substitutions
    edge_min_sim: float = 0.4          # ...but only if they resemble the Quran word they would replace
    vote_floor: float = 0.3            # skip diagonals with under this fraction of the best vote
    max_postings: int = 400


@dataclass
class SessionContext:
    last_end_gid: int | None = None    # end of the previous accepted match in this session


@dataclass
class Candidate:
    start_gid: int
    end_gid: int
    score: float                       # raw alignment score / number of query words, 1.0 = perfect
    coverage: float                    # aligned query words / query words
    mean_sim: float                    # mean similarity over aligned pairs
    ops: list[Op]
    text_key: tuple[str, ...]          # skeleton words of the matched range: equal key = identical text


@dataclass
class MatchResult:
    status: str                        # matched | ambiguous | not_found | too_short
    start_gid: int | None = None
    end_gid: int | None = None
    includes_basmala: bool = False     # play the reciter's 1:1 audio before the range
    score: float = 0.0
    confidence: float = 0.0
    confidence_parts: dict = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)
    alternatives: list[Candidate] = field(default_factory=list)
    n_words: int = 0                   # query words used for matching, after stripping
    istiadha_stripped: bool = False
    internal_basmala: bool = False
    used_session: bool = False


class Matcher:
    def __init__(self, index: WordIndex, config: MatcherConfig | None = None):
        self.index = index
        self.cfg = config or MatcherConfig()
        self.ngrams = NgramIndex(index, self.cfg.max_postings)
        self._surah_starts = [index.surah_range(s)[0] for s in sorted({w.surah for w in index.words})]

    # ---- public -------------------------------------------------------------------------------
    def match(self, text: str | list[str], asr_confidence: float = 1.0,
              session: SessionContext | None = None) -> MatchResult:
        words = normalize_words(text) if isinstance(text, str) else [w for t in text for w in normalize_words(t)]
        q = [skeleton(w) for w in words]
        q, istiadha = strip_istiadha(q)
        q, basmala, internal = strip_basmala(q)
        base = dict(istiadha_stripped=istiadha, internal_basmala=internal, n_words=len(q))

        if not q:
            if basmala:   # the Basmala on its own: play it
                first, last = self.index.ayah_range(1, 1)
                return MatchResult("matched", first, last, False, 1.0, asr_confidence,
                                   {"asr": asr_confidence, "basmala_only": 1.0}, **base)
            return MatchResult("not_found", **base)

        # After a Basmala the reciter usually starts a surah, but not always (isti'adha + Basmala + ayat
        # al-kursi is common). Anchoring is therefore only a way to accept short input, never a filter.
        anchored = basmala and len(q) < self.cfg.min_words
        need = self.cfg.min_words_anchored if basmala else self.cfg.min_words
        cands = self._candidates(q, anchored)
        if basmala and not anchored:
            cands += self._anchored_candidates(q)
        if len(q) < need:
            return MatchResult("too_short", alternatives=self._rank(cands)[:self.cfg.max_alternatives], **base)

        ranked = self._rank(cands)
        if not ranked or ranked[0].score < self.cfg.min_score:
            return MatchResult("not_found", alternatives=ranked[:3], **base)

        top = ranked[0]
        tied = [c for c in ranked if c.score >= top.score - self.cfg.ambiguity_margin]
        used_session = False
        if len(tied) > 1:
            chosen = self._session_pick(tied, session)
            if chosen is None:
                alts = sorted(tied, key=lambda c: (-c.score, c.start_gid))[:self.cfg.max_alternatives]
                return MatchResult("ambiguous", score=top.score, alternatives=alts,
                                   includes_basmala=basmala, **base)
            top, used_session = chosen, True

        second = next((c for c in ranked if c is not top), None)
        margin = top.score - second.score if second else 1.0
        start, end, ops, includes = self._finish(top, basmala)
        parts = {"asr": asr_confidence, "coverage": top.coverage, "mean_sim": top.mean_sim,
                 "uniqueness": min(1.0, margin / (self.cfg.ambiguity_margin * 4)) if not used_session else 0.75}
        conf = asr_confidence * top.coverage * top.mean_sim * parts["uniqueness"]
        return MatchResult("matched", start, end, includes, top.score, conf, parts, ops, [], used_session=used_session, **base)

    # ---- internals ----------------------------------------------------------------------------
    def _candidates(self, q: list[str], anchored: bool) -> list[Candidate]:
        if anchored:
            return self._anchored_candidates(q)
        tokens = self.ngrams.tokens(q)
        votes = self.ngrams.diagonals(tokens, 2)          # bigrams survive one damaged word in a short query
        for d, v in self.ngrams.diagonals(tokens, 3).items():
            votes[d] += 2 * v                              # a trigram hit is worth more than a bigram hit
        out = []
        for d in self.ngrams.top_diagonals(votes, self.cfg.top_n, min_fraction=self.cfg.vote_floor):
            c = self._align_window(q, d - self.cfg.slack, d + len(q) + self.cfg.slack)
            if c is not None:
                out.append(c)
        return out

    def _anchored_candidates(self, q: list[str]) -> list[Candidate]:
        out = []
        for first in self._surah_starts:
            c = self._align_window(q, first, first + len(q) + self.cfg.slack)
            if c is not None and c.start_gid <= first + 1:
                out.append(c)
        return out

    def _align_window(self, q: list[str], lo: int, hi: int) -> Candidate | None:
        lo, hi = max(0, lo), min(len(self.index), hi)
        ref = self.ngrams.keys[lo:hi]
        a = local_align(q, ref, self.cfg.gap)
        if a is None:
            return None
        ops = [replace(o, gid=None if o.gid is None else o.gid + lo) for o in a.ops]
        start, end = a.r_start + lo, a.r_end + lo
        ops, start, end = self._extend_edges(q, ops, a, start, end)
        aligned = [o for o in ops if o.kind in ("match", "substitution")]
        if not aligned:
            return None
        n = len(q)
        return Candidate(start, end, a.raw / n, len(aligned) / n,
                         sum(o.sim for o in aligned) / len(aligned), ops,
                         tuple(self.ngrams.keys[start:end + 1]))

    def _extend_edges(self, q, ops, a: Alignment, start: int, end: int):
        """Adopt up to `edge_extend` unaligned edge query words as substitutions (a mispronounced or
        mis-heard first/last word should not shrink the range)."""
        k, floor, keys = self.cfg.edge_extend, self.cfg.edge_min_sim, self.ngrams.keys
        pre: list[Op] = []
        while len(pre) < k and a.q_start - len(pre) > 0 and start - len(pre) > 0:
            qi, gi = a.q_start - len(pre) - 1, start - len(pre) - 1
            s = similarity(q[qi], keys[gi])
            if s < floor:
                break
            pre.insert(0, Op("substitution", qi, gi, s))
        post: list[Op] = []
        while len(post) < k and a.q_end + len(post) + 1 < len(q) and end + len(post) + 1 < len(self.index):
            qi, gi = a.q_end + len(post) + 1, end + len(post) + 1
            s = similarity(q[qi], keys[gi])
            if s < floor:
                break
            post.append(Op("substitution", qi, gi, s))
        return pre + ops + post, start - len(pre), end + len(post)

    @staticmethod
    def _rank(cands: list[Candidate]) -> list[Candidate]:
        """Best first, dropping candidates that overlap a better one (same place found twice)."""
        kept: list[Candidate] = []
        for c in sorted(cands, key=lambda c: (-c.score, c.start_gid)):
            if all(c.end_gid < k.start_gid or c.start_gid > k.end_gid for k in kept):
                kept.append(c)
        return kept

    def _session_pick(self, tied: list[Candidate], session: SessionContext | None) -> Candidate | None:
        """Among equally good places choose the one that continues from the previous match, if exactly
        one is clearly nearest. Never overrides a textual difference: only candidates already tied."""
        if session is None or session.last_end_gid is None:
            return None
        nxt = session.last_end_gid + 1
        near = sorted(((abs(c.start_gid - nxt), c) for c in tied), key=lambda t: t[0])
        d0, best = near[0]
        if d0 > self.cfg.session_window:
            return None
        if len(near) > 1 and near[1][0] == d0:
            return None
        return best

    def _finish(self, c: Candidate, basmala: bool):
        start, end, ops = c.start_gid, c.end_gid, c.ops
        includes = basmala
        if basmala and start == self.index.gid(1, 2, 1):
            # Basmala + start of Al-Fatiha ayah 2: the range is simply 1:1 onward
            start, includes = self.index.gid(1, 1, 1), False
        return start, end, ops, includes
