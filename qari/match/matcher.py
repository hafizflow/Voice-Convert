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


# Spoken names of the Arabic letters (normalized spelling, alefs kept) -> the letter they stand for.
# "strong" names are not everyday words, so a run containing one is read as spelled-out letters.
_LETTER_NAMES = {
    "الف": "ا", "لام": "ل", "ميم": "م", "صاد": "ص", "راء": "ر", "را": "ر", "كاف": "ك", "ها": "ه",
    "هاء": "ه", "يا": "ي", "ياء": "ي", "عين": "ع", "طا": "ط", "طاء": "ط", "سين": "س", "حا": "ح",
    "حاء": "ح", "قاف": "ق", "نون": "ن",
}
_STRONG_NAMES = {"الف", "لام", "ميم", "صاد", "كاف", "قاف", "نون", "عين", "سين", "راء", "هاء", "ياء", "طاء", "حاء"}
_LONE_NAMES = {"صاد", "قاف", "نون"}        # one spoken name is enough only when it is not an everyday word
_MUQATTAAT_SURAHS = (2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30, 31, 32, 36, 38, 40,
                     41, 42, 43, 44, 45, 46, 50, 68)


@dataclass(frozen=True)
class MatcherConfig:
    min_words: int = 3                 # below this: never auto-play, return candidates
    min_words_anchored: int = 2        # after a leading Basmala the surah start is known
    top_n: int = 25                    # diagonals tried (ties on common phrases need room)
    slack: int = 5                     # extra Quran words either side of the query in the window
    gap: float = -0.7
    ambiguity_margin: float = 0.10     # a second place within this normalized score makes the answer a tie
    min_score: float = 0.45            # normalized score below which nothing is returned
    session_window: int = 150          # words: max distance for the session prior to break a tie
    max_alternatives: int = 8
    edge_extend: int = 2               # unaligned leading/trailing query words adopted as substitutions
    edge_min_sim: float = 0.4          # ...but only if they resemble the Quran word they would replace
    vote_floor: float = 0.3            # skip diagonals with under this fraction of the best vote
    max_postings: int = 400
    max_words: int = 300               # longer input is truncated (cost is quadratic)
    dp_budget: int = 3000              # candidates tried ~ dp_budget / query length (floor 3)
    whole_ayah_max_words: int = 4      # short queries may claim a verse of at most this many words, whole
    whole_ayah_factor: float = 0.9     # confidence discount: one or two words say little
    rare_phrase_max: int = 3           # a short phrase seen in at most this many places is still placeable


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
    vocative_joined: bool = False      # a split يا + word was read as the Quran's single word
    truncated: bool = False


class Matcher:
    def __init__(self, index: WordIndex, config: MatcherConfig | None = None):
        self.index = index
        self.cfg = config or MatcherConfig()
        self.ngrams = NgramIndex(index, self.cfg.max_postings)
        starts = {s: index.surah_range(s)[0] for s in sorted({w.surah for w in index.words})}
        starts[1] = index.gid(1, 2, 1)     # after a spoken Basmala, Al-Fatiha continues at ayah 2
        self._surah_starts = list(starts.values())
        self._surah_start_set = set(self._surah_starts)
        # short verses (55:1 is one word, 20:1 is two): a 1-2 word recitation can only be placed by
        # equating it with a whole verse, since the same words occur inside many longer verses
        self._muqattaat: dict[str, list[int]] = {}      # opening letters (الم, ص, ن ...) -> their gids
        for surah in _MUQATTAAT_SURAHS:
            for ayah in ((1, 2) if surah == 42 else (1,)):
                g = index.gid(surah, ayah, 1)
                self._muqattaat.setdefault(self.ngrams.keys[g], []).append(g)
        self._positions: dict[str, list[int]] = {}
        for g, k in enumerate(self.ngrams.keys):
            self._positions.setdefault(k, []).append(g)
        self._short_ayahs: dict[int, list[tuple[int, int]]] = {}
        seen: dict[tuple[int, int], list[int]] = {}
        for w in index.words:
            seen.setdefault((w.surah, w.ayah), []).append(w.gid)
        for gids in seen.values():
            if len(gids) <= self.cfg.whole_ayah_max_words:
                self._short_ayahs.setdefault(len(gids), []).append((gids[0], gids[-1]))

    # ---- public -------------------------------------------------------------------------------
    def match(self, text: str | list[str], asr_confidence: float = 1.0,
              session: SessionContext | None = None) -> MatchResult:
        asr = asr_confidence if asr_confidence == asr_confidence else 0.0   # NaN -> 0
        asr = min(1.0, max(0.0, asr))
        words = normalize_words(text) if isinstance(text, str) else [w for t in text for w in normalize_words(t)]
        spelled, strong = self._spell_letters(words)
        if spelled == words:
            return self._match_words(words, asr, session)
        # The same recitation heard as letter names (الف لام ميم) or as words; the strong spelling wins ties
        # (صاد is the letter, صد the verb), a weak one (طا ها) only when the words explain nothing.
        a, b = self._match_words(spelled, asr, session), self._match_words(words, asr, session)
        first, second = (a, b) if strong else (b, a)
        rank = {"matched": 3, "ambiguous": 2, "too_short": 1, "not_found": 0}
        return max((first, second), key=lambda r: (rank[r.status], r.score * r.n_words, r is first))

    def _match_words(self, words: list[str], asr: float, session: SessionContext | None) -> MatchResult:
        q = [skeleton(w) for w in words]
        q, istiadha = strip_istiadha(q)
        q, basmala, without_internal = strip_basmala(q)
        truncated = len(q) > self.cfg.max_words
        if truncated:
            q = q[:self.cfg.max_words]
            without_internal = None if without_internal is None else without_internal[:self.cfg.max_words]
        if not q:
            if basmala:   # the Basmala on its own: play it
                first, last = self.index.ayah_range(1, 1)
                return MatchResult("matched", first, last, False, 1.0, asr,
                                   {"asr": asr, "basmala_only": 1.0}, istiadha_stripped=istiadha)
            return MatchResult("not_found", istiadha_stripped=istiadha)

        # Readings of the same recitation, each resolved on its own; the best explained one wins.
        readings: list[tuple[list[str], bool, bool]] = []
        for words_, internal in ((q, False), (without_internal, True)):
            if words_ is None:
                continue
            readings.append((words_, internal, False))
            joined = self._join_vocatives(words_)
            if joined != words_:
                readings.append((joined, internal, True))
        results = []
        for words_, internal, vocative in readings:
            r = self._resolve(words_, basmala, asr, session)
            r.istiadha_stripped, r.internal_basmala, r.vocative_joined, r.truncated = istiadha, internal, vocative, truncated
            results.append(r)
        rank = {"matched": 3, "ambiguous": 2, "too_short": 1, "not_found": 0}
        return max(results, key=lambda r: (rank[r.status], r.score * r.n_words, -results.index(r)))

    # ---- internals ----------------------------------------------------------------------------
    @staticmethod
    def _join_vocatives(q: list[str]) -> list[str]:
        """Simple-script ASR writes يا أيها; the Uthmani text has one word (يَٰٓأَيُّهَا). Join them."""
        out, i = [], 0
        while i < len(q):
            if q[i] in ("ي", "وي") and i + 1 < len(q):
                out.append(q[i] + q[i + 1])
                i += 2
            else:
                out.append(q[i])
                i += 1
        return out

    @staticmethod
    def _letter_run(token: str) -> tuple[str, bool] | None:
        """`token` as a spoken letter name or a fused run of names (ياسين = يا + سين) -> (letters, has a
        strong name); None if it is neither."""
        if token in _LETTER_NAMES:
            return _LETTER_NAMES[token], token in _STRONG_NAMES
        best: dict[int, tuple[str, bool]] = {0: ("", False)}
        for i in range(len(token)):
            if i not in best:
                continue
            for n in (2, 3, 4):
                part = token[i:i + n]
                if part in _LETTER_NAMES:
                    letters, strong = best[i]
                    cand = (letters + _LETTER_NAMES[part], strong or part in _STRONG_NAMES)
                    if i + n not in best or (cand[1] and not best[i + n][1]):
                        best[i + n] = cand
        done = best.get(len(token))
        return done if done and len(done[0]) >= 2 else None

    def _spell_letters(self, words: list[str]) -> tuple[list[str], bool]:
        """Spoken letter names (الف لام ميم, طا ها, ياسين) -> the one word the Uthmani text writes (الم, طه,
        يس). Returns the new word list and whether a strong name was involved."""
        out, i, strong_any = [], 0, False
        while i < len(words):
            j, letters, strong, parts = i, "", False, 0
            while j < len(words):
                one = self._letter_run(words[j])
                if one is None:
                    break
                letters += one[0]
                strong = strong or one[1]
                parts += 1 if words[j] in _LETTER_NAMES else 2
                j += 1
            if j > i and parts >= 2 or (j == i + 1 and words[i] in _LONE_NAMES):
                out.append(letters)
                strong_any = strong_any or strong
                i = j
            else:
                out.append(words[i])
                i += 1
        return out, strong_any

    def _resolve(self, q: list[str], basmala: bool, asr: float, session: SessionContext | None) -> MatchResult:
        base = dict(n_words=len(q), includes_basmala=basmala)
        if not q:
            return MatchResult("not_found", **base)
        cands = self._candidates(q)
        if basmala:
            cands += self._anchored_candidates(q)   # a Basmala usually means a surah start: a hint, not a filter
        ranked = self._rank(cands)
        need = self.cfg.min_words_anchored if basmala else self.cfg.min_words
        short_whole = False
        if len(q) < need:
            whole = self._rank(self._whole_ayah_candidates(q) or self._muqattaat_candidates(q)
                               or self._rare_phrase_candidates(q))
            if not whole:
                alts = ranked[:self.cfg.max_alternatives] or self._first_occurrences(q)
                return MatchResult("too_short", alternatives=alts, **base)
            ranked, short_whole = whole, True
        if not ranked or ranked[0].score < self.cfg.min_score:
            return MatchResult("not_found", alternatives=ranked[:3], **base)

        top = ranked[0]
        tied = [c for c in ranked if c.score >= top.score - self.cfg.ambiguity_margin]
        used_session, uniqueness = False, 1.0
        if len(tied) > 1:
            chosen = self._session_pick(tied, session)
            if chosen is None and basmala:
                # identical wording in several places and the reciter said the Basmala: take the one
                # that starts a surah, if exactly one does
                anchored = [c for c in tied if c.start_gid in self._surah_start_set]
                if len(anchored) == 1:
                    top, uniqueness = anchored[0], 0.8
                    chosen = (top, False)
            if chosen is None:
                alts = sorted(tied, key=lambda c: (-c.score, c.start_gid))[:self.cfg.max_alternatives]
                return MatchResult("ambiguous", score=top.score, alternatives=alts, **base)
            top, used_session = chosen
            if used_session:
                uniqueness = 0.9 if abs(top.start_gid - (session.last_end_gid + 1)) <= 3 else 0.75
        start, end, _, includes = self._finish(top, basmala)
        aligned = sum(o.kind in ("match", "substitution") for o in top.ops)
        missing = sum(o.kind == "missing" for o in top.ops)
        parts = {"asr": asr, "coverage": top.coverage, "mean_sim": top.mean_sim, "uniqueness": uniqueness,
                 "completeness": aligned / max(1, aligned + missing)}
        if short_whole:
            parts["short_query"] = self.cfg.whole_ayah_factor
            uniqueness *= self.cfg.whole_ayah_factor
            parts["uniqueness"] = uniqueness
        conf = asr * top.coverage * top.mean_sim * uniqueness * parts["completeness"]
        return MatchResult("matched", start, end, score=top.score, confidence=conf, confidence_parts=parts,
                           ops=top.ops, used_session=used_session, n_words=len(q), includes_basmala=includes)

    def _candidates(self, q: list[str]) -> list[Candidate]:
        tokens = self.ngrams.tokens(q)
        votes = self.ngrams.diagonals(tokens, 2)          # bigrams survive one damaged word in a short query
        for d, v in self.ngrams.diagonals(tokens, 3).items():
            votes[d] += 2 * v                              # a trigram hit is worth more than a bigram hit
        top_n = min(self.cfg.top_n, max(3, self.cfg.dp_budget // max(1, len(q))))
        out = []
        for d in self.ngrams.top_diagonals(votes, top_n, min_fraction=self.cfg.vote_floor):
            c = self._align_window(q, d - self.cfg.slack, d + len(q) + self.cfg.slack)
            if c is not None:
                out.append(c)
        return out

    def _whole_ayah_candidates(self, q: list[str]) -> list[Candidate]:
        """Verses of about the query's length that the query explains completely, start to end."""
        out = []
        for n in range(max(1, len(q) - 1), len(q) + 2):
            for first, last in self._short_ayahs.get(n, ()):
                c = self._align_window(q, first, last + 1)
                if (c is not None and c.start_gid == first and c.end_gid == last
                        and c.score >= self.cfg.min_score and c.coverage >= 1.0):
                    out.append(c)
        return out

    def _first_occurrences(self, q: list[str]) -> list[Candidate]:
        """Nothing better to offer for a lone word: its earliest exact occurrences (never auto-played)."""
        keys, out = self.ngrams.keys, []
        for g in self._positions.get(q[0], ()):
            if g + len(q) <= len(keys) and all(keys[g + i] == w for i, w in enumerate(q)):
                c = self._align_window(q, g, g + len(q))
                if c is not None and c.start_gid == g:
                    out.append(c)
                    if len(out) >= self.cfg.max_alternatives:
                        break
        return out

    def _muqattaat_candidates(self, q: list[str]) -> list[Candidate]:
        """The opening letters (ن, ص, الم ...) are known places even when the same letter is common."""
        if len(q) != 1:
            return []
        out = []
        for g in self._muqattaat.get(q[0], ()):
            c = self._align_window(q, g, g + 1)
            if c is not None and c.start_gid == g:
                out.append(c)
        return out

    def _rare_phrase_candidates(self, q: list[str]) -> list[Candidate]:
        """A short phrase spelled exactly like the text and found in only a few places (a lone ص or ق:
        the letter opens a longer verse, so only that word plays)."""
        keys = self.ngrams.keys
        starts = [g for g in self._positions.get(q[0], ())
                  if g + len(q) <= len(keys) and all(keys[g + i] == w for i, w in enumerate(q))]
        if not starts or len(starts) > self.cfg.rare_phrase_max:
            return []
        out = []
        for g in starts:
            c = self._align_window(q, g, g + len(q))
            if c is not None and c.coverage >= 1.0 and c.start_gid == g:
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

    def _session_pick(self, tied: list[Candidate], session: SessionContext | None):
        """Among equally good places choose the one that continues forward from the previous match, if it
        is clearly the nearest. Only candidates already tied are considered, so text always wins."""
        if session is None or session.last_end_gid is None:
            return None
        nxt = session.last_end_gid + 1
        fwd = sorted(((c.start_gid - nxt, c) for c in tied if c.start_gid >= nxt - 3), key=lambda t: t[0])
        if not fwd or fwd[0][0] > self.cfg.session_window:
            return None
        if len(fwd) > 1 and fwd[1][0] - fwd[0][0] < 3:
            return None
        return fwd[0][1], True

    def _finish(self, c: Candidate, basmala: bool):
        start, end, ops = c.start_gid, c.end_gid, c.ops
        includes = basmala
        if basmala and start == self.index.gid(1, 2, 1):
            # Basmala + start of Al-Fatiha ayah 2: the range is simply 1:1 onward
            start, includes = self.index.gid(1, 1, 1), False
        return start, end, ops, includes
