# Hear the Qari: Core (text, timestamps, matcher) — design

Sub-project 1 of 5 for the "Hear the Qari" feature (see decomposition at the end).
Status: **draft for review**. Date: 2026-10-13.

## Outcome

Given a list of recognized Arabic words (the output of an ASR we do not build here), return the exact
Quran word range that was recited, as global word ids, plus the Qari time ranges that play exactly
those words. Never confidently return the wrong verse; when unsure return `alternatives`.

Everything here is CPU-only, in-memory, deterministic and unit-testable with no GPU, no app and no
microphone. ASR, API, Flutter client and deployment are separate specs.

## Constraints and decisions already made

- Flutter client, develop on an M4 Mac (no CUDA), production on a separate GPU host from RVC.
- Displayed text is always Tanzil Uthmani, never ASR output.
- Real Qari recordings only, sliced by word timestamps, no server-side audio processing.
- Reciter is a config value in a `reciters/` registry.
- Python 3.11 in this repo (`qari/` package, tests in `tests/qari/`).

## Licensing findings (researched 2026-10-13, from the sources' own pages)

These shape the design. This is not legal advice; the quoted terms should be re-read before launch.

| Source | What it grants | Constraint that matters here |
|---|---|---|
| **Tanzil text** | CC BY 3.0. May be used in any website/app if "Tanzil Project" is clearly credited and linked to tanzil.net. | Verbatim only: "CHANGING IT IS NOT ALLOWED". The copyright notice travels with every copy and with derived files containing a substantial part of the text. So: displayed text is the untouched Tanzil string; the normalized text is a separate derived *index*, never shown. Ship the notice in the data dir and credit in the app. |
| **Quran.com / Quran Foundation API** (Developer Terms, updated 2026-08-10) | Charging for an app, subscriptions, ads are allowed without a separate license. Audio is "QF Content". | Needs OAuth client credentials, and full dataset needs approved production access (prelive sandbox = Al-Fatihah + Al-Baqarah only). No scraping outside the API. No caching beyond 1 week unless via Content Sync. Content Sync does not authorize a prepackaged database or build-time bundle. "Redistribution" (offering QF Content as data to others) needs a signed commercial license. Derived outputs that reproduce or permit reconstruction of QF content stay restricted, which very likely covers a timestamp file built from QF segments. |
| **quran-align (cpfair)** word timings | CC BY 4.0, attribution; author asks to be emailed. | Covers timing data only. Whether it covers Afasy at the bitrate we want must be checked (not verified). The audio it aligns is not covered by that license. |
| **EveryAyah** Afasy audio | No site-wide license published. The only explicit term found is a disclaimer requiring a link back for the timing files. | Redistribution/rehosting rights are unclear; sources I found point to linking to the original rather than rehosting. |
| **Reciter copyright in general** | Recitation is a separate copyrighted recording. | Islamic Network (alquran.cloud) says its audio is licensed to them for free non-commercial redistribution only. A commercial app cannot assume that. |

### What this means

1. **Phases 1 and 3 are unaffected.** They need only Tanzil.
2. **Phase 2 (timestamps) is the licensing-sensitive piece.** Build it behind a source-pluggable
   interface and do not commit a timestamp file to git or ship one until the source is settled.
3. **Do not rehost Qari audio on your own CDN yet.** Return the original source's audio URLs
   (as returned by the QF API) and let the client stream them. That also removes Phase 7's R2/CloudFront
   assumption for audio until you hold written rights.
4. **Action for you (outside code):** email developers@quran.com asking, in writing, for (a) production
   Content API access, (b) permission to hold word-segment timestamps server-side beyond 1 week or via
   Content Sync, (c) whether serving their audio URLs to your app's users is covered for a commercial app.
   Separately, decide if Mishary Afasy's rights holder needs to be asked.

### Timestamp source order (recommendation)

| Option | Timing license | Audio rights | Verdict |
|---|---|---|---|
| A. QF API segments via Content Sync | QF terms (restrictive) | QF audio URLs | Preferred once production access is granted; no bundling in the repo. |
| B. quran-align CC BY 4.0 | Clean, attribution | Audio unclear | Good for development and as cross-check. |
| C. Own CTC forced alignment | Ours | We need rights to the audio we align | Fallback and for future reciters (Phase 2 pipeline). |

Development uses B plus the QF prelive sandbox (Fatihah, Baqarah) so we can compare A and B on the same
verses. Which one ships is gated on the email above, not on engineering.

## Architecture

```
qari/
  text/
    tanzil.py        load Tanzil Uthmani (+ simple-clean), split Basmala, build WordIndex
    normalize.py     normalize(str) -> str, one function used for Quran text AND ASR output
    index.py         WordIndex: words[gid], verse<->gid ranges, surah/ayah/pos lookups
  reciters/
    registry.py      load registry/*.yaml -> Reciter(id, audio_url(surah, ayah), timestamps, license_source)
    registry/afasy.yaml
  align/
    source.py        TimestampSource protocol: segments(reciter) -> {(surah,ayah): [(pos,start_ms,end_ms)]}
    quran_com.py     QF API source (OAuth, Content Sync aware)
    quran_align.py   cpfair JSON source
    forced.py        CTC forced-alignment fallback
    validate.py      stats + clip cutter for listening checks
    store.py         compact per-reciter timestamp file, TimestampStore.range_ms(gid_start, gid_end)
  match/
    ngram.py         inverted index over normalized-word tokens
    align.py         word-level Smith-Waterman with char-similarity
    matcher.py       Matcher.match(words, asr_conf, session) -> MatchResult
    strip.py         isti'adha and Basmala handling
```

### Core types

```python
@dataclass(frozen=True)
class Word:      # one entry per Quran word, gid is its position in the whole mushaf
    gid: int; surah: int; ayah: int; pos: int; uthmani: str; norm: str

@dataclass
class MatchResult:
    status: Literal["matched", "ambiguous", "not_found", "too_short"]
    start_gid: int | None; end_gid: int | None      # inclusive; a range may cross ayah/surah
    includes_basmala: bool                          # play the reciter's 1:1 audio first
    score: float; confidence: float
    ops: list[Op]                                   # match/substitution/missing/extra per word
    alternatives: list[Candidate]
```

A range is just `[gid_start, gid_end]`, so crossing verse and surah boundaries is free.
`TimestampStore.range_ms()` turns it into one or more `{audio_url, start_ms, end_ms}` segments, one per
ayah file touched. Padding is applied there (API phase, configurable).

### Normalizer

One pure function applied to both sides. Strips harakat, tatweel, Quranic annotation marks and the small
Uthmani letters-as-marks; folds alef forms (أ إ آ ٱ → ا), ى→ي, ة→ه, hamza carriers (ؤ ئ → و ي, bare ء kept
as its own letter). The rule list and each folding is covered by a table-driven test that includes the
tricky cases (dagger alef, alef wasla, small waw/yeh, maddah, ending ـٰ). Folding decisions that lose
information are listed in the test file so they are visible.

### Matcher

1. **Strip**: fuzzy-match the leading words against the isti'adha and drop it. Detect the Basmala as a
   prefix; remember it and drop it before search (all surahs except 1 and 9 get it prepended in audio).
2. **Too short** (< `min_words`, default 3): no auto-play; return top candidates only.
3. **Candidate search**: map each normalized word to an int token. Inverted index of 3-grams over the gid
   stream (cross-verse by construction). For short queries also 2-grams. Each query n-gram votes for
   diagonals `gid - query_pos`; keep top-N diagonals. Voting by diagonal tolerates dropped/substituted
   words, which an exact 3-gram lookup does not.
4. **Exact range**: Smith-Waterman over words in a window around each diagonal. Substitution score uses
   normalized char similarity (rapidfuzz), so ASR near-misses score partially; gaps are missing/extra.
   Returns start/end gid, score and per-word ops. Ops feed the "mistakes" output later.
5. **Ambiguity**: candidates whose scores are within `ambiguity_margin` of the best, or whose normalized
   span is identical text, are returned as `alternatives` with `status="ambiguous"`. The Ar-Rahman refrain
   is the canonical test.
6. **Session prior**: a small bonus for candidates adjacent to or continuing from the previous match. It can
   break a tie but is capped so it cannot override a clearly better textual match.
7. **Confidence**: combines ASR confidence (optional input), matched fraction, mean char similarity and the
   margin over the runner-up. Until we have real ASR output, we report the components and use a simple
   product; calibration is a sub-project 2 task.

All thresholds (`min_words`, `ambiguity_margin`, session bonus, top-N, window slack, gap penalty) are
config, not literals scattered in code.

## Verification plan

No ASR exists yet, so evaluation uses **synthetic queries** generated from the Tanzil text, labelled with
the true range:

- full verses; first/second half; arbitrary 3–15 word spans; spans crossing verse and surah boundaries;
- noise: dropped words, substituted words, ASR-style char confusions (ث/س, ذ/ز, ض/ظ, ع/ء, ه/ح), extra words,
  leading isti'adha and Basmala;
- the repeated-phrase set: refrain verses in Ar-Rahman, Al-Mursalat, Al-Qamar, plus repeated formulae across
  surahs.

Reported metrics, per query kind:

| Metric | Why |
|---|---|
| top-1 location accuracy | headline |
| exact start/end boundary accuracy, and within ±1 word | what the user hears |
| **confident-wrong rate** (`matched` but wrong) | the metric the brief says matters most; target 0 on the clean set |
| ambiguous rate on repeated phrases | should be high there, low elsewhere |

Caveat stated up front: synthetic noise is not the real ASR error distribution, so these numbers bound the
matcher's logic, not end-to-end accuracy. Real numbers come from sub-project 2's held-out recitations.

Other tests: normalizer table tests; word-count assertion against the Tanzil file (exact count is asserted
from the file in the test, not from memory); every surah/ayah has contiguous `pos`; timestamp store has no
negative/zero durations and no overlaps within an ayah; coverage check that every word of every ayah has a
segment (report gaps, do not silently fill).

## Milestones (one commit each, README updated)

1. **M1 Text**: Tanzil loader, WordIndex, normalizer, tests. Needs download of Tanzil files.
2. **M2 Timestamps**: source interface, quran-align source, QF sandbox source, store, validate tool.
   Licensing gate: no timestamp file committed.
3. **M3 Matcher**: strip, n-gram, SW, ambiguity, session, confidence, synthetic eval report.

## Out of scope here

ASR/VAD, WebSocket/HTTP API, auth/rate limit/metrics, padding and waqf extension (API spec), Flutter,
Docker, CDN, load test. Forced-alignment (`align/forced.py`) is included as code but exercised only if
a source needs it.

## Open questions

1. Does quran-align cover Afasy at a usable bitrate? (check in M2, not assumed)
2. Does the QF API return segments for Afasy for every ayah? Coverage is verified in M2, not assumed.
3. Does Tanzil's text prepend the Basmala to the first ayah of each surah? M1's loader handles it either
   way and asserts which, rather than relying on this note.
4. Written permission from QF and any needed rights for the reciter (user action).

## Decomposition (remaining sub-projects)

2. ASR service (checkpoint eval, faster-whisper, calibration of confidence) ·
3. API (WS/HTTP, padding, auth, metrics) · 4. Flutter package · 5. Deployment.
