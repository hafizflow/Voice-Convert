import pytest

from qari.match.matcher import Matcher, SessionContext
from qari.text.index import WordIndex


@pytest.fixture(scope="module")
def idx():
    return WordIndex.from_file()


@pytest.fixture(scope="module")
def m(idx):
    return Matcher(idx)


def say(idx, s, a, p1=None, p2=None, to=None):
    """ASR-like text (normalized words) for words p1..p2 of s:a, or up to the end of ayah `to`."""
    first, last = idx.ayah_range(s, a)
    lo = first if p1 is None else idx.gid(s, a, p1)
    hi = last if p2 is None else idx.gid(s, a, p2)
    if to is not None:
        hi = idx.ayah_range(*to)[1]
    return " ".join(idx[g].norm for g in range(lo, hi + 1)), lo, hi


def pos(idx, r):
    s, e = idx[r.start_gid], idx[r.end_gid]
    return (s.surah, s.ayah, s.pos), (e.surah, e.ayah, e.pos)


def test_full_verse(m, idx):
    text, lo, hi = say(idx, 2, 255)
    r = m.match(text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)


def test_half_verse_and_mid_verse_start_play_only_those_words(m, idx):
    text, lo, hi = say(idx, 2, 255, 6, 12)
    r = m.match(text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)


def test_multi_verse_range(m, idx):
    text, lo, hi = say(idx, 112, 1, to=(112, 4))
    r = m.match(text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)


def test_range_crossing_a_surah_boundary(m, idx):
    a, lo, _ = say(idx, 1, 6)
    b, _, hi = say(idx, 2, 1, to=(2, 2))
    r = m.match(a + " " + say(idx, 1, 7)[0] + " " + b)
    assert r.status == "matched" and r.start_gid == lo and r.end_gid == idx.ayah_range(2, 2)[1]


def test_skipped_word_keeps_the_range_and_reports_missing(m, idx):
    text, lo, hi = say(idx, 2, 255, 1, 20)
    words = text.split()
    del words[8]
    r = m.match(" ".join(words))
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)
    assert [o.kind for o in r.ops].count("missing") == 1


def test_wrong_word_keeps_the_range_and_reports_substitution(m, idx):
    text, lo, hi = say(idx, 2, 255, 1, 20)
    words = text.split()
    words[8] = "كتاب"
    r = m.match(" ".join(words))
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)
    assert [o.kind for o in r.ops].count("substitution") >= 1


def test_misheard_first_word_does_not_shrink_the_range(m, idx):
    text, lo, hi = say(idx, 2, 255, 1, 15)
    words = text.split()
    words[0] = words[0][:-1] + "ن"      # damaged but recognizably the same word
    r = m.match(" ".join(words))
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)


def test_unrelated_junk_first_word_costs_at_most_one_word(m, idx):
    text, lo, hi = say(idx, 2, 255, 1, 15)
    words = text.split()
    words[0] = "ززززز"
    r = m.match(" ".join(words))
    assert r.status == "matched" and r.end_gid == hi and 0 <= r.start_gid - lo <= 1


def test_istiadha_and_basmala_are_stripped_and_reported(m, idx):
    text, lo, hi = say(idx, 112, 1, to=(112, 4))
    r = m.match("اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم " + text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi)
    assert r.istiadha_stripped and r.includes_basmala


def test_basmala_inside_a_cross_surah_recitation(m, idx):
    a, lo, _ = say(idx, 114, 5)
    b, _, hi = say(idx, 114, 6)
    last_of_114 = a + " " + b
    r = m.match(last_of_114 + " بسم الله الرحمن الرحيم " + say(idx, 1, 2)[0])
    assert r.internal_basmala and r.status == "matched" and r.start_gid == lo


def test_basmala_then_two_common_words_is_ambiguous_not_a_confident_guess(m, idx):
    # "الله لا" starts 2:255, 4:87, 20:8, 64:13 ... : the Basmala must not turn that into 3:2 at 1.0
    r = m.match("بسم الله الرحمن الرحيم الله لا")
    assert r.status == "ambiguous" and r.start_gid is None
    assert len(r.alternatives) >= 3


def test_basmala_then_four_words_picks_the_surah_start_when_text_ties(m, idx):
    text, lo, hi = say(idx, 112, 1, 1, 4)
    r = m.match("بسم الله الرحمن الرحيم " + text)
    assert r.status == "matched" and r.start_gid == lo and r.includes_basmala


def test_basmala_then_fatiha_resolves_to_fatiha(m, idx):
    r = m.match("بسم الله الرحمن الرحيم الحمد لله رب العالمين")
    assert r.status == "matched" and r.start_gid == idx.gid(1, 1, 1) and not r.includes_basmala
    short = m.match("بسم الله الرحمن الرحيم الحمد لله")
    assert short.status == "ambiguous"
    assert idx.gid(1, 2, 1) in [a.start_gid for a in short.alternatives]


def test_basmala_alone_is_the_basmala(m, idx):
    r = m.match("بسم الله الرحمن الرحيم")
    assert r.status == "matched" and (r.start_gid, r.end_gid) == idx.ayah_range(1, 1)


def test_ar_rahman_refrain_is_ambiguous_without_context(m):
    r = m.match("فبأي الاء ربكما تكذبان")
    assert r.status == "ambiguous" and len(r.alternatives) >= 2
    assert r.start_gid is None


def test_session_context_resolves_the_refrain_to_the_next_occurrence(m, idx):
    prev_end = idx.ayah_range(55, 15)[1]
    r = m.match("فبأي الاء ربكما تكذبان", session=SessionContext(prev_end))
    assert r.status == "matched" and r.used_session
    assert pos(idx, r)[0] == (55, 16, 1)


def test_session_context_never_overrides_a_clear_textual_match(m, idx):
    text, lo, hi = say(idx, 112, 1, to=(112, 4))
    r = m.match(text, session=SessionContext(idx.ayah_range(55, 15)[1]))
    assert (r.start_gid, r.end_gid) == (lo, hi) and not r.used_session


def test_too_short_never_auto_plays(m):
    r = m.match("الحمد لله")
    assert r.status == "too_short" and r.start_gid is None and r.alternatives


def test_gibberish_is_not_found(m):
    assert m.match("قطة تأكل سمكة كبيرة جدا").status == "not_found"


def test_empty_input(m):
    assert m.match("").status == "not_found"
    assert m.match("،.؟ 123").status == "not_found"


def test_accepts_word_list_and_unnormalized_text(m, idx):
    text, lo, hi = say(idx, 112, 1, to=(112, 4))
    r = m.match("قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ")
    assert r.status == "matched" and r.start_gid == lo


def test_confidence_is_between_zero_and_one_and_drops_with_asr_confidence(m, idx):
    text, _, _ = say(idx, 2, 255)
    hi = m.match(text, asr_confidence=1.0).confidence
    lo = m.match(text, asr_confidence=0.5).confidence
    assert 0 < lo < hi <= 1.0


def test_istiadha_with_a_skipped_word_does_not_eat_the_verse(m, idx):
    text, lo, hi = say(idx, 2, 255)
    for prefix in ("اعوذ بالله من الشيطان", "اعوذ بالله من", "اعوذ بالله من الرجيم"):
        r = m.match(prefix + " " + text)
        assert r.status == "matched" and r.start_gid == lo and r.end_gid == hi, prefix


def test_basmala_that_is_part_of_the_verse_is_not_removed(m, idx):
    # 27:30 ends with the Basmala
    text, lo, hi = say(idx, 27, 30)
    r = m.match(text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", lo, hi) and not r.internal_basmala
    r = m.match(text + " " + say(idx, 27, 31)[0])
    assert r.status == "matched" and r.start_gid == lo and r.end_gid == idx.ayah_range(27, 31)[1]


def test_damaged_short_query_with_a_slightly_better_wrong_place_is_not_matched(m):
    r = m.match("الذين امنوا وعملوا الصالحات م")
    assert r.status != "matched"


def test_refrain_said_twice_is_not_a_confident_match(m):
    assert m.match("فباي الاء ربكما تكذبان فباي الاء ربكما تكذبان").status != "matched"


def test_split_vocative_is_joined_and_keeps_full_confidence(m, idx):
    r = m.match("يا ايها الذين امنوا اوفوا بالعقود")
    assert r.status == "matched" and r.vocative_joined
    assert (idx[r.start_gid].surah, idx[r.start_gid].ayah) == (5, 1) and r.confidence >= 0.9


def test_very_long_input_is_truncated_and_fast(m):
    import time
    t = time.time()
    r = m.match(" ".join(["الحمد", "لله"] * 1500))
    assert r.truncated and time.time() - t < 5


def test_session_prior_is_forward_only(m, idx):
    refrain = "فباي الاء ربكما تكذبان"
    r = m.match(refrain, session=SessionContext(idx.ayah_range(55, 13)[1]))
    assert r.status == "matched" and pos(idx, r)[0] == (55, 16, 1)      # not a replay of 55:13
    end = SessionContext(len(idx) - 1)
    assert m.match("قل اعوذ برب", session=end).status == "ambiguous"    # no going backwards


def test_asr_confidence_is_clamped(m, idx):
    text, _, _ = say(idx, 112, 1, to=(112, 4))
    assert m.match(text, asr_confidence=5.0).confidence <= 1.0
    assert m.match(text, asr_confidence=float("nan")).confidence == 0.0
    assert m.match(text, asr_confidence=-1).confidence == 0.0


def test_confidence_drops_when_words_in_range_are_missing(m, idx):
    text, _, _ = say(idx, 2, 255, 1, 20)
    words = text.split()
    clean = m.match(text).confidence
    damaged = m.match(" ".join(words[:5] + words[9:])).confidence
    assert damaged < clean


# ---- one- and two-word verses (55:1 الرحمن, 1:3, 20:1 طه ...) --------------------------------------
@pytest.mark.parametrize("text,ayah", [
    ("ٱلرَّحْمَـٰنُ", (55, 1)),
    ("الرحمن", (55, 1)),            # simple-script spelling an ASR returns
    ("الرحمان", (55, 1)),
    ("الرحمن الرحيم", (1, 3)),
    ("والفجر", (89, 1)),
    ("الحاقة", (69, 1)),
])
def test_short_whole_verse_matches(m, idx, text, ayah):
    first, last = idx.ayah_range(*ayah)
    r = m.match(text)
    assert (r.status, r.start_gid, r.end_gid) == ("matched", first, last), r.status


def test_short_verse_is_not_confident_enough_to_beat_a_longer_exact_match(m, idx):
    # "الرحمن" is a word inside many verses; only the whole-verse reading may claim it
    r = m.match("الرحمن")
    assert r.confidence <= 0.95


def test_every_one_and_two_word_verse_is_found(m, idx):
    miss = []
    seen = {}
    for w in idx.words:
        seen.setdefault((w.surah, w.ayah), []).append(w)
    for (s, a), ws in seen.items():
        if len(ws) > 2:
            continue
        r = m.match(" ".join(w.uthmani for w in ws), 0.95)
        hit = r.start_gid is not None and r.start_gid <= ws[0].gid and r.end_gid >= ws[-1].gid
        # identical wording elsewhere is a legitimate tie: the verse must then be among the alternatives
        tie = r.status == "ambiguous" and any(c.start_gid == ws[0].gid for c in r.alternatives)
        if not (r.status == "matched" and r.start_gid == ws[0].gid and r.end_gid == ws[-1].gid) and not tie:
            miss.append((s, a, r.status))
    assert not miss, miss[:20]


@pytest.mark.parametrize("text,at", [
    ("ص", (38, 1, 1)),              # the letter alone opens a longer verse: only that word plays
    ("ق", (50, 1, 1)),
])
def test_lone_rare_word_is_placed_exactly(m, idx, text, at):
    r = m.match(text)
    assert r.status == "matched" and r.start_gid == r.end_gid == idx.gid(*at)


def test_lone_common_word_is_not_guessed(m):
    r = m.match("الله")
    assert r.status == "too_short" and r.start_gid is None


@pytest.mark.parametrize("text,at", [
    ("الف لام ميم صاد", (7, 1, 1)),
    ("كاف ها يا عين صاد", (19, 1, 1)),
    ("طا ها", (20, 1, 1)),
    ("يا سين", (36, 1, 1)),
    ("ياسين", (36, 1, 1)),
    ("صاد", (38, 1, 1)),
    ("قاف", (50, 1, 1)),
    ("نون", (68, 1, 1)),
])
def test_spoken_letter_names_find_the_opening_letters(m, idx, text, at):
    r = m.match(text)
    assert r.status == "matched" and r.start_gid == idx.gid(*at), (r.status, r.start_gid)


def test_spoken_letter_names_tie_between_surahs_is_ambiguous(m, idx):
    r = m.match("الف لام ميم")
    assert r.status == "ambiguous"
    starts = {c.start_gid for c in r.alternatives}
    assert idx.gid(2, 1, 1) in starts and idx.gid(3, 1, 1) in starts


def test_letter_names_inside_a_verse_are_left_alone(m, idx):
    text, lo, hi = say(idx, 2, 255)
    r = m.match(text)
    assert (r.start_gid, r.end_gid) == (lo, hi)


def test_spoken_letter_names_that_two_surahs_share_are_a_tie(m, idx):
    r = m.match("طا سين ميم")
    assert r.status == "ambiguous"
    assert {c.start_gid for c in r.alternatives} == {idx.gid(26, 1, 1), idx.gid(28, 1, 1)}


def test_an_ordinary_word_that_sounds_like_a_letter_name_is_not_rewritten(m, idx):
    text, lo, hi = say(idx, 4, 55)   # contains صَدَّ
    r = m.match(text)
    assert (r.start_gid, r.end_gid) == (lo, hi)


def test_lone_common_word_still_offers_places_to_play(m, idx):
    r = m.match("الله")
    assert r.status == "too_short" and r.start_gid is None
    assert r.alternatives and all(idx[c.start_gid].norm.replace("ا", "") == "لله" for c in r.alternatives)
    assert r.alternatives[0].start_gid == min(c.start_gid for c in r.alternatives)
