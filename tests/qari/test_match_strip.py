from qari.match.strip import strip_basmala, strip_istiadha


def test_istiadha_is_stripped():
    words, found = strip_istiadha("اعوذ بالله من الشيطان الرجيم قل هو الله".split())
    assert found and words == ["قل", "هو", "الله"]


def test_asr_noise_in_istiadha_still_strips():
    words, found = strip_istiadha("اعوذ بالله من الشيطن الرجيم قل هو".split())
    assert found and words == ["قل", "هو"]


def test_verse_that_starts_with_auzu_billah_is_not_stripped():
    # 2:67 (after قال): اعوذ بالله ان اكون من الجاهلين
    words, found = strip_istiadha("اعوذ بالله ان اكون من الجاهلين".split())
    assert not found and len(words) == 6


def test_leading_and_internal_basmala():
    w, lead, removed = strip_basmala("بسم الله الرحمن الرحيم قل هو الله احد".split())
    assert lead and removed is None and w == ["قل", "هو", "الله", "احد"]
    w, lead, removed = strip_basmala("من الجنه والناس بسم الله الرحمن الرحيم الحمد لله".split())
    assert not lead and w[3] == "بسم"                       # the original is kept...
    assert removed == ["من", "الجنه", "والناس", "الحمد", "لله"]   # ...the removal is only an alternative


def test_istiadha_missing_a_word_does_not_eat_the_verse():
    # reciter skipped الشيطان and went straight to the verse: الله لا اله ...
    words, found = strip_istiadha("اعوذ بالله من الله لا اله الا هو".split())
    assert found and words == ["الله", "لا", "اله", "الا", "هو"]
    words, found = strip_istiadha("اعوذ بالله من الشيطان الله لا اله".split())
    assert found and words == ["الله", "لا", "اله"]


def test_no_basmala_untouched():
    w, lead, removed = strip_basmala("الحمد لله رب العالمين".split())
    assert w == ["الحمد", "لله", "رب", "العالمين"] and not lead and removed is None
