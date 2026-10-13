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
    w, lead, internal = strip_basmala("بسم الله الرحمن الرحيم قل هو الله احد".split())
    assert lead and not internal and w == ["قل", "هو", "الله", "احد"]
    w, lead, internal = strip_basmala("من الجنه والناس بسم الله الرحمن الرحيم الحمد لله".split())
    assert not lead and internal and w == ["من", "الجنه", "والناس", "الحمد", "لله"]


def test_no_basmala_untouched():
    w, lead, internal = strip_basmala("الحمد لله رب العالمين".split())
    assert w == ["الحمد", "لله", "رب", "العالمين"] and not lead and not internal
