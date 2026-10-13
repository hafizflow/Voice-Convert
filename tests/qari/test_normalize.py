import pytest

from qari.text.normalize import normalize, normalize_words

CASES = [
    # (input, expected, why)
    ("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", "بسم الله الرحمان الرحيم", "harakat, alef wasla, dagger alef"),
    ("ٱلْعَٰلَمِينَ", "العالمين", "dagger alef becomes explicit alef"),
    ("عَلَىٰ", "علي", "dagger after alef maqsura is dropped, ى -> ي"),
    ("ءَامَنُوا۟", "امنوا", "hamza+alef = alef-madda, small high zero dropped"),
    ("آمَنُوا", "امنوا", "precomposed alef-madda"),
    ("أُو۟لَٰٓئِكَ", "اولايك", "hamza on alef folds, ئ -> ي"),
    ("رَحْمَةٌ", "رحمه", "taa marbuta -> haa, tanween dropped"),
    ("إِنَّهُۥ", "انه", "small waw dropped, hamza-below alef folds"),
    ("يَسْـَٔلُونَكَ", "يسلونك", "tatweel + hamza above are dropped (known information loss)"),
    ("مَسْـُٔولًا", "مسولا", "same carrier case with ؤ-like spelling"),
    ("ﷲ", "الله", "presentation form via NFKC"),
    ("الله،  ربّ‌ العالمين.", "الله رب العالمين", "ASR punctuation, ZWNJ, shadda, extra spaces"),
    ("قل هو الله أحد 123 abc", "قل هو الله احد", "digits and latin removed"),
    ("", "", "empty"),
    ("َّ", "", "marks only"),
]


@pytest.mark.parametrize("text,expected,why", CASES, ids=[c[2] for c in CASES])
def test_normalize(text, expected, why):
    assert normalize(text) == expected


def test_idempotent():
    for text, _, _ in CASES:
        assert normalize(normalize(text)) == normalize(text)


def test_normalize_words_splits_on_normalized_spaces():
    assert normalize_words("ٱلْحَمْدُ  لِلَّهِ،رَبِّ") == ["الحمد", "لله", "رب"]
