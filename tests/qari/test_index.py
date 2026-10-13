import pytest

from qari.text.index import WordIndex


@pytest.fixture(scope="module")
def idx():
    return WordIndex.from_file()


def test_totals_match_the_committed_tanzil_file(idx):
    # Measured from data/tanzil/quran-uthmani.txt (Tanzil, Basmala prefixes stripped).
    assert len(idx) == 77433
    assert len({w.surah for w in idx.words}) == 114
    assert len({(w.surah, w.ayah) for w in idx.words}) == 6236


def test_no_word_normalizes_to_empty(idx):
    assert all(w.norm for w in idx.words)


def test_basmala_is_gids_0_to_3(idx):
    assert idx.ayah_range(1, 1) == (0, 3)
    assert idx.surah_range(1)[0] == 0


def test_lookup_roundtrip(idx):
    gid = idx.gid(2, 255, 1)
    w = idx[gid]
    assert (w.surah, w.ayah, w.pos) == (2, 255, 1)
    assert w.norm == "الله"


def test_ranges_cross_ayah_and_surah_boundaries(idx):
    last_of_1 = idx.surah_range(1)[1]
    first_of_2 = idx.surah_range(2)[0]
    assert first_of_2 == last_of_1 + 1
    assert (idx[last_of_1].norm, idx[first_of_2].norm) == ("الضالين", "الم")
    assert len(idx.text(last_of_1, first_of_2).split()) == 2


def test_ayah_ranges_tile_the_surah(idx):
    first, last = idx.surah_range(112)
    covered = []
    for a in range(1, 5):
        s, e = idx.ayah_range(112, a)
        covered += list(range(s, e + 1))
    assert covered == list(range(first, last + 1))
