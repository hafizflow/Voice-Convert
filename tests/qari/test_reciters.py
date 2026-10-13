import pytest

from qari.reciters.registry import Reciter, list_reciters, load_reciter


def test_afasy_is_registered_with_license_status():
    r = load_reciter("afasy")
    assert isinstance(r, Reciter)
    assert r.id == "afasy"
    assert r.timestamps_source == "quran-align"
    # licensing must be recorded, and must say audio rights are not settled
    assert r.license["timestamps"].startswith("CC BY 4.0")
    assert r.license["audio_status"] == "unconfirmed"


def test_audio_url_is_built_from_template():
    r = load_reciter("afasy")
    assert r.audio_url(1, 1) == "https://everyayah.com/data/Alafasy_128kbps/001001.mp3"
    assert r.audio_url(114, 6).endswith("/114006.mp3")


def test_unknown_reciter_raises():
    with pytest.raises(KeyError, match="nobody"):
        load_reciter("nobody")


def test_list_reciters_contains_afasy():
    assert "afasy" in list_reciters()


def test_registry_entry_missing_license_fields_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("id: bad\ndisplay_name: Bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad"):
        load_reciter("bad", registry_dir=tmp_path)
