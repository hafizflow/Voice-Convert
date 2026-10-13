from pathlib import Path

from engine.registry import load_registry


def write_voice(root: Path, name: str, yaml_text: str, files: tuple[str, ...] = ()) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "voice.yaml").write_text(yaml_text)
    for f in files:
        (d / f).write_bytes(b"x")


def test_loads_valid_voice(tmp_path):
    write_voice(
        tmp_path, "qari1",
        "name: Qari One\nf0_median_hz: 130.5\nconsent_source: licensed by X, contract 12\n"
        "model: model.pth\nindex: added.index\n",
        files=("model.pth", "added.index"),
    )
    reg = load_registry(tmp_path)
    v = reg["qari1"]
    assert v.name == "Qari One"
    assert v.f0_median_hz == 130.5
    assert v.model_path == tmp_path / "qari1" / "model.pth"
    assert v.index_path == tmp_path / "qari1" / "added.index"


def test_model_is_optional_for_passthrough_voice(tmp_path):
    write_voice(tmp_path, "demo", "name: Demo\nf0_median_hz: 120\nconsent_source: demo\n")
    v = load_registry(tmp_path)["demo"]
    assert v.model_path is None and v.index_path is None


def test_missing_consent_source_is_skipped(tmp_path):
    write_voice(tmp_path, "bad", "name: Bad\nf0_median_hz: 120\n")
    assert load_registry(tmp_path) == {}


def test_blank_consent_source_is_skipped(tmp_path):
    write_voice(tmp_path, "bad", "name: Bad\nf0_median_hz: 120\nconsent_source: '  '\n")
    assert load_registry(tmp_path) == {}


def test_malformed_yaml_does_not_stop_other_voices(tmp_path):
    write_voice(tmp_path, "broken", "name: [unclosed\n")
    write_voice(tmp_path, "ok", "name: Ok\nf0_median_hz: 100\nconsent_source: own recording\n")
    assert set(load_registry(tmp_path)) == {"ok"}


def test_declared_model_file_missing_is_skipped(tmp_path):
    write_voice(tmp_path, "nomodel", "name: N\nf0_median_hz: 100\nconsent_source: c\nmodel: model.pth\n")
    assert load_registry(tmp_path) == {}


def test_nonexistent_root_gives_empty_registry(tmp_path):
    assert load_registry(tmp_path / "nope") == {}


def test_nonpositive_f0_is_skipped(tmp_path):
    write_voice(tmp_path, "z", "name: Z\nf0_median_hz: 0\nconsent_source: c\n")
    assert load_registry(tmp_path) == {}
