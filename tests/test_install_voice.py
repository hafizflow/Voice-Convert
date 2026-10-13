import numpy as np
import pytest
import soundfile as sf

from engine.registry import load_registry
from tools.install_voice import install_voice, latest_weights, median_f0


def make_logs(tmp_path, name="qari"):
    logs = tmp_path / "logs" / name
    logs.mkdir(parents=True)
    for e, steps in [(10, 1170), (20, 2340), (50, 5850)]:
        (logs / f"{name}_{e}e_{steps}s.pth").write_bytes(f"w{e}".encode())
    (logs / f"{name}.index").write_bytes(b"idx")
    (logs / "G_5850.pth").write_bytes(b"big-training-checkpoint")  # must never be picked
    return logs


def test_latest_weights_picks_highest_epoch_and_ignores_training_checkpoints(tmp_path):
    logs = make_logs(tmp_path)
    assert latest_weights(logs, "qari").name == "qari_50e_5850s.pth"


def test_latest_weights_errors_when_none(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        latest_weights(tmp_path / "empty", "qari")


def test_install_creates_a_registry_valid_voice(tmp_path):
    logs = make_logs(tmp_path)
    voices = tmp_path / "voices"
    install_voice("qari", "Qari 012", logs, voices, f0_median_hz=118.0,
                  consent_source="user-supplied recording; user confirmed rights")
    reg = load_registry(voices)
    v = reg["qari"]
    assert v.name == "Qari 012" and v.f0_median_hz == 118.0
    assert v.model_path.read_bytes() == b"w50" and v.index_path.read_bytes() == b"idx"


def test_install_refuses_blank_consent_source(tmp_path):
    logs = make_logs(tmp_path)
    with pytest.raises(ValueError):
        install_voice("qari", "Q", logs, tmp_path / "v", f0_median_hz=100.0, consent_source=" ")


def test_median_f0_of_a_pure_tone(tmp_path):
    pytest.importorskip("librosa")
    sr = 16000
    t = np.arange(sr * 3) / sr
    p = tmp_path / "t.wav"
    sf.write(p, (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    assert abs(median_f0(str(p), seconds=3) - 150) < 5
