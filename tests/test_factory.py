from pathlib import Path

from engine.converter import PassthroughConverter
from engine.factory import default_converter_factory
from engine.registry import Voice
from server.config import Config

CFG = Config("ws://x", "ws://x", "k", "s", "a", "i", "http://w", Path("v"), 4, 30.0,
             rvc_block_ms=160, rvc_extra_s=0.2, rvc_f0="fcpe", rvc_device="cpu", rvc_index_rate=0.3, rvc_pitch_shift=3)


def test_voice_without_model_is_passthrough():
    v = Voice("demo", "Demo", 120.0, "c", None, None)
    assert isinstance(default_converter_factory(CFG)(v), PassthroughConverter)


def test_voice_with_model_builds_rvc_with_configured_settings(monkeypatch):
    seen = {}

    class FakeRVC:
        def __init__(self, model, index, **kw):
            seen.update(model=model, index=index, **kw)

    import engine.rvc as rvc_mod

    monkeypatch.setattr(rvc_mod, "RVCConverter", FakeRVC)
    v = Voice("q", "Q", 110.0, "c", Path("/v/q/model.pth"), Path("/v/q/added.index"))
    default_converter_factory(CFG)(v)
    assert seen["model"] == Path("/v/q/model.pth") and seen["index"] == Path("/v/q/added.index")
    assert (seen["block_ms"], seen["extra_s"], seen["f0_method"], seen["device"], seen["index_rate"]) == (
        160, 0.2, "fcpe", "cpu", 0.3)
    assert seen["pitch_shift"] == 3
