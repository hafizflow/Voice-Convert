from engine.converter import PassthroughConverter


def default_converter_factory(cfg):
    """voice -> StreamingConverter. Voices without a model file stay pass-through."""

    def build(voice):
        if voice.model_path is None:
            return PassthroughConverter()
        from engine.rvc import RVCConverter  # heavy import: only when a real voice is used

        return RVCConverter(
            voice.model_path, voice.index_path,
            block_ms=cfg.rvc_block_ms, extra_s=cfg.rvc_extra_s, f0_method=cfg.rvc_f0,
            device=cfg.rvc_device, index_rate=cfg.rvc_index_rate,
            pitch_shift=cfg.rvc_pitch_shift,
        )

    return build
