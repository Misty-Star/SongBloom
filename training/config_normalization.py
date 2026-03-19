"""Helpers for filling lightweight SongBloom config defaults."""

from omegaconf import open_dict


def ensure_songbloom_config_defaults(cfg):
    if cfg is None:
        return cfg

    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return cfg

    condition_provider_cfg = getattr(model_cfg, "condition_provider_cfg", None)
    if condition_provider_cfg is None:
        return cfg

    lyrics_cfg = getattr(condition_provider_cfg, "lyrics", None)
    max_dur = getattr(cfg, "max_dur", None)
    if lyrics_cfg is None or max_dur is None:
        return cfg

    current_max_duration = getattr(lyrics_cfg, "max_duration", None)
    if current_max_duration not in (None, ""):
        return cfg

    with open_dict(lyrics_cfg):
        lyrics_cfg.max_duration = max_dur
    return cfg
