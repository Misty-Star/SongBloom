"""Utilities for optional structure-duration interpolation."""


def maybe_interpolate_structure_duration(
    interpolate_fn,
    tokens,
    embeds,
    structure_dur,
    batch_index,
):
    if structure_dur is None:
        return embeds

    sample_structure_dur = structure_dur[batch_index]
    if sample_structure_dur is None:
        return embeds

    return interpolate_fn(tokens, embeds, sample_structure_dur)
