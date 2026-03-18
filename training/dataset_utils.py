"""Dataset helper utilities for frame-length alignment."""


def compute_effective_frame_length(
    sketch_frames,
    latent_frames,
    duration_seconds,
    block_size,
    max_duration,
    frame_rate=25,
):
    x_len = min(sketch_frames, latent_frames)
    if duration_seconds is not None:
        x_len = min(x_len, int(duration_seconds * frame_rate))

    max_frames = int(max_duration * frame_rate)
    x_len = min(x_len, max_frames)
    return (x_len // block_size) * block_size
