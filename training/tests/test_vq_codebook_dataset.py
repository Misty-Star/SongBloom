import random
import sys
import types
import unittest
from unittest import mock

import torch

from training.preprocess.vq_codebook_dataset import (
    align_embeddings_to_target_fps,
    build_chunk_spans,
    collect_embedding_samples,
    sample_frames_from_embedding,
    split_audio_paths,
)


class _FakeProgress:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.updates = []
        self.postfixes = []
        self.closed = False

    def update(self, value=1):
        self.updates.append(value)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        payload = dict(ordered_dict or {})
        payload.update(kwargs)
        self.postfixes.append(payload)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class VQCodebookDatasetTest(unittest.TestCase):
    def test_split_audio_paths_is_deterministic_and_disjoint(self):
        audio_paths = [f"/tmp/song_{index}.flac" for index in range(10)]

        train_a, heldout_a = split_audio_paths(audio_paths, heldout_ratio=0.2, seed=7)
        train_b, heldout_b = split_audio_paths(audio_paths, heldout_ratio=0.2, seed=7)

        self.assertEqual(train_a, train_b)
        self.assertEqual(heldout_a, heldout_b)
        self.assertTrue(set(train_a).isdisjoint(set(heldout_a)))
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(heldout_a), 2)

    def test_sample_frames_from_embedding_respects_per_audio_cap(self):
        embedding = torch.arange(0, 120, dtype=torch.float32).view(1, 15, 8)

        sampled = sample_frames_from_embedding(embedding, max_frames=4, rng=random.Random(3))

        self.assertEqual(tuple(sampled.shape), (4, 8))
        self.assertEqual(sampled.dtype, torch.float32)

    def test_align_embeddings_to_target_fps_resamples_temporal_axis(self):
        embedding = torch.arange(24, dtype=torch.float32).view(1, 6, 4)

        aligned = align_embeddings_to_target_fps(
            embedding=embedding,
            audio_num_samples=48000 * 4,
            sample_rate=48000,
            target_fps=2,
        )

        self.assertEqual(tuple(aligned.shape), (1, 8, 4))

    def test_build_chunk_spans_merges_short_tail_into_previous_chunk(self):
        spans = build_chunk_spans(
            total_num_samples=48_000 * 45,
            chunk_num_samples=48_000 * 20,
            min_chunk_num_samples=48_000 * 10,
        )

        self.assertEqual(
            spans,
            [
                (0, 48_000 * 20),
                (48_000 * 20, 48_000 * 45),
            ],
        )

    def test_collect_embedding_samples_tracks_frame_budget_progress(self):
        progress_bars = []

        def fake_tqdm(*args, **kwargs):
            bar = _FakeProgress(*args, **kwargs)
            progress_bars.append(bar)
            return bar

        fake_torchaudio = types.SimpleNamespace(
            load=lambda _path: (torch.ones(1, 48_000), 48_000),
            transforms=types.SimpleNamespace(Resample=lambda _src, _dst: lambda wav: wav),
        )
        fake_extract_sketch = types.SimpleNamespace(
            extract_muq_embeddings=lambda _model, _wav, _sr: torch.arange(6, dtype=torch.float32).view(1, 3, 2),
        )

        with mock.patch("training.preprocess.vq_codebook_dataset.tqdm", side_effect=fake_tqdm), mock.patch.dict(
            sys.modules,
            {
                "torchaudio": fake_torchaudio,
                "training.preprocess.extract_sketch": fake_extract_sketch,
            },
        ):
            samples = collect_embedding_samples(
                audio_paths=["/tmp/song_a.flac", "/tmp/song_b.flac", "/tmp/song_c.flac"],
                muq_model=object(),
                sample_rate=48_000,
                target_fps=3,
                frames_per_audio=3,
                max_total_frames=6,
                seed=0,
                progress_desc="[1/5] Collecting train MuQ frames",
            )

        self.assertEqual(tuple(samples.shape), (6, 2))
        self.assertEqual(len(progress_bars), 1)
        progress = progress_bars[0]
        self.assertEqual(progress.kwargs["total"], 6)
        self.assertEqual(progress.kwargs["unit"], "frame")
        self.assertEqual(progress.updates, [3, 3])
        self.assertTrue(any("songs" in payload for payload in progress.postfixes))
        self.assertTrue(any("audio" in payload for payload in progress.postfixes))

    def test_collect_embedding_samples_treats_zero_budget_as_full_ceiling(self):
        progress_bars = []

        def fake_tqdm(*args, **kwargs):
            bar = _FakeProgress(*args, **kwargs)
            progress_bars.append(bar)
            return bar

        fake_torchaudio = types.SimpleNamespace(
            load=lambda _path: (torch.ones(1, 48_000), 48_000),
            transforms=types.SimpleNamespace(Resample=lambda _src, _dst: lambda wav: wav),
        )
        fake_extract_sketch = types.SimpleNamespace(
            extract_muq_embeddings=lambda _model, _wav, _sr: torch.arange(6, dtype=torch.float32).view(1, 3, 2),
        )

        with mock.patch("training.preprocess.vq_codebook_dataset.tqdm", side_effect=fake_tqdm), mock.patch.dict(
            sys.modules,
            {
                "torchaudio": fake_torchaudio,
                "training.preprocess.extract_sketch": fake_extract_sketch,
            },
        ):
            samples = collect_embedding_samples(
                audio_paths=["/tmp/song_a.flac", "/tmp/song_b.flac", "/tmp/song_c.flac"],
                muq_model=object(),
                sample_rate=48_000,
                target_fps=3,
                frames_per_audio=3,
                max_total_frames=0,
                seed=0,
                progress_desc="[1/5] Collecting train MuQ frames",
            )

        self.assertEqual(tuple(samples.shape), (9, 2))
        self.assertEqual(len(progress_bars), 1)
        progress = progress_bars[0]
        self.assertIsNone(progress.kwargs["total"])
        self.assertEqual(progress.kwargs["unit"], "frame")
        self.assertEqual(progress.updates, [3, 3, 3])
        self.assertTrue(any(payload.get("frames") == "9/full" for payload in progress.postfixes))


if __name__ == "__main__":
    unittest.main()
