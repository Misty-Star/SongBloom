import random
import unittest

import torch

from training.preprocess.vq_codebook_dataset import (
    align_embeddings_to_target_fps,
    build_chunk_spans,
    sample_frames_from_embedding,
    split_audio_paths,
)


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


if __name__ == "__main__":
    unittest.main()
