import unittest
from unittest import mock

import torch

from training.preprocess import vq_codebook_trainer
from training.preprocess.vq_codebook_trainer import StreamingKMeans


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


class StreamingKMeansTest(unittest.TestCase):
    def test_partial_fit_updates_cluster_counts(self):
        trainer = StreamingKMeans(num_codes=4, embed_dim=2, device="cpu", seed=0)
        trainer.initialize(torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))

        trainer.partial_fit(torch.tensor([[0.1, 0.0], [9.9, 10.1]]))

        self.assertEqual(int(trainer.counts.sum().item()), 2)

    def test_refresh_dead_codes_refills_empty_centroids(self):
        trainer = StreamingKMeans(num_codes=3, embed_dim=2, device="cpu", seed=0)
        trainer.centers = torch.zeros(3, 2)
        trainer.counts = torch.tensor([10.0, 0.0, 0.0])

        trainer.refresh_dead_codes(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))

        self.assertTrue(torch.all(trainer.counts > 0))

    def test_usage_stats_reports_dead_code_ratio_and_entropy(self):
        trainer = StreamingKMeans(num_codes=4, embed_dim=2, device="cpu", seed=0)
        trainer.centers = torch.zeros(4, 2)
        trainer.counts = torch.tensor([3.0, 1.0, 0.0, 0.0])

        stats = trainer.usage_stats()

        self.assertAlmostEqual(stats["dead_code_ratio"], 0.5)
        self.assertIn("usage_entropy", stats)
        self.assertIn("top_1_usage_share", stats)

    def test_state_dict_round_trip_preserves_counts_and_centers(self):
        trainer = StreamingKMeans(num_codes=2, embed_dim=2, device="cpu", seed=0)
        trainer.centers = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        trainer.counts = torch.tensor([5.0, 6.0])

        restored = StreamingKMeans(num_codes=2, embed_dim=2, device="cpu", seed=9)
        restored.load_state_dict(trainer.state_dict())

        self.assertTrue(torch.equal(restored.centers, trainer.centers))
        self.assertTrue(torch.equal(restored.counts, trainer.counts))

    def test_fit_tracks_step_progress_and_usage_stats(self):
        progress_bars = []

        def fake_tqdm(*args, **kwargs):
            bar = _FakeProgress(*args, **kwargs)
            progress_bars.append(bar)
            return bar

        trainer = StreamingKMeans(num_codes=2, embed_dim=2, device="cpu", seed=0)
        samples = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
            dtype=torch.float32,
        )

        with mock.patch.object(vq_codebook_trainer, "tqdm", side_effect=fake_tqdm, create=True):
            trainer.fit(samples=samples, batch_size=2, num_steps=3, refresh_every=1)

        self.assertEqual(len(progress_bars), 1)
        progress = progress_bars[0]
        self.assertEqual(progress.kwargs["total"], 3)
        self.assertEqual(progress.kwargs["unit"], "step")
        self.assertEqual(progress.updates, [1, 1, 1])
        self.assertTrue(any("dead" in payload for payload in progress.postfixes))
        self.assertTrue(any("top1" in payload for payload in progress.postfixes))
        self.assertTrue(any("ppl" in payload for payload in progress.postfixes))

    def test_refine_full_tracks_batch_progress(self):
        progress_bars = []

        def fake_tqdm(*args, **kwargs):
            bar = _FakeProgress(*args, **kwargs)
            progress_bars.append(bar)
            return bar

        trainer = StreamingKMeans(num_codes=2, embed_dim=2, device="cpu", seed=0)
        trainer.initialize(torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32))
        samples = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
            dtype=torch.float32,
        )

        with mock.patch.object(vq_codebook_trainer, "tqdm", side_effect=fake_tqdm, create=True):
            centers = trainer.refine_full(samples=samples, batch_size=2)

        self.assertEqual(tuple(centers.shape), (2, 2))
        self.assertEqual(len(progress_bars), 1)
        progress = progress_bars[0]
        self.assertEqual(progress.kwargs["total"], 2)
        self.assertEqual(progress.kwargs["unit"], "batch")
        self.assertEqual(progress.updates, [1, 1])


if __name__ == "__main__":
    unittest.main()
