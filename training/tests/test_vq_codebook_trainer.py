import unittest

import torch

from training.preprocess.vq_codebook_trainer import StreamingKMeans


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


if __name__ == "__main__":
    unittest.main()
