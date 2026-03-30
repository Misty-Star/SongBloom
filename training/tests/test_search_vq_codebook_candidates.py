import json
import os
import tempfile
import unittest
from unittest import mock

from training.preprocess.search_vq_codebook_candidates import rank_candidates


class SearchVQCodebookCandidatesTest(unittest.TestCase):
    def test_rank_candidates_filters_collapsed_codebooks_first(self):
        rows = [
            {"name": "good", "dead_code_ratio": 0.01, "top_1_usage_share": 0.01, "quantization_mse": 0.5},
            {"name": "collapsed", "dead_code_ratio": 0.40, "top_1_usage_share": 0.30, "quantization_mse": 0.1},
        ]

        ranked = rank_candidates(rows, max_dead_code_ratio=0.05, max_top1_share=0.05)

        self.assertEqual(ranked[0]["name"], "good")
        self.assertEqual(len(ranked), 1)

    def test_main_writes_candidate_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-dir", temp_dir,
                "--device", "cpu",
                "--seeds", "11", "17",
                "--frame-budgets", "1000",
            ]

            fake_reports = [
                {
                    "codebook_path": os.path.join(temp_dir, "seed11_frames1000.pt"),
                    "train_metrics": {"dead_code_ratio": 0.01, "top_1_usage_share": 0.02},
                    "heldout_metrics": {"dead_code_ratio": 0.01, "top_1_usage_share": 0.02, "quantization_mse": 0.3},
                    "metadata": {"num_train_frames": 1000, "num_heldout_frames": 100},
                },
                {
                    "codebook_path": os.path.join(temp_dir, "seed17_frames1000.pt"),
                    "train_metrics": {"dead_code_ratio": 0.02, "top_1_usage_share": 0.03},
                    "heldout_metrics": {"dead_code_ratio": 0.02, "top_1_usage_share": 0.03, "quantization_mse": 0.2},
                    "metadata": {"num_train_frames": 1000, "num_heldout_frames": 100},
                },
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.search_vq_codebook_candidates import main

                with mock.patch(
                    "training.preprocess.search_vq_codebook_candidates.resolve_audio_paths",
                    return_value=["/tmp/song_a.flac", "/tmp/song_b.flac"],
                ), mock.patch(
                    "training.preprocess.search_vq_codebook_candidates.train_codebook",
                    side_effect=fake_reports,
                ):
                    main()

            summary_path = os.path.join(temp_dir, "candidate_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["best_candidate"]["seed"], 17)
            self.assertEqual(len(summary["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
