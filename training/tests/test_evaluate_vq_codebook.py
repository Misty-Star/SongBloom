import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from training.preprocess.evaluate_vq_codebook import summarize_assignments


class EvaluateVQCodebookTest(unittest.TestCase):
    def test_summarize_assignments_reports_dead_code_ratio_and_entropy(self):
        assignments = torch.tensor([0, 0, 1, 2, 2, 2])

        summary = summarize_assignments(assignments, num_codes=4)

        self.assertAlmostEqual(summary["dead_code_ratio"], 0.25)
        self.assertIn("usage_entropy", summary)
        self.assertIn("top_1_usage_share", summary)

    def test_main_writes_evaluation_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codebook_path = os.path.join(temp_dir, "vq_codebook.pt")
            torch.save({"codebook.weight": torch.tensor([[0.0, 0.0], [1.0, 1.0]])}, codebook_path)

            report_path = os.path.join(temp_dir, "evaluation.report.json")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--codebook-path", codebook_path,
                "--report-path", report_path,
                "--device", "cpu",
                "--num-codes", "2",
                "--max-total-frames", "4",
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.evaluate_vq_codebook import main

                with mock.patch(
                    "training.preprocess.evaluate_vq_codebook.resolve_audio_paths",
                    return_value=["/tmp/song_a.flac", "/tmp/song_b.flac"],
                ), mock.patch(
                    "training.preprocess.evaluate_vq_codebook.load_muq_model",
                    return_value=mock.Mock(),
                ), mock.patch(
                    "training.preprocess.evaluate_vq_codebook.collect_embedding_samples",
                    return_value=torch.tensor(
                        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [0.9, 1.0]], dtype=torch.float32
                    ),
                ):
                    main()

            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertIn("quantization_mse", report)
            self.assertIn("dead_code_ratio", report)


if __name__ == "__main__":
    unittest.main()
