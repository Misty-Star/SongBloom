import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from training.preprocess.evaluate_vq_codebook import (
    evaluate_codebook_frames,
    summarize_assignments,
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


class EvaluateVQCodebookTest(unittest.TestCase):
    def test_summarize_assignments_reports_dead_code_ratio_and_entropy(self):
        assignments = torch.tensor([0, 0, 1, 2, 2, 2])

        summary = summarize_assignments(assignments, num_codes=4)

        self.assertAlmostEqual(summary["dead_code_ratio"], 0.25)
        self.assertIn("usage_entropy", summary)
        self.assertIn("top_1_usage_share", summary)

    def test_evaluate_codebook_frames_matches_chunked_distance_path(self):
        samples = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [1.0, 1.0],
                [1.2, 1.1],
                [2.0, 2.0],
                [2.2, 2.1],
            ],
            dtype=torch.float32,
        )
        codebook = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [9.0, 9.0],
            ],
            dtype=torch.float32,
        )

        full = evaluate_codebook_frames(samples, codebook)
        chunked = evaluate_codebook_frames(samples, codebook, distance_chunk_size=2)

        self.assertAlmostEqual(chunked["quantization_mse"], full["quantization_mse"])
        self.assertAlmostEqual(chunked["dead_code_ratio"], full["dead_code_ratio"])
        self.assertAlmostEqual(chunked["top_1_usage_share"], full["top_1_usage_share"])
        self.assertAlmostEqual(chunked["usage_entropy"], full["usage_entropy"])

    def test_evaluate_codebook_frames_tracks_chunk_progress_when_requested(self):
        progress_bars = []

        def fake_tqdm(*args, **kwargs):
            bar = _FakeProgress(*args, **kwargs)
            progress_bars.append(bar)
            return bar

        samples = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [1.0, 1.0],
                [1.2, 1.1],
                [2.0, 2.0],
                [2.2, 2.1],
            ],
            dtype=torch.float32,
        )
        codebook = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [9.0, 9.0],
            ],
            dtype=torch.float32,
        )

        with mock.patch("training.preprocess.evaluate_vq_codebook.tqdm", side_effect=fake_tqdm, create=True):
            report = evaluate_codebook_frames(
                samples,
                codebook,
                distance_chunk_size=2,
                progress_desc="[5/5] Evaluating train split",
            )

        self.assertIn("quantization_mse", report)
        self.assertEqual(len(progress_bars), 1)
        progress = progress_bars[0]
        self.assertEqual(progress.kwargs["total"], 3)
        self.assertEqual(progress.kwargs["unit"], "chunk")
        self.assertEqual(progress.updates, [1, 1, 1])

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
