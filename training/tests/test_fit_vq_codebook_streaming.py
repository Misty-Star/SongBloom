import json
import os
import tempfile
import unittest
from unittest import mock

import torch


class FitVQCodebookStreamingCLITest(unittest.TestCase):
    def test_main_forwards_memory_control_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "vq_codebook.pt")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
                "--num-codes", "4",
                "--muq-chunk-seconds", "8",
                "--distance-chunk-size", "16",
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.fit_vq_codebook_streaming import main

                with mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.resolve_audio_paths",
                    return_value=[f"/tmp/song_{index}.flac" for index in range(6)],
                ), mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.train_codebook",
                    return_value={"codebook_path": output_path},
                ) as train_codebook_mock:
                    main()

            train_codebook_mock.assert_called_once()
            call_kwargs = train_codebook_mock.call_args.kwargs
            self.assertEqual(call_kwargs["muq_chunk_seconds"], 8.0)
            self.assertEqual(call_kwargs["distance_chunk_size"], 16)

    def test_main_writes_codebook_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "vq_codebook.pt")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
                "--num-codes", "4",
                "--max-total-train-frames", "8",
                "--max-total-heldout-frames", "4",
                "--train-steps", "2",
                "--batch-size", "2",
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.fit_vq_codebook_streaming import main

                fake_frames = torch.tensor(
                    [
                        [0.0, 0.0],
                        [0.1, 0.0],
                        [10.0, 10.0],
                        [10.1, 10.0],
                        [20.0, 20.0],
                        [20.1, 20.0],
                        [30.0, 30.0],
                        [30.1, 30.0],
                    ],
                    dtype=torch.float32,
                )

                with mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.resolve_audio_paths",
                    return_value=[f"/tmp/song_{index}.flac" for index in range(6)],
                ), mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.load_muq_model",
                    return_value=mock.Mock(),
                ), mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.collect_embedding_samples",
                    side_effect=[fake_frames, fake_frames[:4]],
                ):
                    main()

            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(output_path.replace(".pt", ".meta.json")))
            self.assertTrue(os.path.exists(output_path.replace(".pt", ".report.json")))

            state = torch.load(output_path, map_location="cpu")
            self.assertEqual(tuple(state["codebook.weight"].shape), (4, 2))

            with open(output_path.replace(".pt", ".report.json"), "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertIn("train_metrics", report)
            self.assertIn("heldout_metrics", report)

    def test_main_raises_when_num_codes_exceeds_sampled_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "vq_codebook.pt")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
                "--num-codes", "8",
                "--max-total-train-frames", "4",
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.fit_vq_codebook_streaming import main

                with mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.resolve_audio_paths",
                    return_value=["/tmp/song_a.flac", "/tmp/song_b.flac"],
                ), mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.load_muq_model",
                    return_value=mock.Mock(),
                ), mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.collect_embedding_samples",
                    return_value=torch.ones(4, 2),
                ):
                    with self.assertRaisesRegex(ValueError, "Need at least 8 sampled frames"):
                        main()

    def test_main_raises_when_no_audio_paths_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "vq_codebook.pt")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
            ]

            with mock.patch("sys.argv", argv):
                from training.preprocess.fit_vq_codebook_streaming import main

                with mock.patch(
                    "training.preprocess.fit_vq_codebook_streaming.resolve_audio_paths",
                    return_value=[],
                ):
                    with self.assertRaisesRegex(RuntimeError, "No audio files found"):
                        main()


if __name__ == "__main__":
    unittest.main()
