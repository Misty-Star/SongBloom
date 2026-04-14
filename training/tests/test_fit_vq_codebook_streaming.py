import io
import json
import os
import tempfile
import unittest
from unittest import mock

import torch


class FitVQCodebookStreamingCLITest(unittest.TestCase):
    def test_parse_args_help_describes_zero_as_full_ceiling(self):
        stdout = io.StringIO()
        with mock.patch("sys.argv", ["prog", "--help"]), mock.patch("sys.stdout", stdout):
            from training.preprocess.fit_vq_codebook_streaming import parse_args

            with self.assertRaises(SystemExit) as exit_context:
                parse_args()

        self.assertEqual(exit_context.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--max-total-train-frames", help_text)
        self.assertIn("--max-total-heldout-frames", help_text)
        self.assertGreaterEqual(help_text.count("0 表示 full ceiling"), 2)

    def test_main_forwards_memory_control_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "vq_codebook.pt")
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
                "--num-codes", "4",
                "--muq-cache-dir", "/tmp/muq_cache",
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
            self.assertEqual(call_kwargs["muq_cache_dir"], "/tmp/muq_cache")
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

    def test_train_codebook_uses_numbered_stage_descriptions(self):
        fake_train_frames = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
            ],
            dtype=torch.float32,
        )
        fake_heldout_frames = fake_train_frames[:2]
        trainer = mock.Mock()
        trainer.refine_full.return_value = torch.tensor(
            [[0.0, 0.0], [10.0, 10.0]],
            dtype=torch.float32,
        )

        from training.preprocess.fit_vq_codebook_streaming import train_codebook

        with mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.load_muq_model",
            return_value=mock.Mock(),
        ), mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.collect_embedding_samples",
            side_effect=[fake_train_frames, fake_heldout_frames],
        ) as collect_mock, mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.StreamingKMeans",
            return_value=trainer,
        ), mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.evaluate_codebook_frames",
            side_effect=[
                {"dead_code_ratio": 0.01, "top_1_usage_share": 0.02, "quantization_mse": 0.3},
                {"dead_code_ratio": 0.01, "top_1_usage_share": 0.02, "quantization_mse": 0.4},
            ],
        ) as evaluate_mock, mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.save_codebook",
        ), mock.patch(
            "training.preprocess.fit_vq_codebook_streaming.save_report",
        ):
            train_codebook(
                audio_paths=["/tmp/song_0.flac", "/tmp/song_1.flac", "/tmp/song_2.flac", "/tmp/song_3.flac"],
                output_path="/tmp/vq_codebook.pt",
                muq_model_name="fake-muq",
                device="cpu",
                sample_rate=48_000,
                target_fps=25,
                frames_per_audio=2,
                max_total_train_frames=4,
                max_total_heldout_frames=2,
                num_codes=2,
                heldout_ratio=0.25,
                batch_size=2,
                train_steps=3,
                refresh_every=1,
                seed=7,
                muq_chunk_seconds=0.0,
                distance_chunk_size=2,
            )

        collect_descs = [call.kwargs["progress_desc"] for call in collect_mock.call_args_list]
        self.assertEqual(
            collect_descs,
            [
                "[1/5] Collecting train MuQ frames",
                "[2/5] Collecting heldout MuQ frames",
            ],
        )
        self.assertEqual(trainer.fit.call_args.kwargs["progress_desc"], "[3/5] Streaming K-Means")
        self.assertEqual(trainer.refine_full.call_args.kwargs["progress_desc"], "[4/5] Final refine pass")
        evaluate_descs = [call.kwargs["progress_desc"] for call in evaluate_mock.call_args_list]
        self.assertEqual(
            evaluate_descs,
            [
                "[5/5] Evaluating train split",
                "[5/5] Evaluating heldout split",
            ],
        )


if __name__ == "__main__":
    unittest.main()
