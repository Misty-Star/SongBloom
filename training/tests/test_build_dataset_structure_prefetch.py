import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock


def import_build_dataset_with_mocked_torchaudio():
    sys.modules.pop("training.preprocess.build_dataset", None)
    fake_torchaudio = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
        module = importlib.import_module("training.preprocess.build_dataset")
    return module


class BuildDatasetStructurePrefetchTest(unittest.TestCase):
    def test_main_prefetches_missing_structure_json_before_process_sample(self):
        build_dataset = import_build_dataset_with_mocked_torchaudio()

        raw_items = [
            {"id": "song_a", "audio_path": "/tmp/song_a.flac"},
            {"id": "song_b", "audio_path": "/tmp/song_b.flac", "structure_json": "/tmp/existing_song_b.json"},
        ]
        prefetched_items = [
            {"id": "song_a", "audio_path": "/tmp/song_a.flac", "structure_json": "/tmp/prefetched_song_a.json"},
            {"id": "song_b", "audio_path": "/tmp/song_b.flac", "structure_json": "/tmp/existing_song_b.json"},
        ]
        seen_items = []

        def fake_process_sample(item, args, features, sample_index, total_items):
            seen_items.append(dict(item))
            return {"id": item["id"], "status": "ok", "output_dir": os.path.join(args.output_dir, item["id"])}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            workspace_dir = os.path.join(temp_dir, "workspace")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(workspace_dir, exist_ok=True)

            with mock.patch.object(build_dataset, "load_jsonl", return_value=raw_items), mock.patch.object(
                build_dataset,
                "collect_preflight_issues",
                return_value=[],
            ), mock.patch.object(
                build_dataset,
                "prepare_structure_assets",
                return_value=(
                    prefetched_items,
                    [
                        {"id": "song_a", "structure_status": "ok", "structure_json": "/tmp/prefetched_song_a.json"},
                        {"id": "song_b", "structure_status": "provided", "structure_json": "/tmp/existing_song_b.json"},
                    ],
                ),
            ) as prepare_structure_assets_mock, mock.patch.object(
                build_dataset,
                "FeatureExtractorBundle",
                return_value=object(),
            ), mock.patch.object(
                build_dataset,
                "process_sample",
                side_effect=fake_process_sample,
            ), mock.patch.object(
                build_dataset,
                "write_jsonl",
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "build_dataset",
                    "--input-jsonl",
                    os.path.join(temp_dir, "input.jsonl"),
                    "--output-dir",
                    output_dir,
                    "--workspace-dir",
                    workspace_dir,
                    "--vq-ckpt",
                    os.path.join(temp_dir, "vq.pt"),
                ],
            ):
                build_dataset.main()

        self.assertEqual([item["structure_json"] for item in seen_items], [
            "/tmp/prefetched_song_a.json",
            "/tmp/existing_song_b.json",
        ])
        prepare_structure_assets_mock.assert_called_once()
        self.assertEqual(
            prepare_structure_assets_mock.call_args.kwargs["assets_dir"],
            os.path.join(workspace_dir, "_structure_assets"),
        )


if __name__ == "__main__":
    unittest.main()
