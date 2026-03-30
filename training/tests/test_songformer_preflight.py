import importlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


from training.preprocess import build_dataset
from training.preprocess.extract_structure import collect_songformer_runtime_issues


def import_prepare_assets_with_mocked_torchaudio():
    sys.modules.pop("training.preprocess.prepare_assets", None)
    fake_torchaudio = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
        module = importlib.import_module("training.preprocess.prepare_assets")
    return module


class SongFormerPreflightTest(unittest.TestCase):
    def test_collect_songformer_runtime_issues_detects_numpy_and_gpu_mismatch(self):
        completed = subprocess.CompletedProcess(
            args=["python", "-c", "probe"],
            returncode=0,
            stdout=(
                '{"torch_version": "2.4.0+cu121", '
                '"cuda_available": true, '
                '"device_capability": "sm_120", '
                '"arch_list": ["sm_80", "sm_90"], '
                '"numpy_version": "2.2.6"}\n'
            ),
            stderr="",
        )

        with mock.patch("training.preprocess.extract_structure.subprocess.run", return_value=completed):
            issues = collect_songformer_runtime_issues("python")

        self.assertTrue(any("NumPy 2.2.6" in issue for issue in issues))
        self.assertTrue(any("sm_120" in issue and "2.4.0+cu121" in issue for issue in issues))

    def test_prepare_assets_collect_preflight_issues_includes_songformer_runtime_issues(self):
        prepare_assets = import_prepare_assets_with_mocked_torchaudio()
        args = SimpleNamespace(
            skip_separation=True,
            skip_whisperx=True,
            skip_structure=False,
            separator_cmd="audio-separator",
            separator_model_flag="auto",
            separator_model="model.ckpt",
            separator_model_file_dir="",
            assets_dir="/tmp/assets",
            separator_output_format="FLAC",
            songformer_python="python",
            songformer_root="third_party/SongFormer",
        )
        items = [{"id": "song_a", "audio_path": "/tmp/song_a.flac"}]

        def fake_exists(path):
            return str(path).endswith("src/SongFormer/infer/infer.py")

        with mock.patch.object(prepare_assets, "detect_separator_model_flag", return_value="--model_filename"), mock.patch.object(
            prepare_assets,
            "resolve_separator_model",
            return_value="model.ckpt",
        ), mock.patch.object(
            prepare_assets,
            "command_exists",
            return_value=True,
        ), mock.patch.object(
            prepare_assets,
            "collect_songformer_runtime_issues",
            return_value=["songformer runtime bad"],
        ), mock.patch.object(prepare_assets.os.path, "exists", side_effect=fake_exists):
            issues = prepare_assets.collect_preflight_issues(items, args)

        self.assertIn("songformer runtime bad", issues)

    def test_build_dataset_collect_preflight_issues_includes_songformer_runtime_issues(self):
        args = SimpleNamespace(
            skip_demucs=False,
            demucs_cmd="demucs",
            whisperx_cmd="whisperx",
            songformer_python="python",
            songformer_root="third_party/SongFormer",
        )
        items = [
            {
                "id": "song_a",
                "audio_path": "/tmp/song_a.flac",
                "vocals_path": "/tmp/song_a_vocals.flac",
                "no_vocals_path": "/tmp/song_a_no_vocals.flac",
                "whisperx_json": "/tmp/song_a_whisperx.json",
            }
        ]

        def fake_exists(path):
            return str(path).endswith("src/SongFormer/infer/infer.py")

        with mock.patch.object(build_dataset, "command_exists", return_value=True), mock.patch.object(
            build_dataset,
            "collect_songformer_runtime_issues",
            return_value=["songformer runtime bad"],
        ), mock.patch.object(build_dataset.os.path, "exists", side_effect=fake_exists):
            issues = build_dataset.collect_preflight_issues(items, args)

        self.assertIn("songformer runtime bad", issues)


if __name__ == "__main__":
    unittest.main()
