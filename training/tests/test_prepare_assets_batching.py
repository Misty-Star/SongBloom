import importlib
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


def import_prepare_assets_with_mocked_torchaudio():
    sys.modules.pop("training.preprocess.prepare_assets", None)
    fake_torchaudio = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
        module = importlib.import_module("training.preprocess.prepare_assets")
    return module


class PrepareAssetsBatchingTest(unittest.TestCase):
    def test_run_whisperx_batch_with_recovery_isolates_single_failure(self):
        prepare_assets = import_prepare_assets_with_mocked_torchaudio()

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_root = os.path.join(temp_dir, "whisperx")
            os.makedirs(batch_root, exist_ok=True)
            audio_inputs = {}
            batch_items = []
            for sample_id in ("good_a", "bad_song", "good_b"):
                audio_path = os.path.join(temp_dir, sample_id + ".flac")
                with open(audio_path, "w", encoding="utf-8") as handle:
                    handle.write(sample_id)
                audio_inputs[sample_id] = audio_path
                batch_items.append({"id": sample_id})

            call_sets = []

            def fake_run_whisperx_batch_cli(audio_inputs, output_dir, **kwargs):
                sample_ids = tuple(audio_inputs.keys())
                call_sets.append(sample_ids)
                if "bad_song" in audio_inputs:
                    raise RuntimeError("pure instrumental / no speech")
                os.makedirs(output_dir, exist_ok=True)
                outputs = {}
                for sample_id, audio_path in audio_inputs.items():
                    output_path = prepare_assets.expected_whisperx_output(audio_path, output_dir)
                    with open(output_path, "w", encoding="utf-8") as handle:
                        json.dump({"id": sample_id}, handle)
                    outputs[sample_id] = output_path
                return outputs

            args = SimpleNamespace(
                whisperx_cmd="whisperx",
                whisperx_model="large-v3",
                whisperx_device="cuda",
                whisperx_compute_type="float16",
                whisperx_batch_size=8,
                whisperx_timeout_sec=0.0,
            )

            with mock.patch.object(
                prepare_assets,
                "run_whisperx_batch_cli",
                side_effect=fake_run_whisperx_batch_cli,
            ):
                outputs, errors = prepare_assets.run_whisperx_batch_with_recovery(
                    batch_items=batch_items,
                    audio_inputs=audio_inputs,
                    batch_root=batch_root,
                    args=args,
                    language=None,
                )

        self.assertEqual(set(outputs.keys()), {"good_a", "good_b"})
        self.assertEqual(set(errors.keys()), {"bad_song"})
        self.assertIn("pure instrumental", errors["bad_song"])
        self.assertIn(("good_a", "bad_song", "good_b"), call_sets)
        self.assertIn(("bad_song",), call_sets)

    def test_prepare_separator_assets_merges_batch_results_into_manifest_and_report(self):
        prepare_assets = import_prepare_assets_with_mocked_torchaudio()

        with tempfile.TemporaryDirectory() as temp_dir:
            assets_dir = os.path.join(temp_dir, "assets")
            workspace_dir = os.path.join(temp_dir, "workspace")
            os.makedirs(assets_dir, exist_ok=True)
            os.makedirs(workspace_dir, exist_ok=True)

            records = [
                {"id": "song_a", "audio_path": os.path.join(temp_dir, "song_a.flac")},
                {"id": "song_b", "audio_path": os.path.join(temp_dir, "song_b.flac")},
            ]
            for record in records:
                with open(record["audio_path"], "w", encoding="utf-8") as handle:
                    handle.write(record["id"])

            output_rows, report_rows = prepare_assets.initialize_prepare_assets_rows(records)

            def fake_convert_audio(input_path, output_path, sample_rate, mono=False):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write(os.path.basename(input_path))
                return output_path, 1.0

            batch_calls = []

            def fake_run_audio_separator_batch(batch_items, args):
                batch_calls.append([item["id"] for item in batch_items])
                results = {}
                for item in batch_items:
                    os.makedirs(os.path.dirname(item["vocals_output_path"]), exist_ok=True)
                    with open(item["vocals_output_path"], "w", encoding="utf-8") as handle:
                        handle.write("vocals")
                    with open(item["no_vocals_output_path"], "w", encoding="utf-8") as handle:
                        handle.write("inst")
                    results[item["id"]] = {
                        "vocals_path": item["vocals_output_path"],
                        "no_vocals_path": item["no_vocals_output_path"],
                    }
                return results, {}

            args = SimpleNamespace(
                skip_separation=False,
                skip_existing=False,
                assets_dir=assets_dir,
                workspace_dir=workspace_dir,
                asset_sample_rate=44100,
                separator_output_format="FLAC",
                separator_max_files_per_batch=32,
                separator_python="python",
                separator_cmd="audio-separator",
                separator_model="BS-Roformer-Viperx-1297",
                separator_model_file_dir="",
                separator_numba_cache_dir="",
                separator_timeout_sec=0.0,
            )

            with mock.patch.object(prepare_assets, "convert_audio", side_effect=fake_convert_audio), mock.patch.object(
                prepare_assets,
                "run_audio_separator_batch",
                side_effect=fake_run_audio_separator_batch,
            ):
                prepare_assets.prepare_separator_assets(output_rows, report_rows, args)

        self.assertEqual(batch_calls, [["song_a", "song_b"]])
        self.assertTrue(output_rows[0]["vocals_path"].endswith("song_a/separator/vocals.flac"))
        self.assertTrue(output_rows[1]["no_vocals_path"].endswith("song_b/separator/no_vocals.flac"))
        self.assertEqual(report_rows[0]["separator_status"], "ok")
        self.assertEqual(report_rows[1]["separator_status"], "ok")
        self.assertEqual(report_rows[0]["status"], "ok")
        self.assertEqual(report_rows[1]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
