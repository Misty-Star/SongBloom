import json
import os
import sys
import tempfile
import unittest
import importlib
from unittest import mock


from training.preprocess.extract_structure import prepare_structure_assets
from training.preprocess.structure_assets import merge_structure_preparation_results


class StructureAssetPreparationTest(unittest.TestCase):
    def test_prepare_structure_assets_backfills_missing_structure_json_with_unique_batch_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_root = os.path.join(temp_dir, "audio")
            assets_dir = os.path.join(temp_dir, "assets")
            workspace_dir = os.path.join(temp_dir, "workspace")
            os.makedirs(audio_root, exist_ok=True)

            song_a_audio_dir = os.path.join(audio_root, "song_a")
            song_c_audio_dir = os.path.join(audio_root, "song_c")
            os.makedirs(song_a_audio_dir, exist_ok=True)
            os.makedirs(song_c_audio_dir, exist_ok=True)

            song_a_audio = os.path.join(song_a_audio_dir, "full_audio.flac")
            song_c_audio = os.path.join(song_c_audio_dir, "full_audio.flac")
            with open(song_a_audio, "w", encoding="utf-8") as handle:
                handle.write("song-a")
            with open(song_c_audio, "w", encoding="utf-8") as handle:
                handle.write("song-c")

            existing_dir = os.path.join(assets_dir, "song_b", "structure")
            os.makedirs(existing_dir, exist_ok=True)
            existing_structure_json = os.path.join(existing_dir, "song_b.json")
            with open(existing_structure_json, "w", encoding="utf-8") as handle:
                json.dump([{"label": "verse", "start": 0.0, "end": 10.0}], handle)

            items = [
                {"id": "song_a", "audio_path": song_a_audio},
                {"id": "song_b", "audio_path": song_a_audio, "structure_json": existing_structure_json},
                {"id": "song_c", "audio_path": song_c_audio},
            ]

            def fake_run_songformer_batch(
                audio_inputs,
                output_dir,
                songformer_root,
                python_exec,
                gpu_num,
                num_thread_per_gpu,
                model,
                checkpoint,
                config_path,
                no_rule_post_processing,
                timeout_sec=None,
            ):
                self.assertEqual(set(audio_inputs.keys()), {"song_a", "song_c"})
                self.assertEqual(os.path.basename(audio_inputs["song_a"]), "song_a.flac")
                self.assertEqual(os.path.basename(audio_inputs["song_c"]), "song_c.flac")

                os.makedirs(output_dir, exist_ok=True)
                outputs = {}
                for sample_id in audio_inputs:
                    path = os.path.join(output_dir, sample_id + ".json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump([{"label": "chorus", "start": 1.0, "end": 2.0}], handle)
                    outputs[sample_id] = path
                return outputs

            with mock.patch(
                "training.preprocess.extract_structure.run_songformer_batch",
                side_effect=fake_run_songformer_batch,
            ):
                updated_items, report_rows = prepare_structure_assets(
                    items=items,
                    assets_dir=assets_dir,
                    workspace_dir=workspace_dir,
                    songformer_root="third_party/SongFormer",
                    python_exec="python",
                    gpu_num=1,
                    num_thread_per_gpu=1,
                    model="SongFormer",
                    checkpoint="SongFormer.safetensors",
                    config_path="SongFormer.yaml",
                    no_rule_post_processing=False,
                    skip_existing=True,
                    timeout_sec=None,
                )

            self.assertEqual(updated_items[1]["structure_json"], existing_structure_json)
            self.assertTrue(updated_items[0]["structure_json"].endswith("song_a/structure/song_a.json"))
            self.assertTrue(updated_items[2]["structure_json"].endswith("song_c/structure/song_c.json"))
            self.assertTrue(os.path.exists(updated_items[0]["structure_json"]))
            self.assertTrue(os.path.exists(updated_items[2]["structure_json"]))

            report_by_id = {row["id"]: row for row in report_rows}
            self.assertEqual(report_by_id["song_a"]["structure_status"], "ok")
            self.assertEqual(report_by_id["song_b"]["structure_status"], "provided")
            self.assertEqual(report_by_id["song_c"]["structure_status"], "ok")

    def test_merge_structure_preparation_results_backfills_manifest_and_report_rows(self):
        output_rows = [
            {"id": "song_a", "audio_path": "/tmp/song_a.wav"},
            {"id": "song_b", "audio_path": "/tmp/song_b.wav", "vocals_path": "/tmp/song_b_vocals.wav"},
        ]
        report_rows = [
            {"id": "song_a", "status": "ok", "separator_status": "provided", "whisperx_status": "provided"},
            {
                "id": "song_b",
                "status": "partial",
                "separator_status": "error",
                "whisperx_status": "provided",
                "errors": [{"stage": "separator", "error": "separator failed"}],
            },
        ]
        structure_items = [
            {"id": "song_a", "audio_path": "/tmp/song_a.wav", "structure_json": "/tmp/song_a_structure.json"},
            {"id": "song_b", "audio_path": "/tmp/song_b.wav"},
        ]
        structure_reports = [
            {"id": "song_a", "structure_status": "ok", "structure_json": "/tmp/song_a_structure.json"},
            {"id": "song_b", "structure_status": "error", "error": "SongFormer failed"},
        ]

        merge_structure_preparation_results(
            output_rows=output_rows,
            report_rows=report_rows,
            structure_items=structure_items,
            structure_reports=structure_reports,
        )

        self.assertEqual(output_rows[0]["structure_json"], "/tmp/song_a_structure.json")
        self.assertNotIn("structure_json", output_rows[1])
        self.assertEqual(report_rows[0]["status"], "ok")
        self.assertEqual(report_rows[0]["structure_status"], "ok")
        self.assertEqual(report_rows[0]["structure_json"], "/tmp/song_a_structure.json")
        self.assertEqual(report_rows[1]["status"], "partial")
        self.assertEqual(report_rows[1]["structure_status"], "error")
        self.assertEqual(report_rows[1]["errors"][-1]["stage"], "structure")
        self.assertEqual(report_rows[1]["errors"][-1]["error"], "SongFormer failed")

    def test_prepare_assets_main_writes_structure_json_from_batch_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = os.path.join(temp_dir, "input.jsonl")
            output_manifest = os.path.join(temp_dir, "ready_manifest.jsonl")
            assets_dir = os.path.join(temp_dir, "assets")
            structure_json = os.path.join(temp_dir, "assets", "song_a", "structure", "song_a.json")
            with open(input_jsonl, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "song_a", "audio_path": "/tmp/song_a.wav"}) + "\n")

            sys.modules.pop("training.preprocess.prepare_assets", None)
            with mock.patch.dict(sys.modules, {"torchaudio": mock.MagicMock()}):
                prepare_assets = importlib.import_module("training.preprocess.prepare_assets")

            def fake_prepare_separator_assets(output_rows, report_rows, args):
                output_rows[0]["vocals_path"] = "/tmp/song_a_vocals.flac"
                output_rows[0]["no_vocals_path"] = "/tmp/song_a_no_vocals.flac"
                report_rows[0]["separator_status"] = "ok"
                report_rows[0]["status"] = "ok"

            def fake_prepare_whisperx_assets(output_rows, report_rows, args):
                output_rows[0]["whisperx_json"] = "/tmp/song_a_whisperx.json"
                report_rows[0]["whisperx_status"] = "ok"
                report_rows[0]["status"] = "ok"

            def fake_prepare_structure_assets(**kwargs):
                return (
                    [
                        {
                            "id": "song_a",
                            "audio_path": "/tmp/song_a.wav",
                            "structure_json": structure_json,
                        }
                    ],
                    [
                        {
                            "id": "song_a",
                            "structure_status": "ok",
                            "structure_json": structure_json,
                        }
                    ],
                )

            with mock.patch.object(prepare_assets, "collect_preflight_issues", return_value=[]), mock.patch.object(
                prepare_assets,
                "prepare_separator_assets",
                side_effect=fake_prepare_separator_assets,
            ), mock.patch.object(
                prepare_assets,
                "prepare_whisperx_assets",
                side_effect=fake_prepare_whisperx_assets,
            ), mock.patch.object(
                prepare_assets,
                "prepare_structure_assets",
                side_effect=fake_prepare_structure_assets,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "prepare_assets",
                    "--input-jsonl",
                    input_jsonl,
                    "--output-manifest",
                    output_manifest,
                    "--assets-dir",
                    assets_dir,
                ],
            ):
                prepare_assets.main()

            with open(output_manifest, "r", encoding="utf-8") as handle:
                output_rows = [json.loads(line) for line in handle if line.strip()]
            report_path = os.path.join(temp_dir, "prepare_assets_report.jsonl")
            with open(report_path, "r", encoding="utf-8") as handle:
                report_rows = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(output_rows[0]["structure_json"], structure_json)
            self.assertEqual(report_rows[0]["structure_status"], "ok")
            self.assertEqual(report_rows[0]["structure_json"], structure_json)
            self.assertEqual(report_rows[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
