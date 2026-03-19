import unittest
from types import SimpleNamespace


from training.preprocess.run_preprocess_pipeline import build_prepare_assets_cmd


class PreprocessPipelineCommandTest(unittest.TestCase):
    def test_build_prepare_assets_cmd_forwards_structure_stage_songformer_args(self):
        args = SimpleNamespace(
            input_jsonl="raw.jsonl",
            prepared_manifest="ready.jsonl",
            assets_dir="assets",
            workspace_dir="workspace",
            skip_existing=True,
            keep_intermediate=False,
            skip_separation=False,
            separator_cmd="audio-separator",
            separator_model="BS-Roformer-Viperx-1297",
            separator_model_flag="auto",
            separator_output_format="FLAC",
            separator_model_file_dir="",
            separator_numba_cache_dir="",
            separator_timeout_sec=10.0,
            skip_whisperx=False,
            whisperx_cmd="whisperx",
            whisperx_model="large-v3",
            whisperx_device="cuda",
            whisperx_compute_type="float16",
            language=None,
            whisperx_timeout_sec=20.0,
            skip_structure=False,
            songformer_root="third_party/SongFormer",
            songformer_python="conda run -n songformer python",
            songformer_gpu_num=2,
            songformer_threads=3,
            songformer_model="SongFormer",
            songformer_checkpoint="SongFormer.safetensors",
            songformer_config="SongFormer.yaml",
            songformer_no_rule_post=True,
            songformer_timeout_sec=30.0,
        )

        cmd = build_prepare_assets_cmd(args)

        self.assertIn("--songformer-python", cmd)
        self.assertIn("conda run -n songformer python", cmd)
        self.assertIn("--songformer-root", cmd)
        self.assertIn("third_party/SongFormer", cmd)
        self.assertIn("--songformer-gpu-num", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--songformer-threads", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--songformer-timeout-sec", cmd)
        self.assertIn("30.0", cmd)
        self.assertIn("--songformer-no-rule-post", cmd)


if __name__ == "__main__":
    unittest.main()
