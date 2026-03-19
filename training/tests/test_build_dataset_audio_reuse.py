import importlib
import os
import sys
import tempfile
import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

import torch


def import_build_dataset_with_mocked_torchaudio():
    sys.modules.pop("training.preprocess.build_dataset", None)
    fake_torchaudio = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
        module = importlib.import_module("training.preprocess.build_dataset")
    return module


class BuildDatasetAudioReuseTest(unittest.TestCase):
    def test_save_prompt_audio_uses_preloaded_waveform_without_loading_from_disk(self):
        build_dataset = import_build_dataset_with_mocked_torchaudio()
        build_dataset.torchaudio.load.side_effect = AssertionError("should not load audio again")
        build_dataset.torchaudio.save = mock.Mock()

        wav = torch.ones(2, 48000 * 12)
        build_dataset.save_prompt_audio(
            audio_path="/tmp/full_audio.flac",
            output_path="/tmp/prompt_wav.flac",
            sample_rate=48000,
            prompt_len=10.0,
            structure_segments=[{"label": "[verse]", "start": 0.0, "end": 12.0}],
            wav=wav,
            wav_sample_rate=48000,
        )

        self.assertFalse(build_dataset.torchaudio.load.called)
        self.assertTrue(build_dataset.torchaudio.save.called)

    def test_extract_latent_uses_preloaded_waveform_without_loading_from_disk(self):
        build_dataset = import_build_dataset_with_mocked_torchaudio()
        build_dataset.torchaudio.load.side_effect = AssertionError("should not load audio again")

        args = SimpleNamespace(
            vae_cfg="vae.json",
            vae_ckpt="vae.ckpt",
            sample_rate=48000,
            device="cpu",
            muq_model="muq",
            vq_ckpt="vq.pt",
            target_fps=25,
        )
        bundle = build_dataset.FeatureExtractorBundle(args)
        bundle._vae = mock.Mock()
        bundle._vae.encode.return_value = torch.ones(1, 4, 6)

        wav = torch.ones(2, 3200)
        result = bundle.extract_latent("/tmp/full_audio.flac", wav=wav, sample_rate=48000)

        self.assertEqual(tuple(result.shape), (4, 6))
        self.assertFalse(build_dataset.torchaudio.load.called)

    def test_extract_sketch_uses_preloaded_waveform_without_loading_from_disk(self):
        build_dataset = import_build_dataset_with_mocked_torchaudio()
        build_dataset.torchaudio.load.side_effect = AssertionError("should not load audio again")

        args = SimpleNamespace(
            vae_cfg="vae.json",
            vae_ckpt="vae.ckpt",
            sample_rate=48000,
            device="cpu",
            muq_model="muq",
            vq_ckpt="vq.pt",
            target_fps=25,
        )
        bundle = build_dataset.FeatureExtractorBundle(args)
        bundle._muq = mock.Mock()
        bundle._vq = mock.Mock()
        bundle._vq.encode.return_value = torch.arange(25, dtype=torch.long).unsqueeze(0)

        wav = torch.ones(2, 48000)
        fake_extract_sketch = ModuleType("training.preprocess.extract_sketch")
        fake_extract_sketch.extract_muq_embeddings = mock.Mock(return_value=torch.ones(1, 10, 8))
        with mock.patch.dict(sys.modules, {"training.preprocess.extract_sketch": fake_extract_sketch}):
            result = bundle.extract_sketch("/tmp/full_audio.flac", wav=wav, sample_rate=48000)

        self.assertEqual(tuple(result.shape), (25,))
        self.assertFalse(build_dataset.torchaudio.load.called)

    def test_process_sample_passes_standardized_waveform_to_prompt_and_feature_extractors(self):
        build_dataset = import_build_dataset_with_mocked_torchaudio()

        prepared_wav = torch.ones(2, 128)
        prepared_audio = build_dataset.PreparedAudio(
            path="/tmp/work/full_audio.flac",
            wav=prepared_wav,
            sample_rate=48000,
            duration=120.0,
        )
        features = mock.Mock()
        features.extract_latent.return_value = torch.zeros(64, 3200)
        features.extract_sketch.return_value = torch.zeros(3200, dtype=torch.long)

        args = SimpleNamespace(
            output_dir="",
            workspace_dir="",
            skip_existing=False,
            sample_rate=48000,
            min_duration=80.0,
            max_duration=240.0,
            prompt_len=10.0,
            lyrics_similarity_threshold=0.25,
            whisperx_cmd="whisperx",
            language=None,
            whisperx_model="large-v3",
            whisperx_device="cuda",
            whisperx_compute_type="float16",
            whisperx_timeout_sec=0.0,
            min_whisperx_score=0.15,
            min_structure_duration=1.0,
            disable_vocal_refine=False,
            vocal_threshold=0.3,
            non_vocal_threshold=0.05,
            songformer_root="third_party/SongFormer",
            songformer_python="python",
            songformer_gpu_num=1,
            songformer_threads=1,
            songformer_model="SongFormer",
            songformer_checkpoint="SongFormer.safetensors",
            songformer_config="SongFormer.yaml",
            songformer_no_rule_post=False,
            songformer_timeout_sec=0.0,
            min_structure_segments=2,
            lyric_processor="phoneme",
            keep_intermediate=True,
            target_fps=25,
            block_size=16,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = os.path.join(temp_dir, "output")
            args.workspace_dir = os.path.join(temp_dir, "workspace")
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.workspace_dir, exist_ok=True)

            item = {
                "id": "song_a",
                "audio_path": "/tmp/raw_audio.flac",
                "vocals_path": "/tmp/vocals.flac",
                "no_vocals_path": "/tmp/no_vocals.flac",
            }

            with mock.patch.object(
                build_dataset,
                "convert_audio",
                return_value=prepared_audio,
            ), mock.patch.object(
                build_dataset,
                "resolve_stems",
                return_value=("/tmp/vocals.flac", "/tmp/no_vocals.flac"),
            ), mock.patch.object(
                build_dataset,
                "process_alignment_item",
                return_value={"cleaned_lyrics": "hello world", "whisperx_quality": {"score": 0.9}},
            ), mock.patch.object(
                build_dataset,
                "process_structure_item",
                return_value={
                    "segments": [
                        {"label": "[verse]", "start": 0.0, "end": 60.0},
                        {"label": "[chorus]", "start": 60.0, "end": 120.0},
                    ]
                },
            ), mock.patch.object(
                build_dataset,
                "build_lyrics_and_structure",
                return_value=(
                    "[verse] hello , [chorus] world",
                    [["[verse]", 0.0, 60.0], ["[chorus]", 60.0, 120.0]],
                ),
            ), mock.patch.object(
                build_dataset,
                "write_processed_lyrics",
                return_value="HH AH L OW",
            ), mock.patch.object(
                build_dataset,
                "save_prompt_audio",
            ) as save_prompt_audio_mock, mock.patch.object(
                build_dataset.torch,
                "save",
            ), mock.patch.object(
                build_dataset,
                "save_json",
            ):
                build_dataset.process_sample(
                    item=item,
                    args=args,
                    features=features,
                    sample_index=1,
                    total_items=1,
                )

        save_prompt_audio_mock.assert_called_once()
        self.assertEqual(features.extract_latent.call_count, 1)
        self.assertEqual(features.extract_sketch.call_count, 1)
        self.assertIs(save_prompt_audio_mock.call_args.kwargs["wav"], prepared_wav)
        self.assertEqual(save_prompt_audio_mock.call_args.kwargs["wav_sample_rate"], 48000)
        self.assertIs(features.extract_latent.call_args.kwargs["wav"], prepared_wav)
        self.assertEqual(features.extract_latent.call_args.kwargs["sample_rate"], 48000)
        self.assertIs(features.extract_sketch.call_args.kwargs["wav"], prepared_wav)
        self.assertEqual(features.extract_sketch.call_args.kwargs["sample_rate"], 48000)


if __name__ == "__main__":
    unittest.main()
