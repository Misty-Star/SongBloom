import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


class MuQFeatureCacheTest(unittest.TestCase):
    def test_load_audio_with_fallback_uses_soundfile_when_torchaudio_fails(self):
        from training.preprocess.muq_feature_cache import load_audio_with_fallback

        fake_torchaudio = types.SimpleNamespace(
            load=mock.Mock(side_effect=RuntimeError("torchcodec failed")),
        )
        fake_soundfile = types.SimpleNamespace(
            read=mock.Mock(return_value=(np.ones((48_000, 1), dtype=np.float32), 48_000)),
        )

        with mock.patch.dict(
            'sys.modules',
            {
                'torchaudio': fake_torchaudio,
                'soundfile': fake_soundfile,
            },
        ):
            wav, sr = load_audio_with_fallback('/tmp/fallback.flac')

        self.assertEqual(sr, 48_000)
        self.assertEqual(tuple(wav.shape), (1, 48_000))
        fake_torchaudio.load.assert_called_once_with('/tmp/fallback.flac')
        fake_soundfile.read.assert_called_once_with('/tmp/fallback.flac', always_2d=True, dtype='float32')

    def test_get_or_compute_muq_embedding_reuses_saved_cache(self):
        from training.preprocess.muq_feature_cache import get_or_compute_muq_embedding

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, 'song_a.flac')
            with open(audio_path, 'wb') as handle:
                handle.write(b'fake-audio')

            fake_torchaudio = types.SimpleNamespace(
                load=lambda _path: (torch.ones(1, 48_000), 48_000),
                transforms=types.SimpleNamespace(Resample=lambda _src, _dst: lambda wav: wav),
            )
            extract_mock = mock.Mock(return_value=torch.arange(6, dtype=torch.float32).view(1, 3, 2))
            fake_extract_sketch = types.SimpleNamespace(extract_muq_embeddings=extract_mock)

            with mock.patch.dict(
                'sys.modules',
                {
                    'torchaudio': fake_torchaudio,
                    'training.preprocess.extract_sketch': fake_extract_sketch,
                },
            ):
                first = get_or_compute_muq_embedding(
                    audio_path=audio_path,
                    muq_model=object(),
                    muq_model_name='fake-muq',
                    sample_rate=48_000,
                    target_fps=3,
                    cache_dir=os.path.join(temp_dir, 'muq_cache'),
                )
                second = get_or_compute_muq_embedding(
                    audio_path=audio_path,
                    muq_model=object(),
                    muq_model_name='fake-muq',
                    sample_rate=48_000,
                    target_fps=3,
                    cache_dir=os.path.join(temp_dir, 'muq_cache'),
                )

        self.assertEqual(extract_mock.call_count, 1)
        self.assertTrue(torch.equal(first, second))

    def test_get_or_compute_muq_embedding_invalidates_when_audio_changes(self):
        from training.preprocess.muq_feature_cache import get_or_compute_muq_embedding

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, 'song_a.flac')
            with open(audio_path, 'wb') as handle:
                handle.write(b'fake-audio')

            fake_torchaudio = types.SimpleNamespace(
                load=lambda _path: (torch.ones(1, 48_000), 48_000),
                transforms=types.SimpleNamespace(Resample=lambda _src, _dst: lambda wav: wav),
            )
            outputs = [
                torch.ones(1, 3, 2, dtype=torch.float32),
                torch.full((1, 3, 2), 2.0, dtype=torch.float32),
            ]
            extract_mock = mock.Mock(side_effect=outputs)
            fake_extract_sketch = types.SimpleNamespace(extract_muq_embeddings=extract_mock)

            cache_dir = os.path.join(temp_dir, 'muq_cache')
            with mock.patch.dict(
                'sys.modules',
                {
                    'torchaudio': fake_torchaudio,
                    'training.preprocess.extract_sketch': fake_extract_sketch,
                },
            ):
                first = get_or_compute_muq_embedding(
                    audio_path=audio_path,
                    muq_model=object(),
                    muq_model_name='fake-muq',
                    sample_rate=48_000,
                    target_fps=3,
                    cache_dir=cache_dir,
                )
                with open(audio_path, 'ab') as handle:
                    handle.write(b'-updated')
                stat = os.stat(audio_path)
                os.utime(audio_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                second = get_or_compute_muq_embedding(
                    audio_path=audio_path,
                    muq_model=object(),
                    muq_model_name='fake-muq',
                    sample_rate=48_000,
                    target_fps=3,
                    cache_dir=cache_dir,
                )

        self.assertEqual(extract_mock.call_count, 2)
        self.assertFalse(torch.equal(first, second))


if __name__ == '__main__':
    unittest.main()
