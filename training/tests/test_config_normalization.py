import unittest

from omegaconf import OmegaConf

from training.config_normalization import ensure_songbloom_config_defaults


class ConfigNormalizationTest(unittest.TestCase):
    def test_fill_missing_lyrics_max_duration_from_top_level_max_dur(self):
        cfg = OmegaConf.create(
            {
                "max_dur": 240,
                "model": {
                    "condition_provider_cfg": {
                        "lyrics": {
                            "type": "phoneme_tokenizer",
                            "output_dim": 1536,
                        }
                    }
                },
            }
        )

        result = ensure_songbloom_config_defaults(cfg)

        self.assertEqual(result.model.condition_provider_cfg.lyrics.max_duration, 240)

    def test_keep_existing_lyrics_max_duration(self):
        cfg = OmegaConf.create(
            {
                "max_dur": 240,
                "model": {
                    "condition_provider_cfg": {
                        "lyrics": {
                            "type": "phoneme_tokenizer",
                            "output_dim": 1536,
                            "max_duration": 150,
                        }
                    }
                },
            }
        )

        result = ensure_songbloom_config_defaults(cfg)

        self.assertEqual(result.model.condition_provider_cfg.lyrics.max_duration, 150)


if __name__ == "__main__":
    unittest.main()
