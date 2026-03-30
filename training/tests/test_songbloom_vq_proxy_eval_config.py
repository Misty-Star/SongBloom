import unittest

import yaml


class SongBloomVQProxyEvalConfigTest(unittest.TestCase):
    def test_proxy_eval_config_keeps_num_pitch_and_uses_longer_smoke(self):
        with open("training/configs/songbloom_vq_proxy_eval.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.assertEqual(cfg["model"]["num_pitch"], 16384)
        self.assertEqual(cfg["training"]["max_steps"], 100)
        self.assertEqual(cfg["data"]["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
