import unittest


from training.trainer_config import (
    get_trainer_root_dir,
    get_validation_trainer_kwargs,
)


class TrainerConfigTest(unittest.TestCase):
    def test_trainer_root_dir_uses_training_runs(self):
        self.assertEqual(get_trainer_root_dir(), "training/runs")

    def test_disable_validation_when_val_dir_empty(self):
        kwargs = get_validation_trainer_kwargs(val_dir="", val_check_interval=5000)

        self.assertEqual(kwargs["limit_val_batches"], 0)
        self.assertEqual(kwargs["num_sanity_val_steps"], 0)
        self.assertNotIn("val_check_interval", kwargs)

    def test_keep_validation_when_val_dir_present(self):
        kwargs = get_validation_trainer_kwargs(
            val_dir="/tmp/val_data",
            val_check_interval=5000,
        )

        self.assertEqual(kwargs["val_check_interval"], 5000)
        self.assertNotIn("limit_val_batches", kwargs)
        self.assertNotIn("num_sanity_val_steps", kwargs)


if __name__ == "__main__":
    unittest.main()
