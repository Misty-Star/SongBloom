import unittest


from training.dataset_utils import compute_effective_frame_length


class DatasetUtilsTest(unittest.TestCase):
    def test_effective_frame_length_stays_block_aligned_after_max_duration_clip(self):
        x_len = compute_effective_frame_length(
            sketch_frames=1600,
            latent_frames=1600,
            duration_seconds=999.0,
            block_size=16,
            max_duration=60.0,
        )

        self.assertEqual(x_len, 1488)
        self.assertEqual(x_len % 16, 0)


if __name__ == "__main__":
    unittest.main()
