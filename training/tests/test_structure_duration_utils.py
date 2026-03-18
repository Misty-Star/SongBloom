import unittest


from SongBloom.models.musicgen.conditioners.structure_duration_utils import (
    maybe_interpolate_structure_duration,
)


class StructureDurationUtilsTest(unittest.TestCase):
    def test_skip_interpolation_when_current_sample_structure_duration_is_none(self):
        calls = []

        def interpolate_fn(tokens, embeds, sample_structure_dur):
            calls.append((tokens, embeds, sample_structure_dur))
            return "interpolated"

        result = maybe_interpolate_structure_duration(
            interpolate_fn=interpolate_fn,
            tokens="tokens",
            embeds="embeds",
            structure_dur=[["valid"], None],
            batch_index=1,
        )

        self.assertEqual(result, "embeds")
        self.assertEqual(calls, [])

    def test_use_interpolation_when_current_sample_structure_duration_exists(self):
        calls = []

        def interpolate_fn(tokens, embeds, sample_structure_dur):
            calls.append((tokens, embeds, sample_structure_dur))
            return "interpolated"

        result = maybe_interpolate_structure_duration(
            interpolate_fn=interpolate_fn,
            tokens="tokens",
            embeds="embeds",
            structure_dur=[["valid"], None],
            batch_index=0,
        )

        self.assertEqual(result, "interpolated")
        self.assertEqual(calls, [("tokens", "embeds", ["valid"])])


if __name__ == "__main__":
    unittest.main()
