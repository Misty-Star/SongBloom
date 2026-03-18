import unittest


from training.dropout_utils import apply_condition_dropouts


class _RecordingDropout:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return self.return_value


class DropoutUtilsTest(unittest.TestCase):
    def test_apply_condition_dropouts_passes_full_attribute_list_in_order(self):
        original = ["sample_a", "sample_b"]
        after_cfg = ["cfg_a", "cfg_b"]
        after_att = ["att_a", "att_b"]
        cfg_dropout = _RecordingDropout(after_cfg)
        att_dropout = _RecordingDropout(after_att)

        result = apply_condition_dropouts(cfg_dropout, att_dropout, original)

        self.assertIs(result, after_att)
        self.assertEqual(cfg_dropout.calls, [original])
        self.assertEqual(att_dropout.calls, [after_cfg])


if __name__ == "__main__":
    unittest.main()
