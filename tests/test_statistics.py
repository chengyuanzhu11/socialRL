import unittest

from socialrl.stats import holm_adjust, paired_sign_flip_test
from socialrl.metrics import summarize_training


class StatisticsTests(unittest.TestCase):
    def test_paired_sign_flip_uses_matched_seed_differences(self):
        pvalue = paired_sign_flip_test([2.0, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(pvalue, 0.5)

    def test_holm_adjustment_is_monotonic_in_sorted_order(self):
        adjusted = holm_adjust([0.01, 0.02, 0.5])
        self.assertEqual(adjusted, [0.03, 0.04, 0.5])

    def test_training_summary_uses_unsmoothed_values(self):
        summary = summarize_training(
            [{"bout_success_rate": 0.0}, {"bout_success_rate": 0.5}, {"bout_success_rate": 1.0}],
            threshold=0.75,
            window=2,
        )
        self.assertAlmostEqual(summary["success_auc"], 1.0)
        self.assertEqual(summary["time_to_threshold"], 3)


if __name__ == "__main__":
    unittest.main()
