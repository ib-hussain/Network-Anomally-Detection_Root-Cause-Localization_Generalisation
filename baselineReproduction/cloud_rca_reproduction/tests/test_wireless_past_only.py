"""Independent checks of the chronological Wireless fitting adapter."""
import unittest
import numpy as np
import pandas as pd

from scripts.fit_wireless_past_only import make_counts, fit
from scripts.fit_past_only_thp import past_effects, build_fixture


class TestWirelessPastOnly(unittest.TestCase):
    def test_binning_conserves_counts_and_order(self):
        events = pd.DataFrame({
            "timestamp": list(range(16)),
            "event": [0, 1] * 8,
            "node": [0, 0, 1, 1] * 4,
        })
        x, meta = make_counts(events, n_bins=16)
        self.assertEqual(x.shape, (2, 16, 2))
        self.assertEqual(int(x.sum()), len(events))
        self.assertEqual(x[0, 0, 0], 1)
        self.assertEqual(x[0, 1, 1], 1)
        self.assertEqual(meta["bins"], 16)

    def test_future_event_cannot_influence_past(self):
        x = np.zeros((1, 16, 2), dtype=float)
        x[0, 12, 1] = 10
        history = past_effects(x)
        np.testing.assert_array_equal(history[:, :13, :], 0)
        self.assertGreater(history[0, 13, 1], 0)

    def test_synthetic_parameter_fit(self):
        x = build_fixture(steps=100, devices=2)
        result, metrics = fit(x, maxiter=150)
        self.assertTrue(result.success, result.message)
        self.assertLess(metrics["fitted_train_nll_per_cell"],
                        metrics["baseline_train_nll_per_cell"])
        self.assertEqual(metrics["train_bins"], 75)
        self.assertEqual(metrics["validation_bins"], 25)


if __name__ == "__main__":
    unittest.main()
