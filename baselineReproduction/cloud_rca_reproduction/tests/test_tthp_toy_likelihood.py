import unittest
import numpy as np

from scripts.tthp_toy_likelihood import (
    intensity,
    log_likelihood,
)


class Tests(unittest.TestCase):

    def test_future_event_with_delay(self):
        x = np.zeros((1, 3, 2))
        x[0, 0, 0] = 1
        x[0, 1, 1] = 1

        mu = np.array([0.2, 0.3])
        alpha = np.array([
            [0.0, 0.5],
            [1.0, 0.0],
        ])

        no_delay = intensity(
            x, mu, alpha, np.log(2), 0
        )

        delayed = intensity(
            x, mu, alpha, np.log(2), 1
        )

        # Future type-1 event influences type 0.
        self.assertAlmostEqual(no_delay[0, 0, 0], 0.2)
        self.assertAlmostEqual(delayed[0, 0, 0], 0.7)

        # Earlier type-0 event influences type 1.
        self.assertAlmostEqual(no_delay[0, 1, 1], 0.55)

        self.assertTrue(
            np.isfinite(log_likelihood(x, delayed))
        )

    def test_likelihood_exposure(self):
        x = np.zeros((1, 2, 1))
        rate = np.full((1, 2, 1), 0.5)

        self.assertAlmostEqual(
            log_likelihood(x, rate, bin_width=0.5),
            -0.5
        )

    def test_device_isolation(self):
        x = np.zeros((2, 2, 2))
        x[0, 0, 1] = 2

        result = intensity(
            x,
            mu=[0.1, 0.1],
            alpha=[[0, 0], [1, 0]],
            decay=1.0,
            delay_bins=1,
        )

        # Device 1 must not receive device 0 effects.
        self.assertAlmostEqual(result[1, 0, 0], 0.1)


if __name__ == "__main__":
    unittest.main()
