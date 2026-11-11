import unittest

import numpy as np
import pandas as pd

from scripts.tthp_delay_kernel import event_effects


class Tests(unittest.TestCase):

    def test_delay_kernel(self):
        events = pd.DataFrame({
            "event": [0, 1, 2, 0, 1],
            "timestamp": [10.0, 10.0, 12.0, 15.0, 11.0],
            "node": [0, 0, 0, 0, 1],
        })

        effects = event_effects(
            events,
            delay=2,
            delta=np.log(2),
            n_types=3
        )

        # At t=10: simultaneous type 1 contributes 1.
        # Future type 2 at t=12 contributes 0.25.
        np.testing.assert_allclose(
            effects[0],
            [0.0, 1.0, 0.25]
        )

        # At t=12: both earlier t=10 events contribute.
        np.testing.assert_allclose(
            effects[2],
            [0.25, 0.25, 0.0]
        )

        # Device 1 has no other events.
        np.testing.assert_allclose(
            effects[4],
            [0.0, 0.0, 0.0]
        )

        # Zero collection delay still includes same-time events.
        zero_delay = event_effects(
            events, delay=0,
            delta=np.log(2), n_types=3
        )

        self.assertAlmostEqual(zero_delay[0, 2], 0.0)
        self.assertAlmostEqual(zero_delay[0, 1], 1.0)

    def test_invalid_parameters(self):
        events = pd.DataFrame({
            "event": [0],
            "timestamp": [1],
            "node": [0]
        })

        with self.assertRaises(ValueError):
            event_effects(events, -1, 0.1, 1)


if __name__ == "__main__":
    unittest.main()
