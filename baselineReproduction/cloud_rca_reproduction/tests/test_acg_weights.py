"""Paper Eq. (5) tests that do not use any ground-truth Wireless DAG."""
import unittest

import numpy as np
import pandas as pd

from scripts.build_acg_weights import conditional_acg_weights, presence_in_time_slices


class ACGWeightsTests(unittest.TestCase):
    def test_directed_conditional_probabilities_and_edge_mask(self):
        # Four slices; event 0 in 3 slices, event 2 in 2 slices;
        # both co-occur once. Hence P(2|0)=1/3, P(0|2)=1/2.
        p = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
        graph = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]])
        weights, support = conditional_acg_weights(p, graph)
        np.testing.assert_array_equal(support, [3, 3, 2])
        self.assertAlmostEqual(weights[0, 2], 1 / 3)
        self.assertAlmostEqual(weights[2, 0], 1 / 2)
        self.assertAlmostEqual(weights[1, 0], 2 / 3)
        self.assertEqual(weights[0, 1], 0)  # not a learned graph edge
        self.assertTrue(np.all(np.diag(weights) == 0))

    def test_zero_source_support_and_rejects_self_loop(self):
        p = np.array([[1, 0], [1, 0]])
        graph = np.array([[0, 1], [1, 0]])
        w, support = conditional_acg_weights(p, graph)
        self.assertEqual(w[1, 0], 0)
        self.assertEqual(support[1], 0)
        graph[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "self-loops"):
            conditional_acg_weights(p, graph)

    def test_strictly_excludes_future_events_when_weighting(self):
        data = pd.DataFrame({
            "event": [0, 1, 0, 1, 1, 0],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "node": [0, 0, 0, 0, 0, 0],
        })
        p, meta = presence_in_time_slices(data, n_types=2, n_bins=3, train_fraction=0.6)
        self.assertEqual(meta["train_event_count"], 3)
        self.assertEqual(meta["nontrain_event_count"], 3)
        self.assertEqual(meta["train_time_cutoff_exclusive"], 3)
        p2, _ = presence_in_time_slices(
            pd.concat([data, pd.DataFrame({"event": [1], "timestamp": [4.5], "node": [0]})], ignore_index=True),
            n_types=2, n_bins=3, train_fraction=0.6,
        )
        np.testing.assert_array_equal(p, p2)

    def test_rejects_nonbinary_or_incompatible_input(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            conditional_acg_weights(np.ones((3, 2)), np.array([[0, 0.7], [0, 0]]))
        with self.assertRaisesRegex(ValueError, "square"):
            conditional_acg_weights(np.ones((3, 2)), np.ones((3, 4)))


if __name__ == "__main__":
    unittest.main()
