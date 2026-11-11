"""Paper section 4.3 directed node-alarm sampling checks (synthetic topology)."""
import unittest
import numpy as np
import pandas as pd

from scripts.sample_acg_neighbors import (
    dominant_alarm_types, preview_event_edges, sample_case_edges,
    validate_threshold,
)


class ACGNeighborSamplingTests(unittest.TestCase):
    def test_threshold_preview_retains_boundary(self):
        w = np.array([[0, 0.7, 0.2], [0, 0, 0.95], [0, 0, 0]])
        edges = preview_event_edges(w, 0.7)
        self.assertEqual([(i, j) for i, j, _ in edges], [(0, 1), (1, 2)])

    def test_reverse_alarm_weight_for_directed_topology(self):
        # topology 0->1; alarm(node0)=1, alarm(node1)=0.
        # Correct: ACG[event(node1)=0, event(node0)=1] = 0.9.
        # Wrong forward lookup: ACG[1,0] = 0.1.
        topo = np.array([[0, 1], [0, 0]])
        w = np.array([[0, 0.9], [0.1, 0]])
        self.assertEqual(sample_case_edges(topo, [1, 0], w, 0.7),
                         [(0, 1, 0, 1, 0.9)])
        self.assertEqual(sample_case_edges(topo, [1, 0], w, 0.95), [])

    def test_unalarmed_node_excluded_and_direction_not_reversed(self):
        topo = np.array([[0, 1], [1, 0]])
        w = np.array([[0, 0.8], [0, 0]])
        self.assertEqual(sample_case_edges(topo, [-1, 0], w, 0.7), [])
        self.assertEqual(sample_case_edges(topo, [1, 0], w, 0.7), [(0, 1, 0, 1, 0.8)])

    def test_dominant_type_and_tie_determinism(self):
        alarms = pd.DataFrame({"node_id": [1, 1, 1, 1, 0],
                               "event_type": [2, 1, 2, 1, 0]})
        np.testing.assert_array_equal(dominant_alarm_types(alarms, 3, 4), [0, 1, -1])

    def test_rejects_bad_inputs(self):
        for x in (0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                validate_threshold(x)
        with self.assertRaisesRegex(ValueError, "binary"):
            sample_case_edges(np.array([[0, 0.5], [0, 0]]), [0, 1],
                              np.zeros((2, 2)), 0.7)
        with self.assertRaisesRegex(ValueError, "Invalid case alarm"):
            dominant_alarm_types(pd.DataFrame({"node_id": [2], "event_type": [0]}), 2, 2)


if __name__ == "__main__":
    unittest.main()
