import tempfile
import unittest
from pathlib import Path
import numpy as np

from topokpi.data import generate_demo, load_npz, prepare, save_npz
from topokpi.metrics import average_precision, binary_metrics, rca_metrics, roc_auc
from topokpi.rca import localize, multi_hop_influence, row_normalize


class DataTests(unittest.TestCase):
    def test_roundtrip_and_windows(self):
        data = generate_demo(nodes=8, steps=360, seed=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.npz"
            save_npz(path, data)
            loaded = load_npz(path)
        prepared = prepare(loaded, window=20)
        self.assertEqual(prepared.train.x.shape[1:], (20, 8, 8))
        self.assertEqual(prepared.test.y_next.shape[1:], (8, 8))
        self.assertTrue(np.allclose(prepared.train.x.mean(axis=(0, 1, 2)), 0, atol=.15))

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([.1, .2, .8, .9])
        self.assertEqual(binary_metrics(y, s, .5)["f1"], 1)
        self.assertEqual(roc_auc(y, s), 1)
        self.assertEqual(average_precision(y, s), 1)


class RCATests(unittest.TestCase):
    def test_directed_source_is_preferred(self):
        w = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
        score = np.array([.8, .9, .7])
        result = localize(score, w, node_threshold=.1, candidate_cap=3, depth=2, decay=.7, gamma=.4)
        self.assertEqual(result.ranking[0], 0)
        self.assertAlmostEqual(result.contributions.sum() if result.contributions is not None else 0, 0)
        self.assertTrue(all(i == 0 for i, _, _ in result.explanation_edges))
        self.assertEqual(sorted(result.ranking), [0, 1, 2])

    def test_single_root_metrics_use_complete_rank(self):
        metrics = rca_metrics([[2, 0, 1], [1, 2, 0]], [0, [0]])
        self.assertEqual(metrics["a@1"], 0.0)
        self.assertEqual(metrics["a@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1 / 2 + 1 / 3) / 2)
        self.assertEqual(metrics["mean_rank"], 2.5)

    def test_multiple_roots_are_rejected(self):
        with self.assertRaises(ValueError):
            rca_metrics([[0, 1, 2]], [[0, 1]])

    def test_isolated_rows_remain_zero(self):
        w = np.zeros((3, 3))
        self.assertTrue(np.all(row_normalize(w) == 0))
        self.assertTrue(np.all(multi_hop_influence(w) == 0))


if __name__ == "__main__":
    unittest.main()
