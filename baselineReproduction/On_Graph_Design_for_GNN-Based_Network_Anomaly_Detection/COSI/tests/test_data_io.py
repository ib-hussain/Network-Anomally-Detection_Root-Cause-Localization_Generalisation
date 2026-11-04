import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_adjacency, load_csv_pair


class InputTests(unittest.TestCase):
    def test_order_labels_and_directed_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({'timestamp': [0, 1, 2], 'a': [1, 2, 3], 'b': [3, 2, 1]}).to_csv(root/'train.csv', index=False)
            pd.DataFrame({'b': [3, 4], 'attack': [0, 1], 'a': [1, 2]}).to_csv(root/'test.csv', index=False)
            np.savetxt(root/'adj.csv', np.array([[0, 1], [0, 0]]), delimiter=',')
            train, test, labels = load_csv_pair(root/'train.csv', root/'test.csv')
            self.assertEqual(list(train), ['a', 'b'])
            self.assertEqual(list(test), ['a', 'b'])
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(load_adjacency(root/'adj.csv', list(train)).tolist(), [[0], [1]])

    def test_rejects_test_missing_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({'a': [1], 'b': [2]}).to_csv(root/'train.csv', index=False)
            pd.DataFrame({'a': [1], 'b': [2]}).to_csv(root/'test.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'attack'):
                load_csv_pair(root/'train.csv', root/'test.csv')

    def test_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp)/'adj.csv'
            np.savetxt(f, np.eye(3), delimiter=',')
            with self.assertRaisesRegex(ValueError, 'Adjacency shape'):
                load_adjacency(f, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
