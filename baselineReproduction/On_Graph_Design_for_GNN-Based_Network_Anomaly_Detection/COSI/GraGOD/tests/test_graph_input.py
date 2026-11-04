import tempfile
import unittest
from pathlib import Path
from datasets.graph import get_edge_index
import numpy as np
import torch


class GraphInputTests(unittest.TestCase):
    def test_missing_explicit_graph_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            get_edge_index(torch.zeros(10, 3), 'cpu', 'THIS_FILE_DOES_NOT_EXIST.csv')

    def test_wrong_number_of_features_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'graph.csv'
            np.savetxt(p, np.ones((4, 4)), delimiter=',')
            with self.assertRaisesRegex(ValueError, 'features'):
                get_edge_index(torch.zeros(10, 3), 'cpu', str(p))

if __name__ == '__main__':
    unittest.main()
