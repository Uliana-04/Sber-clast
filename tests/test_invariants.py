"""Research invariants with known answers, independent of the real dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from municipal_economy.quality import network_indices, independent_check
from municipal_economy.dynamics import align_to_previous
from municipal_economy.config import load_config


class Invariants(unittest.TestCase):
    def test_disconnected_pairs(self):
        graph = csr_matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        labels = np.array([0, 0, 1, 1])
        metrics, _, _ = network_indices(graph, labels)
        independent_check(graph, labels, metrics)
        self.assertAlmostEqual(metrics["AVI"], 1)
        self.assertAlmostEqual(metrics["AVU"], 0)
        self.assertAlmostEqual(metrics["MQ"], 0.5)

    def test_complete_graph(self):
        graph = csr_matrix(np.ones((4, 4)) - np.eye(4))
        metrics, _, _ = network_indices(graph, np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(metrics["AVI"], 1 / 3)
        self.assertAlmostEqual(metrics["AVU"], 1)
        self.assertAlmostEqual(metrics["MQ"], -0.5)

    def test_permuted_labels(self):
        previous = pd.Series([0, 0, 1, 1, 2, 3, 4], index=range(7))
        current = previous.map({0: 3, 1: 4, 2: 1, 3: 2, 4: 0})
        aligned, _, overlap = align_to_previous(previous, current)
        np.testing.assert_array_equal(aligned, previous)
        self.assertEqual(overlap, 7)

    def test_config_equivalence(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            load_config(root / "configs/default.json"), load_config(root / "configs/default.yaml")
        )


if __name__ == "__main__":
    unittest.main()
