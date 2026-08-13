from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gem import cali_params


class CalibrationMetricsTest(unittest.TestCase):
    def test_relaxed_metrics_include_connected_fraction(self):
        smiles = {0: "CC", 1: "C.C", 2: "NN", 3: ""}
        build = lambda nodes, edges, decoder: int(nodes.item())
        to_smiles = lambda molecule: smiles[molecule]

        nodes = [torch.tensor(index) for index in range(4)]
        edges = [torch.zeros((1, 1), dtype=torch.long) for _ in nodes]
        dataset_infos = SimpleNamespace(atom_decoder=["C"])

        with patch.object(cali_params, "build_molecule_with_partial_charges", build), patch.object(
            cali_params, "mol2smiles", to_smiles
        ):
            validity, connected, uniqueness, novelty, vun = (
                cali_params._relaxed_validity_and_novelty(
                    nodes,
                    edges,
                    dataset_infos,
                    train_smiles={"CC"},
                )
            )

        self.assertAlmostEqual(validity, 3 / 4)
        self.assertAlmostEqual(connected, 2 / 4)
        self.assertAlmostEqual(uniqueness, 1.0)
        self.assertAlmostEqual(novelty, 2 / 3)
        self.assertAlmostEqual(vun, 1 / 2)
