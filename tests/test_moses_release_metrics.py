import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gem.metrics_over_time import (
    _compute_summary,
    _largest_fragment_smiles,
    _load_or_compute_reference_statistics,
    _moses_split_source,
    _official_moses_split_name,
)
from gem.metrics.molecular_metrics import prepare_fcd_smiles


class MosesReleaseMetricsTest(unittest.TestCase):
    def test_vun_is_exact_product_of_metric_definitions(self):
        records = [
            {"valid": True, "connected": True, "valid_connected": True, "smiles": "C"},
            {"valid": True, "connected": True, "valid_connected": True, "smiles": "C"},
            {"valid": True, "connected": True, "valid_connected": True, "smiles": "N"},
            {"valid": True, "connected": True, "valid_connected": True, "smiles": "O"},
            {"valid": False, "connected": False, "valid_connected": False, "smiles": None},
        ]

        summary = _compute_summary(
            records,
            train_smiles={"C"},
            fcd_reference_statistics=None,
            compute_fcd_enabled=False,
        )

        self.assertAlmostEqual(summary["validity"], 4 / 5)
        self.assertAlmostEqual(summary["uniqueness"], 3 / 4)
        self.assertAlmostEqual(summary["novelty"], 2 / 3)
        self.assertAlmostEqual(summary["vun"], 2 / 5)
        self.assertAlmostEqual(
            summary["vun"],
            summary["validity"] * summary["uniqueness"] * summary["novelty"],
        )

    def test_largest_fragment_preprocessing_is_fcd_only(self):
        largest, rejected = _largest_fragment_smiles(["CC.O", "N"])

        self.assertEqual(rejected, 0)
        self.assertEqual(largest, ["CC", "N"])

    def test_fcd_largest_fragment_does_not_change_validity_metrics(self):
        records = [
            {
                "valid": True,
                "connected": False,
                "valid_connected": False,
                "smiles": "CC.O",
            }
        ]
        reference_statistics = (np.asarray([0.0]), np.asarray([[1.0]]))
        with patch(
            "gem.metrics_over_time.compute_fcd_from_statistics",
            return_value=1.25,
        ) as compute:
            summary = _compute_summary(
                records,
                train_smiles=set(),
                fcd_reference_statistics=reference_statistics,
                compute_fcd_enabled=True,
                fcd_generated_largest_fragment=True,
            )

        self.assertEqual(summary["validity"], 1.0)
        self.assertEqual(summary["connected_fraction"], 0.0)
        self.assertEqual(summary["fcd"], 1.25)
        self.assertEqual(summary["fcd_generated_input"], 1)
        self.assertEqual(summary["fcd_generated_used"], 1)
        self.assertEqual(compute.call_args.args[1], ["CC"])

    def test_raw_reference_smiles_can_be_preserved(self):
        raw, raw_rejected = prepare_fcd_smiles(["C(C)O"], canonicalize=False)
        canonical, canonical_rejected = prepare_fcd_smiles(
            ["C(C)O"],
            canonicalize=True,
        )

        self.assertEqual(raw_rejected, 0)
        self.assertEqual(canonical_rejected, 0)
        self.assertEqual(raw, ["C(C)O"])
        self.assertEqual(canonical, ["CCO"])

    def test_reference_statistics_cache_is_keyed_by_smiles(self):
        expected = (np.asarray([1.0, 2.0]), np.eye(2))
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "reference.npz"
            with patch(
                "gem.metrics_over_time.compute_fcd_statistics",
                return_value=expected,
            ) as compute:
                first, first_hit = _load_or_compute_reference_statistics(
                    ["C", "N"],
                    fingerprint="fingerprint",
                    cache_path=cache_path,
                    device="cpu",
                )
                second, second_hit = _load_or_compute_reference_statistics(
                    ["C", "N"],
                    fingerprint="fingerprint",
                    cache_path=cache_path,
                    device="cpu",
                )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(compute.call_count, 1)
            np.testing.assert_array_equal(first[0], expected[0])
            np.testing.assert_array_equal(second[1], expected[1])

    def test_internal_val_maps_to_raw_official_scaffold_file(self):
        raw_dir = Path("/tmp/moses/raw")

        self.assertEqual(_official_moses_split_name("val"), "test_scaffolds")
        self.assertEqual(
            _moses_split_source(raw_dir, "val", filter_dataset=False),
            raw_dir / "val_moses.csv",
        )


if __name__ == "__main__":
    unittest.main()
