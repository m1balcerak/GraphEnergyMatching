import unittest

from gem.metrics.molecular_metrics import _clean_fcd_smiles


class FcdCleaningTest(unittest.TestCase):
    def test_rejects_unparsable_smiles_and_canonicalizes_valid_entries(self):
        cleaned, rejected = _clean_fcd_smiles([" C(C) ", "not-smiles", "", None])

        self.assertEqual(cleaned, ["CC"])
        self.assertEqual(rejected, 3)


if __name__ == "__main__":
    unittest.main()
