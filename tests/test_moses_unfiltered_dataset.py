import tempfile
import unittest
from pathlib import Path

from gem.datasets.moses_dataset import MOSESDataset


class MosesUnfilteredDatasetTest(unittest.TestCase):
    def test_raw_official_splits_are_processed_without_graph_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            split_smiles = {
                "train_moses.csv": ["CC", "CCO"],
                "val_moses.csv": ["CN", "CO"],
                "test_moses.csv": ["CCC", "CCN"],
            }
            for filename, smiles in split_smiles.items():
                (raw_dir / filename).write_text(
                    "SMILES\n" + "\n".join(smiles) + "\n"
                )

            train = MOSESDataset("train", str(root), filter_dataset=False)
            validation = MOSESDataset("val", str(root), filter_dataset=False)
            test = MOSESDataset("test", str(root), filter_dataset=False)

            self.assertEqual([len(train), len(validation), len(test)], [2, 2, 2])
            self.assertEqual(train[0].idx.item(), 0)
            self.assertEqual(validation[0].idx.item(), 2)
            self.assertEqual(test[0].idx.item(), 4)


if __name__ == "__main__":
    unittest.main()
