import os
import os.path as osp
import pathlib
from functools import lru_cache
from typing import Any, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

from gem import utils
from gem.analysis.rdkit_functions import (
    mol2smiles,
    build_molecule_with_partial_charges,
)
from gem.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]


@lru_cache(maxsize=4)
def _moses_split_sizes(raw_dir: str, filter_dataset: bool) -> dict[str, int]:
    """Return number of molecules per split.

    If filtered SMILES files exist and filtering is enabled, use those counts.
    Otherwise fall back to raw CSVs (header is skipped).
    """

    def _count_lines(path: str, skip_header: bool) -> int:
        if not osp.exists(path):
            raise FileNotFoundError(f"Expected MOSES split file not found: {path}")
        with open(path, "r") as handle:
            count = sum(1 for _ in handle)
        if skip_header and count:
            count -= 1
        return max(count, 0)

    raw_dir = osp.abspath(raw_dir)
    csv_paths = {
        "train": osp.join(raw_dir, "train_moses.csv"),
        "val": osp.join(raw_dir, "val_moses.csv"),
        "test": osp.join(raw_dir, "test_moses.csv"),
    }
    filtered_smiles_paths = {
        "train": osp.join(raw_dir, "new_train.smiles"),
        "val": osp.join(raw_dir, "new_val.smiles"),
        "test": osp.join(raw_dir, "new_test.smiles"),
    }

    split_paths: dict[str, str] = {}
    skip_header_map: dict[str, bool] = {}

    for split in ("train", "val", "test"):
        use_filtered = filter_dataset and osp.exists(filtered_smiles_paths[split])
        if use_filtered:
            split_paths[split] = filtered_smiles_paths[split]
            skip_header_map[split] = False
        else:
            split_paths[split] = csv_paths[split]
            skip_header_map[split] = True  # raw CSVs include a header row

    return {
        split: _count_lines(path, skip_header_map[split]) for split, path in split_paths.items()
    }


def _moses_idx_offsets(raw_dir: str, filter_dataset: bool) -> dict[str, int]:
    """Compute deterministic, non-overlapping index offsets for each split."""
    sizes = _moses_split_sizes(raw_dir, filter_dataset)
    train_size = sizes["train"]
    val_size = sizes["val"]
    return {
        "train": 0,
        "val": train_size,
        "test": train_size + val_size,
    }


class MOSESDataset(InMemoryDataset):
    train_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
    val_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv"
    test_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"

    def __init__(
        self,
        stage,
        root,
        filter_dataset: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        try:
            self.file_idx = {"train": 0, "val": 1, "test": 2}[self.stage]
        except KeyError as exc:
            raise ValueError(f"Unknown MOSES split: {self.stage!r}") from exc
        super().__init__(root, transform, pre_transform, pre_filter)
        self._data, self.slices = torch.load(
            self.processed_paths[self.file_idx], weights_only=False
        )
        self._ensure_global_indices()

    def _ensure_global_indices(self):
        """Ensure idx values are global (non-overlapping) across splits."""
        if self._data is None or not hasattr(self._data, "idx") or self._data.idx is None:
            return
        idx_tensor = self._data.idx
        if idx_tensor.numel() == 0:
            return

        offsets = _moses_idx_offsets(self.raw_dir, self.filter_dataset)
        base_idx = offsets[self.stage]
        expected_max = base_idx + len(self) - 1
        current_min = int(idx_tensor.min())
        current_max = int(idx_tensor.max())

        if current_min == base_idx and current_max == expected_max:
            return  # already global

        if current_min == 0 and current_max == len(self) - 1:
            # Old processed files used local indexing; shift into the global range.
            self._data.idx = idx_tensor + base_idx
            return

        if self.filter_dataset:
            raw_offsets = _moses_idx_offsets(self.raw_dir, False)
            raw_base_idx = raw_offsets[self.stage]
            raw_expected_max = raw_base_idx + len(self) - 1
            if current_min == raw_base_idx and current_max == raw_expected_max:
                # Processed files can be built before all filtered SMILES counts exist,
                # which leaves idx values in raw-split global coordinates.
                self._data.idx = idx_tensor - raw_base_idx + base_idx
                return

        raise ValueError(
            f"Unexpected MOSES idx range for split '{self.stage}': "
            f"min={current_min}, max={current_max}, expected {base_idx}-{expected_max}. "
            "Remove processed files/cache and rebuild the dataset."
        )

    @property
    def raw_file_names(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_file_name(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        # Preserve the cache names used by the original MOSES preprocessing code.
        if self.filter_dataset:
            return [
                "train_filtered.pt",
                "test_filtered.pt",
                "test_scaffold_filtered.pt",
            ]
        return ["train.pt", "test.pt", "test_scaffold.pt"]

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.replace(train_path, osp.join(self.raw_dir, "train_moses.csv"))

        val_path = download_url(self.val_url, self.raw_dir)
        os.replace(val_path, osp.join(self.raw_dir, "val_moses.csv"))

        test_path = download_url(self.test_url, self.raw_dir)
        os.replace(test_path, osp.join(self.raw_dir, "test_moses.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)["SMILES"].values

        offsets = _moses_idx_offsets(self.raw_dir, self.filter_dataset)
        base_idx = offsets[self.stage]

        data_list = []
        smiles_kept = []

        idx_counter = 0
        for smile in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                idx=base_idx + idx_counter,
            )

            if self.filter_dataset:
                # Keep graphs that reconstruct as a single valid molecule.
                dense_data, node_mask = utils.to_dense(
                    data.x, data.edge_index, data.edge_attr, data.batch
                )
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(
                    atom_types, edge_types, atom_decoder
                )
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(
                            mol, asMols=True, sanitizeFrags=True
                        )
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)
                            idx_counter += 1

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                idx_counter += 1

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(
                pathlib.Path(self.raw_paths[0]).parent, f"new_{self.stage}.smiles"
            )
            print(smiles_save_path)
            with open(smiles_save_path, "w") as f:
                f.writelines("%s\n" % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")


class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[3]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": MOSESDataset(
                stage="train", root=root_path, filter_dataset=self.filter_dataset
            ),
            "val": MOSESDataset(
                stage="val", root=root_path, filter_dataset=self.filter_dataset
            ),
            "test": MOSESDataset(
                stage="test", root=root_path, filter_dataset=self.filter_dataset
            ),
        }
        super().__init__(cfg, datasets)


class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = "MOSES"
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        self.compute_fcd = cfg.dataset.compute_fcd

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9, 7: 1}
        self.valencies = [4, 3, 4, 2, 1, 1, 1, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        meta_files = dict(
            n_nodes=f"{self.name}_n_counts.txt",
            node_types=f"{self.name}_atom_types.txt",
            edge_types=f"{self.name}_edge_types.txt",
            valency_distribution=f"{self.name}_valencies.txt",
        )

        self.n_nodes = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.097634362347889692e-06,
                1.858580617408733815e-05,
                5.007842264603823423e-05,
                5.678996240021660924e-05,
                1.244216400664299726e-04,
                4.486406978685408831e-04,
                2.253012731671333313e-03,
                3.231865121051669121e-03,
                6.709992419928312302e-03,
                2.289564721286296844e-02,
                5.411050841212272644e-02,
                1.099515631794929504e-01,
                1.223291903734207153e-01,
                1.280680745840072632e-01,
                1.445975750684738159e-01,
                1.505961418151855469e-01,
                1.436946094036102295e-01,
                9.265746921300888062e-02,
                1.820066757500171661e-02,
                2.065089574898593128e-06,
            ]
        )
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.node_types = torch.tensor(
            [0.722338, 0.13661, 0.163655, 0.103549, 0.1421803, 0.005411, 0.00150, 0.0]
        )
        self.edge_types = torch.tensor(
            [0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486]
        )
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor(
            [0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485]
        )

        if meta is None:
            meta = dict(
                n_nodes=None,
                node_types=None,
                edge_types=None,
                valency_distribution=None,
            )
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_smiles(raw_dir, filter_dataset):

    def _clean(smiles_seq):
        cleaned = []
        for s in smiles_seq:
            if s is None:
                continue
            if not isinstance(s, str):
                try:
                    s = str(s)
                except Exception:
                    continue
            s = s.strip()
            if s:
                cleaned.append(s)
        return cleaned

    if filter_dataset:
        smiles_save_paths = {
            "train": osp.join(raw_dir, "new_train.smiles"),
            "val": osp.join(raw_dir, "new_val.smiles"),
            "test": osp.join(raw_dir, "new_test.smiles"),
        }
        train_smiles = _clean(open(smiles_save_paths["train"]).readlines())
        val_smiles = _clean(open(smiles_save_paths["val"]).readlines())
        test_smiles = _clean(open(smiles_save_paths["test"]).readlines())

    else:
        smiles_save_paths = {
            "train": osp.join(raw_dir, "train_moses.csv"),
            "val": osp.join(raw_dir, "val_moses.csv"),
            "test": osp.join(raw_dir, "test_moses.csv"),
        }
        train_smiles = _clean(extract_smiles_from_csv(smiles_save_paths["train"]))
        val_smiles = _clean(extract_smiles_from_csv(smiles_save_paths["val"]))
        test_smiles = _clean(extract_smiles_from_csv(smiles_save_paths["test"]))

    return {
        "train": train_smiles,
        "val": val_smiles,
        "test": test_smiles,
    }


def extract_smiles_from_csv(csv_path):
    return pd.read_csv(csv_path)["SMILES"].to_list()
