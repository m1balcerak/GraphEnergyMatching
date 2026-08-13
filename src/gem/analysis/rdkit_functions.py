import numpy as np
import torch
import re
import warnings
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, train_smiles=None, args=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.args = args
        default_mode = "vfm_relaxed"
        validity_mode = default_mode
        if args is not None:
            try:
                validity_mode = str(
                    getattr(getattr(args, "general", object()), "validity_mode", default_mode)
                ).lower()
            except AttributeError:
                validity_mode = default_mode
        if validity_mode not in {"vfm_relaxed", "strict"}:
            warnings.warn(
                f"Unknown validity_mode='{validity_mode}', falling back to '{default_mode}'.",
                stacklevel=2,
            )
            validity_mode = default_mode
        self.validity_mode = validity_mode
        self.validity_labels = {
            "vfm_relaxed": "validity_VFM_relaxed",
            "strict": "validity_strict",
        }
        self.primary_validity_label = self.validity_labels[self.validity_mode]
        self._secondary_mode = "strict" if self.validity_mode == "vfm_relaxed" else "vfm_relaxed"
        self.secondary_validity_label = self.validity_labels[self._secondary_mode]
        self.latest_validity_values = {"strict": None, "vfm_relaxed": None}
        self.latest_validity_labels = {
            "primary": self.primary_validity_label,
            "secondary": self.secondary_validity_label,
        }

        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        all_smiles_without_test = []

        for graph in tqdm(
            generated, desc="Generated molecules validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            all_smiles_without_test.append(mol2smilesWithNoSanitize(mol))
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
            except Exception:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        # Persist SMILES; optionally log depending on config
        with open(r"final_smiles_all.txt", "w") as fp:
            for smiles in all_smiles_without_test:
                fp.write("%s\n" % smiles)
        df = pd.DataFrame(all_smiles_without_test, columns=["SMILES"])
        df.to_csv("final_smiles_all.csv", index=False)
        if getattr(getattr(self.args, "sample", object()), "log_smiles", False):
            print("All smiles saved")
            print(all_smiles_without_test)
            print("All SMILES saved to CSV")

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in tqdm(unique, desc="Unique molecules novelty check progress"):
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in tqdm(
            generated, desc="Generated molecules relaxed validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def evaluate(self, generated, input_properties=None, test=False):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        strict_valid, strict_validity, num_components, all_smiles = self.compute_validity(
            generated
        )

        if test:
            # Persist SMILES (optionally log depending on config)
            with open(r"final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    fp.write("%s\n" % smiles)
            df = pd.DataFrame(all_smiles, columns=["SMILES"])
            df.to_csv("final_smiles.csv", index=False)
            if getattr(getattr(self.args, "sample", object()), "log_smiles", False):
                print("All smiles saved")
                print(all_smiles)
                print("All SMILES saved to CSV")

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        total_mols = len(generated)
        strict_label = self.validity_labels["strict"]
        relaxed_label = self.validity_labels["vfm_relaxed"]
        print(f"{strict_label} over {total_mols} molecules: {strict_validity * 100 :.2f}%")
        print(
            f"Number of connected components of {total_mols} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(f"{relaxed_label} over {total_mols} molecules: {relaxed_validity * 100 :.2f}%")

        self.latest_validity_values["strict"] = strict_validity
        self.latest_validity_values["vfm_relaxed"] = relaxed_validity
        self.primary_validity_label = self.validity_labels[self.validity_mode]
        self._secondary_mode = "strict" if self.validity_mode == "vfm_relaxed" else "vfm_relaxed"
        self.secondary_validity_label = self.validity_labels[self._secondary_mode]
        self.latest_validity_labels = {
            "primary": self.primary_validity_label,
            "secondary": self.secondary_validity_label,
        }

        if self.validity_mode == "vfm_relaxed":
            primary_valid, primary_validity = relaxed_valid, relaxed_validity
            secondary_validity = strict_validity
        else:
            primary_valid, primary_validity = strict_valid, strict_validity
            secondary_validity = relaxed_validity

        self.latest_validity_values["primary"] = primary_validity
        self.latest_validity_values["secondary"] = secondary_validity

        if input_properties is not None:
            raise NotImplementedError(
                "Property-conditioned molecular metrics are not included in this release."
            )
        cond_mae = cond_val = -1.0

        if len(primary_valid) > 0:
            unique, uniqueness = self.compute_uniqueness(primary_valid)
            print(
                f"Uniqueness over {len(primary_valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                )
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return (
            [primary_validity, secondary_validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
            [cond_mae, cond_val],
        )


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def mol2smilesWithNoSanitize(mol):
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)
    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def build_molecule_with_partial_charges(
    atom_types, edge_types, atom_decoder, verbose=False
):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)

    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
            # Add positive formal charges for supported over-valent atoms.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def check_stability(
    atom_types, edge_types, dataset_info, debug=False, atom_decoder=None
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(
    molecule_list, train_smiles, dataset_info, labels, args=None, test=False
):
    """molecule_list: (dict)"""

    if not dataset_info.remove_h:
        print("Analyzing molecule stability...")

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for mol in tqdm(molecule_list, desc="Stability computation progress"):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
    else:
        validity_dict = {"mol_stable": -1, "atm_stable": -1}

    metrics = BasicMolecularMetrics(dataset_info, train_smiles, args)
    rdkit_metrics = metrics.evaluate(molecule_list, labels, test)
    all_smiles = rdkit_metrics[-2]

    nc = rdkit_metrics[-3]
    primary_label = getattr(metrics, "primary_validity_label", "validity_VFM_relaxed")
    secondary_label = getattr(metrics, "secondary_validity_label", "validity_strict")
    dic = {
        primary_label: rdkit_metrics[0][0],
        secondary_label: rdkit_metrics[0][1],
        "Uniqueness": rdkit_metrics[0][2],
        "Novelty": rdkit_metrics[0][3],
        "nc_min": nc["nc_min"],
        "nc_max": nc["nc_max"],
        "nc_mu": nc["nc_mu"],
        "cond_mae": rdkit_metrics[-1][0],
        "cond_val": rdkit_metrics[-1][1],
    }
    return validity_dict, rdkit_metrics, all_smiles, dic
