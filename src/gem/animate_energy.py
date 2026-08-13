"""GEM energy trajectory animation utilities.

Builds upon the static visualization in ``visualize_energy`` by running a
small cohort of MCMC chains (noise- and data-initialized) and generating a
matplotlib animation that reveals how their energies evolve step by step.

The module is designed to be mostly self-contained: it reuses the helper
routines from ``visualize_energy`` for collecting data graphs and estimating
energy bands, but otherwise keeps the animation logic independent so it can be
invoked as a standalone Hydra entry point.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List, Sequence, Dict, Any
import sys
import random

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (str(_SRC_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from gem.datasets import moses_dataset
from gem.models.extra_features import ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures
from gem.models.transformer_model import GraphTransformer
from gem.fm_utils import sample_interpolated_graph
from gem.ot_data import initialize_random_graphs_with_counts
from gem.checkpoint_utils import load_model_checkpoint

try:
    from . import sampler
    from .dlangevin_utils import (
        TWO_BETA_ANNEALING_PROPOSALS,
        TWO_BETA_PROPOSALS,
        resolve_chain_warmup,
        resolve_dl_parameters,
    )
    from .visualize_energy import (
        _chunked_energy_mean_std,
        _collect_graphs_from_data,
        _ensure_run_dir,
        _setup_file_logging,
    )
except ImportError:
    from gem import sampler
    from gem.dlangevin_utils import (
        TWO_BETA_ANNEALING_PROPOSALS,
        TWO_BETA_PROPOSALS,
        resolve_chain_warmup,
        resolve_dl_parameters,
    )
    from gem.visualize_energy import (
        _chunked_energy_mean_std,
        _collect_graphs_from_data,
        _ensure_run_dir,
        _setup_file_logging,
    )
from gem.analysis.rdkit_functions import build_molecule, mol2smiles

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    _HAS_RDKIT = True
except Exception:
    Chem = None
    Draw = None
    rdMolDraw2D = None
    _HAS_RDKIT = False


def _build_model_and_data(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None,
    datamodule=None,
    dataset_infos=None,
    extra_features=None,
    domain_features=None,
    device: torch.device | None,
):
    """Mirror ``visualize_energy.run_viz`` setup to reuse trained components."""
    dataset_name = str(cfg.dataset.name).lower()
    if datamodule is None or dataset_infos is None:
        if dataset_name != "moses":
            raise NotImplementedError(
                f"Dataset '{cfg.dataset.name}' is not supported in this trimmed MOSES release."
            )
        datamodule = moses_dataset.MosesDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
    if extra_features is None or domain_features is None:
        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    if model is None:
        act_name = str(getattr(cfg.model, "activation", "relu")).lower()
        act_module = torch.nn.SiLU() if act_name == "silu" else torch.nn.ReLU()
        model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=dataset_infos.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=dataset_infos.output_dims,
            act_fn_in=act_module,
            act_fn_out=act_module,
            tf_activation=act_module,
        ).to(device)
    model.eval()
    return model, datamodule, dataset_infos, extra_features, domain_features, device


def _load_checkpoint_if_any(
    model: torch.nn.Module,
    device: torch.device,
    path: str | None,
    *,
    use_ema: bool = False,
) -> bool:
    if not path:
        return False
    load_model_checkpoint(model, path, map_location=device, use_ema=use_ema)
    weight_label = "EMA" if use_ema else "online"
    print(f"[info] Loaded {weight_label} checkpoint weights: {path}")
    return True


def _clone_graphs(
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    nodes = [nt.detach().clone().cpu() for nt in node_types_list]
    edges = [et.detach().clone().cpu() for et in edge_types_list]
    return nodes, edges


def _render_molecule_grid(
    *,
    noise_nodes,
    noise_edges,
    data_nodes,
    data_edges,
    dataset_info,
    max_display: int,
    mols_per_row: int,
    subimg_size: tuple[int, int],
    show_legends: bool = True,
    draw_padding: float | None = None,
    draw_bond_line_width: float | None = None,
    draw_min_font_size: int | None = None,
    draw_max_font_size: int | None = None,
    draw_same_scale: bool | None = None,
) -> np.ndarray | None:
    if not _HAS_RDKIT:
        return None

    max_display = max(1, int(max_display))
    mols_per_row = max(1, int(mols_per_row))
    subimg_size = tuple(int(x) for x in subimg_size)

    total_noise = len(noise_nodes)
    total_data = len(data_nodes)

    half = max_display // 2 if total_data > 0 and total_noise > 0 else max_display
    n_noise = min(total_noise, half)
    n_data = min(total_data, max_display - n_noise)
    # If one group is undersubscribed, let the other fill remaining slots
    remaining = max_display - (n_noise + n_data)
    if remaining > 0:
        if n_noise < total_noise:
            take = min(remaining, total_noise - n_noise)
            n_noise += take
            remaining -= take
        if remaining > 0 and n_data < total_data:
            take = min(remaining, total_data - n_data)
            n_data += take

    mols = []
    legends = []

    for idx, (nt, et) in enumerate(itertools.islice(zip(noise_nodes, noise_edges), n_noise)):
        try:
            mol = build_molecule(nt, et, dataset_info.atom_decoder)
            if mol is None:
                continue
            if Chem is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() == 0:
                Chem.rdDepictor.Compute2DCoords(mol)
            mols.append(mol)
            legends.append(f"Noise {idx}")
        except Exception:
            continue

    for idx, (nt, et) in enumerate(itertools.islice(zip(data_nodes, data_edges), n_data)):
        try:
            mol = build_molecule(nt, et, dataset_info.atom_decoder)
            if mol is None:
                continue
            if Chem is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() == 0:
                Chem.rdDepictor.Compute2DCoords(mol)
            mols.append(mol)
            legends.append(f"Data {idx}")
        except Exception:
            continue

    if not mols:
        return None

    try:
        total_mols = len(mols)
        mols_per_row = max(1, min(mols_per_row, total_mols))
        grid_kwargs = dict(
            mols=mols,
            molsPerRow=mols_per_row,
            subImgSize=subimg_size,
        )
        draw_options = Draw.MolDrawOptions()
        has_draw_options = False
        if draw_padding is not None:
            draw_options.padding = float(draw_padding)
            has_draw_options = True
        if draw_bond_line_width is not None:
            draw_options.bondLineWidth = float(draw_bond_line_width)
            has_draw_options = True
        if draw_min_font_size is not None:
            draw_options.minFontSize = int(draw_min_font_size)
            has_draw_options = True
        if draw_max_font_size is not None:
            draw_options.maxFontSize = int(draw_max_font_size)
            has_draw_options = True
        if draw_same_scale is not None:
            draw_options.drawMolsSameScale = bool(draw_same_scale)
            has_draw_options = True
        if has_draw_options:
            grid_kwargs["drawOptions"] = draw_options
        if show_legends:
            grid_kwargs["legends"] = legends
        img = Draw.MolsToGridImage(**grid_kwargs)
        return np.array(img)
    except Exception:
        return None


def _compute_metrics(
    molecules,
    dataset_info,
    train_smiles: set[str] | None,
):
    smiles_valid: List[str] = []
    for (nt, et) in molecules:
        try:
            mol = build_molecule(nt, et, dataset_info.atom_decoder)
            s = mol2smiles(mol)
            if s:
                smiles_valid.append(s)
        except Exception:
            continue
    total = len(molecules)
    if total == 0:
        return 0.0, 0.0, 0.0
    validity = len(smiles_valid) / total
    if not smiles_valid:
        return validity, 0.0, 0.0
    unique_smiles = set(smiles_valid)
    uniqueness = len(unique_smiles) / len(smiles_valid)
    if train_smiles:
        novelty = sum(1 for s in unique_smiles if s not in train_smiles) / len(unique_smiles)
    else:
        novelty = 0.0
    return float(validity), float(uniqueness), float(novelty)


def _quality_summary_from_rows(rows: Sequence[dict]) -> dict:
    total = len(rows)
    valid_count = sum(1 for row in rows if row["valid"])
    connected_count = sum(1 for row in rows if row["connected"])
    valid_connected_count = sum(1 for row in rows if row["valid_connected"])
    return {
        "total": int(total),
        "valid_count": int(valid_count),
        "connected_count": int(connected_count),
        "valid_connected_count": int(valid_connected_count),
        "valid_fraction": float(valid_count / total) if total else 0.0,
        "connected_fraction": float(connected_count / total) if total else 0.0,
        "valid_connected_fraction": float(valid_connected_count / total) if total else 0.0,
        "all_valid_connected": bool(total > 0 and valid_connected_count == total),
    }


def _molecule_quality_rows(molecules, dataset_info, indices: Sequence[int] | None = None) -> tuple[list[dict], dict]:
    if indices is None:
        indexed_molecules = list(enumerate(molecules))
    else:
        indexed_molecules = [
            (int(idx), molecules[int(idx)])
            for idx in indices
            if 0 <= int(idx) < len(molecules)
        ]

    rows = []
    for idx, (nt, et) in indexed_molecules:
        row = {
            "index": int(idx),
            "valid": False,
            "connected": False,
            "valid_connected": False,
            "num_fragments": None,
            "smiles": None,
        }
        try:
            mol = build_molecule(nt, et, dataset_info.atom_decoder)
            if mol is not None and Chem is not None:
                try:
                    row["num_fragments"] = int(len(Chem.rdmolops.GetMolFrags(mol)))
                except Exception:
                    row["num_fragments"] = None
                smiles = mol2smiles(mol)
                row["smiles"] = smiles
                row["valid"] = bool(smiles)
                if smiles:
                    row["connected"] = ("." not in smiles) and (row["num_fragments"] in {None, 1})
                    row["valid_connected"] = bool(row["valid"] and row["connected"])
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    return rows, _quality_summary_from_rows(rows)


def _safe_energy_band_stats(
    nodes: Sequence[torch.Tensor],
    edges: Sequence[torch.Tensor],
    *,
    model,
    dataset_info,
    extra_features,
    domain_features,
    device,
    label: str,
) -> tuple[float, float] | None:
    """Compute mean/std for an energy band if graphs are available."""
    if not nodes:
        print(f"[info] Skipping {label} energy band because no graphs were provided.")
        return None
    try:
        mean, std = _chunked_energy_mean_std(
            nodes,
            edges,
            model,
            dataset_info,
            extra_features,
            domain_features,
            device,
        )
        return float(mean), float(std)
    except Exception as exc:
        print(f"[warn] Failed to compute {label} energy band: {exc}")
        return None


def _stepwise_energy(
    *,
    model,
    dataset_info,
    extra_features,
    domain_features,
    device,
    node_types: Sequence[torch.Tensor],
    edge_types: Sequence[torch.Tensor],
) -> np.ndarray:
    """Compute energies for a batch of graphs and return CPU numpy array."""
    with torch.no_grad():
        energies = sampler.energy_batch(
            model=model,
            node_types_list=list(node_types),
            edge_types_list=list(edge_types),
            dataset_info=dataset_info,
            device=device,
            extra_features=extra_features,
            domain_features=domain_features,
            detach=True,
        )
    return energies.detach().cpu().numpy()


def _palette(name: str, n: int) -> List[tuple[float, float, float, float]]:
    if n <= 0:
        return []
    cmap = plt.get_cmap(name)
    if n == 1:
        return [cmap(0.7)]
    return [cmap(0.3 + 0.5 * (i / (n - 1))) for i in range(n)]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_int_list(value) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, ListConfig)):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out or None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        out = []
        for item in parts:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out or None
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return None


def _sanitize_indices(indices: List[int] | None, count: int, label: str) -> list[int] | None:
    if indices is None:
        return None
    clean: list[int] = []
    seen: set[int] = set()
    for idx_raw in indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= count:
            print(f"[warn] Ignoring {label} display index {idx}: valid range is [0, {max(count - 1, 0)}].")
            continue
        if idx in seen:
            continue
        clean.append(idx)
        seen.add(idx)
    if not clean:
        print(f"[warn] No valid {label} display indices were provided; using all {label} chains.")
        return None
    return clean


def _take_indices(items, indices: list[int] | None):
    if indices is None:
        return items
    return [items[idx] for idx in indices if 0 <= idx < len(items)]


def _random_display_indices(total: int, count: int, label: str) -> list[int] | None:
    total = max(0, int(total))
    count = max(0, int(count))
    if total <= 0 or count <= 0 or count >= total:
        return None
    indices = sorted(random.sample(range(total), count))
    print(f"[info] Randomly selected {label} display chains before quality evaluation: {indices}")
    return indices


def _most_likely_node_count(dataset_infos) -> int | None:
    nodes_dist = getattr(dataset_infos, "nodes_dist", None)
    if nodes_dist is None:
        return None
    prob = getattr(nodes_dist, "prob", None)
    if prob is None:
        return None
    if isinstance(prob, torch.Tensor):
        if prob.numel() == 0:
            return None
        scores = prob.clone()
        if scores.numel() > 1:
            scores[0] = -1.0
        return int(torch.argmax(scores).item())
    try:
        arr = np.asarray(prob, dtype=float)
        if arr.size == 0:
            return None
        if arr.size > 1:
            arr[0] = -1.0
        return int(np.argmax(arr))
    except Exception:
        return None


def _format_step_label(step: int, warmup_steps_total: int, *, long: bool = False) -> str:
    if warmup_steps_total > 0 and step <= warmup_steps_total:
        if step == 0:
            return "initial" if long else "w0"
        return f"transport burn-in {step}" if long else f"tb{step}"
    mcmc_step = max(step - warmup_steps_total, 0)
    return f"mcmc {mcmc_step}" if long else f"m{mcmc_step}"


def _render_chain_grid(
    *,
    noise_node_history: List[List[torch.Tensor]],
    noise_edge_history: List[List[torch.Tensor]],
    history_steps: Sequence[int],
    dataset_info,
    step_values: Sequence[int],
    chain_count: int,
    subimg_size: tuple[int, int],
    add_step_labels: bool,
    warmup_steps_total: int,
    title_fontsize: float,
    nv_fontsize: float,
    dpi: int,
    add_row_labels: bool,
    row_label_prefix: str,
    row_label_fontsize: float,
    save_cells: bool,
    cells_dir: Path | None,
    cell_prefix: str,
    mark_invalid: bool,
    cells_format: str,
) -> np.ndarray | None:
    if not _HAS_RDKIT:
        print("[warn] RDKit not available – skipping chain grid rendering.")
        return None
    if not noise_node_history or not noise_edge_history:
        print("[warn] No noise chain history available for chain grid.")
        return None
    if not step_values:
        print("[warn] No chain grid steps provided; skipping chain grid.")
        return None
    step_to_idx = {int(step): idx for idx, step in enumerate(history_steps)}
    missing = [step for step in step_values if int(step) not in step_to_idx]
    if missing:
        print(f"[warn] Missing chain grid steps (not recorded): {missing}")
    selected_steps = [int(step) for step in step_values if int(step) in step_to_idx]
    if not selected_steps:
        print("[warn] No valid chain grid steps found; skipping chain grid.")
        return None

    available_chains = len(noise_node_history[0])
    if available_chains <= 0:
        print("[warn] No noise chains available for chain grid.")
        return None
    chain_count = max(1, min(int(chain_count), available_chains))

    mols = []
    valids = []
    for row in range(chain_count):
        for step in selected_steps:
            hist_idx = step_to_idx[step]
            try:
                nodes = noise_node_history[hist_idx][row]
                edges = noise_edge_history[hist_idx][row]
            except Exception:
                nodes = edges = None
            mol = None
            valid = False
            if nodes is not None and edges is not None:
                try:
                    mol = build_molecule(nodes, edges, dataset_info.atom_decoder)
                    if mol is not None and Chem is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() == 0:
                        Chem.rdDepictor.Compute2DCoords(mol)
                    if mol is not None:
                        try:
                            smiles = mol2smiles(mol)
                            valid = bool(smiles)
                        except Exception:
                            valid = False
                except Exception:
                    mol = None
            if mol is None:
                mol = Chem.Mol() if Chem is not None else None
                valid = False
            mols.append(mol)
            valids.append(valid)

    if not mols:
        print("[warn] No molecules rendered for chain grid.")
        return None

    try:
        cells_format_norm = (cells_format or "png").lower()
        if cells_format_norm not in {"png", "svg"}:
            print(f"[warn] Unknown chain grid cell format '{cells_format}'; defaulting to png.")
            cells_format_norm = "png"
        cells_dir_path = None
        if save_cells:
            cells_dir_path = (cells_dir or Path("chain_grid_cells")).resolve()
            cells_dir_path.mkdir(parents=True, exist_ok=True)
        cols = len(selected_steps)
        rows = chain_count
        width_px = subimg_size[0] * cols
        height_px = subimg_size[1] * rows
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(width_px / max(dpi, 1), height_px / max(dpi, 1)),
            dpi=max(dpi, 1),
        )
        axes = np.atleast_2d(axes)
        blank = np.full((subimg_size[1], subimg_size[0], 3), 255, dtype=np.uint8)
        for row in range(rows):
            for col, step in enumerate(selected_steps):
                ax = axes[row, col]
                ax.axis("off")
                idx = row * cols + col
                mol = mols[idx] if idx < len(mols) else None
                valid = valids[idx] if idx < len(valids) else False
                if mol is not None:
                    try:
                        img = Draw.MolToImage(mol, size=subimg_size)
                        img = np.array(img)
                    except Exception:
                        img = blank
                        valid = False
                else:
                    img = blank
                    valid = False
                ax.imshow(img)
                if add_row_labels and col == 0:
                    label = f"{row_label_prefix} {row + 1}" if row_label_prefix else f"{row + 1}"
                    ax.text(
                        -0.04,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=row_label_fontsize,
                        color="black",
                        rotation=0,
                        clip_on=False,
                    )
                if not valid:
                    ax.text(
                        0.5,
                        -0.08,
                        "NV",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=nv_fontsize,
                        color="red",
                        clip_on=False,
                    )
                if add_step_labels and row == 0:
                    ax.set_title(
                        _format_step_label(step, warmup_steps_total, long=True),
                        fontsize=title_fontsize,
                    )
                if save_cells and cells_dir_path is not None:
                    step_tag = _format_step_label(step, warmup_steps_total, long=False)
                    suffix = "_NV" if (mark_invalid and not valid) else ""
                    ext = "svg" if cells_format_norm == "svg" else "png"
                    filename = f"{cell_prefix}_r{row + 1:02d}_c{col + 1:02d}_{step_tag}{suffix}.{ext}"
                    try:
                        if cells_format_norm == "png":
                            plt.imsave(cells_dir_path / filename, img)
                        else:
                            svg = _mol_to_svg(
                                mol=mol,
                                size=subimg_size,
                                add_invalid=(mark_invalid and not valid),
                            )
                            if svg is not None:
                                (cells_dir_path / filename).write_text(svg)
                    except Exception:
                        pass
        fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.05, top=0.92)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[..., :3].copy()
        plt.close(fig)
        return img
    except Exception as exc:
        print(f"[warn] Failed to render chain grid: {exc}")
        return None


def _mol_to_svg(
    *,
    mol,
    size: tuple[int, int],
    add_invalid: bool,
) -> str | None:
    width, height = int(size[0]), int(size[1])
    if rdMolDraw2D is None:
        return None
    if mol is None:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="white"/></svg>'
        )
    else:
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
        except Exception:
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="white"/></svg>'
            )
    if add_invalid:
        insert = (
            f'<text x="{width / 2:.1f}" y="{height - 5:.1f}" '
            f'font-size="12" text-anchor="middle" fill="red">NV</text>'
        )
        if "</svg>" in svg:
            svg = svg.replace("</svg>", insert + "</svg>")
        else:
            svg = svg + insert
    return svg


def _set_square_axis(ax):
    try:
        ax.set_box_aspect(1.0)
    except AttributeError:
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass


def run_animation(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None = None,
    datamodule=None,
    dataset_infos=None,
    extra_features=None,
    domain_features=None,
    device: torch.device | None = None,
):
    anim_cfg = getattr(cfg, "animation", None)
    if anim_cfg is None:
        print("[warn] cfg.animation not found – nothing to do.")
        return
    if not bool(getattr(anim_cfg, "enabled", True)):
        print("[info] cfg.animation.enabled=false – skipping animation.")
        return

    _ensure_run_dir()
    _setup_file_logging("animate_energy.log")
    print("[info] Using animation config:\n" + OmegaConf.to_yaml(anim_cfg, resolve=True))

    model, datamodule, dataset_infos, extra_features, domain_features, device = _build_model_and_data(
        cfg,
        model=model,
        datamodule=datamodule,
        dataset_infos=dataset_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        device=device,
    )

    checkpoint_path = getattr(anim_cfg, "checkpoint", None)
    if not checkpoint_path:
        raise ValueError("Set viz.checkpoint=/path/to/model.pt.")
    _load_checkpoint_if_any(
        model,
        device,
        checkpoint_path,
        use_ema=bool(getattr(anim_cfg, "use_ema", False)),
    )

    seed_value = None
    seed_cfg = getattr(anim_cfg, "seed", getattr(cfg.general, "seed", None))
    if seed_cfg is not None:
        try:
            seed = int(seed_cfg)
            _set_seed(seed)
            seed_value = seed
            print(f"[info] Using seed={seed}")
        except (TypeError, ValueError):
            print(f"[warn] Invalid seed={seed_cfg!r}; ignoring.")

    viz_cfg = getattr(cfg, "viz", None)

    plot_metrics = bool(
        getattr(
            anim_cfg,
            "plot_metrics",
            getattr(viz_cfg, "plot_molecule_metrics", False),
        )
    )
    show_metric_panel = bool(getattr(anim_cfg, "show_metric_panel", True))
    metrics_only = bool(getattr(anim_cfg, "metrics_only", False))
    if metrics_only and not show_metric_panel:
        print("[warn] animation.show_metric_panel=false ignored because metrics_only=true.")
        show_metric_panel = True
    panel_enabled = plot_metrics and show_metric_panel
    plot_warmup_phase = bool(getattr(anim_cfg, "plot_warmup", True))
    show_noise_distribution_average = bool(getattr(anim_cfg, "show_noise_distribution_average", True))
    show_y_axis_labels = bool(getattr(anim_cfg, "show_y_axis_labels", True))
    log_stats_enabled = bool(getattr(cfg.general, "log_stats", False))
    legend_fontsize_raw = getattr(anim_cfg, "legend_fontsize", None)
    if legend_fontsize_raw is None and viz_cfg is not None:
        legend_fontsize_raw = getattr(viz_cfg, "legend_fontsize", None)
    try:
        legend_fontsize = float(legend_fontsize_raw)
    except (TypeError, ValueError):
        legend_fontsize = 10.0
    try:
        metric_line_width = float(getattr(anim_cfg, "metric_line_width", 2.0))
    except (TypeError, ValueError):
        metric_line_width = 2.0
    def _font_cfg(name: str, default: float) -> float:
        try:
            return float(getattr(anim_cfg, name, default))
        except (TypeError, ValueError):
            return default

    plot_title_fontsize = _font_cfg("plot_title_fontsize", 13.0)
    molecule_title_fontsize = _font_cfg("molecule_title_fontsize", 13.0)
    axis_label_fontsize = _font_cfg("axis_label_fontsize", 11.0)
    tick_fontsize = _font_cfg("tick_fontsize", 10.0)
    step_text_fontsize = _font_cfg("step_text_fontsize", 10.0)
    empty_panel_fontsize = _font_cfg("empty_panel_fontsize", 10.0)
    energy_ylabel_labelpad = getattr(anim_cfg, "energy_ylabel_labelpad", None)
    try:
        energy_ylabel_labelpad = None if energy_ylabel_labelpad is None else float(energy_ylabel_labelpad)
    except (TypeError, ValueError):
        energy_ylabel_labelpad = None
    if plot_metrics:
        dataset_smiles = moses_dataset.get_smiles(
            raw_dir=datamodule.train_dataset.raw_dir,
            filter_dataset=getattr(cfg.dataset, "filter", False),
        )
        train_smiles_seq = dataset_smiles.get("train") if dataset_smiles else None
        train_smiles_set = set(train_smiles_seq) if train_smiles_seq else None
    else:
        train_smiles_set = None

    N_noise = int(getattr(anim_cfg, "N_noise", getattr(viz_cfg, "N_noise", 4)))
    N_data = int(getattr(anim_cfg, "N_data", getattr(viz_cfg, "N_data", 4)))
    steps = int(getattr(anim_cfg, "steps", getattr(viz_cfg, "visualization_steps", 100)))
    record_every_raw = getattr(anim_cfg, "record_every", getattr(viz_cfg, "record_every", 10))
    try:
        record_every = max(int(record_every_raw), 1)
    except (TypeError, ValueError):
        record_every = 10
    proposal = str(getattr(anim_cfg, "proposal", getattr(viz_cfg, "proposal", "random")))
    gwd_beta = float(getattr(anim_cfg, "gwd_beta", getattr(viz_cfg, "gwd_beta", 1.0)))
    simple_n_edits_cfg = getattr(anim_cfg, "simple_n_edits", None)
    if simple_n_edits_cfg is None and viz_cfg is not None:
        simple_n_edits_cfg = getattr(viz_cfg, "simple_n_edits", None)
    simple_n_edits_main: int | None
    if simple_n_edits_cfg is None:
        simple_n_edits_main = None
    else:
        try:
            simple_n_edits_main = int(simple_n_edits_cfg)
        except (TypeError, ValueError):
            print(f"[warn] Invalid simple_n_edits={simple_n_edits_cfg!r}; ignoring override.")
            simple_n_edits_main = None
    base_dl_beta, base_dl_lambda_X, base_dl_lambda_E, dual_kwargs = resolve_dl_parameters(
        anim_cfg, viz_cfg
    )
    is_anneal_proposal = proposal.lower() in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}
    is_twobetas = proposal.lower() in TWO_BETA_PROPOSALS
    is_twobetas_anneal = proposal.lower() in TWO_BETA_ANNEALING_PROPOSALS
    anneal_kwargs_main: Dict[str, Any] = {}
    if is_anneal_proposal:
        beta_init_main = float(getattr(anim_cfg, "dl_beta_init", getattr(viz_cfg, "dl_beta_init", base_dl_beta)))
        beta_final_main = float(getattr(anim_cfg, "dl_beta_final", getattr(viz_cfg, "dl_beta_final", beta_init_main)))
        try:
            beta_anneal_steps_main = int(
                getattr(anim_cfg, "dl_beta_anneal_steps", getattr(viz_cfg, "dl_beta_anneal_steps", steps))
            )
        except (TypeError, ValueError):
            beta_anneal_steps_main = int(steps)
        anneal_kwargs_main.update(
            dl_beta_init=beta_init_main,
            dl_beta_final=beta_final_main,
            dl_beta_anneal_steps=beta_anneal_steps_main,
        )
        base_dl_beta = beta_final_main
        dual_kwargs = {}
    twobeta_kwargs: Dict[str, Any] = {}
    if is_twobetas or is_twobetas_anneal:
        beta_prop_main = getattr(anim_cfg, "dl_beta_prop", getattr(viz_cfg, "dl_beta_prop", None))
        if beta_prop_main is None:
            raise ValueError("dlangevintwobetas requires dl_beta_prop.")
        base_dl_beta = float(beta_prop_main)
        if is_twobetas:
            beta_mh_main = getattr(anim_cfg, "dl_beta_mh", getattr(viz_cfg, "dl_beta_mh", None))
            if beta_mh_main is None:
                raise ValueError("dlangevintwobetas requires dl_beta_mh.")
            twobeta_kwargs.update(
                dl_beta_prop=float(beta_prop_main),
                dl_beta_mh=float(beta_mh_main),
            )
        else:
            beta_mh_init_main = getattr(anim_cfg, "dl_beta_mh_init", getattr(viz_cfg, "dl_beta_mh_init", None))
            beta_mh_final_main = getattr(anim_cfg, "dl_beta_mh_final", getattr(viz_cfg, "dl_beta_mh_final", None))
            beta_mh_anneal_steps_main = getattr(
                anim_cfg, "dl_beta_mh_anneal_steps", getattr(viz_cfg, "dl_beta_mh_anneal_steps", steps)
            )
            missing = [k for k, v in [
                ("dl_beta_mh_init", beta_mh_init_main),
                ("dl_beta_mh_final", beta_mh_final_main),
                ("dl_beta_mh_anneal_steps", beta_mh_anneal_steps_main),
            ] if v is None]
            if missing:
                raise ValueError(f"dlangevintwobetas_annealing requires {', '.join(missing)}.")
            try:
                beta_mh_anneal_steps_main = int(beta_mh_anneal_steps_main)
            except (TypeError, ValueError):
                beta_mh_anneal_steps_main = int(steps)
            twobeta_kwargs.update(
                dl_beta_prop=float(beta_prop_main),
                dl_beta_mh_init=float(beta_mh_init_main),
                dl_beta_mh_final=float(beta_mh_final_main),
                dl_beta_mh_anneal_steps=beta_mh_anneal_steps_main,
            )
        dual_kwargs = {}
        anneal_kwargs_main.update(twobeta_kwargs)
    collect_main_stats = log_stats_enabled and not bool(dual_kwargs)
    main_stats_reason = None
    if log_stats_enabled and not collect_main_stats and (N_noise > 0 or N_data > 0):
        main_stats_reason = "dual DLangevin schedule disables stats collection."
    amp_dtype = getattr(anim_cfg, "amp_dtype", getattr(viz_cfg, "amp_dtype", None))
    chain_warmup_section = getattr(anim_cfg, "chain_warmup", None)
    fallback_chain_cfg = anim_cfg if chain_warmup_section is not None else viz_cfg
    if chain_warmup_section is None and viz_cfg is not None:
        chain_warmup_section = getattr(viz_cfg, "chain_warmup", None)
    chain_warmup_anim = resolve_chain_warmup(
        chain_warmup_section,
        fallback=fallback_chain_cfg,
        default_gwd_beta=gwd_beta,
    )
    vectorized_simple_warmup = sampler.should_vectorize_simple_warmup(
        chain_warmup_anim.proposal,
        vectorized=chain_warmup_anim.vectorized,
    )
    if chain_warmup_anim.enabled:
        print(
            "[info] Animation chain warmup: "
            f"proposal={chain_warmup_anim.proposal}, steps={chain_warmup_anim.steps}, "
            f"vectorized={vectorized_simple_warmup}."
        )
    warmup_steps_total = int(chain_warmup_anim.steps if chain_warmup_anim.enabled else 0)
    record_warmup_phase = plot_warmup_phase and warmup_steps_total > 0
    collect_warmup_stats = log_stats_enabled and chain_warmup_anim.enabled and not bool(chain_warmup_anim.dual_kwargs)
    warmup_stats_reason = None
    if chain_warmup_anim.enabled and log_stats_enabled and not collect_warmup_stats:
        warmup_stats_reason = "chain warmup uses DLangevin dual schedule; stats unavailable."

    record_steps_cfg = _parse_int_list(getattr(anim_cfg, "record_steps", None))
    chain_grid_steps_cfg = _parse_int_list(getattr(anim_cfg, "chain_grid_steps", None))
    record_steps_set = None
    if record_steps_cfg is None and chain_grid_steps_cfg is not None:
        record_steps_cfg = list(chain_grid_steps_cfg)
    if record_steps_cfg is not None:
        total_steps_limit = warmup_steps_total + max(int(steps), 0)
        record_steps_set = {int(s) for s in record_steps_cfg if int(s) >= 0}
        if total_steps_limit > 0:
            record_steps_set = {s for s in record_steps_set if s <= total_steps_limit}
        record_steps_set.add(0)
        print(f"[info] Recording {len(record_steps_set)} selected steps.")

    noise_node_count = None
    noise_node_count_cfg = getattr(anim_cfg, "noise_node_count", None)
    if noise_node_count_cfg is None and viz_cfg is not None:
        noise_node_count_cfg = getattr(viz_cfg, "noise_node_count", None)
    if noise_node_count_cfg is not None:
        if isinstance(noise_node_count_cfg, str) and noise_node_count_cfg.strip().lower() in {
            "mode",
            "most_likely",
            "most_likely_nodes",
        }:
            noise_node_count = _most_likely_node_count(dataset_infos)
        else:
            try:
                noise_node_count = int(noise_node_count_cfg)
            except (TypeError, ValueError):
                noise_node_count = None
        if noise_node_count is not None and noise_node_count <= 0:
            print(f"[warn] Invalid noise_node_count={noise_node_count}; falling back to dataset distribution.")
            noise_node_count = None
    if noise_node_count is not None:
        print(f"[info] Using fixed noise node count={noise_node_count}")
    stats_sum_keys = {
        "total_proposals",
        "total_accepted",
        "nontriv_any",
        "nontriv_node",
        "nontriv_edge",
        "acc_nontriv_any",
        "acc_nontriv_node",
        "acc_nontriv_edge",
        "prop_dist_nodes_sum",
        "prop_dist_edges_sum",
        "acc_dist_nodes_sum",
        "acc_dist_edges_sum",
        "step_prop_nodes_sum",
        "step_prop_edges_sum",
        "step_acc_nodes_sum",
        "step_acc_edges_sum",
        "distance_total_nodes",
        "distance_total_edges",
        "distance_total",
    }

    def _accumulate_stats(acc, new_stats):
        if not new_stats:
            return acc
        if acc is None:
            acc = {}
        for key in stats_sum_keys:
            if key in new_stats:
                acc[key] = acc.get(key, 0.0) + float(new_stats[key])
        return acc

    def _summarize_stats(acc):
        if not acc:
            return None
        stats = {k: float(v) for k, v in acc.items()}
        total_props = stats.get("total_proposals", 0.0)
        if total_props <= 0:
            return None
        total_acc = stats.get("total_accepted", 0.0)

        def _ratio(num_key, denom_key):
            denom = stats.get(denom_key, 0.0)
            if denom <= 0:
                return 0.0
            return stats.get(num_key, 0.0) / denom

        def _mean(sum_key, denom):
            if denom <= 0:
                return 0.0
            return stats.get(sum_key, 0.0) / denom

        summary = dict(
            total_proposals=total_props,
            total_accepted=total_acc,
            overall_accept=(total_acc / total_props) if total_props > 0 else 0.0,
            accept_nontrivial_any=_ratio("acc_nontriv_any", "nontriv_any"),
            accept_nontrivial_node=_ratio("acc_nontriv_node", "nontriv_node"),
            accept_nontrivial_edge=_ratio("acc_nontriv_edge", "nontriv_edge"),
            mean_prop_distance_nodes=_mean("prop_dist_nodes_sum", total_props),
            mean_prop_distance_edges=_mean("prop_dist_edges_sum", total_props),
        )
        summary["mean_prop_distance_total"] = summary["mean_prop_distance_nodes"] + summary["mean_prop_distance_edges"]
        summary["mean_step_distance_nodes"] = _mean("step_prop_nodes_sum", total_props)
        summary["mean_step_distance_edges"] = _mean("step_prop_edges_sum", total_props)
        summary["mean_step_distance_total"] = summary["mean_step_distance_nodes"] + summary["mean_step_distance_edges"]

        acc_denom = total_acc if total_acc > 0 else 0.0
        summary["mean_acc_distance_nodes"] = _mean("acc_dist_nodes_sum", acc_denom)
        summary["mean_acc_distance_edges"] = _mean("acc_dist_edges_sum", acc_denom)
        summary["mean_acc_distance_total"] = summary["mean_acc_distance_nodes"] + summary["mean_acc_distance_edges"]
        summary["mean_step_acc_distance_nodes"] = _mean("step_acc_nodes_sum", acc_denom)
        summary["mean_step_acc_distance_edges"] = _mean("step_acc_edges_sum", acc_denom)
        summary["mean_step_acc_distance_total"] = (
            summary["mean_step_acc_distance_nodes"] + summary["mean_step_acc_distance_edges"]
        )
        summary["move_nodes"] = summary["overall_accept"] * summary["mean_step_acc_distance_nodes"]
        summary["move_edges"] = summary["overall_accept"] * summary["mean_step_acc_distance_edges"]
        summary["move_total"] = summary["overall_accept"] * summary["mean_step_acc_distance_total"]
        return summary

    def _log_stats(phase_name, acc_stats, *, n_noise, n_data, steps_total, reason_disabled=None):
        if acc_stats:
            summary = _summarize_stats(acc_stats)
            if summary is None:
                return
            msg = (
                f"[stats|{phase_name}] chains(noise/data)={n_noise}/{n_data} "
                f"| steps={steps_total} | proposals={summary['total_proposals']:.0f} "
                f"| acc={summary['overall_accept'] * 100:.1f}%"
                f" | acc_nontriv(any,node,edge)="
                f"{summary['accept_nontrivial_any'] * 100:.1f}%/"
                f"{summary['accept_nontrivial_node'] * 100:.1f}%/"
                f"{summary['accept_nontrivial_edge'] * 100:.1f}%"
                f" | prop_d(n/e/t)="
                f"{summary['mean_prop_distance_nodes']:.2f}/"
                f"{summary['mean_prop_distance_edges']:.2f}/"
                f"{summary['mean_prop_distance_total']:.2f}"
                f" | acc_d(n/e/t)="
                f"{summary['mean_acc_distance_nodes']:.2f}/"
                f"{summary['mean_acc_distance_edges']:.2f}/"
                f"{summary['mean_acc_distance_total']:.2f}"
                f" | step_d(n/e/t)="
                f"{summary['mean_step_acc_distance_nodes']:.2f}/"
                f"{summary['mean_step_acc_distance_edges']:.2f}/"
                f"{summary['mean_step_acc_distance_total']:.2f}"
                f" | move_per_step(n/e/t)="
                f"{summary['move_nodes']:.2f}/"
                f"{summary['move_edges']:.2f}/"
                f"{summary['move_total']:.2f}"
            )
            print(msg)
        elif reason_disabled and log_stats_enabled and (n_noise > 0 or n_data > 0):
            print(f"[info] MCMC stats for {phase_name} unavailable: {reason_disabled}")
    single_group = (N_noise > 0) ^ (N_data > 0)
    noise_transition = str(
        getattr(
            anim_cfg,
            "noise_transition",
            getattr(viz_cfg, "noise_transition", getattr(cfg.model, "transition", "marginal")),
        )
    )
    band_M_data = int(getattr(anim_cfg, "band_M_data", getattr(viz_cfg, "M_data", max(64, N_data))))
    band_M_noise = int(getattr(anim_cfg, "band_M_noise", getattr(viz_cfg, "M_noise", max(64, N_noise))))
    if N_noise <= 0:
        band_M_noise = 0
    display_molecules = bool(getattr(anim_cfg, "display_molecules", True)) and _HAS_RDKIT
    if bool(getattr(anim_cfg, "display_molecules", True)) and not _HAS_RDKIT:
        print("[warn] RDKit not available – molecule panel disabled.")
    max_display = int(getattr(anim_cfg, "max_display", 8))
    mols_per_row = int(getattr(anim_cfg, "mols_per_row", 4))
    display_noise_indices_cfg = None
    display_data_indices_cfg = None
    selection_policy = str(getattr(anim_cfg, "selection_policy", "all")).strip().lower()
    valid_selection_policies = {"all", "random"}
    if selection_policy not in valid_selection_policies:
        raise ValueError(
            "animation.selection_policy must be one of "
            f"{sorted(valid_selection_policies)}, got {selection_policy!r}."
        )
    if selection_policy == "random":
        random_display_count = int(getattr(anim_cfg, "random_display_count", max_display))
        random_display_count = max(1, random_display_count)
        if N_noise > 0 and N_data > 0:
            n_noise_random = min(N_noise, max(1, random_display_count // 2))
            n_data_random = min(N_data, max(1, random_display_count - n_noise_random))
            remaining_random = random_display_count - (n_noise_random + n_data_random)
            if remaining_random > 0 and n_noise_random < N_noise:
                take = min(remaining_random, N_noise - n_noise_random)
                n_noise_random += take
                remaining_random -= take
            if remaining_random > 0 and n_data_random < N_data:
                n_data_random += min(remaining_random, N_data - n_data_random)
        elif N_noise > 0:
            n_noise_random = min(N_noise, random_display_count)
            n_data_random = 0
        else:
            n_noise_random = 0
            n_data_random = min(N_data, random_display_count)
        display_noise_indices_cfg = _random_display_indices(N_noise, n_noise_random, "noise")
        display_data_indices_cfg = _random_display_indices(N_data, n_data_random, "data")
    subimg_size_cfg = getattr(anim_cfg, "subimg_size", (260, 260))
    if isinstance(subimg_size_cfg, (list, tuple)) and len(subimg_size_cfg) >= 2:
        mol_subimg_size = (int(subimg_size_cfg[0]), int(subimg_size_cfg[1]))
    else:
        mol_subimg_size = (260, 260)
    def _optional_float_cfg(name: str) -> float | None:
        value = getattr(anim_cfg, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _optional_int_cfg(name: str) -> int | None:
        value = getattr(anim_cfg, name, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    molecule_draw_padding = _optional_float_cfg("molecule_draw_padding")
    molecule_bond_line_width = _optional_float_cfg("molecule_bond_line_width")
    molecule_min_font_size = _optional_int_cfg("molecule_min_font_size")
    molecule_max_font_size = _optional_int_cfg("molecule_max_font_size")
    molecule_draw_same_scale_cfg = getattr(anim_cfg, "molecule_draw_same_scale", None)
    molecule_draw_same_scale = None if molecule_draw_same_scale_cfg is None else bool(molecule_draw_same_scale_cfg)
    use_tight_layout = bool(getattr(anim_cfg, "use_tight_layout", True))
    gridspec_kwargs = {}
    for cfg_name, gridspec_name in (
        ("layout_left", "left"),
        ("layout_right", "right"),
        ("layout_top", "top"),
        ("layout_bottom", "bottom"),
    ):
        value = getattr(anim_cfg, cfg_name, None)
        if value is not None:
            try:
                gridspec_kwargs[gridspec_name] = float(value)
            except (TypeError, ValueError):
                pass
    molecule_legends_cfg = getattr(anim_cfg, "molecule_legends", None)
    if molecule_legends_cfg is None and viz_cfg is not None:
        molecule_legends_cfg = getattr(viz_cfg, "molecule_legends", None)
    show_molecule_legends = (not single_group) if molecule_legends_cfg is None else bool(molecule_legends_cfg)
    molecule_title_cfg = getattr(anim_cfg, "molecule_title", None)
    if molecule_title_cfg is None and viz_cfg is not None:
        molecule_title_cfg = getattr(viz_cfg, "molecule_title", None)
    if molecule_title_cfg is None:
        if N_noise > 0 and N_data <= 0:
            molecule_title = "Samples initialized from noise"
        elif N_data > 0 and N_noise <= 0:
            molecule_title = "Samples initialized from data"
        else:
            molecule_title = "Molecule snapshots"
    else:
        molecule_title = str(molecule_title_cfg)

    def _actual_step_for_frame(frame: int) -> int:
        if history_steps:
            return int(history_steps[min(frame, len(history_steps) - 1)])
        return int(frame)

    def _molecule_panel_title(actual_step: int | None = None) -> str:
        return molecule_title or ""

    def _set_molecule_panel_title(ax, actual_step: int | None = None) -> None:
        ax.set_title(_molecule_panel_title(actual_step), fontsize=molecule_title_fontsize, linespacing=1.25)
    molecule_square_panel = bool(getattr(anim_cfg, "molecule_square_panel", False))

    panel_width_ratios_cfg = getattr(anim_cfg, "panel_width_ratios", None)
    metrics_panel_hspace = float(getattr(anim_cfg, "metrics_panel_hspace", 0.18))

    # Collect data graphs for both the static band and the animated chains.
    data_graph_nodes, data_graph_edges = _collect_graphs_from_data(
        datamodule, max(band_M_data, N_data)
    )
    data_band_nodes = data_graph_nodes[:band_M_data]
    data_band_edges = data_graph_edges[:band_M_data]
    data_nodes = data_graph_nodes[:N_data]
    data_edges = data_graph_edges[:N_data]
    data_tau_raw = getattr(anim_cfg, "data_tau", getattr(viz_cfg, "data_tau", 1.0))
    try:
        data_tau = float(data_tau_raw)
    except (TypeError, ValueError):
        data_tau = 1.0
    data_tau = min(max(data_tau, 0.0), 1.0)
    if data_tau < 1.0 and data_nodes:
        print(f"[info] Applying data_tau interpolation for animation data chains: tau={data_tau:.3f}")
        counts_interp = [int(nt.shape[0]) for nt in data_nodes]
        noise_interp_graphs = initialize_random_graphs_with_counts(
            counts=counts_interp,
            dataset_info=dataset_infos,
            device=torch.device("cpu"),
            transition=noise_transition,
        )
        mixed_nodes: List[torch.Tensor] = []
        mixed_edges: List[torch.Tensor] = []
        for (nt_noise, et_noise), nt_data, et_data in zip(noise_interp_graphs, data_nodes, data_edges):
            nt_i, et_i = sample_interpolated_graph(nt_noise, et_noise, nt_data, et_data, data_tau)
            mixed_nodes.append(nt_i)
            mixed_edges.append(et_i)
        data_nodes = mixed_nodes
        data_edges = mixed_edges

    if band_M_noise > 0:
        noise_band_graphs = sampler.initialize_random_graphs(
            batch_size=band_M_noise,
            dataset_info=dataset_infos,
            device=device,
            transition=noise_transition,
        )
        noise_band_nodes = [nt for (nt, _) in noise_band_graphs]
        noise_band_edges = [et for (_, et) in noise_band_graphs]
    else:
        noise_band_nodes = []
        noise_band_edges = []

    if seed_value is not None:
        # Keep animated chain initialization reproducible even when band sampling changes.
        _set_seed(seed_value)

    if N_noise > 0:
        if noise_node_count is not None:
            noise_graphs = initialize_random_graphs_with_counts(
                counts=[noise_node_count] * N_noise,
                dataset_info=dataset_infos,
                device=device,
                transition=noise_transition,
            )
        else:
            noise_graphs = sampler.initialize_random_graphs(
                batch_size=N_noise,
                dataset_info=dataset_infos,
                device=device,
                transition=noise_transition,
            )
        noise_nodes = [nt for (nt, _) in noise_graphs]
        noise_edges = [et for (_, et) in noise_graphs]
    else:
        noise_nodes = []
        noise_edges = []

    warmup_stats_accum = None
    if chain_warmup_anim.enabled and not record_warmup_phase:
        def _apply_chain_warmup(nodes_seq, edges_seq):
            collect_flag = collect_warmup_stats
            if vectorized_simple_warmup:
                edits_per_step = (
                    5
                    if chain_warmup_anim.simple_n_edits is None
                    else int(chain_warmup_anim.simple_n_edits)
                )
                nodes_w, edges_w, _, _, stats = (
                    sampler.run_simple_v2_warmup_vectorized(
                        model=model,
                        dataset_info=dataset_infos,
                        node_types_list=nodes_seq,
                        edge_types_list=edges_seq,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        steps=chain_warmup_anim.steps,
                        device=device,
                        edits_per_step=edits_per_step,
                        amp_dtype=amp_dtype,
                        stop_when_unchanged=False,
                        collect_stats=collect_flag,
                    )
                )
            else:
                nodes_w, edges_w, _, _, stats = sampler.mcmc_sample_batch(
                    model=model,
                    dataset_info=dataset_infos,
                    node_types_list=nodes_seq,
                    edge_types_list=edges_seq,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    steps=chain_warmup_anim.steps,
                    device=device,
                    proposal=chain_warmup_anim.proposal,
                    gwd_beta=chain_warmup_anim.gwd_beta,
                    dl_beta=chain_warmup_anim.dl_beta,
                    dl_lambda_X=chain_warmup_anim.dl_lambda_X,
                    dl_lambda_E=chain_warmup_anim.dl_lambda_E,
                    simple_n_edits=chain_warmup_anim.simple_n_edits,
                    amp_dtype=amp_dtype,
                    collect_stats=collect_flag,
                    **chain_warmup_anim.dual_kwargs,
                )
            stats_out = stats if collect_flag else {}
            return nodes_w, edges_w, stats_out

        if N_noise > 0 and noise_nodes:
            noise_nodes, noise_edges, stats = _apply_chain_warmup(noise_nodes, noise_edges)
            warmup_stats_accum = _accumulate_stats(warmup_stats_accum, stats)
        if N_data > 0 and data_nodes:
            data_nodes, data_edges, stats = _apply_chain_warmup(data_nodes, data_edges)
            warmup_stats_accum = _accumulate_stats(warmup_stats_accum, stats)
        _log_stats(
            "warmup",
            warmup_stats_accum,
            n_noise=N_noise,
            n_data=N_data,
            steps_total=warmup_steps_total,
            reason_disabled=warmup_stats_reason,
        )

    data_band_stats = _safe_energy_band_stats(
        data_band_nodes,
        data_band_edges,
        model=model,
        dataset_info=dataset_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        device=device,
        label="data",
    )
    noise_band_stats = _safe_energy_band_stats(
        noise_band_nodes,
        noise_band_edges,
        model=model,
        dataset_info=dataset_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        device=device,
        label="noise",
    )
    data_band_mean: float | None
    data_band_std: float | None
    if data_band_stats is not None:
        data_band_mean, data_band_std = data_band_stats
        print(
            f"[info] Data band: mean={data_band_mean:.4f} ± {data_band_std:.4f} over {len(data_band_nodes)} graphs"
        )
    else:
        data_band_mean = data_band_std = None
    if noise_band_stats is not None:
        noise_band_mean, noise_band_std = noise_band_stats
        print(
            f"[info] Noise band: mean={noise_band_mean:.4f} ± {noise_band_std:.4f} over {len(noise_band_nodes)} graphs"
        )
    else:
        noise_band_mean = noise_band_std = None

    history_steps: List[int] = []
    noise_history_list: List[np.ndarray] = []
    data_history_list: List[np.ndarray] = []
    metrics_noise_list: List[np.ndarray] = []
    metrics_data_list: List[np.ndarray] = []
    metrics_noise = metrics_data = None
    noise_node_history: List[List[torch.Tensor]] = []
    noise_edge_history: List[List[torch.Tensor]] = []
    data_node_history: List[List[torch.Tensor]] = []
    data_edge_history: List[List[torch.Tensor]] = []
    main_stats_accum = None

    def _record_state(step_idx: int) -> None:
        history_steps.append(step_idx)

        if N_noise > 0:
            noise_history_list.append(
                _stepwise_energy(
                    model=model,
                    dataset_info=dataset_infos,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    device=device,
                    node_types=noise_nodes,
                    edge_types=noise_edges,
                )
            )
            nodes_snapshot, edges_snapshot = _clone_graphs(noise_nodes, noise_edges)
            noise_node_history.append(nodes_snapshot)
            noise_edge_history.append(edges_snapshot)
            if plot_metrics:
                molecules_noise = list(
                    zip(
                        _take_indices(nodes_snapshot, display_noise_indices_cfg),
                        _take_indices(edges_snapshot, display_noise_indices_cfg),
                    )
                )
                metrics_noise_list.append(
                    np.asarray(
                        _compute_metrics(
                            molecules_noise,
                            dataset_infos,
                            train_smiles_set,
                        ),
                        dtype=np.float64,
                    )
                )
        else:
            noise_history_list.append(np.zeros((max(N_noise, 1),), dtype=np.float64))
            noise_node_history.append([])
            noise_edge_history.append([])
            if plot_metrics:
                metrics_noise_list.append(np.zeros((3,), dtype=np.float64))

        if N_data > 0:
            data_history_list.append(
                _stepwise_energy(
                    model=model,
                    dataset_info=dataset_infos,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    device=device,
                    node_types=data_nodes,
                    edge_types=data_edges,
                )
            )
            nodes_snapshot, edges_snapshot = _clone_graphs(data_nodes, data_edges)
            data_node_history.append(nodes_snapshot)
            data_edge_history.append(edges_snapshot)
            if plot_metrics:
                molecules_data = list(
                    zip(
                        _take_indices(nodes_snapshot, display_data_indices_cfg),
                        _take_indices(edges_snapshot, display_data_indices_cfg),
                    )
                )
                metrics_data_list.append(
                    np.asarray(
                        _compute_metrics(
                            molecules_data,
                            dataset_infos,
                            train_smiles_set,
                        ),
                        dtype=np.float64,
                    )
                )
        else:
            data_history_list.append(np.zeros((max(N_data, 1),), dtype=np.float64))
            data_node_history.append([])
            data_edge_history.append([])
            if plot_metrics:
                metrics_data_list.append(np.zeros((3,), dtype=np.float64))

    def _mcmc_step(
        nodes_seq: list[torch.Tensor],
        edges_seq: list[torch.Tensor],
        *,
        proposal_name: str,
        gwd_beta_value: float,
        dl_beta_value: float,
        dl_lambda_X_value: float,
        dl_lambda_E_value: float,
        simple_edits: int | None = None,
        extra_dual_kwargs: dict | None = None,
        collect_stats: bool = False,
        anneal_kwargs: dict | None = None,
        step_offset: int = 0,
        vectorized_simple: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        if not nodes_seq:
            return nodes_seq, edges_seq, {}
        collect_flag = bool(collect_stats) and not (
            extra_dual_kwargs and len(extra_dual_kwargs) > 0
        )
        if sampler.should_vectorize_simple_warmup(
            proposal_name,
            vectorized=vectorized_simple,
        ):
            edits_per_step = 5 if simple_edits is None else int(simple_edits)
            nodes_out, edges_out, _, _, stats = (
                sampler.run_simple_v2_warmup_vectorized(
                    model=model,
                    dataset_info=dataset_infos,
                    node_types_list=nodes_seq,
                    edge_types_list=edges_seq,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    steps=1,
                    device=device,
                    edits_per_step=edits_per_step,
                    amp_dtype=amp_dtype,
                    stop_when_unchanged=False,
                    collect_stats=collect_flag,
                )
            )
            return nodes_out, edges_out, stats if collect_flag else {}
        kwargs = dict(
            model=model,
            dataset_info=dataset_infos,
            node_types_list=nodes_seq,
            edge_types_list=edges_seq,
            extra_features=extra_features,
            domain_features=domain_features,
            steps=1,
            device=device,
            proposal=proposal_name,
            gwd_beta=gwd_beta_value,
            dl_beta=dl_beta_value,
            dl_lambda_X=dl_lambda_X_value,
            dl_lambda_E=dl_lambda_E_value,
            amp_dtype=amp_dtype,
            collect_stats=False,
            step_offset=int(step_offset),
        )
        if simple_edits is not None:
            kwargs["simple_n_edits"] = simple_edits
        if extra_dual_kwargs:
            kwargs.update(extra_dual_kwargs)
        if anneal_kwargs:
            kwargs.update(anneal_kwargs)
        kwargs["collect_stats"] = collect_flag
        nodes_out, edges_out, _, _, stats = sampler.mcmc_sample_batch(**kwargs)
        stats_out = stats if collect_flag else {}
        return nodes_out, edges_out, stats_out

    global_step_idx = 0
    if record_steps_set is None or global_step_idx in record_steps_set:
        _record_state(global_step_idx)

    if record_warmup_phase:
        for warm_step in range(1, warmup_steps_total + 1):
            if N_noise > 0:
                noise_nodes, noise_edges, stats = _mcmc_step(
                    noise_nodes,
                    noise_edges,
                    proposal_name=chain_warmup_anim.proposal,
                    gwd_beta_value=chain_warmup_anim.gwd_beta,
                    dl_beta_value=chain_warmup_anim.dl_beta,
                    dl_lambda_X_value=chain_warmup_anim.dl_lambda_X,
                    dl_lambda_E_value=chain_warmup_anim.dl_lambda_E,
                    simple_edits=chain_warmup_anim.simple_n_edits,
                    extra_dual_kwargs=chain_warmup_anim.dual_kwargs,
                    collect_stats=collect_warmup_stats,
                    vectorized_simple=vectorized_simple_warmup,
                )
                warmup_stats_accum = _accumulate_stats(warmup_stats_accum, stats)
            if N_data > 0:
                data_nodes, data_edges, stats = _mcmc_step(
                    data_nodes,
                    data_edges,
                    proposal_name=chain_warmup_anim.proposal,
                    gwd_beta_value=chain_warmup_anim.gwd_beta,
                    dl_beta_value=chain_warmup_anim.dl_beta,
                    dl_lambda_X_value=chain_warmup_anim.dl_lambda_X,
                    dl_lambda_E_value=chain_warmup_anim.dl_lambda_E,
                    simple_edits=chain_warmup_anim.simple_n_edits,
                    extra_dual_kwargs=chain_warmup_anim.dual_kwargs,
                    collect_stats=collect_warmup_stats,
                    vectorized_simple=vectorized_simple_warmup,
                )
                warmup_stats_accum = _accumulate_stats(warmup_stats_accum, stats)
            global_step_idx += 1
            if record_steps_set is None:
                if (global_step_idx % record_every == 0) or (warm_step == warmup_steps_total):
                    _record_state(global_step_idx)
            elif global_step_idx in record_steps_set:
                _record_state(global_step_idx)
            if warm_step % 10 == 0 or warm_step == warmup_steps_total:
                print(f"[info] Completed warmup step {warm_step}/{warmup_steps_total}")
        _log_stats(
            "warmup",
            warmup_stats_accum,
            n_noise=N_noise,
            n_data=N_data,
            steps_total=warmup_steps_total,
            reason_disabled=warmup_stats_reason,
        )

    for step in range(1, steps + 1):
        if N_noise > 0:
            noise_nodes, noise_edges, stats = _mcmc_step(
                noise_nodes,
                noise_edges,
                proposal_name=proposal,
                gwd_beta_value=gwd_beta,
                dl_beta_value=base_dl_beta,
                dl_lambda_X_value=base_dl_lambda_X,
                dl_lambda_E_value=base_dl_lambda_E,
                simple_edits=simple_n_edits_main,
                extra_dual_kwargs=dual_kwargs,
                collect_stats=collect_main_stats,
                anneal_kwargs=anneal_kwargs_main,
                step_offset=step - 1,
            )
            main_stats_accum = _accumulate_stats(main_stats_accum, stats)
        if N_data > 0:
            data_nodes, data_edges, stats = _mcmc_step(
                data_nodes,
                data_edges,
                proposal_name=proposal,
                gwd_beta_value=gwd_beta,
                dl_beta_value=base_dl_beta,
                dl_lambda_X_value=base_dl_lambda_X,
                dl_lambda_E_value=base_dl_lambda_E,
                simple_edits=simple_n_edits_main,
                extra_dual_kwargs=dual_kwargs,
                collect_stats=collect_main_stats,
                anneal_kwargs=anneal_kwargs_main,
                step_offset=step - 1,
            )
            main_stats_accum = _accumulate_stats(main_stats_accum, stats)
        global_step_idx += 1
        if record_steps_set is None:
            if (global_step_idx % record_every == 0) or (step == steps):
                _record_state(global_step_idx)
        elif global_step_idx in record_steps_set:
            _record_state(global_step_idx)
        if step % 10 == 0 or step == steps:
            print(f"[info] Completed MCMC step {step}/{steps}")
    _log_stats(
        "mcmc",
        main_stats_accum,
        n_noise=N_noise,
        n_data=N_data,
        steps_total=steps,
        reason_disabled=main_stats_reason,
    )

    history_len = len(history_steps)
    xs = np.array(history_steps, dtype=np.int64)
    total_mcmc_steps = int(history_steps[-1]) if history_steps else 0
    max_frames = history_len
    if noise_history_list:
        max_frames = min(max_frames, len(noise_history_list))
    if data_history_list:
        max_frames = min(max_frames, len(data_history_list))
    if noise_node_history:
        max_frames = min(max_frames, len(noise_node_history))
    if data_node_history:
        max_frames = min(max_frames, len(data_node_history))
    if max_frames <= 0:
        max_frames = history_len

    noise_history = (
        np.stack(noise_history_list, axis=0)
        if noise_history_list
        else np.zeros((0, max(N_noise, 1)), dtype=np.float64)
    )
    data_history = (
        np.stack(data_history_list, axis=0)
        if data_history_list
        else np.zeros((0, max(N_data, 1)), dtype=np.float64)
    )
    if plot_metrics:
        metrics_noise = (
            np.stack(metrics_noise_list, axis=0)
            if metrics_noise_list
            else np.zeros((history_len, 3), dtype=np.float64)
        )
        metrics_data = (
            np.stack(metrics_data_list, axis=0)
            if metrics_data_list
            else np.zeros((history_len, 3), dtype=np.float64)
        )
    else:
        metrics_noise = metrics_data = None

    noise_history = noise_history[:, :N_noise]
    data_history = data_history[:, :N_data]
    num_noise_series_all = int(noise_history.shape[1]) if noise_history.ndim == 2 else 0
    num_data_series_all = int(data_history.shape[1]) if data_history.ndim == 2 else 0
    display_noise_indices = _sanitize_indices(display_noise_indices_cfg, num_noise_series_all, "noise")
    display_data_indices = _sanitize_indices(display_data_indices_cfg, num_data_series_all, "data")
    if display_noise_indices is not None and noise_history.ndim == 2:
        print(f"[info] Displaying selected noise chains: {display_noise_indices}")
        noise_history = noise_history[:, display_noise_indices]
    if display_data_indices is not None and data_history.ndim == 2:
        print(f"[info] Displaying selected data chains: {display_data_indices}")
        data_history = data_history[:, display_data_indices]
    num_noise_series = int(noise_history.shape[1]) if noise_history.ndim == 2 else 0
    num_data_series = int(data_history.shape[1]) if data_history.ndim == 2 else 0
    noise_mean = noise_history.mean(axis=1) if num_noise_series > 0 else None
    noise_std = noise_history.std(axis=1) if num_noise_series > 0 else None
    data_mean = data_history.mean(axis=1) if num_data_series > 0 else None
    data_std = data_history.std(axis=1) if num_data_series > 0 else None
    include_band = bool(getattr(anim_cfg, "include_band", True))
    show_band_spans = bool(getattr(anim_cfg, "show_band_spans", True))

    chain_grid_enabled = bool(getattr(anim_cfg, "chain_grid_enabled", False))
    if chain_grid_enabled:
        chain_grid_steps = chain_grid_steps_cfg or record_steps_cfg
        if chain_grid_steps is None:
            print("[warn] chain_grid_enabled=true but no chain_grid_steps provided.")
        else:
            chain_grid_rows = int(getattr(anim_cfg, "chain_grid_rows", N_noise))
            chain_grid_subimg = getattr(anim_cfg, "chain_grid_subimg_size", None)
            if isinstance(chain_grid_subimg, (list, tuple)) and len(chain_grid_subimg) >= 2:
                chain_grid_subimg_size = (int(chain_grid_subimg[0]), int(chain_grid_subimg[1]))
            else:
                chain_grid_subimg_size = (220, 220)
            chain_grid_labels = bool(getattr(anim_cfg, "chain_grid_step_labels", True))
            chain_grid_label_fontsize = float(getattr(anim_cfg, "chain_grid_label_fontsize", 8))
            chain_grid_nv_fontsize = float(getattr(anim_cfg, "chain_grid_nv_fontsize", 8))
            chain_grid_dpi = int(getattr(anim_cfg, "chain_grid_dpi", 100))
            chain_grid_row_labels = bool(getattr(anim_cfg, "chain_grid_row_labels", False))
            chain_grid_row_label_prefix = str(getattr(anim_cfg, "chain_grid_row_label_prefix", "row"))
            chain_grid_row_label_fontsize = float(getattr(anim_cfg, "chain_grid_row_label_fontsize", 8))
            chain_grid_save_cells = bool(getattr(anim_cfg, "chain_grid_save_cells", False))
            chain_grid_cells_dir = getattr(anim_cfg, "chain_grid_cells_dir", None)
            chain_grid_cell_prefix = str(getattr(anim_cfg, "chain_grid_cell_prefix", "mol"))
            chain_grid_mark_invalid = bool(getattr(anim_cfg, "chain_grid_cells_mark_invalid", True))
            chain_grid_cells_format = str(getattr(anim_cfg, "chain_grid_cells_format", "png"))
            chain_grid_path = Path(getattr(anim_cfg, "chain_grid_filename", "chain_grid.png")).resolve()
            grid_img = _render_chain_grid(
                noise_node_history=noise_node_history,
                noise_edge_history=noise_edge_history,
                history_steps=history_steps,
                dataset_info=dataset_infos,
                step_values=chain_grid_steps,
                chain_count=chain_grid_rows,
                subimg_size=chain_grid_subimg_size,
                add_step_labels=chain_grid_labels,
                warmup_steps_total=warmup_steps_total,
                title_fontsize=chain_grid_label_fontsize,
                nv_fontsize=chain_grid_nv_fontsize,
                dpi=chain_grid_dpi,
                add_row_labels=chain_grid_row_labels,
                row_label_prefix=chain_grid_row_label_prefix,
                row_label_fontsize=chain_grid_row_label_fontsize,
                save_cells=chain_grid_save_cells,
                cells_dir=Path(chain_grid_cells_dir) if chain_grid_cells_dir else None,
                cell_prefix=chain_grid_cell_prefix,
                mark_invalid=chain_grid_mark_invalid,
                cells_format=chain_grid_cells_format,
            )
            if grid_img is not None:
                try:
                    plt.imsave(chain_grid_path, grid_img)
                    print(f"[info] Saved chain grid to {chain_grid_path}")
                except Exception as exc:
                    print(f"[warn] Failed to save chain grid: {exc}")

    energy_min_candidates: list[float] = []
    energy_max_candidates: list[float] = []
    if num_noise_series > 0:
        energy_min_candidates.append(float(np.min(noise_history)))
        energy_max_candidates.append(float(np.max(noise_history)))
    if num_data_series > 0:
        energy_min_candidates.append(float(np.min(data_history)))
        energy_max_candidates.append(float(np.max(data_history)))
    if data_band_mean is not None:
        if show_band_spans and data_band_std is not None:
            energy_min_candidates.append(data_band_mean - data_band_std)
            energy_max_candidates.append(data_band_mean + data_band_std)
        else:
            energy_min_candidates.append(data_band_mean)
            energy_max_candidates.append(data_band_mean)
    if noise_band_mean is not None:
        if show_band_spans and noise_band_std is not None:
            energy_min_candidates.append(noise_band_mean - noise_band_std)
            energy_max_candidates.append(noise_band_mean + noise_band_std)
        else:
            energy_min_candidates.append(noise_band_mean)
            energy_max_candidates.append(noise_band_mean)
    if energy_min_candidates and energy_max_candidates:
        energy_min = min(energy_min_candidates)
        energy_max = max(energy_max_candidates)
    else:
        energy_min, energy_max = -1.0, 1.0
    margin = 0.05 * max(1e-6, energy_max - energy_min)

    def _cfg_float(section, key, default):
        val = getattr(section, key, None) if section is not None else None
        if val is None and viz_cfg is not None and section is not viz_cfg:
            val = getattr(viz_cfg, key, None)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    line_width = _cfg_float(anim_cfg, "line_width", 1.5)
    mean_line_width = _cfg_float(anim_cfg, "mean_line_width", 1.0)
    band_line_width = _cfg_float(anim_cfg, "band_line_width", 0.9)
    chain_alpha = _cfg_float(anim_cfg, "chain_alpha", 0.5)
    panel_wspace = _cfg_float(anim_cfg, "panel_wspace", 0.12)
    metric_panel_height_ratio = _cfg_float(anim_cfg, "metric_panel_height_ratio", 0.3)
    figure_width = _cfg_float(anim_cfg, "figure_width", 13.2)
    figure_height_single = _cfg_float(anim_cfg, "figure_height_single", 5.8)
    figure_height_multi = _cfg_float(anim_cfg, "figure_height_multi", 5.6)
    include_mean = bool(getattr(anim_cfg, "include_mean", True))
    plot_mean_std_only = bool(
        getattr(anim_cfg, "plot_mean_std_only", getattr(viz_cfg, "plot_mean_std_only", False))
    )
    plot_individual_traces = bool(
        getattr(anim_cfg, "plot_individual_traces", getattr(viz_cfg, "plot_individual_traces", True))
    )
    if plot_mean_std_only:
        plot_individual_traces = False
        include_mean = True
    mean_std_band_alpha = _cfg_float(anim_cfg, "mean_std_band_alpha", 0.18)
    # Actual series counts may be smaller than requested if samplers drop chains; guard for safety.
    expected_noise_series = len(display_noise_indices) if display_noise_indices is not None else N_noise
    expected_data_series = len(display_data_indices) if display_data_indices is not None else N_data
    if num_noise_series < expected_noise_series or num_data_series < expected_data_series:
        print(
            "[warn] Series count mismatch: "
            f"requested displayed noise/data={expected_noise_series}/{expected_data_series}, "
            f"available noise/data={num_noise_series}/{num_data_series}. "
            "Using available series to avoid animation errors."
        )
    chain_color = str(getattr(anim_cfg, "chain_color", "#D55E00"))
    data_chain_color = str(getattr(anim_cfg, "data_chain_color", "#56B4E9"))
    mean_color = str(getattr(anim_cfg, "mean_color", "#0072B2"))
    data_mean_color = str(getattr(anim_cfg, "data_mean_color", "#56B4E9"))
    train_expectation_color = str(getattr(anim_cfg, "train_expectation_color", "#333333"))
    noise_expectation_color = str(getattr(anim_cfg, "noise_expectation_color", "#009E73"))
    validity_color = str(getattr(anim_cfg, "validity_color", "#CC79A7"))
    novelty_color = str(getattr(anim_cfg, "novelty_color", "#009E73"))
    plot_novelty = bool(getattr(anim_cfg, "plot_novelty", False))
    metric_legend_loc = str(getattr(anim_cfg, "metric_legend_loc", "lower right"))
    metric_legend_ncol = max(1, int(getattr(anim_cfg, "metric_legend_ncol", 2)))
    warmup_color = str(getattr(anim_cfg, "warmup_color", "#E5E5E5"))
    energy_legend_loc = str(getattr(anim_cfg, "energy_legend_loc", "upper right"))
    noise_palette = [chain_color] * num_noise_series
    data_palette = [data_chain_color] * num_data_series
    metric_panel_height_ratio = max(0.1, metric_panel_height_ratio)
    metrics_title = "Chemical Validity / Novelty [%]" if plot_novelty else "Chemical Validity [%]"
    noise_metrics_title = metrics_title
    data_metrics_title = metrics_title

    metric_panel_count = int(N_noise > 0) + int(N_data > 0) if panel_enabled else 0

    if display_molecules:
        if isinstance(panel_width_ratios_cfg, (list, tuple)) and len(panel_width_ratios_cfg) >= 2:
            width_ratios = [float(panel_width_ratios_cfg[0]), float(panel_width_ratios_cfg[1])]
        elif single_group:
            width_ratios = [1.15, 1.55]
        else:
            width_ratios = [1.2, 1.45]
        figsize = (figure_width, figure_height_single if single_group else figure_height_multi)
        fig = plt.figure(figsize=figsize)
        main_spec = fig.add_gridspec(
            1,
            2,
            width_ratios=width_ratios,
            wspace=panel_wspace,
            **gridspec_kwargs,
        )
        ax_molecules = fig.add_subplot(main_spec[0, 0])
        ax_molecules.axis("off")
        if single_group and molecule_square_panel:
            _set_square_axis(ax_molecules)
        if metric_panel_count > 0:
            rows = 1 + metric_panel_count
            right_spec = main_spec[0, 1].subgridspec(
                rows,
                1,
                height_ratios=[2.0] + [metric_panel_height_ratio] * metric_panel_count,
                hspace=metrics_panel_hspace,
            )
            ax_energy = fig.add_subplot(right_spec[0, 0])
            row_idx = 1
            ax_metrics_noise = None
            ax_metrics_data = None
            if N_noise > 0:
                ax_metrics_noise = fig.add_subplot(right_spec[row_idx, 0], sharex=ax_energy)
                row_idx += 1
            if N_data > 0:
                ax_metrics_data = fig.add_subplot(right_spec[row_idx, 0], sharex=ax_energy)
        else:
            ax_energy = fig.add_subplot(main_spec[0, 1])
            ax_metrics_noise = ax_metrics_data = None
    else:
        ax_molecules = None
        if metric_panel_count > 0:
            total_rows = 1 + metric_panel_count
            height_ratios = [2.0] + [metric_panel_height_ratio] * metric_panel_count
            fig, axes = plt.subplots(
                total_rows,
                1,
                figsize=(9, 5 + 2 * metric_panel_count),
                sharex=True,
                gridspec_kw={
                    "height_ratios": height_ratios,
                    "hspace": metrics_panel_hspace,
                },
            )
            axes = np.atleast_1d(axes)
            ax_energy = axes[0]
            idx = 1
            ax_metrics_noise = axes[idx] if N_noise > 0 else None
            if ax_metrics_noise is not None:
                idx += 1
            ax_metrics_data = axes[idx] if N_data > 0 else None
        else:
            fig, ax_energy = plt.subplots(1, 1, figsize=(9, 5))
            ax_metrics_noise = ax_metrics_data = None

    warmup_display_steps = warmup_steps_total if record_warmup_phase else 0

    if include_band and show_band_spans and data_band_mean is not None and data_band_std is not None:
        ax_energy.axhspan(
            data_band_mean - data_band_std,
            data_band_mean + data_band_std,
            color="gray",
            alpha=0.25,
            label="data band",
            zorder=1,
        )
    if include_band and show_band_spans and num_noise_series > 0 and noise_band_mean is not None and noise_band_std is not None:
        ax_energy.axhspan(
            noise_band_mean - noise_band_std,
            noise_band_mean + noise_band_std,
            color=noise_expectation_color,
            alpha=0.15,
            label="noise band",
            zorder=1,
        )
    if display_molecules and ax_molecules is not None:
        _set_molecule_panel_title(ax_molecules, 0)
    lines_noise = []
    lines_data = []

    if plot_individual_traces:
        for idx in range(num_noise_series):
            label = "Individual trajectories" if idx == 0 else None
            line = ax_energy.plot(
                [],
                [],
                color=noise_palette[idx],
                linewidth=line_width,
                alpha=chain_alpha,
                label=label,
            )[0]
            lines_noise.append(line)

        for idx in range(num_data_series):
            label = "Individual trajectories" if idx == 0 else None
            line = ax_energy.plot(
                [],
                [],
                color=data_palette[idx],
                linewidth=line_width,
                alpha=chain_alpha,
                label=label,
            )[0]
            lines_data.append(line)

    if include_band and data_band_mean is not None:
        ax_energy.plot(
            [0, total_mcmc_steps],
            [data_band_mean, data_band_mean],
            color=train_expectation_color,
            linestyle="--",
            linewidth=band_line_width,
            alpha=0.9,
            label="Training distribution mean",
            zorder=5.5,
        )[0]
    if (
        include_band
        and show_noise_distribution_average
        and num_noise_series > 0
        and noise_band_mean is not None
    ):
        ax_energy.plot(
            [0, total_mcmc_steps],
            [noise_band_mean, noise_band_mean],
            color=noise_expectation_color,
            linestyle="--",
            linewidth=band_line_width,
            alpha=0.9,
            label="Noise distribution mean",
            zorder=5.5,
        )[0]

    metric_specs = [("Chemical Validity", 0, validity_color)]
    if plot_novelty:
        metric_specs.append(("Novelty", 2, novelty_color))
    noise_metric_dim = metrics_noise.shape[1] if metrics_noise is not None and num_noise_series > 0 else 0
    data_metric_dim = metrics_data.shape[1] if metrics_data is not None and num_data_series > 0 else 0
    metric_dims = [d for d in (noise_metric_dim, data_metric_dim) if d > 0]
    available_metric_dim = min(metric_dims) if metric_dims else 0
    metric_specs = [spec for spec in metric_specs if spec[1] < available_metric_dim]
    metric_count = len(metric_specs)
    noise_metric_lines: List[tuple[Any, int]] = []
    data_metric_lines: List[tuple[Any, int]] = []
    if plot_metrics and ax_metrics_noise is not None:
        if metric_count > 0 and num_noise_series > 0:
            for name, metric_idx, color in metric_specs:
                line = ax_metrics_noise.plot([], [], color=color, linewidth=metric_line_width, label=name)[0]
                noise_metric_lines.append((line, metric_idx))
        else:
            ax_metrics_noise.text(
                0.5,
                0.5,
                "No noise chains",
                ha="center",
                va="center",
                fontsize=empty_panel_fontsize,
            )
        ax_metrics_noise.set_ylim(0.0, 100.0)
        ax_metrics_noise.set_ylabel("Validity [%]" if show_y_axis_labels else "", fontsize=axis_label_fontsize)
        ax_metrics_noise.set_title(noise_metrics_title, fontsize=plot_title_fontsize)
        ax_metrics_noise.set_xlabel("MCMC steps", fontsize=axis_label_fontsize)
        ax_metrics_noise.tick_params(axis="both", labelsize=tick_fontsize)
        if metric_count > 1:
            ax_metrics_noise.legend(
                loc=metric_legend_loc,
                ncol=metric_legend_ncol,
                fontsize=legend_fontsize,
            )
    if plot_metrics and ax_metrics_data is not None:
        if metric_count > 0 and num_data_series > 0:
            for name, metric_idx, color in metric_specs:
                line = ax_metrics_data.plot([], [], color=color, linewidth=metric_line_width, label=name)[0]
                data_metric_lines.append((line, metric_idx))
        else:
            ax_metrics_data.text(
                0.5,
                0.5,
                "No data chains",
                ha="center",
                va="center",
                fontsize=empty_panel_fontsize,
            )
        ax_metrics_data.set_ylim(0.0, 100.0)
        ax_metrics_data.set_ylabel("Validity [%]" if show_y_axis_labels else "", fontsize=axis_label_fontsize)
        ax_metrics_data.set_title(data_metrics_title, fontsize=plot_title_fontsize)
        ax_metrics_data.set_xlabel("MCMC steps", fontsize=axis_label_fontsize)
        ax_metrics_data.tick_params(axis="both", labelsize=tick_fontsize)
        if metric_count > 1:
            ax_metrics_data.legend(
                loc=metric_legend_loc,
                ncol=metric_legend_ncol,
                fontsize=legend_fontsize,
            )

    noise_std_band = None
    data_std_band = None
    mean_noise_line = None
    mean_data_line = None
    if include_mean and num_noise_series > 0:
        mean_noise_line = ax_energy.plot(
            [],
            [],
            color=mean_color,
            linestyle="--",
            linewidth=mean_line_width,
            label="Trajectory mean",
            zorder=6,
        )[0]
    if include_mean and num_data_series > 0:
        mean_data_line = ax_energy.plot(
            [],
            [],
            color=data_mean_color,
            linestyle="-.",
            linewidth=mean_line_width,
            label="Trajectory mean",
            zorder=6,
        )[0]
    if plot_mean_std_only and noise_mean is not None and noise_std is not None:
        noise_std_band = ax_energy.fill_between(
            xs[:1],
            (noise_mean[:1] - noise_std[:1]),
            (noise_mean[:1] + noise_std[:1]),
            color=mean_color,
            alpha=mean_std_band_alpha,
            label="Trajectory mean ±1σ",
            zorder=4.5,
        )
    if plot_mean_std_only and data_mean is not None and data_std is not None:
        data_std_band = ax_energy.fill_between(
            xs[:1],
            (data_mean[:1] - data_std[:1]),
            (data_mean[:1] + data_std[:1]),
            color=data_mean_color,
            alpha=mean_std_band_alpha,
            label="Trajectory mean ±1σ",
            zorder=4.5,
        )

    ax_energy.set_xlim(0, total_mcmc_steps)
    ax_energy.set_ylim(energy_min - margin, energy_max + margin)
    ax_energy.set_xlabel("" if metric_panel_count > 0 else "MCMC steps", fontsize=axis_label_fontsize)
    energy_ylabel_kwargs = {"fontsize": axis_label_fontsize}
    if energy_ylabel_labelpad is not None:
        energy_ylabel_kwargs["labelpad"] = energy_ylabel_labelpad
    ax_energy.set_ylabel("Energy" if show_y_axis_labels else "", **energy_ylabel_kwargs)
    ax_energy.set_title("GEM Energies", fontsize=plot_title_fontsize)
    ax_energy.tick_params(axis="both", labelsize=tick_fontsize)
    if record_warmup_phase and warmup_display_steps > 0:
        ax_energy.axvspan(
            0,
            warmup_display_steps,
            color=warmup_color,
            alpha=0.55,
            label="Transport burn-in phase",
            zorder=1,
        )
    ax_energy.legend(loc=energy_legend_loc, fontsize=legend_fontsize)

    step_text = ax_energy.text(
        0.98,
        0.72,
        "",
        transform=ax_energy.transAxes,
        ha="right",
        va="top",
        fontsize=step_text_fontsize,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0),
    )

    mol_image_artist = None
    mol_text_artist = None

    def _frame_label(frame: int) -> str:
        actual_step = _actual_step_for_frame(frame)
        burn_done = min(max(int(actual_step), 0), warmup_display_steps)
        step_num = max(int(actual_step) - warmup_display_steps, 0)
        if steps > 0:
            step_num = min(step_num, steps)
        return f"Transport burn-in steps {burn_done}/{warmup_display_steps}\nMixing steps {step_num}/{steps}"

    def _update(frame: int):
        nonlocal mol_image_artist, mol_text_artist, noise_std_band, data_std_band
        safe_frame = min(frame, max_frames - 1) if max_frames > 0 else 0
        actual_step = _actual_step_for_frame(safe_frame)
        x_data = xs[: safe_frame + 1]
        if plot_individual_traces:
            for idx, line in enumerate(lines_noise):
                if idx < noise_history.shape[1]:
                    line.set_data(x_data, noise_history[: safe_frame + 1, idx])
            for idx, line in enumerate(lines_data):
                if idx < data_history.shape[1]:
                    line.set_data(x_data, data_history[: safe_frame + 1, idx])
        if mean_noise_line is not None and noise_mean is not None:
            mean_noise_line.set_data(x_data, noise_mean[: safe_frame + 1])
        if plot_mean_std_only and noise_std is not None and noise_mean is not None:
            y_low = noise_mean[: safe_frame + 1] - noise_std[: safe_frame + 1]
            y_high = noise_mean[: safe_frame + 1] + noise_std[: safe_frame + 1]
            verts_noise = np.concatenate(
                [np.column_stack([x_data, y_low]), np.column_stack([x_data[::-1], y_high[::-1]])]
            )
            if noise_std_band is None:
                noise_std_band = ax_energy.fill_between(
                    x_data,
                    y_low,
                    y_high,
                    color=mean_color,
                    alpha=mean_std_band_alpha,
                    label="Trajectory mean ±1σ",
                    zorder=4.5,
                )
            else:
                noise_std_band.set_verts([verts_noise])
        if mean_data_line is not None and data_mean is not None:
            mean_data_line.set_data(x_data, data_mean[: safe_frame + 1])
        if plot_mean_std_only and data_std is not None and data_mean is not None:
            y_low = data_mean[: safe_frame + 1] - data_std[: safe_frame + 1]
            y_high = data_mean[: safe_frame + 1] + data_std[: safe_frame + 1]
            verts_data = np.concatenate(
                [np.column_stack([x_data, y_low]), np.column_stack([x_data[::-1], y_high[::-1]])]
            )
            if data_std_band is None:
                data_std_band = ax_energy.fill_between(
                    x_data,
                    y_low,
                    y_high,
                    color=data_mean_color,
                    alpha=mean_std_band_alpha,
                    label="Trajectory mean ±1σ",
                    zorder=4.5,
                )
            else:
                data_std_band.set_verts([verts_data])
        if plot_metrics:
            if metrics_noise is not None and noise_metric_lines:
                for line, metric_idx in noise_metric_lines:
                    if metric_idx < metrics_noise.shape[1]:
                        line.set_data(x_data, 100.0 * metrics_noise[: safe_frame + 1, metric_idx])
            if metrics_data is not None and data_metric_lines:
                for line, metric_idx in data_metric_lines:
                    if metric_idx < metrics_data.shape[1]:
                        line.set_data(x_data, 100.0 * metrics_data[: safe_frame + 1, metric_idx])
        step_text.set_text(_frame_label(safe_frame))
        artists = []
        if display_molecules and ax_molecules is not None:
            mol_array = _render_molecule_grid(
                noise_nodes=_take_indices(noise_node_history[safe_frame], display_noise_indices),
                noise_edges=_take_indices(noise_edge_history[safe_frame], display_noise_indices),
                data_nodes=_take_indices(data_node_history[safe_frame], display_data_indices),
                data_edges=_take_indices(data_edge_history[safe_frame], display_data_indices),
                dataset_info=dataset_infos,
                max_display=max_display,
                mols_per_row=mols_per_row,
                subimg_size=mol_subimg_size,
                show_legends=show_molecule_legends,
                draw_padding=molecule_draw_padding,
                draw_bond_line_width=molecule_bond_line_width,
                draw_min_font_size=molecule_min_font_size,
                draw_max_font_size=molecule_max_font_size,
                draw_same_scale=molecule_draw_same_scale,
            )
            if mol_array is not None:
                if mol_text_artist is not None:
                    mol_text_artist.remove()
                    mol_text_artist = None
                if mol_image_artist is None:
                    mol_image_artist = ax_molecules.imshow(mol_array)
                    ax_molecules.axis("off")
                else:
                    mol_image_artist.set_data(mol_array)
                _set_molecule_panel_title(ax_molecules, actual_step)
                artists.append(mol_image_artist)
            else:
                if mol_image_artist is not None:
                    mol_image_artist.remove()
                    mol_image_artist = None
                if mol_text_artist is None:
                    mol_text_artist = ax_molecules.text(
                        0.5,
                        0.5,
                        "No renderable molecules",
                        ha="center",
                        va="center",
                        fontsize=empty_panel_fontsize,
                    )
                    ax_molecules.axis("off")
                _set_molecule_panel_title(ax_molecules, actual_step)
                artists.append(mol_text_artist)
        return artists

    interval_ms = float(getattr(anim_cfg, "interval_ms", 200.0))
    save_animation = bool(getattr(anim_cfg, "save_animation", True))
    animation_path = Path(getattr(anim_cfg, "filename_animation", "energy_animation.gif")).resolve()
    animation_format = str(getattr(anim_cfg, "animation_format", animation_path.suffix.lstrip("."))).lower()
    dpi = int(getattr(anim_cfg, "dpi", 150))
    fps_cfg = getattr(anim_cfg, "fps", None)
    fps = int(fps_cfg) if fps_cfg is not None else max(1, int(round(1000.0 / max(10.0, interval_ms))))

    if save_animation:
        ani = animation.FuncAnimation(
            fig,
            _update,
            frames=max_frames,
            interval=interval_ms,
            repeat=False,
        )
        try:
            if animation_format in {"gif", ""}:
                ani.save(animation_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
            else:
                ani.save(animation_path, writer=animation_format, dpi=dpi, fps=fps)
            print(f"[info] Saved animation to {animation_path}")
        except Exception as exc:
            import traceback

            print(f"[warn] Failed to save animation ({animation_format}): {exc}")
            traceback.print_exc()
    plt.close(fig)

    save_png = bool(getattr(anim_cfg, "save_png", True))
    png_path = Path(getattr(anim_cfg, "filename_png", "energy_animation.png")).resolve()
    if save_png:
        if display_molecules:
            if isinstance(panel_width_ratios_cfg, (list, tuple)) and len(panel_width_ratios_cfg) >= 2:
                width_ratios_static = [
                    float(panel_width_ratios_cfg[0]),
                    float(panel_width_ratios_cfg[1]),
                ]
            elif single_group:
                width_ratios_static = [1.15, 1.55]
            else:
                width_ratios_static = [1.2, 1.45]
            figsize_static = (figure_width, figure_height_single if single_group else figure_height_multi)
            fig_static = plt.figure(figsize=figsize_static)
            main_spec_static = fig_static.add_gridspec(
                1,
                2,
                width_ratios=width_ratios_static,
                wspace=panel_wspace,
                **gridspec_kwargs,
            )
            ax_mol_static = fig_static.add_subplot(main_spec_static[0, 0])
            ax_mol_static.axis("off")
            if single_group and molecule_square_panel:
                _set_square_axis(ax_mol_static)
            if metric_panel_count > 0:
                rows_static = 1 + metric_panel_count
                right_spec_static = main_spec_static[0, 1].subgridspec(
                    rows_static,
                    1,
                    height_ratios=[2.0] + [metric_panel_height_ratio] * metric_panel_count,
                    hspace=metrics_panel_hspace,
                )
                ax_energy_static = fig_static.add_subplot(right_spec_static[0, 0])
                row_idx_static = 1
                ax_metrics_noise_static = None
                ax_metrics_data_static = None
                if N_noise > 0:
                    ax_metrics_noise_static = fig_static.add_subplot(
                        right_spec_static[row_idx_static, 0], sharex=ax_energy_static
                    )
                    row_idx_static += 1
                if N_data > 0:
                    ax_metrics_data_static = fig_static.add_subplot(
                        right_spec_static[row_idx_static, 0], sharex=ax_energy_static
                    )
            else:
                ax_energy_static = fig_static.add_subplot(main_spec_static[0, 1])
                ax_metrics_noise_static = ax_metrics_data_static = None
            mol_static_array = _render_molecule_grid(
                noise_nodes=_take_indices(noise_node_history[-1], display_noise_indices),
                noise_edges=_take_indices(noise_edge_history[-1], display_noise_indices),
                data_nodes=_take_indices(data_node_history[-1], display_data_indices),
                data_edges=_take_indices(data_edge_history[-1], display_data_indices),
                dataset_info=dataset_infos,
                max_display=max_display,
                mols_per_row=mols_per_row,
                subimg_size=mol_subimg_size,
                show_legends=show_molecule_legends,
                draw_padding=molecule_draw_padding,
                draw_bond_line_width=molecule_bond_line_width,
                draw_min_font_size=molecule_min_font_size,
                draw_max_font_size=molecule_max_font_size,
                draw_same_scale=molecule_draw_same_scale,
            )
            if mol_static_array is not None:
                ax_mol_static.imshow(mol_static_array)
                _set_molecule_panel_title(ax_mol_static, total_mcmc_steps)
            else:
                ax_mol_static.text(
                    0.5,
                    0.5,
                    "No renderable molecules",
                    ha="center",
                    va="center",
                    fontsize=empty_panel_fontsize,
                )
        else:
            if metric_panel_count > 0:
                total_rows_static = 1 + metric_panel_count
                height_ratios_static = [2.0] + [metric_panel_height_ratio] * metric_panel_count
                fig_static, axes_static = plt.subplots(
                    total_rows_static,
                    1,
                    figsize=(9, 5 + 2 * metric_panel_count),
                    sharex=True,
                    gridspec_kw={
                        "height_ratios": height_ratios_static,
                        "hspace": metrics_panel_hspace,
                    },
                )
                axes_static = np.atleast_1d(axes_static)
                ax_energy_static = axes_static[0]
                idx_static = 1
                ax_metrics_noise_static = axes_static[idx_static] if N_noise > 0 else None
                if ax_metrics_noise_static is not None:
                    idx_static += 1
                ax_metrics_data_static = axes_static[idx_static] if N_data > 0 else None
            else:
                fig_static, ax_energy_static = plt.subplots(1, 1, figsize=(9, 5))
                ax_metrics_noise_static = ax_metrics_data_static = None
            ax_mol_static = None

        if include_band and show_band_spans and data_band_mean is not None and data_band_std is not None:
            ax_energy_static.axhspan(
                data_band_mean - data_band_std,
                data_band_mean + data_band_std,
                color=train_expectation_color,
                alpha=0.25,
                label="data band",
                zorder=1,
            )
        if include_band and show_band_spans and num_noise_series > 0 and noise_band_mean is not None and noise_band_std is not None:
            ax_energy_static.axhspan(
                noise_band_mean - noise_band_std,
                noise_band_mean + noise_band_std,
                color=noise_expectation_color,
                alpha=0.15,
                label="noise band",
                zorder=1,
            )
        if include_band and data_band_mean is not None:
            ax_energy_static.plot(
                [0, total_mcmc_steps],
                [data_band_mean, data_band_mean],
                color=train_expectation_color,
                linestyle="--",
                linewidth=band_line_width,
                alpha=0.9,
                label="Training distribution mean",
                zorder=5.5,
            )
        if (
            include_band
            and show_noise_distribution_average
            and num_noise_series > 0
            and noise_band_mean is not None
        ):
            ax_energy_static.plot(
                [0, total_mcmc_steps],
                [noise_band_mean, noise_band_mean],
                color=noise_expectation_color,
                linestyle="--",
                linewidth=band_line_width,
                alpha=0.9,
                label="Noise distribution mean",
                zorder=5.5,
            )
        if plot_individual_traces:
            for idx in range(num_noise_series):
                label = "Individual trajectories" if idx == 0 else None
                ax_energy_static.plot(
                    xs,
                    noise_history[:, idx],
                    color=noise_palette[idx],
                    linewidth=line_width,
                    alpha=chain_alpha,
                    label=label,
                )
            for idx in range(num_data_series):
                label = "Individual trajectories" if idx == 0 else None
                ax_energy_static.plot(
                    xs,
                    data_history[:, idx],
                    color=data_palette[idx],
                    linewidth=line_width,
                    alpha=chain_alpha,
                    label=label,
                )
        if plot_mean_std_only and noise_mean is not None and noise_std is not None:
            ax_energy_static.fill_between(
                xs,
                noise_mean - noise_std,
                noise_mean + noise_std,
                color=mean_color,
                alpha=mean_std_band_alpha,
                label="Trajectory mean ±1σ",
                zorder=4.5,
            )
        if plot_mean_std_only and data_mean is not None and data_std is not None:
            ax_energy_static.fill_between(
                xs,
                data_mean - data_std,
                data_mean + data_std,
                color=data_mean_color,
                alpha=mean_std_band_alpha,
                label="Trajectory mean ±1σ",
                zorder=4.5,
            )
        if include_mean and noise_mean is not None:
            ax_energy_static.plot(
                xs,
                noise_mean,
                color=mean_color,
                linestyle="--",
                linewidth=mean_line_width,
                label="Trajectory mean",
                zorder=6,
            )
        if include_mean and data_mean is not None:
            ax_energy_static.plot(
                xs,
                data_mean,
                color=data_mean_color,
                linestyle="-.",
                linewidth=mean_line_width,
                label="Trajectory mean",
                zorder=6,
            )
        ax_energy_static.set_xlim(0, total_mcmc_steps)
        ax_energy_static.set_ylim(energy_min - margin, energy_max + margin)
        ax_energy_static.set_xlabel("" if metric_panel_count > 0 else "MCMC steps", fontsize=axis_label_fontsize)
        ax_energy_static.set_ylabel("Energy" if show_y_axis_labels else "", **energy_ylabel_kwargs)
        ax_energy_static.set_title("GEM Energies", fontsize=plot_title_fontsize)
        ax_energy_static.tick_params(axis="both", labelsize=tick_fontsize)
        if record_warmup_phase and warmup_display_steps > 0:
            ax_energy_static.axvspan(
                0,
                warmup_display_steps,
                color=warmup_color,
                alpha=0.55,
                label="Transport burn-in phase",
                zorder=1,
            )
        ax_energy_static.legend(loc=energy_legend_loc, fontsize=legend_fontsize)
        ax_energy_static.text(
            0.98,
            0.72,
            f"Transport burn-in steps {warmup_display_steps}/{warmup_display_steps}\nMixing steps {steps}/{steps}",
            transform=ax_energy_static.transAxes,
            ha="right",
            va="top",
            fontsize=step_text_fontsize,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0),
        )
        if plot_metrics and ax_metrics_noise_static is not None:
            if num_noise_series > 0 and metric_count > 0:
                for name, metric_idx, color in metric_specs:
                    ax_metrics_noise_static.plot(
                        xs,
                        100.0 * metrics_noise[:, metric_idx],
                        color=color,
                        linewidth=metric_line_width,
                        label=name,
                    )
            else:
                ax_metrics_noise_static.text(
                    0.5,
                    0.5,
                    "No noise chains",
                    ha="center",
                    va="center",
                    fontsize=empty_panel_fontsize,
                )
            ax_metrics_noise_static.set_ylim(0.0, 100.0)
            ax_metrics_noise_static.set_ylabel(
                "Validity [%]" if show_y_axis_labels else "",
                fontsize=axis_label_fontsize,
            )
            ax_metrics_noise_static.set_title(noise_metrics_title, fontsize=plot_title_fontsize)
            ax_metrics_noise_static.set_xlabel("MCMC steps", fontsize=axis_label_fontsize)
            ax_metrics_noise_static.tick_params(axis="both", labelsize=tick_fontsize)
            if metric_count > 1:
                ax_metrics_noise_static.legend(
                    loc=metric_legend_loc,
                    ncol=metric_legend_ncol,
                    fontsize=legend_fontsize,
                )
        if plot_metrics and ax_metrics_data_static is not None:
            if num_data_series > 0 and metric_count > 0:
                for name, metric_idx, color in metric_specs:
                    ax_metrics_data_static.plot(
                        xs,
                        100.0 * metrics_data[:, metric_idx],
                        color=color,
                        linewidth=metric_line_width,
                        label=name,
                    )
            else:
                ax_metrics_data_static.text(
                    0.5,
                    0.5,
                    "No data chains",
                    ha="center",
                    va="center",
                    fontsize=empty_panel_fontsize,
                )
            ax_metrics_data_static.set_ylim(0.0, 100.0)
            ax_metrics_data_static.set_ylabel(
                "Validity [%]" if show_y_axis_labels else "",
                fontsize=axis_label_fontsize,
            )
            ax_metrics_data_static.set_xlabel("MCMC steps", fontsize=axis_label_fontsize)
            ax_metrics_data_static.set_title(data_metrics_title, fontsize=plot_title_fontsize)
            ax_metrics_data_static.tick_params(axis="both", labelsize=tick_fontsize)
            if metric_count > 1:
                ax_metrics_data_static.legend(
                    loc=metric_legend_loc,
                    ncol=metric_legend_ncol,
                    fontsize=legend_fontsize,
                )

        if use_tight_layout:
            fig_static.tight_layout()
        fig_static.savefig(png_path, dpi=dpi)
        plt.close(fig_static)
        print(f"[info] Saved static plot to {png_path}")

    save_quality = bool(getattr(anim_cfg, "save_final_quality", True))
    if save_quality:
        quality_path = Path(getattr(anim_cfg, "filename_final_quality", "final_quality.json")).resolve()
        payload = {
            "seed": seed_value,
            "selection_policy": selection_policy,
            "total_steps": int(total_mcmc_steps),
            "transport_burn_in_steps": int(warmup_steps_total),
            "mixing_steps": int(steps),
            "display_noise_indices": display_noise_indices,
            "display_data_indices": display_data_indices,
            "groups": {},
        }
        if noise_node_history:
            final_noise_molecules = list(zip(noise_node_history[-1], noise_edge_history[-1]))
            rows, summary = _molecule_quality_rows(
                final_noise_molecules,
                dataset_infos,
            )
            payload["groups"]["noise"] = {"summary": summary, "molecules": rows}
            if display_noise_indices is not None:
                selected_rows, selected_summary = _molecule_quality_rows(
                    final_noise_molecules,
                    dataset_infos,
                    indices=display_noise_indices,
                )
                payload["groups"]["noise"]["selected_summary"] = selected_summary
                payload["groups"]["noise"]["selected_molecules"] = selected_rows
        if data_node_history and N_data > 0:
            final_data_molecules = list(zip(data_node_history[-1], data_edge_history[-1]))
            rows, summary = _molecule_quality_rows(
                final_data_molecules,
                dataset_infos,
            )
            payload["groups"]["data"] = {"summary": summary, "molecules": rows}
            if display_data_indices is not None:
                selected_rows, selected_summary = _molecule_quality_rows(
                    final_data_molecules,
                    dataset_infos,
                    indices=display_data_indices,
                )
                payload["groups"]["data"]["selected_summary"] = selected_summary
                payload["groups"]["data"]["selected_molecules"] = selected_rows
        try:
            quality_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"[info] Saved final quality report to {quality_path}")
        except Exception as exc:
            print(f"[warn] Failed to save final quality report to {quality_path}: {exc}")


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="gem_ebm_animation_moses_public",
)
def main(cfg: DictConfig):
    run_animation(cfg)


if __name__ == "__main__":
    main()
