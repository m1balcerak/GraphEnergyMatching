import os
import sys
import time
import random
import pathlib
import math
import json
from datetime import timedelta
from contextlib import nullcontext
from typing import List, Optional, Dict, Any

# Ensure the package root is importable when running this file directly.
_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    Chem = None
    pass
try:
    from gem.analysis.rdkit_functions import build_molecule, mol2smiles
except Exception:
    build_molecule = None
    mol2smiles = None
from gem.metrics.molecular_metrics import SamplingMolecularMetrics
from gem.models.transformer_model import GraphTransformer
from gem.models.extra_features import DummyExtraFeatures, ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures
from gem.datasets.dataset_context import build_dataset_context
from gem import sampler
from gem.dlangevin_utils import (
    resolve_chain_warmup,
    resolve_dl_parameters,
    resolve_two_beta_annealing_kwargs,
    resolve_two_beta_kwargs,
)
from gem.checkpoint_utils import load_checkpoint, load_model_checkpoint
from gem.ema import ExponentialMovingAverage
from gem.visualize_energy import run_viz as run_energy_viz
from gem.ot_matching import minibatch_ot_pairs
from gem.ot_data import initialize_random_graphs_with_counts
from gem.fm_utils import (
    sample_interpolated_graph,
    grad_scalar_strength,
    mean_std,
    evaluate_interpolation_path,
)
from gem.ot_plotting import plot_interpolation_energy
from gem import utils
import matplotlib.pyplot as plt


def _set_module_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(flag)


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_simple_n_edits(section, label: str) -> Optional[int]:
    if section is None:
        return None
    value = getattr(section, "simple_n_edits", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[warn] Invalid {label}.simple_n_edits={value!r}; ignoring override.")
        return None


def _load_model_weights(
    model: nn.Module,
    device: torch.device,
    path: str | None,
    *,
    use_ema: bool = False,
    log: bool = True,
) -> Any:
    """Load model weights and return the raw checkpoint object when available."""
    if not path:
        return None
    try:
        ckpt = load_model_checkpoint(
            model,
            path,
            map_location=device,
            use_ema=use_ema,
        )
        if log:
            source = "EMA" if use_ema else "online"
            print(f"[info] Loaded {source} checkpoint weights: {path}")
        return ckpt
    except Exception as exc:
        source = "EMA" if use_ema else "online"
        raise RuntimeError(
            f"Could not load {source} checkpoint weights from '{path}': {exc}"
        ) from exc


def _validate_required_initialization(
    *,
    required: bool,
    resume_path: str,
    init_ckpt_path: str,
) -> None:
    if required and not resume_path and not init_ckpt_path:
        raise ValueError(
            "This training configuration requires train.init_ckpt to contain "
            "phase-1 online weights, or train.resume to continue phase 2."
        )


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            cuda_state = state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                torch.cuda.set_rng_state(cuda_state)
        print("[info] Restored RNG state from checkpoint.")
    except Exception as e:
        print(f"[warn] Could not restore RNG state: {e}")


def _gather_rng_states(device: torch.device, dist_ctx: dict) -> Optional[List[Dict[str, Any]]]:
    """Collect each process's RNG state on rank zero before checkpointing."""
    local_state = _capture_rng_state(device)
    if not dist_ctx.get("is_distributed", False):
        return [local_state]

    rank = int(dist_ctx.get("rank", 0))
    world_size = int(dist_ctx.get("world_size", 1))
    gathered: Optional[List[Dict[str, Any]]] = (
        [None] * world_size if rank == 0 else None  # type: ignore[list-item]
    )
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def _checkpoint_rng_state_for_rank(
    checkpoint: Any,
    *,
    rank: int,
    is_distributed: bool,
) -> Any:
    """Select a rank-specific RNG state, with safe legacy-checkpoint behavior."""
    if not isinstance(checkpoint, dict):
        return None
    rank_states = checkpoint.get("rng_state_by_rank")
    if isinstance(rank_states, (list, tuple)) and 0 <= rank < len(rank_states):
        state = rank_states[rank]
        if isinstance(state, dict):
            return state

    legacy_state = checkpoint.get("rng_state")
    if not is_distributed or rank == 0:
        return legacy_state
    return None


def _save_training_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: DictConfig,
    iteration: int,
    history: Dict[str, Any],
    last_plot_iter: int,
    train_stats: Dict[str, Any],
    rng_state_by_rank: List[Dict[str, Any]],
    ema: Optional[ExponentialMovingAverage] = None,
) -> None:
    try:
        cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_payload = None
    if not rng_state_by_rank:
        raise ValueError("rng_state_by_rank must contain at least rank zero.")
    payload = {
        "format": "gem_train_checkpoint_v3",
        "iteration": int(iteration),
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "last_plot_iter": int(last_plot_iter),
        "train_stats": train_stats,
        "rng_state": rng_state_by_rank[0],
        "rng_state_by_rank": rng_state_by_rank,
        "config": cfg_payload,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _build_dataset_context(cfg: DictConfig):
    """Instantiate the configured dataset and its metadata."""
    return build_dataset_context(cfg)


def _safe_float(val, default=float("nan")) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _negative_molecule_diagnostics(
    node_types_list: List[torch.Tensor],
    edge_types_list: List[torch.Tensor],
    dataset_info,
) -> Dict[str, Any]:
    total = len(node_types_list)
    out: Dict[str, Any] = {
        "total": int(total),
        "valid": 0,
        "connected": 0,
        "valid_mask": [False] * total,
        "connected_mask": [False] * total,
    }
    if total == 0 or build_molecule is None or mol2smiles is None:
        return out

    atom_decoder = getattr(dataset_info, "atom_decoder", None)
    for idx, (node_types, edge_types) in enumerate(zip(node_types_list, edge_types_list)):
        try:
            nt = node_types.detach().cpu() if torch.is_tensor(node_types) else node_types
            et = edge_types.detach().cpu() if torch.is_tensor(edge_types) else edge_types
            mol = build_molecule(nt, et, atom_decoder)
            smiles = mol2smiles(mol)
        except Exception:
            continue
        if not smiles:
            continue
        out["valid"] += 1
        out["valid_mask"][idx] = True

        connected = "." not in smiles
        if connected and Chem is not None and mol is not None:
            try:
                connected = len(Chem.rdmolops.GetMolFrags(mol)) == 1
            except Exception:
                connected = False
        if connected:
            out["connected"] += 1
            out["connected_mask"][idx] = True
    return out


def _negative_sample_diagnostics(
    node_types_list: List[torch.Tensor],
    edge_types_list: List[torch.Tensor],
    dataset_info,
) -> Dict[str, Any]:
    return _negative_molecule_diagnostics(
        node_types_list,
        edge_types_list,
        dataset_info,
    )


def _pairwise_cl_masks(
    valid_mask,
    connected_mask,
    count: int,
    device: torch.device,
    clip_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(valid_mask) == count and len(connected_mask) == count:
        good_mask_list = [bool(v) and bool(c) for v, c in zip(valid_mask, connected_mask)]
    else:
        good_mask_list = [False] * count
    good_mask = torch.tensor(good_mask_list, dtype=torch.bool, device=device)
    cap_mask = torch.ones_like(good_mask) if clip_mode == "paired_all" else good_mask
    return good_mask, cap_mask


def _plot_loss_curves(history: dict, out_path: str = "training_losses.png") -> None:
    its = history.get("iter", [])
    if not its:
        return
    fm_losses = history.get("loss_fm", [])
    cl_losses = history.get("loss_cl", [])
    total_losses = history.get("loss_total", [])
    if not fm_losses or not total_losses:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("Training losses")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.plot(its, fm_losses, label="Transport loss", color="C0")
    if any(math.isfinite(x) for x in cl_losses):
        ax.plot(its, cl_losses, label="CL loss", color="C1")
    ax.plot(its, total_losses, label="Total loss", color="C3", linestyle="--")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_acceptance_metrics(history: dict, out_path: str = "mcmc_metrics.png") -> None:
    its = history.get("iter", [])
    if not its:
        return
    acc_rates = history.get("acc_rate", [])
    if not any(math.isfinite(x) for x in acc_rates):
        return

    prop_total = history.get("prop_total", [])
    acc_total = history.get("acc_total", [])
    step_total = history.get("step_total", [])
    move_total = history.get("move_total", [])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.0))
    axes[0].set_title("CL MCMC acceptance & move diagnostics")
    axes[0].set_ylabel("acceptance (%)")
    axes[0].plot(its, acc_rates, color="C0", label="MCMC acceptance")
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("distance / move")
    metric_series = [
        ("proposal distance", prop_total, "C1"),
        ("accepted distance", acc_total, "C2"),
        ("step distance", step_total, "C3"),
        ("move per step", move_total, "C4"),
    ]
    for label, series, color in metric_series:
        if series and any(math.isfinite(x) for x in series):
            axes[1].plot(its, series, label=label, color=color)
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _record_expected_energy(
    *,
    model: nn.Module,
    datamodule,
    dataset_infos,
    device: torch.device,
    extra_features,
    domain_features,
    transition: str,
    num_samples: int | None = None,
    batch_size: int | None = None,
    output_path: str = "expected_energy.json",
    iteration: int | None = None,
) -> dict:
    if batch_size is None:
        return {}
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        return {}
    if batch_size <= 0:
        return {}
    num_samples = batch_size
    num_batches = num_samples // batch_size
    if num_batches <= 0:
        return {}
    num_samples = num_batches * batch_size
    num_workers = int(getattr(datamodule.cfg.train, "num_workers", 0))
    pin_memory = bool(getattr(datamodule.cfg.dataset, "pin_memory", False))
    data_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    def _energy_for_data() -> torch.Tensor:
        energies = []
        iterator = iter(data_loader)
        for _ in range(num_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                batch = next(iterator)
            dense_data, node_mask = utils.to_dense(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            graphs = dense_data.mask(node_mask, collapse=True).split(node_mask)
            node_list = [g.X.long().cpu() for g in graphs]
            edge_list = [g.E.long().cpu() for g in graphs]
            energy = sampler.energy_batch(
                model=model,
                node_types_list=node_list,
                edge_types_list=edge_list,
                dataset_info=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=True,
                apply_property_conditioner=False,
            )
            energies.append(energy.cpu())
        return torch.cat(energies, dim=0)

    def _energy_for_noise() -> torch.Tensor:
        energies = []
        for _ in range(num_batches):
            rand_graphs = sampler.initialize_random_graphs(
                batch_size=batch_size,
                dataset_info=dataset_infos,
                device=device,
                transition=transition,
            )
            rand_nodes = [nt for (nt, _) in rand_graphs]
            rand_edges = [et for (_, et) in rand_graphs]
            energy = sampler.energy_batch(
                model=model,
                node_types_list=rand_nodes,
                edge_types_list=rand_edges,
                dataset_info=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=True,
                apply_property_conditioner=False,
            )
            energies.append(energy.cpu())
        return torch.cat(energies, dim=0)

    prev_training = model.training
    model.eval()
    with torch.no_grad():
        data_energy = _energy_for_data()
        noise_energy = _energy_for_noise()
    if prev_training:
        model.train()

    data_mean = float(data_energy.mean().item())
    data_std = float(data_energy.std(unbiased=False).item())
    noise_mean = float(noise_energy.mean().item())
    noise_std = float(noise_energy.std(unbiased=False).item())

    label = f"iter {iteration}" if iteration is not None else "final"
    print(
        f"[energy] ({label}) Expected energy (data, n={num_samples}, batch={batch_size}): "
        f"mean={data_mean:.6f}, std={data_std:.6f}"
    )
    print(
        f"[energy] ({label}) Expected energy (noise, n={num_samples}, batch={batch_size}): "
        f"mean={noise_mean:.6f}, std={noise_std:.6f}"
    )

    summary = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "data": {"mean": data_mean, "std": data_std},
        "noise": {"mean": noise_mean, "std": noise_std},
    }
    if iteration is not None:
        summary["iteration"] = int(iteration)
    try:
        with open(output_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[energy] Saved expected energy summary to {output_path}")
    except Exception as exc:
        print(f"[warn] Failed to write expected energy summary to {output_path}: {exc}")
    return summary


def _extract_expected_energy(payload: object) -> float | None:
    if isinstance(payload, dict):
        data_block = payload.get("data")
        if isinstance(data_block, dict) and "mean" in data_block:
            try:
                return float(data_block["mean"])
            except (TypeError, ValueError):
                return None
        for key in (
            "data_mean",
            "energy_data_mean",
            "expected_energy",
            "energy_mean",
            "mean",
        ):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    return None
    return None


def _coerce_energy_threshold(raw_val) -> float | None:
    if raw_val is False or raw_val is None:
        return None
    if raw_val is True:
        return None
    if isinstance(raw_val, str) and raw_val.strip().lower() in {"", "false", "none", "null"}:
        return None
    try:
        return float(raw_val)
    except (TypeError, ValueError):
        return None


def _resolve_warmup_energy_threshold(cfg: DictConfig) -> tuple[float | None, str | None]:
    cfg_train = getattr(cfg, "train", None)
    if cfg_train is not None:
        for key in ("warmup_energy_threshold", "warmup_energy", "expected_energy"):
            raw_val = getattr(cfg_train, key, None)
            if raw_val is not None:
                value = _coerce_energy_threshold(raw_val)
                if value is None:
                    return None, f"cfg.train.{key}"
                return value, f"cfg.train.{key}"

    ckpt_path_raw = None
    if cfg_train is not None:
        ckpt_path_raw = getattr(cfg_train, "resume", None) or getattr(cfg_train, "init_ckpt", None)
    if not ckpt_path_raw:
        ckpt_path_raw = getattr(getattr(cfg, "general", None), "resume", None)
    if not ckpt_path_raw:
        return None, None

    ckpt_path = pathlib.Path(str(ckpt_path_raw)).expanduser()
    candidates = [
        ckpt_path.with_suffix(".json"),
        ckpt_path.parent / "expected_energy.json",
        ckpt_path.parent.parent / "expected_energy.json",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                with open(candidate, "r") as fp:
                    payload = json.load(fp)
                value = _extract_expected_energy(payload)
                if value is not None:
                    return value, str(candidate)
        except Exception:
            continue

    if ckpt_path.is_file():
        try:
            payload = load_checkpoint(str(ckpt_path), map_location="cpu")
            value = _extract_expected_energy(payload)
            if value is not None:
                return value, str(ckpt_path)
        except Exception:
            pass

    return None, None


def _count_graph_changes(
    prev_nodes: List[torch.Tensor],
    prev_edges: List[torch.Tensor],
    next_nodes: List[torch.Tensor],
    next_edges: List[torch.Tensor],
) -> int:
    if len(prev_nodes) != len(next_nodes) or len(prev_edges) != len(next_edges):
        return max(len(prev_nodes), len(next_nodes), len(prev_edges), len(next_edges))
    changed = 0
    for p_nt, p_et, n_nt, n_et in zip(prev_nodes, prev_edges, next_nodes, next_edges):
        if p_nt.shape != n_nt.shape or p_et.shape != n_et.shape:
            changed += 1
        elif not torch.equal(p_nt, n_nt) or not torch.equal(p_et, n_et):
            changed += 1
    return changed


def _run_chain_warmup(
    *,
    init_nodes: List[torch.Tensor],
    init_edges: List[torch.Tensor],
    model: nn.Module,
    dataset_infos,
    device: torch.device,
    extra_features,
    domain_features,
    chain_cfg,
    amp_dtype: str | None,
    energy_threshold: float | None,
) -> tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    int,
    int,
    int,
    str,
    float | None,
]:
    if not chain_cfg.enabled or chain_cfg.steps <= 0:
        return init_nodes, init_edges, 0, 0, 0, "disabled", None

    if (
        sampler.should_vectorize_simple_warmup(
            chain_cfg.proposal,
            vectorized=bool(getattr(chain_cfg, "vectorized", True)),
        )
        and energy_threshold is None
    ):
        edits_per_step = (
            5
            if chain_cfg.simple_n_edits is None
            else int(chain_cfg.simple_n_edits)
        )
        (
            warm_nodes,
            warm_edges,
            warmup_moves,
            warmup_attempts,
            warmup_stats,
        ) = sampler.run_simple_v2_warmup_vectorized(
            model=model,
            dataset_info=dataset_infos,
            node_types_list=init_nodes,
            edge_types_list=init_edges,
            extra_features=extra_features,
            domain_features=domain_features,
            steps=int(chain_cfg.steps),
            device=device,
            edits_per_step=edits_per_step,
            amp_dtype=amp_dtype,
            stop_when_unchanged=True,
        )
        return (
            warm_nodes,
            warm_edges,
            int(warmup_moves),
            int(warmup_attempts),
            int(warmup_stats["steps_executed"]),
            str(warmup_stats["stop_reason"]),
            None,
        )

    current_nodes = init_nodes
    current_edges = init_edges
    warmup_moves = 0
    warmup_steps_total = 0
    warmup_steps_done = 0
    warmup_stop = "max"
    warmup_energy_mean = None

    for _ in range(int(chain_cfg.steps)):
        prev_nodes = current_nodes
        prev_edges = current_edges
        current_nodes, current_edges, n_accepts, n_steps_total, _ = sampler.mcmc_sample_batch(
            model=model,
            dataset_info=dataset_infos,
            node_types_list=current_nodes,
            edge_types_list=current_edges,
            extra_features=extra_features,
            domain_features=domain_features,
            steps=1,
            device=device,
            proposal=chain_cfg.proposal,
            gwd_beta=chain_cfg.gwd_beta,
            dl_beta=chain_cfg.dl_beta,
            dl_lambda_X=chain_cfg.dl_lambda_X,
            dl_lambda_E=chain_cfg.dl_lambda_E,
            simple_n_edits=chain_cfg.simple_n_edits,
            amp_dtype=amp_dtype,
            collect_stats=False,
            property_lambda_prop=chain_cfg.property_lambda_prop,
            **chain_cfg.dual_kwargs,
        )
        changed_graphs = _count_graph_changes(
            prev_nodes,
            prev_edges,
            current_nodes,
            current_edges,
        )
        warmup_moves += changed_graphs
        warmup_steps_total += int(n_steps_total)
        warmup_steps_done += 1

        if changed_graphs == 0:
            warmup_stop = "stuck"
            break

        if energy_threshold is not None:
            energy = sampler.energy_batch(
                model=model,
                node_types_list=current_nodes,
                edge_types_list=current_edges,
                dataset_info=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=True,
                apply_property_conditioner=False,
            )
            warmup_energy_mean = float(energy.mean().item())
            if warmup_energy_mean <= energy_threshold:
                warmup_stop = "energy"
                break

    return (
        current_nodes,
        current_edges,
        warmup_moves,
        warmup_steps_total,
        warmup_steps_done,
        warmup_stop,
        warmup_energy_mean,
    )


def _plot_energy_gap(history: dict, out_path: str = "training_energy_gap.png") -> None:
    its = history.get("iter", [])
    ediff_mu = history.get("fm_ediff_mu", [])
    if not its or not ediff_mu:
        return
    if not any(math.isfinite(x) for x in ediff_mu):
        return

    ediff_sd = history.get("fm_ediff_sd", [])
    mu = np.array(ediff_mu, dtype=float)
    sd = np.array(ediff_sd if ediff_sd else [float("nan")] * len(mu), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("E_data - E_noise over training")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy difference")
    ax.plot(its, mu, color="C1", label="ΔE mean")
    if sd.size == mu.size and np.isfinite(sd).any():
        ax.fill_between(its, mu - 3.0 * sd, mu + 3.0 * sd, color="C1", alpha=0.2, label="±3σ")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=0.9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_energy_trajectories(
    history: dict,
    out_path: str = "training_energy_trajectories.png",
) -> None:
    its = history.get("iter", [])
    data_mean = history.get("energy_data_mean", [])
    noise_mean = history.get("energy_noise_mean", [])
    if not its or not data_mean or not noise_mean:
        return
    if not any(math.isfinite(x) for x in data_mean + noise_mean):
        return

    data_mu = np.asarray(data_mean, dtype=float)
    noise_mu = np.asarray(noise_mean, dtype=float)
    data_sd = np.asarray(history.get("energy_data_std", []), dtype=float)
    noise_sd = np.asarray(history.get("energy_noise_std", []), dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("Energy trajectories")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.plot(its, data_mu, color="C2", label="Training graphs")
    ax.plot(its, noise_mu, color="C0", label="Noise graphs")
    if data_sd.size == data_mu.size and np.isfinite(data_sd).any():
        ax.fill_between(
            its,
            data_mu - data_sd,
            data_mu + data_sd,
            color="C2",
            alpha=0.18,
        )
    if noise_sd.size == noise_mu.size and np.isfinite(noise_sd).any():
        ax.fill_between(
            its,
            noise_mu - noise_sd,
            noise_mu + noise_sd,
            color="C0",
            alpha=0.18,
        )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_transport_gradient_strength(
    history: dict,
    out_path: str = "training_gradient_strength.png",
) -> None:
    its = history.get("iter", [])
    grad_mean = history.get("fm_grad_mu", [])
    if not its or not grad_mean or not any(math.isfinite(x) for x in grad_mean):
        return

    mean = np.asarray(grad_mean, dtype=float)
    std = np.asarray(history.get("fm_grad_sd", []), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("Transport gradient strength")
    ax.set_xlabel("iteration")
    ax.set_ylabel("gradient strength")
    ax.plot(its, mean, color="C4", label="Mean")
    if std.size == mean.size and np.isfinite(std).any():
        ax.fill_between(
            its,
            mean - std,
            mean + std,
            color="C4",
            alpha=0.18,
            label="Standard deviation",
        )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_all_training_plots(
    history: dict,
    cl_active: bool,
    iteration: Optional[int] = None,
    interp_payload: Optional[dict] = None,
    interp_context: Optional[dict] = None,
) -> None:
    """Helper to persist every available training diagnostic plot."""
    if not history or not history.get("iter"):
        return
    resolved_iter = iteration
    if resolved_iter is None:
        try:
            resolved_iter = int(history.get("iter")[-1])
        except Exception:
            resolved_iter = None
    suffix = f"_it{resolved_iter:06d}" if resolved_iter is not None else ""
    iter_note = f" at iter {resolved_iter}" if resolved_iter is not None else ""
    scope = "intermediate " if iteration is not None else ""
    plot_jobs = [
        ("training losses", "training_losses", _plot_loss_curves, True),
        ("CL acceptance", "mcmc_metrics", _plot_acceptance_metrics, cl_active),
        ("energy gap", "training_energy_gap", _plot_energy_gap, True),
        (
            "energy trajectories",
            "training_energy_trajectories",
            _plot_energy_trajectories,
            True,
        ),
        (
            "transport gradient strength",
            "training_gradient_strength",
            _plot_transport_gradient_strength,
            True,
        ),
    ]
    for human_label, stem, fn, enabled in plot_jobs:
        if not enabled:
            continue
        filename = f"{stem}{suffix}.png"
        try:
            fn(history, out_path=filename)
            print(f"[info] Saved {scope}{human_label} plot{iter_note}: {filename}")
        except Exception as e:
            print(f"[warn] Could not plot {human_label}{iter_note}: {e}")
    if interp_payload and interp_context:
        try:
            _plot_interpolation_from_cache(
                interp_payload=interp_payload,
                iteration=resolved_iter,
                **interp_context,
            )
        except Exception as e:
            print(f"[warn] Could not plot interpolation energy{iter_note}: {e}")


def _plot_interpolation_from_cache(
    interp_payload: dict,
    *,
    model: nn.Module,
    dataset_info,
    extra_features,
    domain_features,
    device: torch.device,
    iteration: Optional[int],
) -> None:
    if not interp_payload:
        return
    taus = torch.linspace(0.0, 1.0, steps=21)
    energy_means, energy_stds, _, _ = evaluate_interpolation_path(
        interp_payload["noise_nodes"],
        interp_payload["noise_edges"],
        interp_payload["data_nodes"],
        interp_payload["data_edges"],
        taus,
        model=model,
        dataset_info=dataset_info,
        extra_features=extra_features,
        domain_features=domain_features,
        device=device,
        log_progress=False,
    )
    suffix = f"_it{iteration:06d}" if iteration is not None else ""
    out_path = f"training_energy_interpolation{suffix}.png"
    plot_interpolation_energy(
        taus,
        energy_means,
        energy_stds,
        mu_noise=interp_payload["mu_noise"],
        sd_noise=interp_payload["sd_noise"],
        mu_data=interp_payload["mu_data"],
        sd_data=interp_payload["sd_data"],
        out_path=out_path,
        sigma_k=3.0,
        dpi=150,
    )
    iter_note = f" at iter {iteration}" if iteration is not None else ""
    print(f"[info] Saved {'intermediate ' if iteration is not None else ''}interpolation energy plot{iter_note}: {out_path}")


def _init_distributed(cfg: DictConfig) -> dict:
    dist_cfg = getattr(getattr(cfg, "train", None), "distributed", None)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    context = {
        "is_distributed": False,
        "backend": None,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "find_unused_parameters": False,
        "sampler_shuffle": True,
    }
    if env_world_size <= 1 or not dist.is_available():
        if env_world_size <= 1 and dist_cfg and bool(getattr(dist_cfg, "enabled", False)):
            print("[warn] cfg.train.distributed.enabled=true but WORLD_SIZE=1; running single-process.")
        if not dist.is_available() and env_world_size > 1:
            print("[warn] torch.distributed not available; running single-process.")
        return context

    backend = str(getattr(dist_cfg, "backend", "nccl")) if dist_cfg else "nccl"
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    init_method = getattr(dist_cfg, "init_method", None) if dist_cfg else None
    init_kwargs = {"init_method": init_method} if init_method else {}
    timeout_minutes_cfg = None
    if dist_cfg is not None:
        try:
            timeout_minutes_cfg = float(getattr(dist_cfg, "barrier_timeout_minutes", 0.0))
        except (TypeError, ValueError):
            timeout_minutes_cfg = 0.0
    default_timeout_minutes = 120.0
    timeout_minutes = timeout_minutes_cfg if timeout_minutes_cfg and timeout_minutes_cfg > 0 else default_timeout_minutes
    timeout_delta = timedelta(minutes=timeout_minutes)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timeout_delta, **init_kwargs)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    default_local = rank % n_gpus if n_gpus > 0 else 0
    local_rank = int(os.environ.get("LOCAL_RANK", default_local))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    context.update(
        {
            "is_distributed": world_size > 1,
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "find_unused_parameters": bool(getattr(dist_cfg, "find_unused_parameters", False)) if dist_cfg else False,
            "sampler_shuffle": bool(getattr(dist_cfg, "sampler_shuffle", True)) if dist_cfg else True,
        }
    )
    return context


def _build_distributed_train_loader(
    template_loader,
    *,
    dist_ctx: dict,
    sampler_seed: int,
    shuffle: bool,
):
    if not dist_ctx.get("is_distributed", False):
        return template_loader, None

    dataset = template_loader.dataset
    drop_last = getattr(template_loader, "drop_last", False)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_ctx["world_size"],
        rank=dist_ctx["rank"],
        shuffle=shuffle,
        seed=int(sampler_seed),
        drop_last=drop_last,
    )

    loader_kwargs = {
        "batch_size": template_loader.batch_size,
        "num_workers": template_loader.num_workers,
        "pin_memory": template_loader.pin_memory,
        "drop_last": drop_last,
        "timeout": getattr(template_loader, "timeout", 0),
    }
    if template_loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = getattr(template_loader, "persistent_workers", False)
        prefetch_factor = getattr(template_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    pin_memory_device = getattr(template_loader, "pin_memory_device", None)
    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = pin_memory_device
    collate_fn = getattr(template_loader, "collate_fn", None)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    worker_init_fn = getattr(template_loader, "worker_init_fn", None)
    if worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn
    generator = getattr(template_loader, "generator", None)
    if generator is not None:
        loader_kwargs["generator"] = generator
    follow_batch = getattr(template_loader, "follow_batch", None)
    if follow_batch is not None:
        loader_kwargs["follow_batch"] = follow_batch
    exclude_keys = getattr(template_loader, "exclude_keys", None)
    if exclude_keys is not None:
        loader_kwargs["exclude_keys"] = exclude_keys

    loader_cls = type(template_loader)
    loader = loader_cls(
        dataset,
        sampler=sampler,
        shuffle=False,
        **loader_kwargs,
    )
    return loader, sampler


def _barrier(dist_ctx: dict) -> None:
    if dist_ctx.get("is_distributed", False):
        dist.barrier()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="gem_ebm_fm_moses_ver3",
)
def main(cfg: DictConfig):
    """Train GEM with transport and optional contrastive objectives."""
    resume_path = str(getattr(cfg.train, "resume", "") or "")
    init_ckpt_path = str(getattr(cfg.train, "init_ckpt", "") or "")
    init_use_ema = bool(getattr(cfg.train, "init_use_ema", False))

    dist_ctx = _init_distributed(cfg)
    is_main_process = (not dist_ctx.get("is_distributed", False)) or dist_ctx.get("rank", 0) == 0

    # Loss weights (needed early to configure DDP behavior)
    lambda_cl_cfg = float(getattr(cfg.train, "lambda_cl", getattr(cfg.train, "lambda_cd", 1.0)))
    lambda_fm_cfg = float(getattr(cfg.train, "lambda_fm", 1.0))
    cl_steps_cfg = int(getattr(cfg.train, "cl_steps", getattr(cfg.train, "cd_steps", 0)))
    cl_active_cfg = (not math.isclose(lambda_cl_cfg, 0.0)) and cl_steps_cfg > 0
    _validate_required_initialization(
        required=cl_active_cfg,
        resume_path=resume_path,
        init_ckpt_path=init_ckpt_path,
    )
    if dist_ctx.get("is_distributed", False) and not cl_active_cfg:
        if not dist_ctx.get("find_unused_parameters", False):
            dist_ctx["find_unused_parameters"] = True
        if is_main_process:
            reason = "lambda_cl=0" if math.isclose(lambda_cl_cfg, 0.0) else "cl_steps<=0"
            print(f"[info] Enabling DDP find_unused_parameters ({reason}).")

    # Mirror all stdout/stderr to a log file in the run directory (rank 0 only)
    if is_main_process:
        try:
            log_file = open("train_gem_ebm_fm.log", "a", buffering=1)

            class _Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for s in self.streams:
                        try:
                            s.write(data)
                            s.flush()
                        except Exception:
                            pass

                def flush(self):
                    for s in self.streams:
                        try:
                            s.flush()
                        except Exception:
                            pass

            sys.stdout = _Tee(sys.stdout, log_file)
            sys.stderr = _Tee(sys.stderr, log_file)
            print("[info] Logging to train_gem_ebm_fm.log")
        except Exception as e:
            print(f"[warn] Could not set up file logging: {e}")

    pl.seed_everything(int(cfg.train.seed) + dist_ctx.get("rank", 0))

    # Persist the fully-resolved configuration for reproducibility
    if is_main_process:
        try:
            with open("settings.txt", "w") as fp:
                fp.write(OmegaConf.to_yaml(cfg))
        except Exception as e:
            print(f"[warn] Could not write settings.txt: {e}")

    # Data & dataset info
    datamodule, dataset_infos, dataset_smiles = _build_dataset_context(cfg)

    # Features & dimensions
    extra_features = ExtraFeatures(
        cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos
    )
    is_molecular = dataset_smiles is not None
    domain_features = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if is_molecular
        else DummyExtraFeatures()
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    # Metrics & references are only needed by the optional final evaluation.
    do_evaluate = bool(getattr(cfg.sample, "evaluate", True))
    sampling_metrics = None
    if do_evaluate:
        if not is_molecular:
            raise ValueError("cfg.sample.evaluate=true requires a molecular dataset.")
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, dataset_smiles, cfg)
        if dist_ctx.get("is_distributed", False):
            if is_main_process:
                dataset_infos.compute_reference_metrics(
                    datamodule=datamodule,
                    sampling_metrics=sampling_metrics,
                )
            _barrier(dist_ctx)
            if not is_main_process:
                dataset_infos.compute_reference_metrics(
                    datamodule=datamodule,
                    sampling_metrics=sampling_metrics,
                )
            _barrier(dist_ctx)
        else:
            dataset_infos.compute_reference_metrics(
                datamodule=datamodule,
                sampling_metrics=sampling_metrics,
            )
        if is_main_process:
            print("Reference metrics:", dataset_infos.ref_metrics)
    elif is_main_process:
        print("[info] Evaluation disabled; skipping reference metrics.")

    # Device & precision
    if torch.cuda.is_available():
        if dist_ctx.get("is_distributed", False):
            device = torch.device("cuda", dist_ctx.get("local_rank", 0))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("medium")

    # Model activation (configurable: relu | silu). Default: relu
    act_name = str(getattr(cfg.model, "activation", "relu")).lower()
    if act_name == "silu":
        act_module = nn.SiLU()
    else:
        if act_name != "relu":
            if is_main_process:
                print(f"[warn] Unknown activation '{act_name}', defaulting to ReLU.")
        act_module = nn.ReLU()

    # Model
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

    frozen_heads: List[str] = []
    for name in ("mlp_out_X", "mlp_out_E", "mlp_out_y"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            _set_module_requires_grad(module, False)
            frozen_heads.append(name)
    if not cl_active_cfg:
        # The terminal energy bias never influences FM-only gradients (∇E invariant to constant shift)
        energy_tail = getattr(model, "energy_mlp", None)
        if isinstance(energy_tail, nn.Sequential) and len(energy_tail) >= 3:
            last_linear = energy_tail[-1]
            if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
                last_linear.bias.requires_grad_(False)
                frozen_heads.append("energy_mlp[-1].bias")
    if is_main_process and frozen_heads:
        frozen_str = ", ".join(frozen_heads)
        print(f"[info] Frozen outputs unused by GEM energy objectives: {frozen_str}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_millions = trainable_params / 1.0e6
    if is_main_process:
        print(
            f"[info] Trainable parameters: {trainable_params:,} ({params_millions:.2f}M)"
        )

    # Optional: resume/init from checkpoint. resume is full-state when possible;
    # init_ckpt is intentionally weight-only initialization.
    resume_ckpt = None
    if resume_path:
        resume_ckpt = _load_model_weights(
            model,
            device,
            resume_path,
            use_ema=False,
            log=is_main_process,
        )
    elif init_ckpt_path:
        _load_model_weights(
            model,
            device,
            init_ckpt_path,
            use_ema=init_use_ema,
            log=is_main_process,
        )
    elif init_use_ema and is_main_process:
        print("[warn] cfg.train.init_use_ema=true has no effect without train.init_ckpt.")

    # Optional compile (guarded)
    if getattr(cfg.train, "torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=getattr(cfg.train, "compile_mode", "max-autotune"))
            if is_main_process:
                print("[info] torch.compile enabled.")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}. Continuing without compile.")

    if dist_ctx.get("is_distributed", False):
        ddp_kwargs = {
            "broadcast_buffers": False,
            "find_unused_parameters": bool(dist_ctx.get("find_unused_parameters", False)),
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DDP(model, **ddp_kwargs)

    # Optimizer
    opt_name = str(getattr(cfg.train, "optimizer", "adam")).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=float(getattr(cfg.train, "weight_decay", 0.0) or 0.0))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Track parameters for numerical guards (preserves ordering for rollbacks)
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    try:
        ema_decay = float(getattr(cfg.train, "ema_decay", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cfg.train.ema_decay must be numeric; got "
            f"{getattr(cfg.train, 'ema_decay', None)!r}."
        ) from exc
    if not math.isfinite(ema_decay) or ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError(
            f"cfg.train.ema_decay must be 0 (disabled) or in (0, 1); got {ema_decay}."
        )
    ema_use_for_eval = bool(getattr(cfg.train, "ema_use_for_eval", True))
    ema: Optional[ExponentialMovingAverage] = None
    if ema_decay > 0.0:
        ema = ExponentialMovingAverage(_unwrap_model(model), decay=ema_decay)
        ema_payload = resume_ckpt.get("ema") if isinstance(resume_ckpt, dict) else None
        if isinstance(ema_payload, dict):
            saved_decay = float(ema_payload.get("decay", ema_decay))
            ema.load_state_dict(ema_payload)
            if is_main_process:
                print(
                    f"[info] Restored EMA state with {ema.num_updates} updates "
                    f"(saved decay={saved_decay:.6g}, active decay={ema_decay:.6g})."
                )
        elif is_main_process:
            print(
                f"[info] EMA enabled with decay={ema_decay:.6g}; initialized from "
                "the loaded online model."
            )
    elif is_main_process:
        print("[info] EMA disabled (cfg.train.ema_decay=0).")

    # AMP setup for CL (FM runs in FP32 for stability/second-order)
    amp_dtype_str = (cfg.train.amp_dtype or "").lower() if getattr(cfg.train, "amp_dtype", None) else ""
    use_cuda = (device.type == "cuda")
    amp_ctx = nullcontext()
    scaler = None
    if use_cuda and amp_dtype_str in {"fp16", "float16"}:
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        if is_main_process:
            print("[info] AMP fp16 enabled (CL only).")
    elif use_cuda and amp_dtype_str in {"bf16", "bfloat16"}:
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if is_main_process:
            print("[info] AMP bf16 enabled (CL only).")
    else:
        if is_main_process:
            print("[info] AMP disabled.")

    resume_state: Dict[str, Any] = {}
    if isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                if is_main_process:
                    print("[info] Restored optimizer state from checkpoint.")
                if bool(getattr(cfg.train, "override_optimizer_lr_on_resume", False)):
                    resume_lr = float(getattr(cfg.train, "lr", optimizer.param_groups[0].get("lr", 0.0)))
                    for pg in optimizer.param_groups:
                        pg["lr"] = resume_lr
                    if is_main_process:
                        print(f"[info] Overrode restored optimizer LR with cfg.train.lr={resume_lr:.3e}.")
            except Exception as e:
                print(f"[warn] Could not restore optimizer state: {e}")
        if scaler is not None and "scaler" in resume_ckpt and resume_ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(resume_ckpt["scaler"])
                if is_main_process:
                    print("[info] Restored AMP scaler state from checkpoint.")
            except Exception as e:
                print(f"[warn] Could not restore AMP scaler state: {e}")
        resume_rng_state = _checkpoint_rng_state_for_rank(
            resume_ckpt,
            rank=int(dist_ctx.get("rank", 0)),
            is_distributed=bool(dist_ctx.get("is_distributed", False)),
        )
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
        elif (
            dist_ctx.get("is_distributed", False)
            and "rng_state" in resume_ckpt
        ):
            print(
                f"[warn] Legacy checkpoint has no RNG state for rank "
                f"{dist_ctx.get('rank', 0)}; retaining this rank's seeded RNG state."
            )
        resume_state = {
            "start_iter": int(resume_ckpt.get("iteration", resume_ckpt.get("iter", 0)) or 0),
            "history": resume_ckpt.get("history"),
            "last_plot_iter": int(resume_ckpt.get("last_plot_iter", 0) or 0),
            "train_stats": resume_ckpt.get("train_stats"),
        }

    # Data loader iterators (separate for CL and FM by default)
    base_loader = datamodule.train_dataloader()
    if dist_ctx.get("is_distributed", False):
        cl_loader, cl_sampler = _build_distributed_train_loader(
            base_loader,
            dist_ctx=dist_ctx,
            sampler_seed=int(cfg.train.seed),
            shuffle=dist_ctx.get("sampler_shuffle", True),
        )
        fm_loader, fm_sampler = _build_distributed_train_loader(
            base_loader,
            dist_ctx=dist_ctx,
            sampler_seed=int(cfg.train.seed) + 1337,
            shuffle=dist_ctx.get("sampler_shuffle", True),
        )
    else:
        cl_loader = base_loader
        fm_loader = datamodule.train_dataloader()
        cl_sampler = None
        fm_sampler = None
    cl_iter = iter(cl_loader)
    fm_iter = iter(fm_loader)
    cl_sampler_state = {"sampler": cl_sampler, "epoch": 0} if cl_sampler is not None else None
    fm_sampler_state = {"sampler": fm_sampler, "epoch": 0} if fm_sampler is not None else None

    def _next(iter_ref, loader, sampler_state=None):
        try:
            batch = next(iter_ref)
        except StopIteration:
            if sampler_state and sampler_state.get("sampler") is not None:
                sampler_state["epoch"] += 1
                sampler_state["sampler"].set_epoch(sampler_state["epoch"])
            iter_ref = iter(loader)
            batch = next(iter_ref)
        return batch, iter_ref

    # ----------------------------- Training loop -----------------------------
    model.train()
    max_iters = int(cfg.train.max_iters)

    history = {
        "iter": [],
        "loss_cl": [],
        "loss_fm": [],
        "loss_total": [],
        "acc_rate": [],
        "warmup_move_rate": [],
        "prop_total": [],
        "acc_total": [],
        "step_total": [],
        "move_total": [],
        "fm_loss_nodes": [],
        "fm_loss_edges": [],
        "fm_ediff_mu": [],
        "fm_ediff_sd": [],
        "fm_grad_mu": [],
        "fm_grad_sd": [],
        "energy_noise_mean": [],
        "energy_noise_std": [],
        "energy_data_mean": [],
        "energy_data_std": [],
        "neg_valid_pct": [],
        "neg_connected_pct": [],
    }
    last_interp = None
    start_iter = max(0, int(resume_state.get("start_iter", 0) or 0))
    resume_history = resume_state.get("history")
    if isinstance(resume_history, dict):
        for key, value in resume_history.items():
            if isinstance(value, list):
                history[key] = value
        if is_main_process:
            print(f"[info] Restored training history through iteration {start_iter}.")

    plot_hist_cfg = getattr(cfg.train, "plot_history", None)
    plot_history_enabled = bool(getattr(plot_hist_cfg, "enabled", False)) if plot_hist_cfg is not None else False
    plot_history_every = 0
    plot_history_on_checkpoint = False
    run_viz_during_training = bool(getattr(cfg.viz, "run_during_training", False))
    if plot_hist_cfg is not None:
        try:
            plot_history_every = int(getattr(plot_hist_cfg, "every", 0) or 0)
        except (TypeError, ValueError):
            print(
                f"[warn] Ignoring invalid cfg.train.plot_history.every="
                f"{getattr(plot_hist_cfg, 'every', None)!r}; expected integer."
            )
            plot_history_every = 0
        if plot_history_every < 0:
            print(f"[warn] cfg.train.plot_history.every={plot_history_every} < 0; disabling periodic plots.")
            plot_history_every = 0
        plot_history_on_checkpoint = bool(getattr(plot_hist_cfg, "on_checkpoint", False))
    last_plot_iter = int(resume_state.get("last_plot_iter", 0) or 0)

    log_stats_enabled = bool(getattr(cfg.general, "log_stats", True))
    cl_batch_size = int(getattr(cfg.train, "batch_size", 1) or 1)

    # Loss weights & FM config
    lambda_cl = lambda_cl_cfg
    lambda_fm = lambda_fm_cfg
    cl_active = (not math.isclose(lambda_cl, 0.0)) and cl_steps_cfg > 0
    try:
        cl_warmup_iters = int(
            getattr(
                cfg.train,
                "cl_warmup_iters",
                getattr(cfg.train, "cd_warmup_iters", 0),
            )
            or 0
        )
    except (TypeError, ValueError):
        print(
            f"[warn] Invalid cfg.train.cl_warmup_iters={getattr(cfg.train, 'cl_warmup_iters', None)!r}; disabling CL warmup."
        )
        cl_warmup_iters = 0
    if cl_warmup_iters < 0:
        print(f"[warn] cfg.train.cl_warmup_iters={cl_warmup_iters} < 0; disabling CL warmup.")
        cl_warmup_iters = 0
    if cl_active and cl_warmup_iters > 0:
        saved_cl_warmup_start = None
        saved_train_stats = resume_state.get("train_stats")
        if isinstance(saved_train_stats, dict):
            saved_cl_warmup_start = saved_train_stats.get("cl_warmup_start_iter")
        try:
            cl_warmup_start_iter = int(saved_cl_warmup_start)
        except (TypeError, ValueError):
            cl_warmup_start_iter = start_iter
        if is_main_process:
            print(
                f"[info] CL loss weight warmup enabled for {cl_warmup_iters} iterations "
                f"after iteration {cl_warmup_start_iter}."
            )
    else:
        cl_warmup_start_iter = start_iter

    raw_cl_threshold = getattr(
        cfg.train,
        "cl_loss_threshold",
        getattr(cfg.train, "cd_loss_threshold", False),
    )
    cl_loss_threshold = None
    if raw_cl_threshold not in (False, None):
        try:
            cl_loss_threshold = float(raw_cl_threshold)
        except (TypeError, ValueError):
            print(
                f"[warn] Ignoring invalid cfg.train.cl_loss_threshold={raw_cl_threshold!r}; expected numeric."
            )
            cl_loss_threshold = None
    if cl_loss_threshold is not None and cl_loss_threshold <= 0.0:
        print(
            f"[warn] cfg.train.cl_loss_threshold={cl_loss_threshold} <= 0; disabling CL loss capping."
        )
        cl_loss_threshold = None
    cl_clip_mode = str(getattr(cfg.train, "cl_clip_mode", "batch")).strip().lower()
    cl_clip_mode = cl_clip_mode.replace("-", "_")
    valid_cl_clip_modes = {
        "batch",
        "paired_all",
        "paired_connected",
        "size_matched_connected",
    }
    if cl_clip_mode not in valid_cl_clip_modes:
        print(
            f"[warn] Unknown cfg.train.cl_clip_mode={cl_clip_mode!r}; "
            "using 'batch'."
        )
        cl_clip_mode = "batch"
    try:
        cl_bad_negative_weight = float(getattr(cfg.train, "cl_bad_negative_weight", 1.0))
    except (TypeError, ValueError):
        print(
            f"[warn] Invalid cfg.train.cl_bad_negative_weight="
            f"{getattr(cfg.train, 'cl_bad_negative_weight', None)!r}; using 1.0."
        )
        cl_bad_negative_weight = 1.0
    if cl_bad_negative_weight < 1.0:
        print(
            f"[warn] cfg.train.cl_bad_negative_weight={cl_bad_negative_weight} < 1; using 1.0."
        )
        cl_bad_negative_weight = 1.0

    nan_guard_cfg = None
    try:
        nan_guard_cfg = cfg.train.get("nan_guard", None)
    except AttributeError:
        nan_guard_cfg = None
    nan_guard_enabled = True
    nan_guard_lr_backoff = 0.5
    nan_guard_min_lr = 1.0e-6
    nan_guard_reset_state = False
    if nan_guard_cfg is not None:
        nan_guard_enabled = bool(getattr(nan_guard_cfg, "enabled", True))
        try:
            nan_guard_lr_backoff = float(getattr(nan_guard_cfg, "lr_backoff", 0.5))
        except (TypeError, ValueError):
            print(
                f"[warn] Invalid cfg.train.nan_guard.lr_backoff={getattr(nan_guard_cfg, 'lr_backoff', None)!r}; using 0.5"
            )
            nan_guard_lr_backoff = 0.5
        try:
            nan_guard_min_lr = float(getattr(nan_guard_cfg, "min_lr", nan_guard_min_lr))
        except (TypeError, ValueError):
            print(
                f"[warn] Invalid cfg.train.nan_guard.min_lr={getattr(nan_guard_cfg, 'min_lr', None)!r}; using {nan_guard_min_lr}"
            )
        nan_guard_reset_state = bool(getattr(nan_guard_cfg, "reset_optimizer_state", nan_guard_reset_state))
    if nan_guard_lr_backoff <= 0.0 or nan_guard_lr_backoff >= 1.0:
        print(
            f"[warn] cfg.train.nan_guard.lr_backoff={nan_guard_lr_backoff} outside (0,1); resetting to 0.5"
        )
        nan_guard_lr_backoff = 0.5
    if nan_guard_min_lr <= 0.0:
        print(
            f"[warn] cfg.train.nan_guard.min_lr={nan_guard_min_lr} <= 0; resetting to 1e-6"
        )
        nan_guard_min_lr = 1.0e-6

    # OT/FM hyperparams
    ocfg = getattr(cfg, "ot", None)
    cost_mode = str(getattr(ocfg, "cost", "hist")) if ocfg is not None else "hist"
    alpha = float(getattr(ocfg, "alpha", 1.0)) if ocfg is not None else 1.0
    beta = float(getattr(ocfg, "beta", 1.0)) if ocfg is not None else 1.0
    gamma = float(getattr(ocfg, "gamma", 1.0)) if ocfg is not None else 1.0
    ncfg = getattr(cfg, "noise", None)
    fm_noise_transition = str(getattr(ncfg, "transition", getattr(cfg.model, "transition", "marginal")))

    fcfg = getattr(cfg, "fm", None)
    fm_enabled = bool(getattr(fcfg, "enabled", True)) if fcfg is not None else True
    fm_bs_cap = int(getattr(fcfg, "batch_size", int(getattr(cfg.train, "batch_size", 512))))
    fm_k = float(getattr(fcfg, "k", 1.0))
    fm_w_node = float(getattr(fcfg, "node_weight", 1.0))
    fm_w_edge = float(getattr(fcfg, "edge_weight", 1.0))
    fm_grad_clip_val = float(getattr(fcfg, "grad_value_clip", 10.0))
    fm_loss_type = str(getattr(fcfg, "loss", "huber")).lower()
    if fm_loss_type in {"mse", "l2", "l2_loss"}:
        fm_loss_type = "mse"
    elif fm_loss_type in {"huber", "smoothl1", "smooth_l1"}:
        fm_loss_type = "huber"
    else:
        print(f"[warn] Unknown cfg.fm.loss={fm_loss_type!r}; defaulting to 'huber'.")
        fm_loss_type = "huber"
    fm_huber_delta = float(getattr(fcfg, "robust_delta", 1.0))
    fm_detect_anomaly = bool(getattr(fcfg, "detect_anomaly", False))
    global_clip_norm = float(getattr(cfg.train, "clip_grad", 0.0) or 0.0)

    # Optional chain warmup (pre-CL MCMC) configuration
    chain_warmup_train = resolve_chain_warmup(
        getattr(cfg.train, "chain_warmup", None),
        fallback=cfg.train,
        default_gwd_beta=float(getattr(cfg.train, "gwd_beta", 1.0)),
    )
    warmup_energy_threshold, warmup_energy_source = _resolve_warmup_energy_threshold(cfg)
    warmup_energy_disabled = (
        warmup_energy_source is not None and warmup_energy_threshold is None
    )
    if chain_warmup_train.enabled:
        vectorized_simple_warmup = (
            sampler.should_vectorize_simple_warmup(
                chain_warmup_train.proposal,
                vectorized=chain_warmup_train.vectorized,
            )
            and warmup_energy_threshold is None
        )
        warmup_msg = (
            "[info] Chain warmup enabled: "
            f"proposal={chain_warmup_train.proposal}, steps={chain_warmup_train.steps}, "
            f"vectorized={vectorized_simple_warmup}"
        )
        if chain_warmup_train.proposal in {"dlangevin", "dlang", "dl", "dlangevin_nomh", "dlang_nomh", "dl_nomh"}:
            warmup_msg += (
                f", dl_beta={chain_warmup_train.dl_beta}"
                f", dl_lambda_X={chain_warmup_train.dl_lambda_X}"
                f", dl_lambda_E={chain_warmup_train.dl_lambda_E}"
            )
        if chain_warmup_train.proposal in {"simple", "simple_ver2", "simple_v2"} and chain_warmup_train.simple_n_edits is not None:
            warmup_msg += f", simple_n_edits={chain_warmup_train.simple_n_edits}"
        print(warmup_msg)

    if chain_warmup_train.enabled:
        if warmup_energy_threshold is not None:
            src_note = f" (from {warmup_energy_source})" if warmup_energy_source else ""
            print(
                f"[info] Chain warmup early-stop energy threshold={warmup_energy_threshold:.6f}{src_note}"
            )
        elif warmup_energy_disabled:
            print(
                f"[info] Chain warmup early-stop energy threshold disabled by {warmup_energy_source}."
            )
        else:
            print("[info] Chain warmup early-stop energy threshold not set.")

    train_proposal = str(getattr(cfg.train, "proposal", "random") or "random")
    train_gwd_beta = float(getattr(cfg.train, "gwd_beta", 1.0))
    simple_n_edits_train = _parse_simple_n_edits(cfg.train, "cfg.train")
    (
        base_dl_beta_train,
        base_dl_lambda_X_train,
        base_dl_lambda_E_train,
        dual_kwargs_train,
    ) = resolve_dl_parameters(cfg.train)
    two_beta_kwargs_train = (
        resolve_two_beta_kwargs(train_proposal, cfg.train) if cl_active else {}
    )
    two_beta_annealing_kwargs_train = (
        resolve_two_beta_annealing_kwargs(train_proposal, cfg.train)
        if cl_active
        else {}
    )
    if (two_beta_kwargs_train or two_beta_annealing_kwargs_train) and dual_kwargs_train:
        raise ValueError(
            "Two-beta DLangevin does not support the near/far energy-threshold parameter split."
        )
    anneal_kwargs_train: Dict[str, Any] = {}
    if train_proposal.lower() in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}:
        dl_beta_init_train = float(getattr(cfg.train, "dl_beta_init", base_dl_beta_train))
        dl_beta_final_train = float(getattr(cfg.train, "dl_beta_final", base_dl_beta_train))
        try:
            dl_beta_anneal_steps_train = int(getattr(cfg.train, "dl_beta_anneal_steps", cl_steps_cfg))
        except (TypeError, ValueError):
            dl_beta_anneal_steps_train = int(cl_steps_cfg)
        anneal_kwargs_train.update(
            dl_beta_init=dl_beta_init_train,
            dl_beta_final=dl_beta_final_train,
            dl_beta_anneal_steps=dl_beta_anneal_steps_train,
        )
    if cl_active:
        sampler_msg = (
            "[info] CL sampler config: "
            f"proposal={train_proposal}, cl_steps={cl_steps_cfg}, gwd_beta={train_gwd_beta:.4g}, "
        )
        if two_beta_annealing_kwargs_train:
            sampler_msg += (
                f"dl_beta_prop={two_beta_annealing_kwargs_train['dl_beta_prop']:.4g}, "
                f"dl_beta_mh="
                f"{two_beta_annealing_kwargs_train['dl_beta_mh_init']:.4g}->"
                f"{two_beta_annealing_kwargs_train['dl_beta_mh_final']:.4g}, "
                f"dl_beta_mh_anneal_steps="
                f"{two_beta_annealing_kwargs_train['dl_beta_mh_anneal_steps']}, "
            )
        elif two_beta_kwargs_train:
            sampler_msg += (
                f"dl_beta_prop={two_beta_kwargs_train['dl_beta_prop']:.4g}, "
                f"dl_beta_mh={two_beta_kwargs_train['dl_beta_mh']:.4g}, "
            )
        else:
            sampler_msg += f"dl_beta={base_dl_beta_train:.4g}, "
        sampler_msg += (
            f"dl_lambda_X={base_dl_lambda_X_train:.4g}, "
            f"dl_lambda_E={base_dl_lambda_E_train:.4g}"
        )
        if simple_n_edits_train is not None:
            sampler_msg += f", simple_n_edits={simple_n_edits_train}"
        if dual_kwargs_train:
            sampler_msg += f", energy_threshold={dual_kwargs_train.get('energy_split_threshold')}"
        print(sampler_msg)


    # Accumulators for training summary
    train_total_time = 0.0
    train_total_accepts = 0
    train_total_proposals = 0
    train_throughput_sum = 0.0
    train_iters = 0
    train_stats = resume_state.get("train_stats")
    if isinstance(train_stats, dict):
        train_total_time = float(train_stats.get("train_total_time", train_total_time) or 0.0)
        train_total_accepts = int(train_stats.get("train_total_accepts", train_total_accepts) or 0)
        train_total_proposals = int(train_stats.get("train_total_proposals", train_total_proposals) or 0)
        train_throughput_sum = float(train_stats.get("train_throughput_sum", train_throughput_sum) or 0.0)
        train_iters = int(train_stats.get("train_iters", train_iters) or 0)
        if is_main_process:
            print(f"[info] Restored training summary accumulators from checkpoint at iteration {start_iter}.")

    # FM warmup schedule (applies to the unified optimizer)
    wcfg = getattr(cfg.fm, "warmup", None)
    warmup_enabled = bool(getattr(wcfg, "enabled", False)) if wcfg is not None else False
    warmup_iters = int(getattr(wcfg, "iters", 1000)) if wcfg is not None else 1000
    warmup_start_factor = float(getattr(wcfg, "start_factor", 0.1)) if wcfg is not None else 0.1
    base_lr = float(getattr(cfg.train, "lr", 2.0e-4))

    if is_main_process and start_iter > 0:
        print(f"[info] Resuming training at iteration {start_iter + 1}/{max_iters}.")
    if is_main_process and start_iter >= max_iters:
        print(f"[info] Checkpoint iteration {start_iter} is already >= train.max_iters={max_iters}; no training steps to run.")

    for it in range(start_iter, max_iters):
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cl_loss_clipped = False
        cl_capped_pairs = 0
        cl_cap_eligible_pairs = 0
        neg_diag = {"total": 0, "valid": 0, "connected": 0}

        # ----------------------- LR warmup (global) -----------------------
        if warmup_enabled and warmup_iters > 0:
            if it < warmup_iters:
                warmup_t = float(it + 1) / float(warmup_iters)
                scale = warmup_start_factor + (1.0 - warmup_start_factor) * warmup_t
            else:
                scale = 1.0
            lr_now = base_lr * scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        if cl_warmup_iters > 0 and lambda_cl != 0.0:
            cl_warmup_progress = max(0, (it + 1) - cl_warmup_start_iter)
            cl_weight_scale = min(
                1.0,
                float(cl_warmup_progress) / float(cl_warmup_iters),
            )
        else:
            cl_weight_scale = 1.0
        lambda_cl_iter = lambda_cl * cl_weight_scale

        cl_steps = int(cl_steps_cfg)
        run_cl = lambda_cl_iter != 0.0 and cl_steps > 0
        warmup_energy_threshold_iter = warmup_energy_threshold
        if run_cl and not warmup_energy_disabled:
            if is_main_process:
                try:
                    summary = _record_expected_energy(
                        model=_unwrap_model(model),
                        datamodule=datamodule,
                        dataset_infos=dataset_infos,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        transition=cfg.model.transition,
                        batch_size=cl_batch_size,
                        output_path="expected_energy.json",
                        iteration=it + 1,
                    )
                    warmup_energy_threshold_iter = float(summary["data"]["mean"])
                except Exception as exc:
                    print(f"[warn] Expected energy update failed at iter {it+1}: {exc}")
            if dist_ctx.get("is_distributed", False):
                value = (
                    warmup_energy_threshold_iter
                    if warmup_energy_threshold_iter is not None
                    else float("nan")
                )
                tensor = torch.tensor([value], device=device, dtype=torch.float32)
                dist.broadcast(tensor, src=0)
                warmup_energy_threshold_iter = float(tensor.item())
                if math.isnan(warmup_energy_threshold_iter):
                    warmup_energy_threshold_iter = None
        if run_cl:
            # ----------------------- CL batch & loss (as-is) -----------------------
            batch_cl, cl_iter = _next(cl_iter, cl_loader, cl_sampler_state)
            dense_data_cl, node_mask_cl = utils.to_dense(
                batch_cl.x, batch_cl.edge_index, batch_cl.edge_attr, batch_cl.batch
            )
            graphs_cl = dense_data_cl.mask(node_mask_cl, collapse=True).split(node_mask_cl)
            node_list_cl = [g.X.long().cpu() for g in graphs_cl]
            edge_list_cl = [g.E.long().cpu() for g in graphs_cl]
            B_cl = len(node_list_cl)

            # Positive energies (with gradients)
            _sync_if_cuda(device)
            t_pos0 = time.perf_counter()
            with amp_ctx:
                pos_E = sampler.energy_batch(
                    model=model,
                    node_types_list=node_list_cl,
                    edge_types_list=edge_list_cl,
                    dataset_info=dataset_infos,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    detach=False,
                )  # (B,)
            _sync_if_cuda(device)
            t_pos1 = time.perf_counter()

            # Negative phase: batched CL
            _sync_if_cuda(device)
            t_mcmc0 = time.perf_counter()
            init_nodes = [t.clone() for t in node_list_cl]
            init_edges = [t.clone() for t in edge_list_cl]
            gamma_train = float(getattr(cfg.train, "gamma_train", 0.0))
            n_rand = max(0, min(B_cl, int(round(B_cl * gamma_train))))
            if n_rand > 0:
                replace_idx = torch.randperm(B_cl)[:n_rand]
                rand_counts = [int(init_nodes[int(idx)].shape[0]) for idx in replace_idx]
                rand_graphs = initialize_random_graphs_with_counts(
                    counts=rand_counts,
                    dataset_info=dataset_infos,
                    device=device,
                    transition=cfg.model.transition,
                )
                rand_nodes = [nt for (nt, _) in rand_graphs]
                rand_edges = [et for (_, et) in rand_graphs]
                for i, idx in enumerate(replace_idx):
                    init_nodes[idx] = rand_nodes[i]
                    init_edges[idx] = rand_edges[i]

            with torch.no_grad():
                warmup_moves = 0
                warmup_steps_total = 0
                warmup_steps_done = 0
                warmup_stop_reason = "disabled"
                warmup_energy_mean = None
                if chain_warmup_train.enabled:
                    (
                        init_nodes,
                        init_edges,
                        warmup_moves,
                        warmup_steps_total,
                        warmup_steps_done,
                        warmup_stop_reason,
                        warmup_energy_mean,
                    ) = _run_chain_warmup(
                        init_nodes=init_nodes,
                        init_edges=init_edges,
                        model=model,
                        dataset_infos=dataset_infos,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        chain_cfg=chain_warmup_train,
                        amp_dtype=amp_dtype_str,
                        energy_threshold=warmup_energy_threshold_iter,
                    )
                neg_nodes, neg_edges, n_accepts, n_steps_total, mcmc_stats = sampler.mcmc_sample_batch(
                    model=model,
                    dataset_info=dataset_infos,
                    node_types_list=init_nodes,
                    edge_types_list=init_edges,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    steps=cl_steps,
                    device=device,
                    proposal=train_proposal,
                    gwd_beta=train_gwd_beta,
                    simple_n_edits=simple_n_edits_train,
                    dl_beta=base_dl_beta_train,
                    dl_lambda_X=base_dl_lambda_X_train,
                    dl_lambda_E=base_dl_lambda_E_train,
                    amp_dtype=amp_dtype_str,
                    collect_stats=log_stats_enabled,
                    **dual_kwargs_train,
                    **anneal_kwargs_train,
                    **two_beta_kwargs_train,
                    **two_beta_annealing_kwargs_train,
                )
            _sync_if_cuda(device)
            t_mcmc1 = time.perf_counter()
            neg_diag = _negative_sample_diagnostics(
                neg_nodes,
                neg_edges,
                dataset_infos,
            )

            # Negative energies (with gradients)
            _sync_if_cuda(device)
            t_neg0 = time.perf_counter()
            with amp_ctx:
                neg_E = sampler.energy_batch(
                    model=model,
                    node_types_list=neg_nodes,
                    edge_types_list=neg_edges,
                    dataset_info=dataset_infos,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    detach=False,
                )  # (B,)
                if cl_clip_mode in {
                    "paired_all",
                    "paired_connected",
                    "size_matched_connected",
                }:
                    loss_cl_terms = pos_E - neg_E
                    valid_mask = neg_diag.get("valid_mask", [])
                    connected_mask = neg_diag.get("connected_mask", [])
                    good_negative_mask, cap_mask = _pairwise_cl_masks(
                        valid_mask,
                        connected_mask,
                        int(loss_cl_terms.numel()),
                        loss_cl_terms.device,
                        cl_clip_mode,
                    )
                    if cl_loss_threshold is not None:
                        unclipped_cl_terms = loss_cl_terms.detach()
                        capped_pair_mask = cap_mask & (
                            unclipped_cl_terms < -cl_loss_threshold
                        )
                        cl_cap_eligible_pairs = int(cap_mask.sum().item())
                        cl_capped_pairs = int(capped_pair_mask.sum().item())
                        if cap_mask.any():
                            capped_terms = loss_cl_terms.clone()
                            capped_terms[cap_mask] = torch.clamp(
                                capped_terms[cap_mask],
                                min=-cl_loss_threshold,
                            )
                            loss_cl_terms = capped_terms
                        cl_loss_clipped = cl_capped_pairs > 0
                    if cl_bad_negative_weight > 1.0 and loss_cl_terms.numel() > 0:
                        weights = torch.ones_like(loss_cl_terms)
                        weights[~good_negative_mask] = cl_bad_negative_weight
                        loss_cl = (weights * loss_cl_terms).sum() / weights.sum().clamp_min(1.0)
                    else:
                        loss_cl = loss_cl_terms.mean()
                else:
                    loss_cl = (pos_E - neg_E).mean()
                    if cl_loss_threshold is not None:
                        unclipped_cl = loss_cl.detach()
                        loss_cl = torch.clamp(loss_cl, min=-cl_loss_threshold)
                        cl_cap_eligible_pairs = 1
                        if unclipped_cl.item() < -cl_loss_threshold:
                            cl_loss_clipped = True
                            cl_capped_pairs = 1

            # Pos-Neg edit distance (training only)
            posneg_stats = None
            if log_stats_enabled:
                try:
                    total_node_edits = 0
                    total_edge_edits = 0
                    cnt = 0
                    for p_nt, p_et, n_nt, n_et in zip(node_list_cl, edge_list_cl, neg_nodes, neg_edges):
                        if p_nt.shape == n_nt.shape and p_et.shape == n_et.shape:
                            node_edits = int((p_nt != n_nt).sum().item())
                            if n_et.numel() == 0:
                                edge_edits = 0
                            else:
                                diff = (p_et != n_et)
                                edge_edits = int(torch.triu(diff, diagonal=1).sum().item())
                            total_node_edits += node_edits
                            total_edge_edits += edge_edits
                            cnt += 1
                    if cnt > 0:
                        posneg_stats = {
                            "mean_posneg_nodes": total_node_edits / cnt,
                            "mean_posneg_edges": total_edge_edits / cnt,
                            "mean_posneg_total": (total_node_edits + total_edge_edits) / cnt,
                        }
                except Exception:
                    posneg_stats = None
            _sync_if_cuda(device)
            t_neg1 = time.perf_counter()
        else:
            neg_nodes = []
            neg_edges = []
            n_accepts = 0
            n_steps_total = 0
            mcmc_stats = None
            loss_cl = torch.tensor(0.0, device=device)
            cl_loss_clipped = False
            posneg_stats = None
            warmup_moves = 0
            warmup_steps_total = 0
            warmup_steps_done = 0
            warmup_stop_reason = "disabled"
            warmup_energy_mean = None
            # Initialize timing values when contrastive sampling is disabled.
            t_pos0 = t_pos1 = t_mcmc0 = t_mcmc1 = t_neg0 = t_neg1 = time.perf_counter()
            B_cl = 0

        # ------------------------ FM batch & loss (OT) -------------------------
        t_fm0 = time.perf_counter()
        loss_fm = torch.tensor(0.0, device=device)
        fm_pairs_count = 0
        fm_loss_nodes = torch.tensor(0.0, device=device)
        fm_loss_edges = torch.tensor(0.0, device=device)
        fm_ediff_mu = float("nan")
        fm_ediff_sd = float("nan")
        fm_gs_mu = float("nan")
        fm_gs_sd = float("nan")
        mu_noise_it = float("nan")
        sd_noise_it = float("nan")
        mu_data_it = float("nan")
        sd_data_it = float("nan")
        if fm_enabled and lambda_fm != 0.0:
            batch_fm, fm_iter = _next(fm_iter, fm_loader, fm_sampler_state)
            dense_data_fm, node_mask_fm = utils.to_dense(
                batch_fm.x, batch_fm.edge_index, batch_fm.edge_attr, batch_fm.batch
            )
            graphs_fm = dense_data_fm.mask(node_mask_fm, collapse=True).split(node_mask_fm)
            data_nodes = [g.X.long().cpu() for g in graphs_fm]
            data_edges = [g.E.long().cpu() for g in graphs_fm]

            # Noise graphs matching sizes
            counts_batch = [int(t.shape[0]) for t in data_nodes]
            noise_graphs = initialize_random_graphs_with_counts(
                counts=counts_batch,
                dataset_info=dataset_infos,
                device=device,
                transition=fm_noise_transition,
            )
            noise_nodes = [nt for (nt, _) in noise_graphs]
            noise_edges = [et for (_, et) in noise_graphs]

            # OT pairing (bucketed by graph size)
            dx = int(dataset_infos.output_dims["X"])
            de = int(dataset_infos.output_dims["E"])
            pairs, _ = minibatch_ot_pairs(
                noise_nodes, noise_edges, data_nodes, data_edges,
                dx=dx, de=de, cost_mode=cost_mode, alpha=alpha, beta=beta, gamma=gamma, verbose=False
            )

            if pairs:
                # Subsample to cap compute
                if fm_bs_cap < len(pairs):
                    sel_idxs = random.sample(range(len(pairs)), fm_bs_cap)
                else:
                    sel_idxs = list(range(len(pairs)))
                fm_pairs_count = len(sel_idxs)

                taus_batch = torch.rand(fm_pairs_count).tolist()
                it_nodes: List[torch.Tensor] = []
                it_edges: List[torch.Tensor] = []
                target_X_list: List[torch.Tensor] = []
                target_E_list: List[torch.Tensor] = []

                for b, pidx in enumerate(sel_idxs):
                    ia, jb = pairs[pidx]
                    nt_n = noise_nodes[ia]
                    et_n = noise_edges[ia]
                    nt_d = data_nodes[jb]
                    et_d = data_edges[jb]
                    tau = float(taus_batch[b])
                    nt_i, et_i = sample_interpolated_graph(nt_n, et_n, nt_d, et_d, tau)
                    it_nodes.append(nt_i)
                    it_edges.append(et_i)

                    Xn = F.one_hot(nt_n, num_classes=dx).to(torch.float32)
                    Xd = F.one_hot(nt_d, num_classes=dx).to(torch.float32)
                    target_X_list.append((Xd - Xn).to(device))

                    En = F.one_hot(et_n, num_classes=de).to(torch.float32)
                    Ed = F.one_hot(et_d, num_classes=de).to(torch.float32)
                    target_E_list.append((Ed - En).to(device))

                # Compute gradients wrt inputs; keep graph for second-order
                E_tau, gX_list, gE_list = sampler.energy_and_grads_batch(
                    model=model,
                    node_types_list=it_nodes,
                    edge_types_list=it_edges,
                    dataset_info=dataset_infos,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    create_graph=True,
                    detach_energies=True,
                    detach_grads=False,
                )

                node_losses: List[torch.Tensor] = []
                edge_losses: List[torch.Tensor] = []
                for gX, gE, tX, tE in zip(gX_list, gE_list, target_X_list, target_E_list):
                    # sanitize + clip grads
                    gX = torch.nan_to_num(gX).clamp_(-fm_grad_clip_val, fm_grad_clip_val)
                    gE = torch.nan_to_num(gE).clamp_(-fm_grad_clip_val, fm_grad_clip_val)

                    predX = -fm_k * gX
                    if fm_loss_type == "mse":
                        node_losses.append(F.mse_loss(predX, tX))
                    else:
                        node_losses.append(F.huber_loss(predX, tX, delta=fm_huber_delta))

                    nloc = int(gE.shape[0])
                    if nloc > 1:
                        mask = torch.triu(torch.ones((nloc, nloc), dtype=torch.bool, device=device), diagonal=1)
                        predE = -fm_k * gE[mask]
                        targE = tE[mask]
                        if fm_loss_type == "mse":
                            edge_losses.append(F.mse_loss(predE, targE))
                        else:
                            edge_losses.append(F.huber_loss(predE, targE, delta=fm_huber_delta))
                    else:
                        edge_losses.append(torch.tensor(0.0, device=device))

                node_stack = torch.stack(node_losses) if node_losses else torch.zeros(0, device=device)
                edge_stack = torch.stack(edge_losses) if edge_losses else torch.zeros(0, device=device)
                loss_nodes = node_stack.mean() if node_stack.numel() else torch.tensor(0.0, device=device)
                loss_edges = edge_stack.mean() if edge_stack.numel() else torch.tensor(0.0, device=device)
                loss_fm = fm_w_node * loss_nodes + fm_w_edge * loss_edges
                fm_loss_nodes = loss_nodes.detach()
                fm_loss_edges = loss_edges.detach()

                # Extra FM diagnostics: energy gap (data - noise) and grad strength
                sel_noise_nodes = [noise_nodes[pairs[pidx][0]] for pidx in sel_idxs]
                sel_noise_edges = [noise_edges[pairs[pidx][0]] for pidx in sel_idxs]
                sel_data_nodes = [data_nodes[pairs[pidx][1]] for pidx in sel_idxs]
                sel_data_edges = [data_edges[pairs[pidx][1]] for pidx in sel_idxs]
                with torch.no_grad():
                    E_noise_it = sampler.energy_batch(
                        model=model,
                        node_types_list=sel_noise_nodes,
                        edge_types_list=sel_noise_edges,
                        dataset_info=dataset_infos,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        detach=True,
                    )
                    E_data_it = sampler.energy_batch(
                        model=model,
                        node_types_list=sel_data_nodes,
                        edge_types_list=sel_data_edges,
                        dataset_info=dataset_infos,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        detach=True,
                    )
                mu_noise_it, sd_noise_it = mean_std(E_noise_it)
                mu_data_it, sd_data_it = mean_std(E_data_it)
                diff_dn_it = E_data_it - E_noise_it
                fm_ediff_mu, fm_ediff_sd = mean_std(diff_dn_it)

                strengths = []
                for gX, gE in zip(gX_list, gE_list):
                    strengths.append(grad_scalar_strength(-gX, -gE))
                if strengths:
                    s_t = torch.tensor(strengths, dtype=torch.float32, device=device)
                    fm_gs_mu = float(s_t.mean().item())
                    fm_gs_sd = float(s_t.std(unbiased=False).item())
                last_interp = {
                    "noise_nodes": [nt.cpu() for nt in sel_noise_nodes],
                    "noise_edges": [et.cpu() for et in sel_noise_edges],
                    "data_nodes": [nt.cpu() for nt in sel_data_nodes],
                    "data_edges": [et.cpu() for et in sel_data_edges],
                    "mu_noise": mu_noise_it,
                    "sd_noise": sd_noise_it,
                    "mu_data": mu_data_it,
                    "sd_data": sd_data_it,
                }

        t_fm1 = time.perf_counter()

        # ------------------------------- Update -------------------------------
        _sync_if_cuda(device)
        t_back0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = lambda_cl_iter * loss_cl + lambda_fm * loss_fm
        # Cache weighted losses for logging without touching autograd graph
        loss_cl_weighted = (lambda_cl_iter * loss_cl).detach()
        loss_fm_weighted = (lambda_fm * loss_fm).detach()
        total_loss_detached = total_loss.detach()
        cl_clip_suffix = ""
        if cl_loss_clipped and cl_loss_threshold is not None:
            cl_clip_suffix = (
                f" | cl_cap=-{cl_loss_threshold:.4f} "
                f"pairs={cl_capped_pairs}/{cl_cap_eligible_pairs}"
            )

        nan_guard_reason = ""
        nan_guard_triggered = False
        prev_params = None
        if nan_guard_enabled and named_params:
            prev_params = [param.data.detach().clone() for _, param in named_params]

        ctx = torch.autograd.detect_anomaly() if fm_detect_anomaly else nullcontext()
        with ctx:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

        grads_finite = True
        if nan_guard_enabled and named_params:
            for _, param in named_params:
                grad = param.grad
                if grad is not None and not torch.isfinite(grad).all():
                    grads_finite = False
                    break

        if grads_finite:
            if global_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=global_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            nan_guard_triggered = True
            nan_guard_reason = "grads"
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()

        if nan_guard_enabled and named_params and prev_params and grads_finite:
            bad_params = [name for name, param in named_params if not torch.isfinite(param.data).all()]
            if bad_params:
                nan_guard_triggered = True
                nan_guard_reason = "params"
                for (_, param), prev in zip(named_params, prev_params):
                    param.data.copy_(prev)
                optimizer.zero_grad(set_to_none=True)

        if nan_guard_triggered and nan_guard_enabled:
            if nan_guard_reset_state:
                try:
                    for state in optimizer.state.values():
                        for value in state.values():
                            if torch.is_tensor(value):
                                value.zero_()
                except Exception:
                    pass
            if nan_guard_lr_backoff > 0.0:
                for pg in optimizer.param_groups:
                    old_lr = pg.get("lr", nan_guard_min_lr)
                    new_lr = max(old_lr * nan_guard_lr_backoff, nan_guard_min_lr)
                    pg["lr"] = new_lr
            try:
                lr_now = optimizer.param_groups[0]["lr"]
                msg_reason = nan_guard_reason or "unknown"
                reset_note = " and cleared optimizer state" if nan_guard_reset_state else ""
                print(
                    f"[diag] NaN guard rollback at iter {it+1}: reason={msg_reason}{reset_note}; lr={lr_now:.3e}."
                )
            except Exception:
                pass

        if ema is not None and grads_finite and not nan_guard_triggered:
            ema.update(_unwrap_model(model))

        nan_guard_suffix = f" | nan_guard={nan_guard_reason}" if nan_guard_triggered else ""

        _sync_if_cuda(device)
        t_back1 = time.perf_counter()

        # ------------------------------- Logging ------------------------------
        t1 = time.perf_counter()

        acc_rate = (n_accepts / n_steps_total) if n_steps_total > 0 else 0.0
        iter_time = t1 - t0
        t_pos = t_pos1 - t_pos0
        t_mcmc = t_mcmc1 - t_mcmc0
        t_neg = t_neg1 - t_neg0
        t_back = t_back1 - t_back0
        t_fm = t_fm1 - t_fm0
        throughput_graphs = B_cl if run_cl else fm_pairs_count
        throughput = (
            throughput_graphs / iter_time
            if iter_time > 0
            else float("inf")
        )

        # Update training accumulators
        train_total_time += iter_time
        train_total_accepts += int(n_accepts)
        train_total_proposals += int(n_steps_total)
        train_throughput_sum += throughput
        train_iters += 1

        prop_total_val = float("nan")
        acc_total_val = float("nan")
        step_total_val = float("nan")
        move_total_val = float("nan")
        if mcmc_stats:
            try:
                prop_total_val = float(mcmc_stats.get("mean_prop_distance_total", float("nan")))
                acc_total_val = float(mcmc_stats.get("mean_acc_distance_total", float("nan")))
                step_total_val = float(mcmc_stats.get("mean_step_acc_distance_total", float("nan")))
                overall_acc = float(mcmc_stats.get("overall_accept", acc_rate))
                move_total_val = overall_acc * mcmc_stats.get("mean_step_acc_distance_total", 0.0)
            except Exception:
                prop_total_val = acc_total_val = step_total_val = move_total_val = float("nan")
        log_vector = torch.tensor(
            [
                loss_cl_weighted.item() if run_cl else 0.0,
                loss_fm_weighted.item(),
                total_loss_detached.item(),
                throughput,
                prop_total_val,
                acc_total_val,
                step_total_val,
                move_total_val,
                fm_loss_nodes.item(),
                fm_loss_edges.item(),
                fm_ediff_mu,
                fm_ediff_sd,
                fm_gs_mu,
                fm_gs_sd,
                mu_noise_it,
                sd_noise_it,
                mu_data_it,
                sd_data_it,
            ],
            device=device,
            dtype=torch.float64,
        )
        if dist_ctx.get("is_distributed", False):
            dist.all_reduce(log_vector, op=dist.ReduceOp.SUM)
            log_vector /= dist_ctx["world_size"]
        acceptance_counts = torch.tensor(
            [
                int(n_accepts),
                int(n_steps_total),
                int(warmup_moves),
                int(warmup_steps_total),
            ],
            device=device,
            dtype=torch.long,
        )
        if dist_ctx.get("is_distributed", False):
            dist.all_reduce(acceptance_counts, op=dist.ReduceOp.SUM)
        (
            log_mcmc_accepts,
            log_mcmc_proposals,
            log_warmup_moves,
            log_warmup_attempts,
        ) = [int(x) for x in acceptance_counts.tolist()]
        log_acc_rate = (
            float(log_mcmc_accepts) / float(log_mcmc_proposals)
            if log_mcmc_proposals > 0
            else 0.0
        )
        log_warmup_move_rate = (
            float(log_warmup_moves) / float(log_warmup_attempts)
            if log_warmup_attempts > 0
            else 0.0
        )
        neg_diag_vector = torch.tensor(
            [
                float(neg_diag.get("total", 0)),
                float(neg_diag.get("valid", 0)),
                float(neg_diag.get("connected", 0)),
            ],
            device=device,
            dtype=torch.float64,
        )
        if dist_ctx.get("is_distributed", False):
            dist.all_reduce(neg_diag_vector, op=dist.ReduceOp.SUM)
        log_neg_total, log_neg_valid, log_neg_connected = [
            int(round(x)) for x in neg_diag_vector.tolist()
        ]
        log_neg_valid_pct = (
            100.0 * float(log_neg_valid) / float(log_neg_total)
            if log_neg_total > 0 else float("nan")
        )
        log_neg_connected_pct = (
            100.0 * float(log_neg_connected) / float(log_neg_total)
            if log_neg_total > 0 else float("nan")
        )
        (
            log_loss_cl,
            log_loss_fm,
            log_loss_total,
            log_throughput,
            log_prop_total,
            log_acc_total,
            log_step_total,
            log_move_total,
            log_fm_nodes,
            log_fm_edges,
            log_fm_ediff_mu,
            log_fm_ediff_sd,
            log_fm_grad_mu,
            log_fm_grad_sd,
            log_mu_noise,
            log_sd_noise,
            log_mu_data,
            log_sd_data,
        ) = log_vector.tolist()

        suffix = f"{cl_clip_suffix}{nan_guard_suffix}"
        if is_main_process:
            history["iter"].append(it + 1)
            history["loss_cl"].append(
                _safe_float(log_loss_cl) if cl_active else float("nan")
            )
            history["loss_fm"].append(_safe_float(log_loss_fm))
            history["loss_total"].append(_safe_float(log_loss_total))
            history["acc_rate"].append(_safe_float(log_acc_rate * 100.0))
            history["warmup_move_rate"].append(
                _safe_float(log_warmup_move_rate * 100.0)
            )
            history["prop_total"].append(_safe_float(log_prop_total))
            history["acc_total"].append(_safe_float(log_acc_total))
            history["step_total"].append(_safe_float(log_step_total))
            history["move_total"].append(_safe_float(log_move_total))
            history["fm_loss_nodes"].append(_safe_float(log_fm_nodes))
            history["fm_loss_edges"].append(_safe_float(log_fm_edges))
            history["fm_ediff_mu"].append(_safe_float(log_fm_ediff_mu))
            history["fm_ediff_sd"].append(_safe_float(log_fm_ediff_sd))
            history["fm_grad_mu"].append(_safe_float(log_fm_grad_mu))
            history["fm_grad_sd"].append(_safe_float(log_fm_grad_sd))
            history["energy_noise_mean"].append(_safe_float(log_mu_noise))
            history["energy_noise_std"].append(_safe_float(log_sd_noise))
            history["energy_data_mean"].append(_safe_float(log_mu_data))
            history["energy_data_std"].append(_safe_float(log_sd_data))
            history["neg_valid_pct"].append(_safe_float(log_neg_valid_pct))
            history["neg_connected_pct"].append(_safe_float(log_neg_connected_pct))

            print(
                f"[loss] it {it+1}/{max_iters} | cl={log_loss_cl:.4f} "
                f"| fm={log_loss_fm:.4f} | total={log_loss_total:.4f}{suffix}"
            )

            if (it + 1) % int(cfg.train.log_interval) == 0:
                acc_display = (
                    f"| mcmc_acc={log_acc_rate*100:.1f}% "
                    if run_cl
                    else ""
                )
                warmup_msg = ""
                if chain_warmup_train.enabled:
                    warmup_msg = (
                        f" | warmup_steps={warmup_steps_done}/"
                        f"{chain_warmup_train.steps}"
                        f" warmup_move={log_warmup_move_rate*100:.1f}%"
                    )
                    if warmup_stop_reason:
                        warmup_msg += f" stop={warmup_stop_reason}"
                    if warmup_stop_reason == "energy" and warmup_energy_mean is not None and warmup_energy_threshold_iter is not None:
                        warmup_msg += f" energy={warmup_energy_mean:.3f}<={warmup_energy_threshold_iter:.3f}"
                neg_diag_msg = ""
                if run_cl:
                    neg_diag_msg = (
                        f" | neg_valid={log_neg_valid}/{log_neg_total}({log_neg_valid_pct:.1f}%)"
                        f" neg_connected={log_neg_connected}/{log_neg_total}({log_neg_connected_pct:.1f}%)"
                    )
                gpu_memory_msg = ""
                if device.type == "cuda":
                    peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
                    gpu_memory_msg = (
                        f" | gpu_peak_alloc={peak_allocated_gb:.2f}GiB"
                        f" reserved={peak_reserved_gb:.2f}GiB"
                    )
                base_msg = (
                    f"[it {it+1}/{max_iters}] "
                    f"loss_cl={log_loss_cl:.4f} "
                    f"| loss_fm={log_loss_fm:.4f} "
                    f"| total={log_loss_total:.4f}{suffix} "
                    f"{acc_display}"
                    f"| batch_cl={B_cl} | t_iter={iter_time:.3f}s "
                    f"{neg_diag_msg} "
                    f"(pos={t_pos:.3f}s mcmc={t_mcmc:.3f}s neg={t_neg:.3f}s fm={t_fm:.3f}s back={t_back:.3f}s) "
                    f"| throughput={log_throughput:.1f} graphs/s"
                    f"{gpu_memory_msg}"
                    f"{warmup_msg}"
                )
                fm_msg = (
                    f" | fm_pairs={fm_pairs_count} "
                    f"| fm_nodes={log_fm_nodes:.4f} fm_edges={log_fm_edges:.4f}"
                    f" | fm_ediff={log_fm_ediff_mu:.4f}±{log_fm_ediff_sd:.4f}"
                    f" | fm_grad={log_fm_grad_mu:.4f}±{log_fm_grad_sd:.4f}"
                    if (fm_enabled and lambda_fm != 0.0)
                    else ""
                )
                if log_stats_enabled and mcmc_stats:
                    s_acc = (
                        f" | acc_nontriv(any,node,edge)="
                        f"{mcmc_stats.get('accept_nontrivial_any',0.0)*100:.1f}%/"
                        f"{mcmc_stats.get('accept_nontrivial_node',0.0)*100:.1f}%/"
                        f"{mcmc_stats.get('accept_nontrivial_edge',0.0)*100:.1f}%"
                    )
                    s_dist = (
                        f" | prop_d(n/e/t)="
                        f"{mcmc_stats.get('mean_prop_distance_nodes',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_prop_distance_edges',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_prop_distance_total',0.0):.2f}"
                        f" | acc_d(n/e/t)="
                        f"{mcmc_stats.get('mean_acc_distance_nodes',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_acc_distance_edges',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_acc_distance_total',0.0):.2f}"
                    )
                    s_step = (
                        f" | step_d(n/e/t)="
                        f"{mcmc_stats.get('mean_step_acc_distance_nodes',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_step_acc_distance_edges',0.0):.2f}/"
                        f"{mcmc_stats.get('mean_step_acc_distance_total',0.0):.2f}"
                    )
                    _overall_acc = mcmc_stats.get('overall_accept', acc_rate)
                    move_nodes = _overall_acc * mcmc_stats.get('mean_step_acc_distance_nodes', 0.0)
                    move_edges = _overall_acc * mcmc_stats.get('mean_step_acc_distance_edges', 0.0)
                    move_total = _overall_acc * mcmc_stats.get('mean_step_acc_distance_total', 0.0)
                    s_move = (
                        f" | move_per_step(n/e/t)="
                        f"{move_nodes:.2f}/{move_edges:.2f}/{move_total:.2f}"
                    )
                    s_posneg = (
                        "" if posneg_stats is None else
                        f" | pos-neg_d(n/e/t)="
                        f"{posneg_stats.get('mean_posneg_nodes',0.0):.2f}/"
                        f"{posneg_stats.get('mean_posneg_edges',0.0):.2f}/"
                        f"{posneg_stats.get('mean_posneg_total',0.0):.2f}"
                    )
                    print(base_msg + fm_msg + s_acc + s_dist + s_step + s_move + s_posneg)
                else:
                    print(base_msg + fm_msg)

        # Optional checkpointing
        checkpoint_saved = False
        save_every = int(getattr(cfg.train, "save_every", 0) or 0)
        recovery_save_every = int(
            getattr(cfg.train, "recovery_save_every", 0) or 0
        )
        checkpoint_due = save_every > 0 and ((it + 1) % save_every == 0)
        recovery_due = (
            not checkpoint_due
            and recovery_save_every > 0
            and ((it + 1) % recovery_save_every == 0)
        )
        save_due = checkpoint_due or recovery_due
        checkpoint_rng_states = (
            _gather_rng_states(device, dist_ctx) if save_due else None
        )
        if is_main_process and save_due:
            try:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_name = (
                    f"model_it{it+1:06d}.pt"
                    if checkpoint_due
                    else "model_recovery.pt"
                )
                path = os.path.join("checkpoints", checkpoint_name)
                _save_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                    iteration=it + 1,
                    history=history,
                    last_plot_iter=last_plot_iter,
                    train_stats={
                        "train_total_time": train_total_time,
                        "train_total_accepts": train_total_accepts,
                        "train_total_proposals": train_total_proposals,
                        "train_throughput_sum": train_throughput_sum,
                        "train_iters": train_iters,
                        "cl_warmup_start_iter": (
                            cl_warmup_start_iter if cl_active else None
                        ),
                    },
                    rng_state_by_rank=checkpoint_rng_states,
                    ema=ema,
                )
                checkpoint_saved = checkpoint_due
                checkpoint_kind = "checkpoint" if checkpoint_due else "recovery checkpoint"
                print(f"[info] Saved {checkpoint_kind}: {path}")
            except Exception as e:
                print(f"[warn] Could not save checkpoint: {e}")

        iter_idx = it + 1
        if is_main_process and plot_history_enabled:
            trigger_plot = False
            if plot_history_every > 0 and (iter_idx % plot_history_every == 0):
                trigger_plot = True
            if plot_history_on_checkpoint and checkpoint_saved:
                trigger_plot = True
            if trigger_plot and iter_idx != last_plot_iter:
                _save_all_training_plots(
                    history,
                    cl_active=cl_active,
                    iteration=iter_idx,
                    interp_payload=last_interp,
                    interp_context={
                        "model": _unwrap_model(model),
                        "dataset_info": dataset_infos,
                        "extra_features": extra_features,
                        "domain_features": domain_features,
                        "device": device,
                    },
                )
                last_plot_iter = iter_idx
                if run_viz_during_training and bool(getattr(cfg.viz, "enabled", True)):
                    print(f"[viz] Running energy visualization at iter {iter_idx} ...")
                    prev_training_mode = model.training
                    model.eval()
                    try:
                        run_energy_viz(
                            cfg,
                            model=_unwrap_model(model),
                            datamodule=datamodule,
                            dataset_infos=dataset_infos,
                            extra_features=extra_features,
                            domain_features=domain_features,
                            device=device,
                            iteration=iter_idx,
                        )
                    except Exception as viz_err:
                        print(f"[warn] Iter {iter_idx} visualization failed: {viz_err}")
                    finally:
                        if prev_training_mode:
                            model.train()

    # --------------------------- Training summary ---------------------------
    summary_tensor = torch.tensor(
        [
            train_total_time,
            float(train_total_accepts),
            float(train_total_proposals),
            train_throughput_sum,
            float(train_iters),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist_ctx.get("is_distributed", False):
        dist.all_reduce(summary_tensor, op=dist.ReduceOp.SUM)
    (
        train_total_time,
        train_total_accepts,
        train_total_proposals,
        train_throughput_sum,
        train_iters,
    ) = summary_tensor.tolist()
    avg_accept = (train_total_accepts / train_total_proposals) * 100.0 if train_total_proposals > 0 else 0.0
    avg_throughput = (train_throughput_sum / train_iters) if train_iters > 0 else 0.0
    if is_main_process:
        print(
            f"[train] Summary: total_time={train_total_time:.3f}s | avg_accept={avg_accept:.1f}% | avg_graphs/s={avg_throughput:.1f}"
        )

    if is_main_process:
        _save_all_training_plots(
            history,
            cl_active=cl_active,
            iteration=(history.get("iter")[-1] if history.get("iter") else None),
            interp_payload=last_interp,
            interp_context={
                "model": _unwrap_model(model),
                "dataset_info": dataset_infos,
                "extra_features": extra_features,
                "domain_features": domain_features,
                "device": device,
            },
        )

    # Save final
    save_last = bool(getattr(cfg.train, "save_last", True))
    final_rng_states = _gather_rng_states(device, dist_ctx) if save_last else None
    if is_main_process and save_last:
        try:
            os.makedirs("checkpoints", exist_ok=True)
            path = os.path.join("checkpoints", "model_last.pt")
            final_iteration = int(history.get("iter", [start_iter])[-1] if history.get("iter") else start_iter)
            _save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                iteration=final_iteration,
                history=history,
                last_plot_iter=last_plot_iter,
                train_stats={
                    "train_total_time": train_total_time,
                    "train_total_accepts": train_total_accepts,
                    "train_total_proposals": train_total_proposals,
                    "train_throughput_sum": train_throughput_sum,
                    "train_iters": train_iters,
                    "cl_warmup_start_iter": (
                        cl_warmup_start_iter if cl_active else None
                    ),
                },
                rng_state_by_rank=final_rng_states,
                ema=ema,
            )
            print(f"[info] Saved final checkpoint: {path}")
        except Exception as e:
            print(f"[warn] Could not save final checkpoint: {e}")

    # -------------------------------- Evaluate -------------------------------
    if ema is not None and ema_use_for_eval:
        ema.copy_to(_unwrap_model(model))
        if is_main_process:
            print(
                f"[info] Using EMA weights for final evaluation "
                f"(decay={ema.decay:.6g}, updates={ema.num_updates})."
            )
    model.eval()
    eval_model = _unwrap_model(model)

    _barrier(dist_ctx)

    if do_evaluate and is_main_process:
        eval_bs = int(getattr(cfg.sample, "eval_batch_size", cfg.train.batch_size))
        init_nodes: List[torch.Tensor] = []
        init_edges: List[torch.Tensor] = []
        ev_iter = iter(datamodule.train_dataloader())
        while len(init_nodes) < eval_bs:
            try:
                batch = next(ev_iter)
            except StopIteration:
                ev_iter = iter(datamodule.train_dataloader())
                batch = next(ev_iter)
            dense_data, node_mask = utils.to_dense(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            graphs = dense_data.mask(node_mask, collapse=True).split(node_mask)
            for g in graphs:
                init_nodes.append(g.X.long().cpu())
                init_edges.append(g.E.long().cpu())
                if len(init_nodes) >= eval_bs:
                    break

        gamma_eval = float(getattr(cfg.sample, "gamma_evaluate", 1.0))
        n_rand = int(round(eval_bs * gamma_eval))
        if n_rand > 0:
            rand_graphs = sampler.initialize_random_graphs(
                batch_size=n_rand,
                dataset_info=dataset_infos,
                device=device,
                transition=cfg.model.transition,
            )
            rand_nodes = [nt for (nt, _) in rand_graphs]
            rand_edges = [et for (_, et) in rand_graphs]
            replace_idx = torch.randperm(eval_bs)[:n_rand]
            for i, idx in enumerate(replace_idx):
                init_nodes[idx] = rand_nodes[i]
                init_edges[idx] = rand_edges[i]

        eval_t0 = time.perf_counter()
        base_dl_beta_sample, base_dl_lambda_X_sample, base_dl_lambda_E_sample, dual_kwargs_sample = (
            resolve_dl_parameters(cfg.sample)
        )
        simple_n_edits_sample = _parse_simple_n_edits(cfg.sample, "cfg.sample")
        sample_proposal = str(getattr(cfg.sample, "proposal", "random"))
        two_beta_kwargs_sample = resolve_two_beta_kwargs(sample_proposal, cfg.sample)
        two_beta_annealing_kwargs_sample = resolve_two_beta_annealing_kwargs(
            sample_proposal,
            cfg.sample,
        )
        if two_beta_kwargs_sample or two_beta_annealing_kwargs_sample:
            dual_kwargs_sample = {}
        anneal_kwargs_sample: Dict[str, Any] = {}
        if sample_proposal.lower() in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}:
            dl_beta_init_sample = float(getattr(cfg.sample, "dl_beta_init", base_dl_beta_sample))
            dl_beta_final_sample = float(getattr(cfg.sample, "dl_beta_final", base_dl_beta_sample))
            try:
                dl_beta_anneal_steps_sample = int(
                    getattr(cfg.sample, "dl_beta_anneal_steps", cfg.sample.sample_steps)
                )
            except (TypeError, ValueError):
                dl_beta_anneal_steps_sample = int(cfg.sample.sample_steps)
            anneal_kwargs_sample.update(
                dl_beta_init=dl_beta_init_sample,
                dl_beta_final=dl_beta_final_sample,
                dl_beta_anneal_steps=dl_beta_anneal_steps_sample,
            )
        with torch.no_grad():
            final_nodes, final_edges, eval_accepts, eval_steps_total, eval_stats = sampler.mcmc_sample_batch(
                model=eval_model,
                dataset_info=dataset_infos,
                node_types_list=init_nodes,
                edge_types_list=init_edges,
                extra_features=extra_features,
                domain_features=domain_features,
                steps=cfg.sample.sample_steps,
                device=device,
                proposal=sample_proposal,
                gwd_beta=float(getattr(cfg.sample, "gwd_beta", 1.0)),
                simple_n_edits=simple_n_edits_sample,
                dl_beta=base_dl_beta_sample,
                dl_lambda_X=base_dl_lambda_X_sample,
                dl_lambda_E=base_dl_lambda_E_sample,
                amp_dtype=amp_dtype_str,
                collect_stats=log_stats_enabled,
                **dual_kwargs_sample,
                **anneal_kwargs_sample,
                **two_beta_kwargs_sample,
                **two_beta_annealing_kwargs_sample,
            )

        eval_acc_rate = (eval_accepts / max(eval_steps_total, 1)) if eval_steps_total > 0 else 0.0
        if log_stats_enabled and eval_stats:
            _eval_overall_acc = eval_stats.get('overall_accept', eval_acc_rate)
            _eval_move_nodes = _eval_overall_acc * eval_stats.get('mean_step_acc_distance_nodes', 0.0)
            _eval_move_edges = _eval_overall_acc * eval_stats.get('mean_step_acc_distance_edges', 0.0)
            _eval_move_total = _eval_overall_acc * eval_stats.get('mean_step_acc_distance_total', 0.0)
            print(
                f"[eval] MCMC acceptance={eval_acc_rate*100:.1f}% over {eval_steps_total} proposals. "
                f"nontriv(any,node,edge)={eval_stats.get('accept_nontrivial_any',0.0)*100:.1f}%/"
                f"{eval_stats.get('accept_nontrivial_node',0.0)*100:.1f}%/"
                f"{eval_stats.get('accept_nontrivial_edge',0.0)*100:.1f}% | prop_d(n/e/t)="
                f"{eval_stats.get('mean_prop_distance_nodes',0.0):.2f}/"
                f"{eval_stats.get('mean_prop_distance_edges',0.0):.2f}/"
                f"{eval_stats.get('mean_prop_distance_total',0.0):.2f} | acc_d(n/e/t)="
                f"{eval_stats.get('mean_acc_distance_nodes',0.0):.2f}/"
                f"{eval_stats.get('mean_acc_distance_edges',0.0):.2f}/"
                f"{eval_stats.get('mean_acc_distance_total',0.0):.2f} | step_d(n/e/t)="
                f"{eval_stats.get('mean_step_acc_distance_nodes',0.0):.2f}/"
                f"{eval_stats.get('mean_step_acc_distance_edges',0.0):.2f}/"
                f"{eval_stats.get('mean_step_acc_distance_total',0.0):.2f} | move_per_step(n/e/t)="
                f"{_eval_move_nodes:.2f}/{_eval_move_edges:.2f}/{_eval_move_total:.2f}"
            )
        else:
            print(f"[eval] MCMC acceptance={eval_acc_rate*100:.1f}% over {eval_steps_total} proposals.")

        molecules = list(zip(final_nodes, final_edges))

        sampling_metrics(
            molecules=molecules,
            ref_metrics=dataset_infos.ref_metrics,
            name="GEM",
            current_epoch=0,
            val_counter=0,
            local_rank=0,
            test=True,
        )
        eval_t1 = time.perf_counter()

        eval_total_time = eval_t1 - eval_t0
        eval_graphs_per_s = (eval_bs / eval_total_time) if eval_total_time > 0 else float("inf")
        print(
            f"[eval] Summary: total_time={eval_total_time:.3f}s | avg_accept={eval_acc_rate*100:.1f}% | graphs/s={eval_graphs_per_s:.1f}"
        )
    elif is_main_process:
        print("[info] Evaluation disabled by cfg.sample.evaluate=false. Skipping evaluation.")

    if is_main_process:
        try:
            _record_expected_energy(
                model=eval_model,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                transition=cfg.model.transition,
                batch_size=cl_batch_size,
                output_path="expected_energy.json",
            )
        except Exception as exc:
            print(f"[warn] Expected energy estimation failed: {exc}")

    _barrier(dist_ctx)

    # -------------------------------- Visualization ---------------------------
    if is_main_process and bool(getattr(cfg.viz, "enabled", True)):
        print("[viz] Running energy landscape visualization ...")
        final_iter = history.get("iter")[-1] if history.get("iter") else None
        run_energy_viz(
            cfg,
            model=_unwrap_model(model),
            datamodule=datamodule,
            dataset_infos=dataset_infos,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
            iteration=final_iter,
        )
    elif is_main_process:
        print("[viz] Visualization disabled (cfg.viz.enabled=false).")

    if dist_ctx.get("is_distributed", False):
        _barrier(dist_ctx)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
