from __future__ import annotations

import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (str(_SRC_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from gem.models.transformer_model import GraphTransformer
from gem.models.extra_features import ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures

from gem.datasets import moses_dataset
from gem.checkpoint_utils import load_model_checkpoint
from gem import utils

try:
    from . import sampler
    from .ot_data import initialize_random_graphs_with_counts
    from .fm_utils import sample_interpolated_graph
    from .visualize_energy import run_viz as _run_energy_viz
    from .dlangevin_utils import (
        TWO_BETA_ANNEALING_PROPOSALS,
        TWO_BETA_PROPOSALS,
        resolve_chain_warmup,
    )
except ImportError:
    from gem import sampler
    from gem.ot_data import initialize_random_graphs_with_counts
    from gem.fm_utils import sample_interpolated_graph
    from gem.visualize_energy import run_viz as _run_energy_viz
    from gem.dlangevin_utils import (
        TWO_BETA_ANNEALING_PROPOSALS,
        TWO_BETA_PROPOSALS,
        resolve_chain_warmup,
    )

try:
    from gem.analysis.rdkit_functions import build_molecule_with_partial_charges, mol2smiles
except Exception:
    build_molecule_with_partial_charges = None
    mol2smiles = None


def _setup_file_logging(log_name: str = "cali_params.log") -> None:
    """Mirror stdout+stderr into a file in the current working directory (Hydra run dir)."""
    try:
        f = open(log_name, "a", buffering=1)

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

        sys.stdout = _Tee(sys.stdout, f)
        sys.stderr = _Tee(sys.stderr, f)
        print(f"[info] Logging to {log_name}")
    except Exception as e:
        print(f"[warn] Could not set up file logging: {e}")


def _activation_from_cfg(cfg: DictConfig) -> nn.Module:
    act_name = str(getattr(cfg.model, "activation", "relu")).lower()
    if act_name == "silu":
        return nn.SiLU()
    if act_name != "relu":
        print(f"[warn] Unknown activation '{act_name}', defaulting to ReLU.")
    return nn.ReLU()


def _make_feature_builders(cfg: DictConfig, dataset_infos):
    dname = str(cfg.dataset.name)
    if dname == "moses":
        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        raise NotImplementedError("This release supports sampler calibration for dataset.name=moses only.")
    return extra_features, domain_features


def _load_checkpoint_if_any(
    model: nn.Module,
    device: torch.device,
    path: Optional[str],
    *,
    use_ema: bool = False,
) -> bool:
    if not path:
        return False
    load_model_checkpoint(model, path, map_location=device, use_ema=use_ema)
    weight_label = "EMA" if use_ema else "online"
    print(f"[info] Loaded {weight_label} checkpoint weights: {path}")
    return True


def _collect_graphs_from_data(datamodule, max_graphs: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Collect up to `max_graphs` graphs (node_types, edge_types) from the training data."""
    node_types_list: List[torch.Tensor] = []
    edge_types_list: List[torch.Tensor] = []

    try:
        for batch in datamodule.train_dataloader():
            dense, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(node_mask, collapse=True)
            graphs = dense.split(node_mask)
            for g in graphs:
                # g.X: (n,), g.E: (n,n) with integer type indices
                node_types_list.append(g.X.long().detach().cpu())
                edge_types_list.append(g.E.long().detach().cpu())
                if len(node_types_list) >= max_graphs:
                    break
            if len(node_types_list) >= max_graphs:
                break
    except Exception as e:
        print(f"[warn] Failed collecting data graphs: {e}")

    return node_types_list, edge_types_list


def _collect_counts_from_data(datamodule, max_graphs: int) -> List[int]:
    """Collect only node counts from the training data (empirical distribution)."""
    counts: List[int] = []
    try:
        for batch in datamodule.train_dataloader():
            # counts of graphs in this batch
            _, c = torch.unique(batch.batch, return_counts=True)
            for n in c.tolist():
                counts.append(int(n))
                if len(counts) >= max_graphs:
                    break
            if len(counts) >= max_graphs:
                break
    except Exception as e:
        print(f"[warn] Failed collecting counts from data: {e}")
    return counts


def _mean_energy_of_graphs(
    *,
    model,
    dataset_infos,
    extra_features,
    domain_features,
    device: torch.device,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    chunk_size: int = 128,
) -> Tuple[float, float]:
    """Compute mean and std energy over a list of graphs in chunks to control memory."""
    assert len(node_types_list) == len(edge_types_list)
    N = len(node_types_list)
    if N == 0:
        return float("nan"), float("nan")

    total = 0.0
    total_sq = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, N, max(1, int(chunk_size))):
            j = min(N, i + max(1, int(chunk_size)))
            E = sampler.energy_batch(
                model=model,
                node_types_list=node_types_list[i:j],
                edge_types_list=edge_types_list[i:j],
                dataset_info=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=True,
            )
            E = E.to(torch.float64)
            total += float(E.sum().item())
            total_sq += float((E * E).sum().item())
            count += int(E.numel())

    if count == 0:
        return float("nan"), float("nan")

    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    return mean, std


def _relaxed_validity_and_novelty(
    nodes: Sequence[torch.Tensor],
    edges: Sequence[torch.Tensor],
    dataset_infos,
    train_smiles: Optional[Set[str]] = None,
) -> Tuple[float, float, float, float, float]:
    """Return relaxed validity, connectedness, uniqueness, novelty, and VUN."""
    if build_molecule_with_partial_charges is None or mol2smiles is None:
        return (float("nan"),) * 5
    decoder = getattr(dataset_infos, "atom_decoder", None)
    if decoder is None:
        return (float("nan"),) * 5
    total = len(nodes)
    if total == 0:
        return (float("nan"),) * 5

    valid_smiles: List[str] = []
    connected_count = 0
    for nt, et in zip(nodes, edges):
        try:
            mol = build_molecule_with_partial_charges(nt.cpu(), et.cpu(), decoder)
            if mol is None:
                continue
            smiles_raw = mol2smiles(mol)
            if not smiles_raw:
                continue
            smiles = str(smiles_raw).strip()
            if smiles:
                valid_smiles.append(smiles)
                if "." not in smiles:
                    connected_count += 1
        except Exception:
            continue

    validity = float(len(valid_smiles)) / float(total)
    connected_fraction = float(connected_count) / float(total)
    if not valid_smiles:
        return validity, connected_fraction, 0.0, 0.0, 0.0

    unique_smiles = set(valid_smiles)
    uniqueness = float(len(unique_smiles)) / float(len(valid_smiles)) if valid_smiles else 0.0
    novelty = float("nan")
    if train_smiles is not None:
        if unique_smiles:
            novel = sum(1 for s in unique_smiles if s not in train_smiles)
            novelty = float(novel) / float(len(unique_smiles))
        else:
            novelty = 0.0
    vun = validity * uniqueness * novelty if math.isfinite(novelty) else float("nan")
    return validity, connected_fraction, uniqueness, novelty, vun


def _build_dataset_context(cfg: DictConfig):
    dataset_name = str(getattr(cfg.dataset, "name", "moses")).lower()
    if dataset_name != "moses":
        raise ValueError("This release supports sampler calibration for dataset.name=moses only.")
    datamodule = moses_dataset.MosesDataModule(cfg)
    dataset_infos = moses_dataset.MOSESinfos(datamodule=datamodule, cfg=cfg)
    return datamodule, dataset_infos


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, float]
    mean_energy: float
    std_energy: float
    distance_total: float
    distance_per_step: float
    validity_relaxed: float
    connected_fraction: float
    uniqueness: float
    objective: float
    real_accept: float
    novelty: float
    vun: float


def _evaluate_one(
    *,
    model,
    dataset_infos,
    extra_features,
    domain_features,
    device: torch.device,
    init_nodes: Sequence[torch.Tensor],
    init_edges: Sequence[torch.Tensor],
    steps: int,
    proposal: str,
    params: Dict[str, float],
    energy_threshold: Optional[float],
    train_smiles: Optional[Set[str]],
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """Run MCMC from fixed initial states for `steps`, return energy + molecule metrics."""
    # Keep states on device across inner MCMC, but inputs are CPU lists
    proposal = (proposal or "dlangevin").lower()
    is_anneal = proposal in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}
    is_twobetas = proposal in TWO_BETA_PROPOSALS
    is_twobetas_anneal = proposal in TWO_BETA_ANNEALING_PROPOSALS
    base_dl_beta = float(
        params.get("dl_beta", params.get("dl_beta_far", params.get("dl_beta_near", 1.0)))
    )
    base_dl_lambda_X = float(
        params.get("dl_lambda_X", params.get("dl_lambda_X_far", params.get("dl_lambda_X_near", 1.0)))
    )
    base_dl_lambda_E = float(
        params.get("dl_lambda_E", params.get("dl_lambda_E_far", params.get("dl_lambda_E_near", 1.0)))
    )

    required_dual = {
        "dl_beta_near",
        "dl_lambda_X_near",
        "dl_lambda_E_near",
        "dl_beta_far",
        "dl_lambda_X_far",
        "dl_lambda_E_far",
    }
    has_dual = energy_threshold is not None and required_dual.issubset(params.keys())

    dual_kwargs: Dict[str, float] = {}
    if has_dual:
        dual_kwargs = dict(
            energy_split_threshold=float(energy_threshold),
            dl_beta_near=float(params["dl_beta_near"]),
            dl_lambda_X_near=float(params["dl_lambda_X_near"]),
            dl_lambda_E_near=float(params["dl_lambda_E_near"]),
            dl_beta_far=float(params["dl_beta_far"]),
            dl_lambda_X_far=float(params["dl_lambda_X_far"]),
            dl_lambda_E_far=float(params["dl_lambda_E_far"]),
        )
        base_dl_beta = float(params["dl_beta_far"])
        base_dl_lambda_X = float(params["dl_lambda_X_far"])
        base_dl_lambda_E = float(params["dl_lambda_E_far"])
    if has_dual and is_anneal:
        # Annealed DLangevin currently uses a single parameter set; ignore dual configs.
        has_dual = False
        dual_kwargs = {}

    beta_prop = base_dl_beta
    beta_mh = None
    beta_mh_init = None
    beta_mh_final = None
    beta_mh_anneal_steps = None
    if is_twobetas or is_twobetas_anneal:
        beta_prop = params.get("dl_beta_prop")
        if beta_prop is None:
            raise ValueError("dlangevintwobetas requires dl_beta_prop.")
        beta_prop = float(beta_prop)
        if is_twobetas:
            beta_mh = params.get("dl_beta_mh")
            if beta_mh is None:
                raise ValueError("dlangevintwobetas requires dl_beta_mh.")
            beta_mh = float(beta_mh)
        else:
            beta_mh_init = params.get("dl_beta_mh_init")
            beta_mh_final = params.get("dl_beta_mh_final")
            beta_mh_anneal_steps = params.get("dl_beta_mh_anneal_steps", params.get("dl_anneal_steps", 0))
            missing = [k for k, v in [
                ("dl_beta_mh_init", beta_mh_init),
                ("dl_beta_mh_final", beta_mh_final),
                ("dl_beta_mh_anneal_steps", beta_mh_anneal_steps),
            ] if v is None]
            if missing:
                raise ValueError(f"dlangevintwobetas_annealing requires {', '.join(missing)}.")
            beta_mh_init = float(beta_mh_init)
            beta_mh_final = float(beta_mh_final)
            try:
                beta_mh_anneal_steps = max(int(beta_mh_anneal_steps), 0)
            except (TypeError, ValueError):
                beta_mh_anneal_steps = 0

    beta_init = float(params.get("dl_beta_init", base_dl_beta))
    beta_final = float(params.get("dl_beta_final", beta_init))
    anneal_steps_raw = params.get("dl_beta_anneal_steps", params.get("dl_anneal_steps", 0))
    try:
        anneal_steps = max(int(anneal_steps_raw), 0)
    except (TypeError, ValueError):
        anneal_steps = 0

    nodes, edges, _, _, stats = sampler.mcmc_sample_batch(
        model=model,
        dataset_info=dataset_infos,
        node_types_list=list(init_nodes),  # shallow copy is fine; sampler keeps device-resident copies
        edge_types_list=list(init_edges),
        extra_features=extra_features,
        domain_features=domain_features,
        steps=int(steps),
        device=device,
        proposal=proposal,
        dl_beta=beta_prop if (is_twobetas or is_twobetas_anneal) else base_dl_beta,
        dl_beta_init=beta_init if is_anneal else base_dl_beta,
        dl_beta_final=beta_final if is_anneal else base_dl_beta,
        dl_beta_anneal_steps=anneal_steps if is_anneal else None,
        dl_beta_prop=beta_prop if (is_twobetas or is_twobetas_anneal) else None,
        dl_beta_mh=beta_mh if is_twobetas else None,
        dl_beta_mh_init=beta_mh_init if is_twobetas_anneal else None,
        dl_beta_mh_final=beta_mh_final if is_twobetas_anneal else None,
        dl_beta_mh_anneal_steps=beta_mh_anneal_steps if is_twobetas_anneal else None,
        dl_lambda_X=base_dl_lambda_X,
        dl_lambda_E=base_dl_lambda_E,
        collect_stats=True,
        **dual_kwargs,
    )
    with torch.no_grad():
        E = sampler.energy_batch(
            model=model,
            node_types_list=nodes,
            edge_types_list=edges,
            dataset_info=dataset_infos,
            device=device,
            extra_features=extra_features,
            domain_features=domain_features,
            detach=True,
        )
    total_distance = float(stats.get("distance_total", 0.0))
    distance_per_step_val = float(
        stats.get("distance_per_step", stats.get("distance_per_chain_step", 0.0))
    )
    total_props = float(stats.get("total_proposals", 0.0))
    acc_nontriv = float(stats.get("acc_nontriv_any", 0.0))
    real_accept = acc_nontriv / total_props if total_props > 0 else 0.0
    (
        validity_relaxed,
        connected_fraction,
        uniqueness,
        novelty_fraction,
        vun,
    ) = _relaxed_validity_and_novelty(nodes, edges, dataset_infos, train_smiles=train_smiles)
    return (
        float(E.mean().item()),
        float(E.std(unbiased=False).item()),
        total_distance,
        distance_per_step_val,
        validity_relaxed,
        connected_fraction,
        uniqueness,
        real_accept,
        novelty_fraction,
        vun,
    )


def _get_optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except Exception:
        return None


def _search_optuna(
    *,
    n_trials: int,
    seed: int,
    log_scale: bool,
    ranges: Dict[str, Tuple[float, float]],
    n_jobs: int,
    eval_fn,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    load_if_exists: bool = True,
    int_params: Optional[Set[str]] = None,
) -> List[TrialResult]:
    optuna = _get_optuna()
    if optuna is None:
        print("[warn] Optuna not available. Falling back to random search.")
        return _search_random(
            n_trials=n_trials,
            seed=seed,
            log_scale=log_scale,
            ranges=ranges,
            n_jobs=n_jobs,
            eval_fn=eval_fn,
            int_params=int_params,
        )

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    create_kwargs: Dict[str, Any] = dict(direction="minimize", sampler=sampler)
    if storage:
        create_kwargs["storage"] = storage
        if study_name:
            create_kwargs["study_name"] = study_name
        create_kwargs["load_if_exists"] = bool(load_if_exists)
    else:
        if study_name:
            create_kwargs["study_name"] = study_name
    study = optuna.create_study(**create_kwargs)

    param_names = list(ranges.keys())
    int_params = set() if int_params is None else set(int_params)
    results: List[TrialResult] = []

    def objective(trial: Any) -> float:
        params: Dict[str, float] = {}
        for name in param_names:
            low, high = ranges[name]
            use_log = log_scale and name not in int_params
            if use_log:
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high)
        for name in int_params:
            if name in params:
                try:
                    params[name] = float(int(round(params[name])))
                except Exception:
                    try:
                        params[name] = float(params[name])
                    except Exception:
                        pass

        (
            objective_val,
            mean_e,
            std_e,
            distance_total,
            distance_per_step,
            validity_relaxed,
            connected_fraction,
            uniqueness,
            real_accept,
            novelty_fraction,
            vun,
        ) = eval_fn(params)
        try:
            trial.set_user_attr("mean_energy", float(mean_e))
            trial.set_user_attr("std_energy", float(std_e))
            trial.set_user_attr("distance_total", float(distance_total))
            trial.set_user_attr("distance_per_step", float(distance_per_step))
            trial.set_user_attr("validity_relaxed", float(validity_relaxed))
            trial.set_user_attr("connected_fraction", float(connected_fraction))
            trial.set_user_attr("uniqueness", float(uniqueness))
            trial.set_user_attr("real_accept", float(real_accept))
            trial.set_user_attr("novelty", float(novelty_fraction))
            trial.set_user_attr("vun", float(vun))
        except Exception:
            pass
        results.append(
            TrialResult(
                trial_id=trial.number,
                params={k: float(v) for k, v in params.items()},
                mean_energy=float(mean_e),
                std_energy=float(std_e),
                distance_total=float(distance_total),
                distance_per_step=float(distance_per_step),
                validity_relaxed=float(validity_relaxed),
                connected_fraction=float(connected_fraction),
                uniqueness=float(uniqueness),
                objective=float(objective_val),
                real_accept=float(real_accept),
                novelty=float(novelty_fraction),
                vun=float(vun),
            )
        )
        print(
            f"[trial {trial.number:03d}] objective={objective_val:.6f} | mean_energy={mean_e:.6f} "
            f"mean_distance={distance_per_step:.6f} validity={validity_relaxed:.4f} "
            f"connected={connected_fraction:.4f} uniqueness={uniqueness:.4f} "
            f"real_accept={real_accept:.4f} "
            f"novelty={novelty_fraction:.4f} vun={vun:.4f}"
        )
        return objective_val

    # Parallelize with threads inside a single process. Beware of GPU memory when n_jobs>1.
    study.optimize(objective, n_trials=n_trials, n_jobs=max(int(n_jobs), 1), show_progress_bar=False)
    print(f"[info] Optuna best value: {study.best_value:.6f}, params: {study.best_params}")
    return results


def _search_random(
    *,
    n_trials: int,
    seed: int,
    log_scale: bool,
    ranges: Dict[str, Tuple[float, float]],
    n_jobs: int,
    eval_fn,
    int_params: Optional[Set[str]] = None,
) -> List[TrialResult]:
    rng = random.Random(seed)
    int_params = set() if int_params is None else set(int_params)
    results: List[TrialResult] = []

    param_names = list(ranges.keys())

    def _sample_param(low: float, high: float, use_log: bool) -> float:
        if use_log:
            if low <= 0 or high <= 0:
                raise ValueError("Log-scale sampling requires positive bounds.")
            la = math.log(low)
            lb = math.log(high)
            return float(math.exp(rng.uniform(la, lb)))
        return float(rng.uniform(low, high))

    params_list: List[Dict[str, float]] = []
    for _ in range(n_trials):
        sample_dict: Dict[str, float] = {}
        for name in param_names:
            low, high = ranges[name]
            use_log = log_scale and name not in int_params
            sample_dict[name] = _sample_param(low, high, use_log)
        for name in int_params:
            if name in sample_dict:
                try:
                    sample_dict[name] = float(int(round(sample_dict[name])))
                except Exception:
                    try:
                        sample_dict[name] = float(sample_dict[name])
                    except Exception:
                        pass
        params_list.append(sample_dict)

    if int(n_jobs) > 1:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = [ex.submit(eval_fn, dict(p)) for p in params_list]
            for t, (param_dict, fut) in enumerate(zip(params_list, futs)):
                (
                    objective_val,
                    mean_e,
                    std_e,
                    distance_total,
                    distance_per_step,
                    validity_relaxed,
                    connected_fraction,
                    uniqueness,
                    real_accept,
                    novelty_fraction,
                    vun,
                ) = fut.result()
                results.append(
                    TrialResult(
                        trial_id=t,
                        params={k: float(v) for k, v in param_dict.items()},
                        mean_energy=float(mean_e),
                        std_energy=float(std_e),
                        distance_total=float(distance_total),
                        distance_per_step=float(distance_per_step),
                        validity_relaxed=float(validity_relaxed),
                        connected_fraction=float(connected_fraction),
                        uniqueness=float(uniqueness),
                        objective=float(objective_val),
                        real_accept=float(real_accept),
                        novelty=float(novelty_fraction),
                        vun=float(vun),
                    )
                )
                print(
                    f"[trial {t:03d}] objective={objective_val:.6f} | mean_energy={mean_e:.6f} "
                    f"mean_distance={distance_per_step:.6f} validity={validity_relaxed:.4f} "
                    f"connected={connected_fraction:.4f} uniqueness={uniqueness:.4f} "
                    f"real_accept={real_accept:.4f} "
                    f"novelty={novelty_fraction:.4f} vun={vun:.4f}"
                )
    else:
        for t, param_dict in enumerate(params_list):
            (
                objective_val,
                mean_e,
                std_e,
                distance_total,
                distance_per_step,
                validity_relaxed,
                connected_fraction,
                uniqueness,
                real_accept,
                novelty_fraction,
                vun,
            ) = eval_fn(param_dict)
            results.append(
                TrialResult(
                    trial_id=t,
                    params={k: float(v) for k, v in param_dict.items()},
                    mean_energy=float(mean_e),
                    std_energy=float(std_e),
                    distance_total=float(distance_total),
                    distance_per_step=float(distance_per_step),
                    validity_relaxed=float(validity_relaxed),
                    connected_fraction=float(connected_fraction),
                    uniqueness=float(uniqueness),
                    objective=float(objective_val),
                    real_accept=float(real_accept),
                    novelty=float(novelty_fraction),
                    vun=float(vun),
                )
            )
            print(
                f"[trial {t:03d}] objective={objective_val:.6f} | mean_energy={mean_e:.6f} "
                f"mean_distance={distance_per_step:.6f} validity={validity_relaxed:.4f} "
                f"connected={connected_fraction:.4f} uniqueness={uniqueness:.4f} "
                f"real_accept={real_accept:.4f} "
                f"novelty={novelty_fraction:.4f} vun={vun:.4f}"
            )
    return results


def _write_results_csv(path: str, results: List[TrialResult], param_names: Sequence[str]) -> None:
    try:
        with open(path, "w", newline="") as fp:
            writer = csv.writer(fp)
            header = [
                "trial",
                *param_names,
                "objective",
                "mean_energy",
                "std_energy",
                "distance_total",
                "distance_per_step",
                "validity_relaxed",
                "connected_fraction",
                "uniqueness",
                "real_accept",
                "novelty",
                "vun",
            ]
            writer.writerow(header)
            for r in results:
                row = [int(r.trial_id)]
                for name in param_names:
                    val = r.params.get(name)
                    row.append(f"{float(val):.6g}" if val is not None else "")
                row.append(f"{r.objective:.6f}")
                row.append(f"{r.mean_energy:.6f}")
                row.append(f"{r.std_energy:.6f}")
                row.append(f"{r.distance_total:.6f}")
                row.append(f"{r.distance_per_step:.6f}")
                row.append(f"{r.validity_relaxed:.6f}")
                row.append(f"{r.connected_fraction:.6f}")
                row.append(f"{r.uniqueness:.6f}")
                row.append(f"{r.real_accept:.6f}")
                row.append(f"{r.novelty:.6f}")
                row.append(f"{r.vun:.6f}")
                writer.writerow(row)
        print(f"[info] Wrote results CSV: {path}")
    except Exception as e:
        print(f"[warn] Could not write CSV '{path}': {e}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="cali_params_moses")
def main(cfg: DictConfig):
    # Log to file and console
    _setup_file_logging("cali_params.log")

    # Repro
    seed = int(getattr(cfg.train, "seed", 0) or 0)
    random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    dataset_name = str(getattr(cfg.dataset, "name", "moses"))
    dataset_name_lower = dataset_name.lower()
    print(f"[info] Loading dataset context for '{dataset_name}'.")
    datamodule, dataset_infos = _build_dataset_context(cfg)

    extra_features, domain_features = _make_feature_builders(cfg, dataset_infos)
    try:
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    except Exception:
        pass

    # Cached SMILES for novelty (best-effort)
    train_smiles_set: Optional[Set[str]] = None
    dataset_smiles = None
    try:
        if dataset_name_lower == "moses":
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=getattr(cfg.dataset, "filter", False),
            )
        else:
            dataset_smiles = None
    except Exception as e:
        dataset_smiles = None
        print(f"[warn] Could not load dataset SMILES for novelty: {e}")
    if dataset_smiles:
        train_smiles_raw = dataset_smiles.get("train")
        if train_smiles_raw:
            train_smiles_set = {
                str(s).strip()
                for s in train_smiles_raw
                if isinstance(s, str) and str(s).strip()
            }
            if not train_smiles_set:
                train_smiles_set = None

    # Model
    act = _activation_from_cfg(cfg)
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=dataset_infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=dataset_infos.output_dims,
        act_fn_in=act,
        act_fn_out=act,
        tf_activation=act,
    ).to(device)
    model.eval()

    # Optional resume
    resume_path = str(getattr(cfg.train, "resume", "") or getattr(cfg.train, "init_ckpt", "") or "")
    if not resume_path:
        raise ValueError("Set train.resume=/path/to/transport/checkpoint.pt.")
    _load_checkpoint_if_any(
        model,
        device,
        resume_path,
        use_ema=bool(getattr(cfg.cali, "use_ema", False)),
    )

    # Report baseline energies before starting the search.
    data_energy_threshold: Optional[float] = None
    data_energy_mean: Optional[float] = None
    data_energy_std: Optional[float] = None
    try:
        BASE_N = 1024
        noise_transition = str(getattr(cfg.cali, "noise_transition", "uniform"))
        print(f"[info] Sampling {BASE_N} data graphs for baseline energy…")
        data_nodes, data_edges = _collect_graphs_from_data(datamodule, BASE_N)
        data_mean, data_std = _mean_energy_of_graphs(
            model=model,
            dataset_infos=dataset_infos,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
            node_types_list=data_nodes,
            edge_types_list=data_edges,
            chunk_size=int(os.getenv("CALI_BASELINE_CHUNK", "128")),
        )
        data_energy_mean = float(data_mean)
        data_energy_std = float(data_std)
        data_energy_threshold = float(data_mean + 3.0 * data_std)
        print(
            f"[info] Data energy stats (n={len(data_nodes)}): mean={data_mean:.6f} std={data_std:.6f} "
            f"threshold(mean+3*std)={data_energy_threshold:.6f}"
        )

        print(f"[info] Sampling {BASE_N} noise graphs (transition='{noise_transition}') for baseline energy, with counts matched to data…")
        # Empirical node counts from the collected data sample
        counts = [int(t.shape[0]) for t in data_nodes[:BASE_N]]
        noise_graphs = initialize_random_graphs_with_counts(
            counts=counts,
            dataset_info=dataset_infos,
            device=torch.device("cpu"),
            transition=noise_transition,
        )
        noise_nodes = [nt for (nt, _) in noise_graphs]
        noise_edges = [et for (_, et) in noise_graphs]
        noise_mean, noise_std = _mean_energy_of_graphs(
            model=model,
            dataset_infos=dataset_infos,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
            node_types_list=noise_nodes,
            edge_types_list=noise_edges,
            chunk_size=int(os.getenv("CALI_BASELINE_CHUNK", "128")),
        )
        print(
            f"[info] Noise energy stats (transition='{noise_transition}', n={len(noise_nodes)}): "
            f"mean={noise_mean:.6f} std={noise_std:.6f}"
        )
    except Exception as e:
        print(f"[warn] Failed to compute baseline energies: {e}")

    if data_energy_threshold is None or data_energy_mean is None or data_energy_std is None:
        raise RuntimeError("Failed to compute data energy statistics required for dual-parameter calibration.")

    # Fixed initial states for fairness across trials
    N_noise = max(int(getattr(cfg.cali, "N_noise", 256)), 0)
    N_data = max(int(getattr(cfg.cali, "N_data", 0)), 0)
    if N_noise == 0 and N_data == 0:
        raise ValueError("cali.N_noise and cali.N_data cannot both be zero.")

    noise_transition = str(getattr(cfg.cali, "noise_transition", "uniform"))
    noise_init_nodes: List[torch.Tensor] = []
    noise_init_edges: List[torch.Tensor] = []
    if N_noise > 0:
        print(
            f"[info] Initializing {N_noise} noise graphs for calibration (counts sampled from empirical data; transition='{noise_transition}')"
        )
        counts_init = _collect_counts_from_data(datamodule, N_noise)
        if len(counts_init) < N_noise:
            # Fallback to dataset_infos.nodes_dist if we couldn't collect enough counts
            print(
                f"[warn] Only collected {len(counts_init)} counts from data; falling back to sampler.initialize_random_graphs for the remainder"
            )
            noise_graphs = sampler.initialize_random_graphs(
                batch_size=N_noise,
                dataset_info=dataset_infos,
                device=torch.device("cpu"),
                transition=noise_transition,
            )
        else:
            noise_graphs = initialize_random_graphs_with_counts(
                counts=counts_init,
                dataset_info=dataset_infos,
                device=torch.device("cpu"),
                transition=noise_transition,
            )
        noise_init_nodes = [nt.clone() for (nt, _) in noise_graphs]
        noise_init_edges = [et.clone() for (_, et) in noise_graphs]
    else:
        print("[info] Skipping noise initializations (cali.N_noise=0).")

    data_init_nodes: List[torch.Tensor] = []
    data_init_edges: List[torch.Tensor] = []
    data_tau_raw = getattr(cfg.cali, "data_tau", 1.0)
    try:
        data_tau = float(data_tau_raw)
    except (TypeError, ValueError):
        data_tau = 1.0
    data_tau = min(max(data_tau, 0.0), 1.0)
    if N_data > 0:
        print(f"[info] Collecting {N_data} data graphs for calibration initial states…")
        data_nodes_raw, data_edges_raw = _collect_graphs_from_data(datamodule, N_data)
        if len(data_nodes_raw) == 0:
            print("[warn] Failed to collect data graphs; proceeding without data-initialized chains.")
        else:
            data_init_nodes = [nt.clone() for nt in data_nodes_raw]
            data_init_edges = [et.clone() for et in data_edges_raw]
            collected = len(data_init_nodes)
            if collected >= N_data:
                data_init_nodes = data_init_nodes[:N_data]
                data_init_edges = data_init_edges[:N_data]
            else:
                print(
                    f"[warn] Only collected {collected} data graphs; repeating samples to reach N_data={N_data}."
                )
                base_nodes = list(data_init_nodes)
                base_edges = list(data_init_edges)
                idx = 0
                while len(data_init_nodes) < N_data and collected > 0:
                    source_idx = idx % collected
                    data_init_nodes.append(base_nodes[source_idx].clone())
                    data_init_edges.append(base_edges[source_idx].clone())
                    idx += 1
    else:
        print("[info] Skipping data initializations (cali.N_data=0).")
    if data_tau < 1.0 and data_init_nodes:
        print(f"[info] Applying data_tau interpolation for data-initialized chains: tau={data_tau:.3f}")
        counts_interp = [int(nt.shape[0]) for nt in data_init_nodes]
        noise_interp_graphs = initialize_random_graphs_with_counts(
            counts=counts_interp,
            dataset_info=dataset_infos,
            device=torch.device("cpu"),
            transition=noise_transition,
        )
        interp_nodes: List[torch.Tensor] = []
        interp_edges: List[torch.Tensor] = []
        for (nt_noise, et_noise), nt_data, et_data in zip(noise_interp_graphs, data_init_nodes, data_init_edges):
            nt_i, et_i = sample_interpolated_graph(nt_noise, et_noise, nt_data, et_data, data_tau)
            interp_nodes.append(nt_i)
            interp_edges.append(et_i)
        data_init_nodes = interp_nodes
        data_init_edges = interp_edges

    init_nodes = noise_init_nodes + data_init_nodes
    init_edges = noise_init_edges + data_init_edges
    num_init_noise = len(noise_init_nodes)
    num_init_data = len(data_init_nodes)
    if len(init_nodes) == 0:
        raise RuntimeError("Failed to prepare initial graphs for calibration.")

    chain_warmup_cfg = getattr(cfg.cali, "chain_warmup", None)
    default_gwd_beta = float(getattr(cfg.cali, "gwd_beta", 1.0))
    chain_warmup = resolve_chain_warmup(
        chain_warmup_cfg,
        fallback=getattr(cfg, "cali", None),
        default_gwd_beta=default_gwd_beta,
    )
    if chain_warmup.enabled and chain_warmup.steps > 0:
        vectorized_simple_warmup = sampler.should_vectorize_simple_warmup(
            chain_warmup.proposal,
            vectorized=chain_warmup.vectorized,
        )
        print(
            f"[info] Calibration chain warmup: proposal={chain_warmup.proposal}, "
            f"steps={chain_warmup.steps}, vectorized={vectorized_simple_warmup}."
        )
        with torch.no_grad():
            if vectorized_simple_warmup:
                edits_per_step = (
                    5
                    if chain_warmup.simple_n_edits is None
                    else int(chain_warmup.simple_n_edits)
                )
                init_nodes, init_edges, moves, attempts, warmup_stats = (
                    sampler.run_simple_v2_warmup_vectorized(
                        model=model,
                        dataset_info=dataset_infos,
                        node_types_list=[nt.clone() for nt in init_nodes],
                        edge_types_list=[et.clone() for et in init_edges],
                        extra_features=extra_features,
                        domain_features=domain_features,
                        steps=int(chain_warmup.steps),
                        device=device,
                        edits_per_step=edits_per_step,
                        amp_dtype=None,
                        stop_when_unchanged=True,
                    )
                )
                move_rate = float(moves / attempts) if attempts else 0.0
                print(
                    "[info] Calibration transport warmup completed: "
                    f"steps={warmup_stats['steps_executed']}/{chain_warmup.steps}, "
                    f"move_rate={100.0 * move_rate:.1f}%, "
                    f"stop={warmup_stats['stop_reason']}."
                )
            else:
                init_nodes, init_edges, _, _, _ = sampler.mcmc_sample_batch(
                    model=model,
                    dataset_info=dataset_infos,
                    node_types_list=[nt.clone() for nt in init_nodes],
                    edge_types_list=[et.clone() for et in init_edges],
                    extra_features=extra_features,
                    domain_features=domain_features,
                    steps=int(chain_warmup.steps),
                    device=device,
                    proposal=chain_warmup.proposal,
                    gwd_beta=chain_warmup.gwd_beta,
                    dl_beta=chain_warmup.dl_beta,
                    dl_lambda_X=chain_warmup.dl_lambda_X,
                    dl_lambda_E=chain_warmup.dl_lambda_E,
                    simple_n_edits=chain_warmup.simple_n_edits,
                    amp_dtype=None,
                    collect_stats=False,
                    **chain_warmup.dual_kwargs,
                )

    steps = int(getattr(cfg.cali, "steps", 30))
    proposal = str(getattr(cfg.cali, "proposal", "dlangevin") or "dlangevin")
    proposal_lower = proposal.lower()
    is_twobetas = proposal_lower in TWO_BETA_PROPOSALS
    is_twobetas_anneal = proposal_lower in TWO_BETA_ANNEALING_PROPOSALS
    fixed_params: Dict[str, float] = {}
    if is_twobetas:
        fixed_beta_mh_raw = getattr(cfg.cali, "dl_beta_mh", None)
        if fixed_beta_mh_raw is not None:
            try:
                fixed_beta_mh = float(fixed_beta_mh_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cali.dl_beta_mh must be numeric; got {fixed_beta_mh_raw!r}."
                ) from exc
            if not math.isfinite(fixed_beta_mh) or fixed_beta_mh <= 0.0:
                raise ValueError(
                    f"cali.dl_beta_mh must be finite and > 0; got {fixed_beta_mh_raw!r}."
                )
            fixed_params["dl_beta_mh"] = fixed_beta_mh
            print(f"[info] Fixed MH target beta: dl_beta_mh={fixed_beta_mh:.6g}")
    if is_twobetas_anneal:
        fixed_beta_mh_final_raw = getattr(cfg.cali, "dl_beta_mh_final", None)
        if fixed_beta_mh_final_raw is not None:
            try:
                fixed_beta_mh_final = float(fixed_beta_mh_final_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cali.dl_beta_mh_final must be numeric; "
                    f"got {fixed_beta_mh_final_raw!r}."
                ) from exc
            if not math.isfinite(fixed_beta_mh_final) or fixed_beta_mh_final <= 0.0:
                raise ValueError(
                    "cali.dl_beta_mh_final must be finite and > 0; "
                    f"got {fixed_beta_mh_final_raw!r}."
                )
            fixed_params["dl_beta_mh_final"] = fixed_beta_mh_final
            print(
                "[info] Fixed terminal MH target beta: "
                f"dl_beta_mh_final={fixed_beta_mh_final:.6g}"
            )
        anneal_steps_raw = getattr(cfg.cali, "dl_beta_mh_anneal_steps", None)
        if anneal_steps_raw is None:
            raise ValueError(
                "Two-beta annealing calibration requires fixed "
                "cali.dl_beta_mh_anneal_steps."
            )
        try:
            anneal_steps = int(anneal_steps_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cali.dl_beta_mh_anneal_steps must be an integer; "
                f"got {anneal_steps_raw!r}."
            ) from exc
        if anneal_steps < 0:
            raise ValueError(
                "cali.dl_beta_mh_anneal_steps must be >= 0; "
                f"got {anneal_steps_raw!r}."
            )
        fixed_params["dl_beta_mh_anneal_steps"] = float(anneal_steps)
        print(
            "[info] Fixed MH annealing duration: "
            f"dl_beta_mh_anneal_steps={anneal_steps}"
        )
    total_chains = len(init_nodes)
    print(
        f"[info] Calibration budget: steps={steps}, N_noise={num_init_noise}, N_data={num_init_data}, "
        f"total={total_chains}, proposal='{proposal}'"
    )

    objective_cfg = getattr(cfg.cali, "objective", None)
    if objective_cfg is None:
        energy_weight = 1.0
        distance_weight = float(getattr(cfg.cali, "distance_weight", 0.0))
        validity_weight = 0.0
        connected_weight = 0.0
        accept_weight = 0.0
        novelty_weight = 0.0
    else:
        energy_weight = float(getattr(objective_cfg, "mean_energy_weight", 1.0))
        distance_weight = float(getattr(objective_cfg, "distance_weight", 0.0))
        validity_weight = float(getattr(objective_cfg, "validity_weight", 0.0))
        connected_weight = float(getattr(objective_cfg, "connected_weight", 0.0))
        accept_weight = float(getattr(objective_cfg, "real_accept_weight", 0.0))
        novelty_weight = float(getattr(objective_cfg, "novelty_weight", 0.0))
    print(
        f"[info] Objective weights | energy={energy_weight:.4f} distance={distance_weight:.4f} "
        f"validity={validity_weight:.4f} connected={connected_weight:.4f} "
        f"acceptance={accept_weight:.4f} novelty={novelty_weight:.4f}"
    )
    validity_warning_emitted = False
    connected_warning_emitted = False
    novelty_warning_emitted = False

    # Evaluation closure
    def eval_fn(
        param_dict: Dict[str, float],
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        nonlocal validity_warning_emitted, connected_warning_emitted, novelty_warning_emitted
        eval_params = {**param_dict, **fixed_params}
        (
            mean_e,
            std_e,
            distance_total,
            distance_per_step,
            validity_relaxed,
            connected_fraction,
            uniqueness,
            real_accept,
            novelty_fraction,
            vun,
        ) = _evaluate_one(
            model=model,
            dataset_infos=dataset_infos,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
            init_nodes=init_nodes,
            init_edges=init_edges,
            steps=steps,
            proposal=proposal,
            params=eval_params,
            energy_threshold=data_energy_threshold,
            train_smiles=train_smiles_set,
        )
        validity_term = validity_relaxed
        if not math.isfinite(validity_term):
            if validity_weight != 0.0 and not validity_warning_emitted:
                print(
                    "[warn] Relaxed validity could not be computed (missing RDKit?). "
                    "Validity weight ignored for objective."
                )
                validity_warning_emitted = True
            validity_term = 0.0
        connected_term = connected_fraction
        if not math.isfinite(connected_term):
            if connected_weight != 0.0 and not connected_warning_emitted:
                print(
                    "[warn] Connectedness could not be computed (missing RDKit?). "
                    "Connectedness weight ignored for objective."
                )
                connected_warning_emitted = True
            connected_term = 0.0
        novelty_term = novelty_fraction
        if not math.isfinite(novelty_term):
            if novelty_weight != 0.0 and not novelty_warning_emitted:
                print(
                    "[warn] Novelty could not be computed (missing SMILES cache?). "
                    "Novelty weight ignored for objective."
                )
                novelty_warning_emitted = True
            novelty_term = 0.0
        objective = (
            energy_weight * mean_e
            - distance_weight * distance_per_step
            - validity_weight * validity_term
            - connected_weight * connected_term
            - accept_weight * real_accept
            - novelty_weight * novelty_term
        )
        return (
            objective,
            mean_e,
            std_e,
            distance_total,
            distance_per_step,
            validity_relaxed,
            connected_term,
            uniqueness,
            real_accept,
            novelty_term,
            vun,
        )

    # Search config
    search_cfg = cfg.cali.search
    method = str(getattr(search_cfg, "method", "optuna")).lower()
    n_trials = int(getattr(search_cfg, "n_trials", 30))
    log_scale = bool(getattr(search_cfg, "log_scale", True))
    n_jobs = int(getattr(search_cfg, "n_jobs", 1))

    beta_min = float(getattr(search_cfg, "beta_min", 1.0))
    beta_max = float(getattr(search_cfg, "beta_max", 1000.0))
    beta_near_min = float(getattr(search_cfg, "beta_near_min", beta_min))
    beta_near_max = float(getattr(search_cfg, "beta_near_max", beta_max))
    beta_far_min = float(getattr(search_cfg, "beta_far_min", beta_min))
    beta_far_max = float(getattr(search_cfg, "beta_far_max", beta_max))

    lam_min_bc = float(getattr(search_cfg, "lambda_min", 1.0))
    lam_max_bc = float(getattr(search_cfg, "lambda_max", 1000.0))

    lamX_min = float(getattr(search_cfg, "lambda_X_min", lam_min_bc))
    lamX_max = float(getattr(search_cfg, "lambda_X_max", lam_max_bc))
    lamE_min = float(getattr(search_cfg, "lambda_E_min", lam_min_bc))
    lamE_max = float(getattr(search_cfg, "lambda_E_max", lam_max_bc))

    lamX_near_min = float(getattr(search_cfg, "lambda_X_near_min", lamX_min))
    lamX_near_max = float(getattr(search_cfg, "lambda_X_near_max", lamX_max))
    lamE_near_min = float(getattr(search_cfg, "lambda_E_near_min", lamE_min))
    lamE_near_max = float(getattr(search_cfg, "lambda_E_near_max", lamE_max))

    lamX_far_min = float(getattr(search_cfg, "lambda_X_far_min", lamX_min))
    lamX_far_max = float(getattr(search_cfg, "lambda_X_far_max", lamX_max))
    lamE_far_min = float(getattr(search_cfg, "lambda_E_far_min", lamE_min))
    lamE_far_max = float(getattr(search_cfg, "lambda_E_far_max", lamE_max))

    beta_init_min = float(getattr(search_cfg, "beta_init_min", beta_min))
    beta_init_max = float(getattr(search_cfg, "beta_init_max", beta_max))
    beta_final_min = float(getattr(search_cfg, "beta_final_min", beta_min))
    beta_final_max = float(getattr(search_cfg, "beta_final_max", beta_max))
    anneal_steps_min_raw = getattr(search_cfg, "anneal_steps_min", 0)
    anneal_steps_max_raw = getattr(search_cfg, "anneal_steps_max", steps)
    try:
        anneal_steps_min = max(int(anneal_steps_min_raw), 0)
    except (TypeError, ValueError):
        anneal_steps_min = 0
    try:
        anneal_steps_max = max(int(anneal_steps_max_raw), anneal_steps_min)
    except (TypeError, ValueError):
        anneal_steps_max = anneal_steps_min
    if anneal_steps_max < anneal_steps_min:
        anneal_steps_max = anneal_steps_min
    beta_prop_min = getattr(search_cfg, "beta_prop_min", None)
    beta_prop_max = getattr(search_cfg, "beta_prop_max", None)
    beta_mh_init_min = getattr(search_cfg, "beta_mh_init_min", None)
    beta_mh_init_max = getattr(search_cfg, "beta_mh_init_max", None)
    beta_mh_final_min = getattr(search_cfg, "beta_mh_final_min", None)
    beta_mh_final_max = getattr(search_cfg, "beta_mh_final_max", None)
    beta_mh_anneal_steps_min = getattr(search_cfg, "beta_mh_anneal_steps_min", None)
    beta_mh_anneal_steps_max = getattr(search_cfg, "beta_mh_anneal_steps_max", None)

    optuna_storage = getattr(search_cfg, "optuna_storage", None)
    optuna_study_name = getattr(search_cfg, "optuna_study_name", None)
    optuna_load_if_exists = bool(getattr(search_cfg, "optuna_load_if_exists", True))

    is_anneal_proposal = proposal_lower in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}
    single_param_mode = bool(getattr(search_cfg, "single_params", False))
    if (is_anneal_proposal or is_twobetas_anneal) and not single_param_mode:
        print("[warn] Annealing proposals use single-parameter search; overriding search.single_params=True.")
        single_param_mode = True

    if single_param_mode:
        if is_twobetas_anneal:
            ranges = {
                "dl_beta_prop": (float(beta_prop_min), float(beta_prop_max)),
                "dl_beta_mh_init": (float(beta_mh_init_min), float(beta_mh_init_max)),
                "dl_lambda_X": (lamX_min, lamX_max),
                "dl_lambda_E": (lamE_min, lamE_max),
            }
            if "dl_beta_mh_final" not in fixed_params:
                if beta_mh_final_min is None or beta_mh_final_max is None:
                    raise ValueError(
                        "Two-beta annealing calibration requires fixed "
                        "cali.dl_beta_mh_final or search bounds."
                    )
                ranges["dl_beta_mh_final"] = (
                    float(beta_mh_final_min),
                    float(beta_mh_final_max),
                )
            if "dl_beta_mh_anneal_steps" not in fixed_params:
                if (
                    beta_mh_anneal_steps_min is None
                    or beta_mh_anneal_steps_max is None
                ):
                    raise ValueError(
                        "Two-beta annealing calibration requires fixed "
                        "cali.dl_beta_mh_anneal_steps or search bounds."
                    )
                ranges["dl_beta_mh_anneal_steps"] = (
                    float(beta_mh_anneal_steps_min),
                    float(beta_mh_anneal_steps_max),
                )
        elif is_twobetas:
            ranges = {
                "dl_beta_prop": (float(beta_prop_min), float(beta_prop_max)),
                "dl_lambda_X": (lamX_min, lamX_max),
                "dl_lambda_E": (lamE_min, lamE_max),
            }
            if "dl_beta_mh" not in fixed_params:
                if beta_mh_init_min is None or beta_mh_init_max is None:
                    raise ValueError(
                        "Two-beta calibration requires fixed cali.dl_beta_mh or both "
                        "cali.search.beta_mh_init_min/max."
                    )
                ranges["dl_beta_mh"] = (
                    float(beta_mh_init_min),
                    float(beta_mh_init_max),
                )
        elif is_anneal_proposal:
            ranges = {
                "dl_beta_init": (beta_init_min, beta_init_max),
                "dl_beta_final": (beta_final_min, beta_final_max),
                "dl_beta_anneal_steps": (float(anneal_steps_min), float(anneal_steps_max)),
                "dl_lambda_X": (lamX_min, lamX_max),
                "dl_lambda_E": (lamE_min, lamE_max),
            }
        else:
            ranges = {
                "dl_beta": (beta_min, beta_max),
                "dl_lambda_X": (lamX_min, lamX_max),
                "dl_lambda_E": (lamE_min, lamE_max),
            }
    else:
        ranges = {
            "dl_beta_near": (beta_near_min, beta_near_max),
            "dl_lambda_X_near": (lamX_near_min, lamX_near_max),
            "dl_lambda_E_near": (lamE_near_min, lamE_near_max),
            "dl_beta_far": (beta_far_min, beta_far_max),
            "dl_lambda_X_far": (lamX_far_min, lamX_far_max),
            "dl_lambda_E_far": (lamE_far_min, lamE_far_max),
        }
    search_param_order = list(ranges.keys())
    param_order = search_param_order + [
        name for name in fixed_params if name not in ranges
    ]
    int_params = {name for name in search_param_order if "steps" in name}

    range_summary = ", ".join(
        f"{name}:[{low},{high}]" for name, (low, high) in ranges.items()
    )
    print(
        f"[info] Search method={method} | trials={n_trials} | log_scale={log_scale} | ranges={{ {range_summary} }}"
    )

    if method == "optuna":
        results = _search_optuna(
            n_trials=n_trials,
            seed=seed,
            log_scale=log_scale,
            ranges=ranges,
            n_jobs=n_jobs,
            eval_fn=eval_fn,
            storage=optuna_storage,
            study_name=optuna_study_name,
            load_if_exists=optuna_load_if_exists,
            int_params=int_params,
        )
    else:
        if n_jobs > 1:
            print("[warn] Random search parallelization uses threads; watch GPU memory when n_jobs>1.")
        results = _search_random(
            n_trials=n_trials,
            seed=seed,
            log_scale=log_scale,
            ranges=ranges,
            n_jobs=n_jobs,
            eval_fn=eval_fn,
            int_params=int_params,
        )

    if not results:
        print("[warn] No results collected.")
        return

    for result in results:
        result.params.update(fixed_params)

    # Pick best
    best = min(results, key=lambda r: r.objective)
    for name in param_order:
        if name not in best.params:
            raise KeyError(f"Best trial missing parameter '{name}' in results.")
    param_summary = " ".join(
        f"{name}={best.params.get(name, float('nan')):.6g}" for name in param_order
    )
    print(
        f"[best] objective={best.objective:.6f} mean_energy={best.mean_energy:.6f} "
        f"std={best.std_energy:.6f} distance_per_step={best.distance_per_step:.6f} "
        f"distance_total={best.distance_total:.6f} validity={best.validity_relaxed:.6f} "
        f"connected={best.connected_fraction:.4f} uniqueness={best.uniqueness:.4f} "
        f"real_accept={best.real_accept:.4f} "
        f"novelty={best.novelty:.4f} vun={best.vun:.4f} | {param_summary}"
    )

    # Persist CSV and best YAML-like summary
    out_csv = str(getattr(cfg.cali.output, "csv", "cali_results.csv"))
    _write_results_csv(out_csv, results, param_order)

    try:
        with open(str(getattr(cfg.cali.output, "best_txt", "best_params.txt")), "w") as fp:
            fp.write(f"best_mean_energy: {best.mean_energy:.6f}\n")
            fp.write(f"best_std_energy: {best.std_energy:.6f}\n")
            fp.write(f"best_objective: {best.objective:.6f}\n")
            fp.write(f"data_mean_energy: {data_energy_mean:.6f}\n")
            fp.write(f"data_std_energy: {data_energy_std:.6f}\n")
            fp.write(f"energy_threshold: {data_energy_threshold:.6f}\n")
            fp.write(f"distance_total: {best.distance_total:.6f}\n")
            fp.write(f"distance_per_step: {best.distance_per_step:.6f}\n")
            fp.write(f"best_validity_relaxed: {best.validity_relaxed:.6f}\n")
            fp.write(f"best_connected_fraction: {best.connected_fraction:.6f}\n")
            fp.write(f"best_real_accept: {best.real_accept:.6f}\n")
            fp.write(f"best_novelty: {best.novelty:.6f}\n")
            for name in param_order:
                fp.write(f"{name}: {best.params.get(name, float('nan')):.6g}\n")
        print("[info] Wrote best params to best_params.txt")
    except Exception as e:
        print(f"[warn] Could not write best params: {e}")

    # Visualization of energy vs MCMC steps for the best configuration
    try:
        with open_dict(cfg):
            cfg.viz = getattr(cfg, "viz", {}) or {}
            cfg.viz["enabled"] = True
            cfg.viz["proposal"] = proposal
            if single_param_mode:
                cfg.viz["energy_threshold"] = None
                if is_anneal_proposal:
                    cfg.viz["dl_beta_init"] = float(
                        best.params.get("dl_beta_init", best.params.get("dl_beta", float("nan")))
                    )
                    cfg.viz["dl_beta_final"] = float(
                        best.params.get(
                            "dl_beta_final",
                            best.params.get("dl_beta", best.params.get("dl_beta_far", float("nan"))),
                        )
                    )
                    cfg.viz["dl_beta_anneal_steps"] = int(
                        best.params.get("dl_beta_anneal_steps", best.params.get("dl_anneal_steps", 0))
                    )
                    cfg.viz["dl_lambda_X"] = float(
                        best.params.get(
                            "dl_lambda_X",
                            best.params.get("dl_lambda_X_far", float("nan")),
                        )
                    )
                    cfg.viz["dl_lambda_E"] = float(
                        best.params.get(
                            "dl_lambda_E",
                            best.params.get("dl_lambda_E_far", float("nan")),
                        )
                    )
                    cfg.viz["dl_beta"] = cfg.viz["dl_beta_final"]
                elif is_twobetas_anneal:
                    cfg.viz["dl_beta_prop"] = float(
                        best.params.get("dl_beta_prop", float("nan"))
                    )
                    cfg.viz["dl_beta_mh_init"] = float(
                        best.params.get("dl_beta_mh_init", float("nan"))
                    )
                    cfg.viz["dl_beta_mh_final"] = float(
                        best.params.get("dl_beta_mh_final", float("nan"))
                    )
                    cfg.viz["dl_beta_mh_anneal_steps"] = int(
                        best.params.get("dl_beta_mh_anneal_steps", 0)
                    )
                    cfg.viz["dl_beta"] = cfg.viz["dl_beta_prop"]
                    cfg.viz["dl_lambda_X"] = float(
                        best.params.get("dl_lambda_X", float("nan"))
                    )
                    cfg.viz["dl_lambda_E"] = float(
                        best.params.get("dl_lambda_E", float("nan"))
                    )
                elif is_twobetas:
                    cfg.viz["dl_beta_prop"] = float(
                        best.params.get("dl_beta_prop", float("nan"))
                    )
                    cfg.viz["dl_beta_mh"] = float(
                        best.params.get("dl_beta_mh", float("nan"))
                    )
                    cfg.viz["dl_beta"] = cfg.viz["dl_beta_prop"]
                    cfg.viz["dl_lambda_X"] = float(
                        best.params.get("dl_lambda_X", float("nan"))
                    )
                    cfg.viz["dl_lambda_E"] = float(
                        best.params.get("dl_lambda_E", float("nan"))
                    )
                else:
                    cfg.viz["dl_beta"] = float(
                        best.params.get("dl_beta", best.params.get("dl_beta_far", float("nan")))
                    )
                    cfg.viz["dl_lambda_X"] = float(
                        best.params.get(
                            "dl_lambda_X",
                            best.params.get("dl_lambda_X_far", float("nan")),
                        )
                    )
                    cfg.viz["dl_lambda_E"] = float(
                        best.params.get(
                            "dl_lambda_E",
                            best.params.get("dl_lambda_E_far", float("nan")),
                        )
                    )
                    cfg.viz["dl_beta_near"] = cfg.viz["dl_beta"]
                    cfg.viz["dl_lambda_X_near"] = cfg.viz["dl_lambda_X"]
                    cfg.viz["dl_lambda_E_near"] = cfg.viz["dl_lambda_E"]
                    cfg.viz["dl_beta_far"] = cfg.viz["dl_beta"]
                    cfg.viz["dl_lambda_X_far"] = cfg.viz["dl_lambda_X"]
                    cfg.viz["dl_lambda_E_far"] = cfg.viz["dl_lambda_E"]
            else:
                cfg.viz["energy_threshold"] = float(data_energy_threshold)
                cfg.viz["dl_beta_near"] = float(best.params.get("dl_beta_near", float('nan')))
                cfg.viz["dl_lambda_X_near"] = float(best.params.get("dl_lambda_X_near", float('nan')))
                cfg.viz["dl_lambda_E_near"] = float(best.params.get("dl_lambda_E_near", float('nan')))
                cfg.viz["dl_beta_far"] = float(best.params.get("dl_beta_far", float('nan')))
                cfg.viz["dl_lambda_X_far"] = float(best.params.get("dl_lambda_X_far", float('nan')))
                cfg.viz["dl_lambda_E_far"] = float(best.params.get("dl_lambda_E_far", float('nan')))
            cfg.viz["noise_transition"] = noise_transition
            cfg.viz["chain_warmup"] = OmegaConf.to_container(
                cfg.cali.chain_warmup,
                resolve=True,
            )
            cfg.viz["N_noise"] = int(min(64, num_init_noise))
            cfg.viz["N_data"] = int(min(64, num_init_data))
            cfg.viz["visualization_steps"] = int(max(steps, 1))
            cfg.viz["save_png"] = True
            cfg.viz["save_csv"] = True
            cfg.viz["filename_png"] = str(getattr(cfg.cali.output, "viz_png", "cali_energy_trajectories.png"))
            cfg.viz["filename_csv"] = str(getattr(cfg.cali.output, "viz_csv", "cali_energy_trajectories.csv"))
            # Keep it lightweight by default
            cfg.viz["plot_molecule_metrics"] = False
            # Build reference bands from a moderate sample for speed
            cfg.viz["M_data"] = 512
            cfg.viz["M_noise"] = 512
        print("[info] Generating energy trajectory visualization for best params …")
        _run_energy_viz(
            cfg,
            model=model,
            datamodule=datamodule,
            dataset_infos=dataset_infos,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
        )
    except Exception as e:
        print(f"[warn] Could not generate energy trajectory visualization: {e}")


if __name__ == "__main__":
    main()
