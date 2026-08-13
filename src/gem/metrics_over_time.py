"""MOSES unconditional GEM sampling metrics over MCMC time.

This is the slim public evaluator for the MOSES release. It initializes graphs
from noise or training data, optionally applies the same transport burn-in used
by the animation workflow, records requested MCMC steps, and reports molecular
validity, uniqueness, novelty, connectedness, and FCD.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (str(_SRC_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    Chem = None

from gem.analysis.rdkit_functions import build_molecule, mol2smiles
from gem.datasets import moses_dataset
from gem import sampler
from gem.animate_energy import _build_model_and_data
from gem.checkpoint_utils import load_model_checkpoint
from gem.dlangevin_utils import resolve_chain_warmup, resolve_dl_parameters
from gem.ot_data import collect_graphs_from_data
from gem.metrics.molecular_metrics import (
    compute_fcd_from_statistics,
    compute_fcd_statistics,
    prepare_fcd_smiles,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _resolve_optional_path(value: Any) -> Path | None:
    if _is_nullish(value):
        return None
    return Path(to_absolute_path(str(value))).resolve()


def _as_int_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, ListConfig)):
        steps = [int(v) for v in value]
    elif isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        steps = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        steps = [int(value)]
    steps = sorted(set(max(0, int(step)) for step in steps))
    return steps or [0]


def _optional_int(value: Any) -> int | None:
    if _is_nullish(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if _is_nullish(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset_raw_dir(datamodule, cfg: DictConfig) -> Path:
    raw_dir = getattr(getattr(datamodule, "train_dataset", None), "raw_dir", None)
    if raw_dir is None:
        raw_dir = Path(to_absolute_path(str(cfg.dataset.datadir))) / "raw"
    return Path(raw_dir).resolve()


def _dataset_smiles(
    datamodule,
    cfg: DictConfig,
    *,
    filter_dataset: bool | None = None,
) -> Dict[str, List[str]]:
    raw_dir = _dataset_raw_dir(datamodule, cfg)
    if filter_dataset is None:
        filter_dataset = bool(getattr(cfg.dataset, "filter", False))
    return moses_dataset.get_smiles(
        raw_dir=str(raw_dir),
        filter_dataset=bool(filter_dataset),
    )


def _moses_split_source(raw_dir: Path, split: str, *, filter_dataset: bool) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unknown MOSES split {split!r}; expected train, val, or test."
        )
    filename = (
        f"new_{split}.smiles" if filter_dataset else f"{split}_moses.csv"
    )
    return raw_dir / filename


def _official_moses_split_name(split: str) -> str:
    return {
        "train": "train",
        "val": "test_scaffolds",
        "test": "test",
    }.get(split, split)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smiles_fingerprint(smiles: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in smiles:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_or_compute_reference_statistics(
    reference_smiles: Sequence[str],
    *,
    fingerprint: str,
    cache_path: Path | None,
    device: str,
) -> tuple[tuple[np.ndarray, np.ndarray], bool]:
    if cache_path is not None and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_fingerprint = str(cached["fingerprint"].item())
                cached_count = int(cached["count"].item())
                if (
                    cached_fingerprint == fingerprint
                    and cached_count == len(reference_smiles)
                ):
                    return (
                        np.asarray(cached["mean"]),
                        np.asarray(cached["covariance"]),
                    ), True
        except (OSError, KeyError, ValueError):
            pass

    statistics = compute_fcd_statistics(reference_smiles, device=device)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=statistics[0],
                covariance=statistics[1],
                fingerprint=np.asarray(fingerprint),
                count=np.asarray(len(reference_smiles), dtype=np.int64),
            )
        os.replace(temporary_path, cache_path)
    return statistics, False


def _largest_fragment_smiles(smiles: Sequence[str]) -> tuple[List[str], int]:
    if Chem is None:
        raise RuntimeError("RDKit is required for largest-fragment FCD preprocessing.")

    largest_smiles: List[str] = []
    rejected = 0
    for value in smiles:
        try:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError("RDKit could not parse SMILES")
            fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
            largest = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
            Chem.SanitizeMol(largest)
            largest_smiles.append(Chem.MolToSmiles(largest))
        except Exception:
            rejected += 1
    return largest_smiles, rejected


def _random_subset(items: Sequence[str], size: int, seed: int | None) -> List[str]:
    items = list(items)
    if size <= 0 or size >= len(items):
        return items
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(items), size=size, replace=False))
    return [items[int(idx)] for idx in indices]


def _molecule_records(
    node_types: Sequence[torch.Tensor],
    edge_types: Sequence[torch.Tensor],
    dataset_info,
    *,
    step: int,
    sample_offset: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for local_idx, (nt, et) in enumerate(zip(node_types, edge_types)):
        record: Dict[str, Any] = {
            "step": int(step),
            "index": int(sample_offset + local_idx),
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
                    record["num_fragments"] = int(len(Chem.rdmolops.GetMolFrags(mol)))
                except Exception:
                    record["num_fragments"] = None
            smiles = mol2smiles(mol)
            record["smiles"] = smiles
            record["valid"] = bool(smiles)
            if smiles:
                connected = ("." not in smiles) and (record["num_fragments"] in {None, 1})
                record["connected"] = bool(connected)
                record["valid_connected"] = bool(connected)
        except Exception as exc:
            record["error"] = str(exc)
        records.append(record)
    return records


def _compute_summary(
    records: Sequence[Dict[str, Any]],
    *,
    train_smiles: set[str],
    fcd_reference_statistics: tuple[np.ndarray, np.ndarray] | None,
    compute_fcd_enabled: bool,
    fcd_device: str = "cpu",
    fcd_generated_largest_fragment: bool = False,
) -> Dict[str, Any]:
    total = len(records)
    valid_smiles = [str(row["smiles"]) for row in records if row.get("valid") and row.get("smiles")]
    unique_smiles = sorted(set(valid_smiles))
    connected_count = sum(1 for row in records if row.get("connected"))
    valid_connected_count = sum(1 for row in records if row.get("valid_connected"))
    novel_smiles = [smiles for smiles in unique_smiles if smiles not in train_smiles]

    fcd_score: float | None
    fcd_generated_input = 0
    fcd_generated_used = 0
    fcd_generated_rejected = 0
    fcd_largest_fragment_rejected = 0
    if compute_fcd_enabled:
        if fcd_reference_statistics is None:
            raise ValueError("FCD reference statistics are required when FCD is enabled.")
        fcd_smiles = valid_smiles
        fcd_generated_input = len(fcd_smiles)
        if fcd_generated_largest_fragment:
            fcd_smiles, fcd_largest_fragment_rejected = _largest_fragment_smiles(
                fcd_smiles
            )
        fcd_smiles, fcd_generated_rejected = prepare_fcd_smiles(
            fcd_smiles,
            canonicalize=True,
        )
        fcd_generated_used = len(fcd_smiles)
        if fcd_smiles:
            fcd_score = float(
                compute_fcd_from_statistics(
                    fcd_reference_statistics,
                    fcd_smiles,
                    device=fcd_device,
                )
            )
        else:
            fcd_score = -1.0
    else:
        fcd_score = None

    return {
        "total": int(total),
        "valid_count": int(len(valid_smiles)),
        "validity": float(len(valid_smiles) / total) if total else 0.0,
        "validity_pct": float(100.0 * len(valid_smiles) / total) if total else 0.0,
        "unique_count": int(len(unique_smiles)),
        "uniqueness": float(len(unique_smiles) / len(valid_smiles)) if valid_smiles else 0.0,
        "uniqueness_pct": float(100.0 * len(unique_smiles) / len(valid_smiles)) if valid_smiles else 0.0,
        "novel_count": int(len(novel_smiles)),
        "novelty": float(len(novel_smiles) / len(unique_smiles)) if unique_smiles else 0.0,
        "novelty_pct": float(100.0 * len(novel_smiles) / len(unique_smiles)) if unique_smiles else 0.0,
        "vun_count": int(len(novel_smiles)),
        "vun": float(len(novel_smiles) / total) if total else 0.0,
        "vun_pct": float(100.0 * len(novel_smiles) / total) if total else 0.0,
        "connected_count": int(connected_count),
        "connected_fraction": float(connected_count / total) if total else 0.0,
        "connected_pct": float(100.0 * connected_count / total) if total else 0.0,
        "valid_connected_count": int(valid_connected_count),
        "valid_connected_fraction": float(valid_connected_count / total) if total else 0.0,
        "valid_connected_pct": float(100.0 * valid_connected_count / total) if total else 0.0,
        "fcd": fcd_score,
        "fcd_generated_input": int(fcd_generated_input),
        "fcd_generated_used": int(fcd_generated_used),
        "fcd_generated_rejected": int(fcd_generated_rejected),
        "fcd_generated_largest_fragment": bool(fcd_generated_largest_fragment),
        "fcd_largest_fragment_rejected": int(fcd_largest_fragment_rejected),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in keys})


def _run_mcmc(
    *,
    model,
    dataset_infos,
    node_types: Sequence[torch.Tensor],
    edge_types: Sequence[torch.Tensor],
    extra_features,
    domain_features,
    device: torch.device,
    run_cfg,
    steps: int,
    step_offset: int,
):
    base_beta, base_lambda_x, base_lambda_e, dual_kwargs = resolve_dl_parameters(run_cfg)
    proposal = str(getattr(run_cfg, "proposal", "dlangevin"))
    anneal_kwargs: Dict[str, Any] = {}
    if proposal.lower() in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}:
        anneal_kwargs.update(
            dl_beta_init=float(getattr(run_cfg, "dl_beta_init", base_beta)),
            dl_beta_final=float(getattr(run_cfg, "dl_beta_final", base_beta)),
            dl_beta_anneal_steps=int(getattr(run_cfg, "dl_beta_anneal_steps", max(steps, 1))),
        )
    return sampler.mcmc_sample_batch(
        model=model,
        dataset_info=dataset_infos,
        node_types_list=node_types,
        edge_types_list=edge_types,
        extra_features=extra_features,
        domain_features=domain_features,
        steps=int(steps),
        device=device,
        proposal=proposal,
        gwd_beta=float(getattr(run_cfg, "gwd_beta", 1.0)),
        dl_beta=base_beta,
        dl_beta_prop=_optional_float(getattr(run_cfg, "dl_beta_prop", None)),
        dl_beta_mh=_optional_float(getattr(run_cfg, "dl_beta_mh", None)),
        dl_beta_mh_init=_optional_float(getattr(run_cfg, "dl_beta_mh_init", None)),
        dl_beta_mh_final=_optional_float(getattr(run_cfg, "dl_beta_mh_final", None)),
        dl_beta_mh_anneal_steps=_optional_int(getattr(run_cfg, "dl_beta_mh_anneal_steps", None)),
        dl_lambda_X=base_lambda_x,
        dl_lambda_E=base_lambda_e,
        simple_n_edits=_optional_int(getattr(run_cfg, "simple_n_edits", None)),
        amp_dtype=getattr(run_cfg, "amp_dtype", None),
        collect_stats=bool(getattr(run_cfg, "collect_sampler_stats", False)),
        step_offset=int(step_offset),
        **dual_kwargs,
        **anneal_kwargs,
    )


def run_metrics(cfg: DictConfig) -> Dict[str, Any] | None:
    run_cfg = getattr(cfg, "metrics_run", None)
    if run_cfg is None:
        raise ValueError("Missing cfg.metrics_run section.")
    if not bool(getattr(run_cfg, "enabled", True)):
        print("[info] metrics_run.enabled=false; skipping metrics evaluation.")
        return None

    print("[info] Using metrics config:\n" + OmegaConf.to_yaml(run_cfg, resolve=True))
    seed_cfg = getattr(run_cfg, "seed", None)
    if seed_cfg is not None:
        seed = int(seed_cfg)
        _set_seed(seed)
        print(f"[info] Using seed={seed}")

    checkpoint = _resolve_optional_path(getattr(run_cfg, "checkpoint", None))
    if checkpoint is None:
        raise ValueError("Set metrics_run.checkpoint=/path/to/model.pt.")

    model, datamodule, dataset_infos, extra_features, domain_features, device = _build_model_and_data(
        cfg,
        model=None,
        datamodule=None,
        dataset_infos=None,
        extra_features=None,
        domain_features=None,
        device=None,
    )
    use_ema = bool(getattr(run_cfg, "use_ema", False))
    load_model_checkpoint(model, checkpoint, map_location=device, use_ema=use_ema)
    print(f"[info] Loaded checkpoint weights: {checkpoint}")
    model.eval()

    dataset_filter = bool(getattr(cfg.dataset, "filter", False))
    raw_dir = _dataset_raw_dir(datamodule, cfg)
    smiles_by_split = _dataset_smiles(
        datamodule,
        cfg,
        filter_dataset=dataset_filter,
    )
    canonical_train_smiles, train_smiles_rejected = prepare_fcd_smiles(
        smiles_by_split.get("train", []),
        canonicalize=True,
    )
    train_smiles = set(canonical_train_smiles)
    del canonical_train_smiles

    compute_fcd_enabled = bool(getattr(run_cfg, "compute_fcd", True))
    reference_split = str(getattr(run_cfg, "fcd_reference_split", "val"))
    reference_filter = bool(getattr(run_cfg, "fcd_reference_filter", False))
    reference_size = int(getattr(run_cfg, "fcd_reference_size", 0) or 0)
    reference_seed = None
    reference_smiles: List[str] = []
    reference_rejected = 0
    reference_statistics: tuple[np.ndarray, np.ndarray] | None = None
    reference_cache_hit = False
    reference_fingerprint = None
    reference_source = _moses_split_source(
        raw_dir,
        reference_split,
        filter_dataset=reference_filter,
    )
    reference_source_sha256 = None
    reference_cache_path = _resolve_optional_path(
        getattr(run_cfg, "fcd_reference_stats_cache", None)
    )
    fcd_device = str(getattr(run_cfg, "fcd_device", "auto"))
    reference_canonicalize = bool(
        getattr(run_cfg, "fcd_reference_canonicalize", False)
    )

    if compute_fcd_enabled:
        reference_by_split = (
            smiles_by_split
            if reference_filter == dataset_filter
            else _dataset_smiles(
                datamodule,
                cfg,
                filter_dataset=reference_filter,
            )
        )
        if reference_split not in reference_by_split:
            raise ValueError(f"MOSES reference split {reference_split!r} is unavailable.")
        reference_smiles = list(reference_by_split[reference_split])
        if reference_size > 0:
            reference_seed = _optional_int(
                getattr(run_cfg, "fcd_reference_seed", getattr(run_cfg, "seed", 0))
            )
            reference_smiles = _random_subset(
                reference_smiles,
                reference_size,
                reference_seed,
            )

        expected_reference_size = _optional_int(
            getattr(run_cfg, "fcd_reference_expected_size", None)
        )
        if (
            reference_size == 0
            and expected_reference_size is not None
            and len(reference_smiles) != expected_reference_size
        ):
            raise ValueError(
                "Full MOSES FCD reference has an unexpected size: "
                f"expected {expected_reference_size}, found {len(reference_smiles)} "
                f"in {reference_source}."
            )

        reference_smiles, reference_rejected = prepare_fcd_smiles(
            reference_smiles,
            canonicalize=reference_canonicalize,
        )
        if not reference_smiles:
            raise ValueError("No valid SMILES remain in the FCD reference split.")
        reference_fingerprint = _smiles_fingerprint(reference_smiles)
        reference_statistics, reference_cache_hit = (
            _load_or_compute_reference_statistics(
                reference_smiles,
                fingerprint=reference_fingerprint,
                cache_path=reference_cache_path,
                device=fcd_device,
            )
        )
        reference_source_sha256 = _sha256_file(reference_source)
        print(
            "[fcd-reference] "
            f"internal_split={reference_split} "
            f"official_split={_official_moses_split_name(reference_split)} "
            f"filter={reference_filter} used={len(reference_smiles)} "
            f"rejected={reference_rejected} cache_hit={reference_cache_hit}"
        )
        del reference_by_split
    del smiles_by_split

    steps = _as_int_list(getattr(run_cfg, "steps", [1000]))
    total_samples = int(getattr(run_cfg, "total_samples", 25000))
    batch_size = int(getattr(run_cfg, "batch_size", 512))
    if total_samples <= 0 or batch_size <= 0:
        raise ValueError("metrics_run.total_samples and metrics_run.batch_size must be positive.")

    init_source = str(getattr(run_cfg, "init_source", "noise")).strip().lower()
    if init_source not in {"noise", "data"}:
        raise ValueError("metrics_run.init_source must be either 'noise' or 'data'.")

    data_init_nodes: List[torch.Tensor] = []
    data_init_edges: List[torch.Tensor] = []
    if init_source == "data":
        print(f"[info] Collecting {total_samples} training-data graphs for initialization.")
        data_init_nodes, data_init_edges = collect_graphs_from_data(
            datamodule,
            total_samples,
        )
        if len(data_init_nodes) < total_samples or len(data_init_edges) < total_samples:
            raise RuntimeError(
                f"Collected only {min(len(data_init_nodes), len(data_init_edges))} "
                f"data graphs for {total_samples} requested samples."
            )

    noise_transition = str(getattr(run_cfg, "noise_transition", getattr(cfg.model, "transition", "uniform")))
    chain_warmup = resolve_chain_warmup(
        getattr(run_cfg, "chain_warmup", None),
        fallback=run_cfg,
        default_gwd_beta=float(getattr(run_cfg, "gwd_beta", 1.0)),
    )
    vectorized_simple_warmup = sampler.should_vectorize_simple_warmup(
        chain_warmup.proposal,
        vectorized=chain_warmup.vectorized,
    )
    if chain_warmup.enabled:
        print(
            "[info] Metrics transport warmup: "
            f"proposal={chain_warmup.proposal}, steps={chain_warmup.steps}, "
            f"vectorized={vectorized_simple_warmup}."
        )

    records_by_step: Dict[int, List[Dict[str, Any]]] = {step: [] for step in steps}
    cumulative_mcmc: Dict[int, Dict[str, int]] = {
        step: {"accepted": 0, "proposals": 0} for step in steps
    }
    warmup_totals = {"accepted": 0, "proposals": 0}

    sample_offset = 0
    while sample_offset < total_samples:
        batch_n = min(batch_size, total_samples - sample_offset)
        if init_source == "noise":
            graphs = sampler.initialize_random_graphs(
                batch_size=batch_n,
                dataset_info=dataset_infos,
                device=device,
                transition=noise_transition,
            )
            nodes = [nt for nt, _ in graphs]
            edges = [et for _, et in graphs]
        else:
            batch_end = sample_offset + batch_n
            nodes = [nt.clone() for nt in data_init_nodes[sample_offset:batch_end]]
            edges = [et.clone() for et in data_init_edges[sample_offset:batch_end]]

        if chain_warmup.enabled:
            if vectorized_simple_warmup:
                edits_per_step = (
                    5
                    if chain_warmup.simple_n_edits is None
                    else int(chain_warmup.simple_n_edits)
                )
                nodes, edges, accepts, proposals, _ = (
                    sampler.run_simple_v2_warmup_vectorized(
                        model=model,
                        dataset_info=dataset_infos,
                        node_types_list=nodes,
                        edge_types_list=edges,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        steps=chain_warmup.steps,
                        device=device,
                        edits_per_step=edits_per_step,
                        amp_dtype=getattr(run_cfg, "amp_dtype", None),
                        stop_when_unchanged=True,
                        collect_stats=bool(
                            getattr(run_cfg, "collect_sampler_stats", False)
                        ),
                    )
                )
            else:
                nodes, edges, accepts, proposals, _ = sampler.mcmc_sample_batch(
                    model=model,
                    dataset_info=dataset_infos,
                    node_types_list=nodes,
                    edge_types_list=edges,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    steps=chain_warmup.steps,
                    device=device,
                    proposal=chain_warmup.proposal,
                    gwd_beta=chain_warmup.gwd_beta,
                    dl_beta=chain_warmup.dl_beta,
                    dl_lambda_X=chain_warmup.dl_lambda_X,
                    dl_lambda_E=chain_warmup.dl_lambda_E,
                    simple_n_edits=chain_warmup.simple_n_edits,
                    amp_dtype=getattr(run_cfg, "amp_dtype", None),
                    collect_stats=bool(
                        getattr(run_cfg, "collect_sampler_stats", False)
                    ),
                    **chain_warmup.dual_kwargs,
                )
            warmup_totals["accepted"] += int(accepts)
            warmup_totals["proposals"] += int(proposals)

        current_step = 0
        batch_accepts = 0
        batch_proposals = 0
        for target_step in steps:
            delta = int(target_step - current_step)
            if delta > 0:
                nodes, edges, accepts, proposals, _ = _run_mcmc(
                    model=model,
                    dataset_infos=dataset_infos,
                    node_types=nodes,
                    edge_types=edges,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    device=device,
                    run_cfg=run_cfg,
                    steps=delta,
                    step_offset=current_step,
                )
                batch_accepts += int(accepts)
                batch_proposals += int(proposals)
                current_step = int(target_step)
            cumulative_mcmc[target_step]["accepted"] += batch_accepts
            cumulative_mcmc[target_step]["proposals"] += batch_proposals
            records_by_step[target_step].extend(
                _molecule_records(
                    nodes,
                    edges,
                    dataset_infos,
                    step=target_step,
                    sample_offset=sample_offset,
                )
            )

        sample_offset += batch_n
        print(f"[info] Generated {sample_offset}/{total_samples} samples for metrics.")

    summaries: List[Dict[str, Any]] = []
    fcd_generated_largest_fragment = bool(
        getattr(run_cfg, "fcd_generated_largest_fragment", False)
    )
    for step in steps:
        summary = _compute_summary(
            records_by_step[step],
            train_smiles=train_smiles,
            fcd_reference_statistics=reference_statistics,
            compute_fcd_enabled=compute_fcd_enabled,
            fcd_device=fcd_device,
            fcd_generated_largest_fragment=fcd_generated_largest_fragment,
        )
        proposals = cumulative_mcmc[step]["proposals"]
        accepts = cumulative_mcmc[step]["accepted"]
        warmup_proposals = warmup_totals["proposals"]
        warmup_accepts = warmup_totals["accepted"]
        summary.update(
            {
                "mcmc_steps": int(step),
                "transport_burn_in_steps": int(chain_warmup.steps if chain_warmup.enabled else 0),
                "transport_burn_in_vectorized": bool(vectorized_simple_warmup),
                "mcmc_accepted": int(accepts),
                "mcmc_proposals": int(proposals),
                "mcmc_acceptance": float(accepts / proposals) if proposals else 0.0,
                "transport_burn_in_accepted": int(warmup_accepts),
                "transport_burn_in_proposals": int(warmup_proposals),
                "transport_burn_in_acceptance": (
                    float(warmup_accepts / warmup_proposals) if warmup_proposals else 0.0
                ),
                "transport_burn_in_moves": int(warmup_accepts),
                "transport_burn_in_attempts": int(warmup_proposals),
                "transport_burn_in_move_rate": (
                    float(warmup_accepts / warmup_proposals)
                    if warmup_proposals
                    else 0.0
                ),
            }
        )
        summaries.append(summary)
        print(
            "[metrics] "
            f"steps={step} valid={summary['validity_pct']:.2f}% "
            f"unique={summary['uniqueness_pct']:.2f}% novel={summary['novelty_pct']:.2f}% "
            f"vun={summary['vun_pct']:.2f}% connected={summary['connected_pct']:.2f}% "
            f"fcd={summary['fcd']}"
        )

    output_json = Path(str(getattr(run_cfg, "output_json", "metrics_over_time.json")))
    output_csv = Path(str(getattr(run_cfg, "output_csv", "metrics_over_time.csv")))
    checkpoint_sha256 = _sha256_file(checkpoint)
    release_checkpoint_cfg = getattr(cfg, "checkpoint", None)
    release_checkpoint_sha256 = (
        str(getattr(release_checkpoint_cfg, "sha256", ""))
        if release_checkpoint_cfg is not None
        else ""
    )
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "matches_release_checkpoint": bool(
            release_checkpoint_sha256
            and checkpoint_sha256 == release_checkpoint_sha256
        ),
        "use_ema": bool(use_ema),
        "dataset_filter": dataset_filter,
        "init_source": init_source,
        "noise_transition": noise_transition,
        "total_samples": int(total_samples),
        "batch_size": int(batch_size),
        "fcd_reference_split": reference_split,
        "fcd_reference_size": int(len(reference_smiles)),
        "fcd_reference_seed": reference_seed,
        "novelty_reference": {
            "internal_split": "train",
            "filter_dataset": dataset_filter,
            "source": str(
                _moses_split_source(
                    raw_dir,
                    "train",
                    filter_dataset=dataset_filter,
                )
            ),
            "canonical_unique_size": int(len(train_smiles)),
            "rejected": int(train_smiles_rejected),
        },
        "fcd_reference": {
            "internal_split": reference_split,
            "official_split": _official_moses_split_name(reference_split),
            "filter_dataset": reference_filter,
            "full_split": bool(reference_size == 0),
            "source": str(reference_source),
            "source_sha256": reference_source_sha256,
            "selected_size": int(len(reference_smiles) + reference_rejected),
            "used_size": int(len(reference_smiles)),
            "rejected": int(reference_rejected),
            "canonicalized": reference_canonicalize,
            "fingerprint": reference_fingerprint,
            "statistics_cache": (
                str(reference_cache_path) if reference_cache_path is not None else None
            ),
            "statistics_cache_hit": bool(reference_cache_hit),
            "device": fcd_device,
            "generated_largest_fragment": fcd_generated_largest_fragment,
        },
        "summaries": summaries,
    }
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    _write_rows_csv(output_csv, summaries)
    print(f"[info] Wrote metrics JSON: {output_json.resolve()}")
    print(f"[info] Wrote metrics CSV: {output_csv.resolve()}")

    if bool(getattr(run_cfg, "save_smiles", True)):
        smiles_dir = Path(str(getattr(run_cfg, "smiles_dir", "generated_smiles")))
        smiles_dir.mkdir(parents=True, exist_ok=True)
        for step, records in records_by_step.items():
            _write_rows_csv(smiles_dir / f"generated_smiles_step{step:04d}.csv", records)
        print(f"[info] Wrote generated SMILES tables under: {smiles_dir.resolve()}")

    return payload


@hydra.main(version_base="1.3", config_path="../../configs", config_name="gem_metrics_over_time_moses")
def main(cfg: DictConfig):
    run_metrics(cfg)


if __name__ == "__main__":
    main()
