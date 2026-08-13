"""
Energy landscape visualization for GEM-EBM.

Computes data energy mean/std from M_data data graphs and runs two groups of
MCMC chains (starting from noise and from data) for a small number of steps,
plotting the energy trajectories against the data/noise mean±std bands.

Configuration lives under cfg.viz (separate from train/sample).

Can be used in two modes:
- as a script (Hydra entry): builds its own datamodule/model
- as a library: call run_viz(...) from training to reuse the trained model
"""

import os
import sys
import pathlib
from pathlib import Path
from typing import List, Tuple, Optional

import hydra
import torch
from omegaconf import DictConfig
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package root is importable when running this file directly.
_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from gem.datasets import moses_dataset
from gem.models.transformer_model import GraphTransformer
from gem.models.extra_features import ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures
from gem.checkpoint_utils import load_model_checkpoint
try:
    from . import sampler
    from .sampler_energy import register_property_conditioner
    from .dlangevin_utils import (
        resolve_chain_warmup,
        resolve_dl_parameters,
        resolve_two_beta_annealing_kwargs,
        resolve_two_beta_kwargs,
    )
except ImportError:
    from gem import sampler
    from gem.sampler_energy import register_property_conditioner
    from gem.dlangevin_utils import (
        resolve_chain_warmup,
        resolve_dl_parameters,
        resolve_two_beta_annealing_kwargs,
        resolve_two_beta_kwargs,
    )
from gem.analysis.rdkit_functions import (
    build_molecule,
    build_molecule_with_partial_charges,
    mol2smiles,
)
from gem import utils


def _ensure_run_dir():
    try:
        os.makedirs(".", exist_ok=True)
    except Exception:
        pass


def _setup_file_logging(log_name: str):
    try:
        log_file = open(log_name, "a", buffering=1)

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
        print(f"[info] Logging to {log_name}")
    except Exception as e:
        print(f"[warn] Could not set up file logging: {e}")


def _load_checkpoint_if_any(
    model: torch.nn.Module,
    device: torch.device,
    path: str | None,
    *,
    use_ema: bool = False,
) -> bool:
    """Load model weights if `path` is provided. Accepts raw state_dict or wrapped checkpoints."""
    if not path:
        return False
    load_model_checkpoint(model, path, map_location=device, use_ema=use_ema)
    weight_label = "EMA" if use_ema else "online"
    print(f"[info] Loaded {weight_label} checkpoint weights: {path}")
    return True


def _collect_graphs_from_data(datamodule, M: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Collect at least M graphs from the training loader as discrete label tensors."""
    node_list: List[torch.Tensor] = []
    edge_list: List[torch.Tensor] = []
    loader = datamodule.train_dataloader()
    it = iter(loader)
    while len(node_list) < M:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(datamodule.train_dataloader())
            batch = next(it)
        dense_data, node_mask = utils.to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        graphs = dense_data.mask(node_mask, collapse=True).split(node_mask)
        for g in graphs:
            node_list.append(g.X.long().cpu())
            edge_list.append(g.E.long().cpu())
            if len(node_list) >= M:
                break
    return node_list, edge_list


def _chunked_energy_mean_std(
    node_list: List[torch.Tensor],
    edge_list: List[torch.Tensor],
    model,
    dataset_info,
    extra_features,
    domain_features,
    device: torch.device,
    chunk_size: int = 256,
):
    import torch

    assert len(node_list) == len(edge_list)
    n = len(node_list)
    energies = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        E = sampler.energy_batch(
            model=model,
            node_types_list=node_list[start:end],
            edge_types_list=edge_list[start:end],
            dataset_info=dataset_info,
            device=device,
            extra_features=extra_features,
            domain_features=domain_features,
            detach=True,
            apply_property_conditioner=False,
        )
        energies.append(E.detach().cpu())
    energies = torch.cat(energies, dim=0)
    return energies.mean().item(), energies.std(unbiased=False).item()


def run_viz(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None = None,
    datamodule=None,
    dataset_infos=None,
    extra_features=None,
    domain_features=None,
    device: torch.device | None = None,
    iteration: Optional[int] = None,
):
    """Run the energy visualization using provided objects or by creating them.

    When called from training, pass the trained `model`, and the existing
    `datamodule`, `dataset_infos`, `extra_features`, `domain_features`, and `device`
    to reuse them and visualize the trained energy.
    """
    if not bool(getattr(cfg.viz, "enabled", True)):
        print("[info] Visualization disabled by cfg.viz.enabled=false. Skipping.")
        return

    _ensure_run_dir()
    _setup_file_logging("visualize_energy.log")

    dataset_name = str(getattr(cfg.dataset, "name", "moses")).lower()

    # Build missing components if not provided (script mode)
    if datamodule is None:
        if dataset_name != "moses":
            raise NotImplementedError(
                f"Dataset '{cfg.dataset.name}' is not supported. Only 'moses' is available."
            )
        datamodule = moses_dataset.MosesDataModule(cfg)

    if dataset_infos is None:
        if dataset_name != "moses":
            raise NotImplementedError(
                f"Dataset '{cfg.dataset.name}' is not supported. Only 'moses' is available."
            )
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
    if extra_features is None or domain_features is None:
        extra_features = ExtraFeatures(
            cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        # Respect activation setting if constructing the model here
        act_name = str(getattr(cfg.model, "activation", "relu")).lower()
        if act_name == "silu":
            act_module = torch.nn.SiLU()
        else:
            act_module = torch.nn.ReLU()
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
    general_cfg = getattr(cfg, "general", None)
    general_resume = getattr(general_cfg, "resume", "") if general_cfg is not None else ""
    checkpoint_path = str(
        getattr(cfg.viz, "checkpoint", "") or general_resume or ""
    ).strip()
    if checkpoint_path:
        _load_checkpoint_if_any(
            model,
            device,
            checkpoint_path,
            use_ema=bool(getattr(cfg.viz, "use_ema", False)),
        )

    property_cfg = getattr(cfg, "property_condition", None)
    conditioner = None
    property_enabled = False
    property_png_path = None
    save_property_png = False
    property_hist_png_path = None
    save_property_hist_png = False
    make_property_hist_animation = False
    property_hist_animation_path: str | None = None
    property_hist_animation_steps_per_sec = 100.0
    property_hist_animation_fps = 30.0
    property_hist_animation_interval_ms = 200.0
    property_hist_animation_dpi = 150
    property_hist_history = None
    if property_cfg is not None and bool(getattr(property_cfg, "enabled", False)):
        raise NotImplementedError(
            "Property conditioning is not included in this slim MOSES release."
        )
    register_property_conditioner(conditioner)

    try:
        # --- Static energy bands (data and noise) ---
        M_data = int(getattr(cfg.viz, "M_data", 1024))
        M_noise = int(getattr(cfg.viz, "M_noise", 1024))
        viz_noise_transition = str(
            getattr(cfg.viz, "noise_transition", getattr(cfg.model, "transition", "marginal"))
        )
        print(f"[info] Visualization noise transition: {viz_noise_transition}")

        node_M, edge_M = _collect_graphs_from_data(datamodule, M_data)
        data_mean, data_std = _chunked_energy_mean_std(
            node_M, edge_M, model, dataset_infos, extra_features, domain_features, device
        )
        print(f"[info] Data energy mean/std over M_data={M_data}: {data_mean:.4f} ± {data_std:.4f}")

        noise_graphs = sampler.initialize_random_graphs(
            batch_size=M_noise,
            dataset_info=dataset_infos,
            device=device,
            transition=viz_noise_transition,
        )
        noise_nodes_M = [nt for (nt, _) in noise_graphs]
        noise_edges_M = [et for (_, et) in noise_graphs]
        noise_band_mean, noise_band_std = _chunked_energy_mean_std(
            noise_nodes_M, noise_edges_M, model, dataset_infos, extra_features, domain_features, device
        )
        print(
            f"[info] Noise energy mean/std over M_noise={M_noise}: {noise_band_mean:.4f} ± {noise_band_std:.4f}"
        )

        # --- Initialize chains ---
        N_noise = int(getattr(cfg.viz, "N_noise", 64))
        N_data = int(getattr(cfg.viz, "N_data", 64))
        steps = int(getattr(cfg.viz, "visualization_steps", 100))
        amp_dtype = getattr(cfg.viz, "amp_dtype", None)
        chain_warmup_viz = resolve_chain_warmup(
            getattr(cfg.viz, "chain_warmup", None),
            fallback=cfg.viz,
            default_gwd_beta=float(getattr(cfg.viz, "gwd_beta", 1.0)),
        )
        if chain_warmup_viz.enabled:
            print(
                "[info] Visualization chain warmup: "
                f"proposal={chain_warmup_viz.proposal}, steps={chain_warmup_viz.steps}."
            )
        warmup_steps_total = chain_warmup_viz.steps if chain_warmup_viz.enabled else 0
        record_warmup_phase = warmup_steps_total > 0
        history_len = steps + warmup_steps_total + 1

        noise_nodes_edges = sampler.initialize_random_graphs(
            batch_size=N_noise,
            dataset_info=dataset_infos,
            device=device,
            transition=viz_noise_transition,
        )
        noise_nodes = [nt for (nt, _) in noise_nodes_edges]
        noise_edges = [et for (_, et) in noise_nodes_edges]
        data_nodes, data_edges = _collect_graphs_from_data(datamodule, N_data)

        noise_mean = np.zeros(history_len, dtype=np.float64)
        noise_std = np.zeros(history_len, dtype=np.float64)
        data_start_mean = np.zeros(history_len, dtype=np.float64)
        data_start_std = np.zeros(history_len, dtype=np.float64)

        property_hist_init = None
        property_hist_final = None

        def _property_values(_nodes_seq, _edges_seq):
            return np.asarray([], dtype=np.float64)

        def _property_stats(_nodes_seq, _edges_seq):
            values = _property_values(_nodes_seq, _edges_seq)
            if isinstance(values, np.ndarray) and values.size > 0:
                return values, float(values.mean())
            return values, float("nan")

        if property_enabled:
            property_noise = np.zeros(history_len, dtype=np.float64)
            property_data = np.zeros(history_len, dtype=np.float64)
            property_target_scalar = float(conditioner.target_value.detach().cpu().item())
            property_hist_history = [] if make_property_hist_animation else None

            def _property_values(nodes_seq, edges_seq):
                if len(nodes_seq) == 0:
                    return np.asarray([], dtype=np.float64)
                preds = conditioner.predict_property(nodes_seq, edges_seq, device=device)
                return preds.detach().cpu().numpy().astype(np.float64, copy=False)
            def _property_stats(nodes_seq, edges_seq):
                values = _property_values(nodes_seq, edges_seq)
                if values.size == 0:
                    return values, float("nan")
                return values, float(values.mean())
        else:
            property_noise = property_data = None
            property_target_scalar = float("nan")

        plot_metrics = bool(getattr(cfg.viz, "plot_molecule_metrics", True))
        metrics_only = bool(getattr(cfg.viz, "metrics_only", False))
        if metrics_only and not plot_metrics:
            print("[warn] cfg.viz.metrics_only=true requires plot_molecule_metrics=true. Enabling metric plotting.")
            plot_metrics = True
        if plot_metrics:
            validity_mode = str(getattr(cfg.general, "validity_mode", "vfm_relaxed")).lower()
            if validity_mode not in {"vfm_relaxed", "strict"}:
                validity_mode = "vfm_relaxed"
            validity_label_primary = "validity_VFM_relaxed" if validity_mode == "vfm_relaxed" else "validity_strict"
            validity_label_display = (
                "Validity (VFM relaxed)" if validity_mode == "vfm_relaxed" else "Validity (strict)"
            )

            def _metric_group():
                return {
                    "validity": np.zeros(history_len, dtype=np.float64),
                    "uniqueness": np.zeros(history_len, dtype=np.float64),
                    "novelty": np.zeros(history_len, dtype=np.float64),
                }

            metric_groups = {
                "combined": _metric_group(),
                "noise": _metric_group(),
                "data": _metric_group(),
            }

            def _store_metrics(group: str, idx: int, values: Tuple[float, float, float]):
                metric_groups[group]["validity"][idx] = values[0]
                metric_groups[group]["uniqueness"][idx] = values[1]
                metric_groups[group]["novelty"][idx] = values[2]

        else:
            metric_groups = None
            validity_mode = "vfm_relaxed"
            validity_label_primary = "validity_VFM_relaxed"
            validity_label_display = "Validity (VFM relaxed)"

            def _store_metrics(*_args, **_kwargs):
                return

        legend_fontsize_raw = getattr(cfg.viz, "legend_fontsize", None)
        try:
            legend_fontsize = float(legend_fontsize_raw)
        except (TypeError, ValueError):
            legend_fontsize = 10.0
        mean_line_width = float(getattr(cfg.viz, "mean_line_width", 1.0))
        band_line_width = float(getattr(cfg.viz, "band_line_width", 0.9))

        def _compute_metrics(molecules, dataset_info, train_smiles):
            use_partial = validity_mode == "vfm_relaxed"
            build_fn = (
                lambda nt, et: build_molecule_with_partial_charges(nt, et, dataset_info.atom_decoder)
                if use_partial
                else build_molecule(nt, et, dataset_info.atom_decoder)
            )

            smiles_valid = []
            for nt, et in molecules:
                try:
                    mol = build_fn(nt, et)
                    s = mol2smiles(mol)
                    if s is not None:
                        smiles_valid.append(s)
                except Exception:
                    continue

            total = len(molecules)
            v = (len(smiles_valid) / total) if total > 0 else 0.0
            if len(smiles_valid) > 0:
                uniq_set = set(smiles_valid)
                u = len(uniq_set) / len(smiles_valid)
                if train_smiles is not None and len(uniq_set) > 0:
                    n = sum(1 for s in uniq_set if s not in train_smiles) / len(uniq_set)
                else:
                    n = 0.0
            else:
                u, n = 0.0, 0.0
            return float(v), float(u), float(n)

        if plot_metrics:
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=getattr(cfg.dataset, "filter", False),
            )
            train_smiles = dataset_smiles.get("train")
        else:
            train_smiles = None

        def _mcmc_step(
            nodes_seq,
            edges_seq,
            *,
            proposal_name: str,
            gwd_beta_value: float,
            dl_beta_value: float,
            dl_lambda_X_value: float,
            dl_lambda_E_value: float,
            simple_edits: Optional[int] = None,
            extra_dual_kwargs: Optional[dict] = None,
            anneal_kwargs: Optional[dict] = None,
            proposal_kwargs: Optional[dict] = None,
            step_offset: int = 0,
        ):
            if not nodes_seq:
                return nodes_seq, edges_seq
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
            if proposal_kwargs:
                kwargs.update(proposal_kwargs)
            nodes_out, edges_out, *_ = sampler.mcmc_sample_batch(**kwargs)
            return nodes_out, edges_out

        def _record_state(step_idx: int):
            nonlocal property_hist_init, property_hist_final
            if N_noise > 0:
                e_noise = sampler.energy_batch(
                    model=model,
                    node_types_list=noise_nodes,
                    edge_types_list=noise_edges,
                    dataset_info=dataset_infos,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    detach=True,
                    apply_property_conditioner=False,
                ).cpu().numpy()
                noise_mean[step_idx] = e_noise.mean()
                noise_std[step_idx] = e_noise.std()
            else:
                noise_mean[step_idx] = float("nan")
                noise_std[step_idx] = float("nan")

            if N_data > 0:
                e_data = sampler.energy_batch(
                    model=model,
                    node_types_list=data_nodes,
                    edge_types_list=data_edges,
                    dataset_info=dataset_infos,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    detach=True,
                    apply_property_conditioner=False,
                ).cpu().numpy()
                data_start_mean[step_idx] = e_data.mean()
                data_start_std[step_idx] = e_data.std()
            else:
                data_start_mean[step_idx] = float("nan")
                data_start_std[step_idx] = float("nan")

            if plot_metrics:
                molecules_noise = list(zip(noise_nodes, noise_edges))
                molecules_data = list(zip(data_nodes, data_edges))
                molecules_combined = molecules_noise + molecules_data
                _store_metrics("noise", step_idx, _compute_metrics(molecules_noise, dataset_infos, train_smiles))
                _store_metrics("data", step_idx, _compute_metrics(molecules_data, dataset_infos, train_smiles))
                _store_metrics("combined", step_idx, _compute_metrics(molecules_combined, dataset_infos, train_smiles))

            if property_enabled:
                noise_vals, noise_mean_val = _property_stats(noise_nodes, noise_edges)
                data_vals, data_mean_val = _property_stats(data_nodes, data_edges)
                property_noise[step_idx] = noise_mean_val
                property_data[step_idx] = data_mean_val
                hist_entry = {"noise": noise_vals, "data": data_vals}
                if step_idx == 0:
                    property_hist_init = hist_entry
                if step_idx == history_len - 1:
                    property_hist_final = hist_entry
                if property_hist_history is not None:
                    property_hist_history.append(
                        {
                            "step": step_idx,
                            "noise": noise_vals,
                            "data": data_vals,
                        }
                    )

        base_dl_beta, base_dl_lambda_X, base_dl_lambda_E, dual_kwargs = resolve_dl_parameters(cfg.viz)
        main_proposal = str(getattr(cfg.viz, "proposal", "random"))
        two_beta_kwargs_main = resolve_two_beta_kwargs(main_proposal, cfg.viz)
        two_beta_annealing_kwargs_main = resolve_two_beta_annealing_kwargs(
            main_proposal,
            cfg.viz,
        )
        proposal_kwargs_main = {
            **two_beta_kwargs_main,
            **two_beta_annealing_kwargs_main,
        }
        if proposal_kwargs_main:
            dual_kwargs = {}
        is_main_anneal = (
            main_proposal.lower()
            in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}
            or bool(two_beta_annealing_kwargs_main)
        )
        anneal_kwargs_main: Optional[dict] = None
        if is_main_anneal:
            beta_init_viz = float(getattr(cfg.viz, "dl_beta_init", base_dl_beta))
            beta_final_viz = float(getattr(cfg.viz, "dl_beta_final", beta_init_viz))
            try:
                beta_anneal_steps_viz = max(int(getattr(cfg.viz, "dl_beta_anneal_steps", steps)), 0)
            except (TypeError, ValueError):
                beta_anneal_steps_viz = 0
            anneal_kwargs_main = dict(
                dl_beta_init=beta_init_viz,
                dl_beta_final=beta_final_viz,
                dl_beta_anneal_steps=beta_anneal_steps_viz,
            )
            base_dl_beta = beta_final_viz
            dual_kwargs = {}

        current_idx = 0
        _record_state(current_idx)

        if record_warmup_phase:
            warm_proposal = chain_warmup_viz.proposal
            warm_simple = chain_warmup_viz.simple_n_edits
            warm_dual = chain_warmup_viz.dual_kwargs
            for _ in range(1, warmup_steps_total + 1):
                if N_noise > 0:
                    noise_nodes, noise_edges = _mcmc_step(
                        noise_nodes,
                        noise_edges,
                        proposal_name=warm_proposal,
                        gwd_beta_value=chain_warmup_viz.gwd_beta,
                        dl_beta_value=chain_warmup_viz.dl_beta,
                        dl_lambda_X_value=chain_warmup_viz.dl_lambda_X,
                        dl_lambda_E_value=chain_warmup_viz.dl_lambda_E,
                        simple_edits=warm_simple,
                        extra_dual_kwargs=warm_dual,
                    )
                if N_data > 0:
                    data_nodes, data_edges = _mcmc_step(
                        data_nodes,
                        data_edges,
                        proposal_name=warm_proposal,
                        gwd_beta_value=chain_warmup_viz.gwd_beta,
                        dl_beta_value=chain_warmup_viz.dl_beta,
                        dl_lambda_X_value=chain_warmup_viz.dl_lambda_X,
                        dl_lambda_E_value=chain_warmup_viz.dl_lambda_E,
                        simple_edits=warm_simple,
                        extra_dual_kwargs=warm_dual,
                    )
                current_idx += 1
                _record_state(current_idx)

        main_gwd_beta = float(getattr(cfg.viz, "gwd_beta", 1.0))
        for step_idx in range(steps):
            step_offset_val = step_idx if is_main_anneal else 0
            if N_noise > 0:
                noise_nodes, noise_edges = _mcmc_step(
                    noise_nodes,
                    noise_edges,
                    proposal_name=main_proposal,
                    gwd_beta_value=main_gwd_beta,
                    dl_beta_value=base_dl_beta,
                    dl_lambda_X_value=base_dl_lambda_X,
                    dl_lambda_E_value=base_dl_lambda_E,
                    extra_dual_kwargs=dual_kwargs,
                    anneal_kwargs=anneal_kwargs_main,
                    proposal_kwargs=proposal_kwargs_main,
                    step_offset=step_offset_val,
                )
            if N_data > 0:
                data_nodes, data_edges = _mcmc_step(
                    data_nodes,
                    data_edges,
                    proposal_name=main_proposal,
                    gwd_beta_value=main_gwd_beta,
                    dl_beta_value=base_dl_beta,
                    dl_lambda_X_value=base_dl_lambda_X,
                    dl_lambda_E_value=base_dl_lambda_E,
                    extra_dual_kwargs=dual_kwargs,
                    anneal_kwargs=anneal_kwargs_main,
                    proposal_kwargs=proposal_kwargs_main,
                    step_offset=step_offset_val,
                )
            current_idx += 1
            _record_state(current_idx)

        iter_suffix = f"_it{int(iteration):06d}" if iteration is not None else ""
        energy_png_base = getattr(cfg.viz, "filename_png", "energy_viz.png")
        metrics_png_base = getattr(cfg.viz, "filename_metrics_png", "energy_metrics.png")
        csv_base = getattr(cfg.viz, "filename_csv", "energy_viz.csv")
        def _with_suffix(path, suffix):
            path = str(path)
            if not suffix:
                return path
            stem, dot, ext = path.partition('.')
            if dot:
                return f"{stem}{suffix}.{ext}"
            return f"{path}{suffix}"
        energy_png_path = _with_suffix(energy_png_base, iter_suffix)
        metrics_png_path = _with_suffix(metrics_png_base, iter_suffix)
        csv_path = _with_suffix(csv_base, iter_suffix)
        save_energy_png = bool(getattr(cfg.viz, "save_png", True))
        save_metrics_png = bool(getattr(cfg.viz, "save_metrics_png", save_energy_png))
        save_csv = bool(getattr(cfg.viz, "save_csv", True))

        xs = np.arange(history_len)

        fig_energy = ax_energy = None
        if not metrics_only:
            fig_energy, ax_energy = plt.subplots(1, 1, figsize=(9, 5))

        fig_metrics = None
        ax_noise = ax_data = None
        if plot_metrics:
            metric_rows = int(N_noise > 0) + int(N_data > 0)
            if metric_rows > 0:
                fig_metrics, axes_metrics = plt.subplots(
                    metric_rows,
                    1,
                    figsize=(9, 4 + 2 * metric_rows) if not metrics_only else (9, 3 + 2 * metric_rows),
                    sharex=True,
                )
                axes_metrics = np.atleast_1d(axes_metrics)
                idx = 0
                if N_noise > 0:
                    ax_noise = axes_metrics[idx]
                    idx += 1
                if N_data > 0:
                    ax_data = axes_metrics[idx]

        if ax_energy is not None:
            if record_warmup_phase:
                ax_energy.axvspan(
                    0,
                    warmup_steps_total,
                    color="gray",
                    alpha=0.08,
                    label="warmup phase",
                    zorder=0.5,
                )
            if data_mean is not None and data_std is not None:
                ax_energy.axhspan(
                    data_mean - data_std,
                    data_mean + data_std,
                    color="gray",
                    alpha=0.25,
                    label=f"data band (M={M_data})",
                    zorder=1,
                )
            ax_energy.axhspan(
                noise_band_mean - noise_band_std,
                noise_band_mean + noise_band_std,
                color="tab:orange",
                alpha=0.15,
                label=f"noise band (M={M_noise})",
                zorder=1,
            )
            if data_mean is not None:
                ax_energy.axhline(
                    data_mean,
                    color="dimgray",
                    linestyle="--",
                    linewidth=band_line_width,
                    label="data band mid",
                    zorder=5.5,
                )
            ax_energy.axhline(
                noise_band_mean,
                color="darkorange",
                linestyle="--",
                linewidth=band_line_width,
                label="noise band mid",
                zorder=5.5,
            )
            ax_energy.plot(
                xs,
                noise_mean,
                color="tab:red",
                linewidth=mean_line_width,
                label=f"samples mean (noise init, N={N_noise})",
                zorder=6,
            )
            ax_energy.fill_between(
                xs,
                noise_mean - noise_std,
                noise_mean + noise_std,
                color="tab:red",
                alpha=0.2,
                zorder=2,
            )
            if N_data > 0:
                ax_energy.plot(
                    xs,
                    data_start_mean,
                    color="tab:blue",
                    linewidth=mean_line_width,
                    label=f"samples mean (data init, N={N_data})",
                    zorder=6,
                )
                ax_energy.fill_between(
                    xs,
                    data_start_mean - data_start_std,
                    data_start_mean + data_start_std,
                    color="tab:blue",
                    alpha=0.2,
                    zorder=2,
                )
            ax_energy.set_xlabel("Chain steps")
            ax_energy.set_ylabel("Energy (unconditional prior)")
            ax_energy.set_title("GEM-EBM unconditional energy trajectories vs. data/noise bands")
            ax_energy.legend(loc="best", fontsize=legend_fontsize)

        if plot_metrics and fig_metrics is not None:
            try:
                y_marks = list(getattr(cfg.viz, "metric_y_lines", [0.95, 0.90]))
            except Exception:
                y_marks = [0.95, 0.90]

            def _plot_metric_panel(ax, metrics, title: str):
                ax.plot(xs, metrics["validity"], color="tab:green", label=validity_label_display)
                ax.plot(xs, metrics["uniqueness"], color="tab:purple", label="uniqueness")
                ax.plot(xs, metrics["novelty"], color="tab:brown", label="novelty")
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Metric")
                ax.set_title(title)
                for y_val in y_marks:
                    try:
                        y_float = float(y_val)
                    except Exception:
                        continue
                    ax.axhline(y_float, color="gray", linestyle="--", linewidth=1, alpha=0.6)
                    ax.text(
                        0.995,
                        y_float,
                        f"{y_float:.2f}",
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="bottom",
                        color="gray",
                        fontsize=8,
                    )
                ax.legend(loc="best", fontsize=legend_fontsize)

            if ax_noise is not None:
                _plot_metric_panel(ax_noise, metric_groups["noise"], "Noise-initialized metrics")
            if ax_data is not None:
                _plot_metric_panel(ax_data, metric_groups["data"], "Data-initialized metrics")
                ax_data.set_xlabel("Chain steps")
            fig_metrics.tight_layout()
            if save_metrics_png:
                fig_metrics.savefig(metrics_png_path, dpi=150)
                print(f"[info] Saved metrics figure to {metrics_png_path}")
            plt.close(fig_metrics)

        if fig_energy is not None:
            fig_energy.tight_layout()
            if save_energy_png:
                fig_energy.savefig(energy_png_path, dpi=150)
                print(f"[info] Saved figure to {energy_png_path}")
            plt.close(fig_energy)

        if property_enabled and save_property_png and property_png_path:
            fig_property, ax_property = plt.subplots(1, 1, figsize=(9, 4.5))
            ax_property.plot(xs, property_noise, color="tab:red", label="noise-init")
            ax_property.plot(xs, property_data, color="tab:blue", label="data-init")
            if not np.isnan(property_target_scalar):
                ax_property.axhline(
                    property_target_scalar,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"target ({property_target_scalar:.2f})",
                )
            ax_property.set_xlabel("Chain steps")
            ax_property.set_ylabel(f"{conditioner.property_name}")
            ax_property.set_title(f"{conditioner.property_name.upper()} trajectory vs target")
            ax_property.legend(loc="best", fontsize=legend_fontsize)
            ax_property.grid(alpha=0.2)
            fig_property.tight_layout()
            property_png_path_str = str(property_png_path)
            fig_property.savefig(property_png_path_str, dpi=150)
            print(f"[info] Saved property figure to {property_png_path_str}")
            plt.close(fig_property)

        if (
            property_enabled
            and save_property_hist_png
            and property_hist_init is not None
            and property_hist_final is not None
            and property_hist_png_path
        ):
            hist_cols = [
                ("noise", "Noise-init", "tab:red"),
                ("data", "Data-init", "tab:blue"),
            ]
            row_specs = [
                ("Step 0", property_hist_init),
                (f"Step {warmup_steps_total + steps}", property_hist_final),
            ]
            all_hist_arrays = []
            for _, hist_dict in row_specs:
                for key, _, _ in hist_cols:
                    arr = hist_dict.get(key, np.asarray([]))
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        all_hist_arrays.append(arr)
            if property_hist_history:
                for frame_data in property_hist_history:
                    for key in ("noise", "data"):
                        arr = frame_data.get(key, np.asarray([]))
                        if isinstance(arr, np.ndarray) and arr.size > 0:
                            all_hist_arrays.append(arr)
            if all_hist_arrays:
                global_min = min(arr.min() for arr in all_hist_arrays if arr.size > 0)
                global_max = max(arr.max() for arr in all_hist_arrays if arr.size > 0)
                if not np.isfinite(global_min) or not np.isfinite(global_max):
                    global_min, global_max = -1.0, 1.0
                if global_min == global_max:
                    global_max = global_min + 1e-3
                hist_bins = np.linspace(global_min, global_max, 41)
            else:
                hist_bins = np.linspace(-1.0, 1.0, 41)

            max_hist_height = 0.0
            hist_inputs = []
            for row_label, hist_dict in row_specs:
                row_data = {}
                for key, _, _ in hist_cols:
                    values = hist_dict.get(key, np.asarray([]))
                    values = values if isinstance(values, np.ndarray) else np.asarray(values)
                    if values.size > 0:
                        weights = np.ones_like(values) / values.size
                        counts, _ = np.histogram(values, bins=hist_bins, weights=weights)
                        if counts.size > 0:
                            max_hist_height = max(max_hist_height, float(counts.max()))
                    else:
                        weights = None
                    row_data[key] = (values, weights)
                hist_inputs.append((row_label, row_data))

            if max_hist_height <= 0.0:
                max_hist_height = 1.0

            n_hist_rows = len(hist_inputs)
            n_hist_cols = len(hist_cols)
            fig_hist_width = 4.0 * n_hist_cols + 1.0
            fig_hist_height = 3.0 * n_hist_rows
            fig_hist, axes_hist = plt.subplots(
                n_hist_rows,
                n_hist_cols,
                figsize=(fig_hist_width, fig_hist_height),
                sharex=True,
                sharey=True,
            )
            if n_hist_rows == 1:
                axes_hist = np.expand_dims(axes_hist, axis=0)
            if n_hist_cols == 1:
                axes_hist = np.expand_dims(axes_hist, axis=1)
            for row_idx, (row_label, row_data) in enumerate(hist_inputs):
                for col_idx, (key, title, color) in enumerate(hist_cols):
                    ax = axes_hist[row_idx, col_idx]
                    values, weights = row_data.get(key, (np.asarray([]), None))
                    if isinstance(values, np.ndarray) and values.size > 0 and weights is not None:
                        ax.hist(
                            values,
                            bins=hist_bins,
                            weights=weights,
                            color=color,
                            alpha=0.7,
                            edgecolor="black",
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No samples",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="gray",
                        )
                    if not np.isnan(property_target_scalar):
                        ax.axvline(
                            property_target_scalar,
                            color="black",
                            linestyle="--",
                            linewidth=1.2,
                            label="target",
                        )
                    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
                    ax.set_ylim(0.0, max_hist_height * 1.05)
                    if col_idx == 0:
                        ax.set_ylabel(f"{row_label}\nFrequency")
                    else:
                        ax.set_ylabel("")
                    if row_idx == len(hist_inputs) - 1:
                        ax.set_xlabel(f"{conditioner.property_name}")
                    if row_idx == 0:
                        ax.set_title(title)
            fig_hist.suptitle(f"{conditioner.property_name.upper()} distribution at MCMC endpoints", fontsize=14)
            fig_hist.tight_layout(rect=[0, 0, 1, 0.95])
            hist_path_str = str(property_hist_png_path)
            fig_hist.savefig(hist_path_str, dpi=150)
            print(f"[info] Saved property histogram figure to {hist_path_str}")
            plt.close(fig_hist)

        if (
            property_enabled
            and make_property_hist_animation
            and property_hist_history
            and property_hist_animation_path
        ):
            hist_cols = [
                ("noise", "Noise-init", "tab:red"),
                ("data", "Data-init", "tab:blue"),
            ]
            all_hist_arrays = []
            for frame_data in property_hist_history:
                for key in ("noise", "data"):
                    arr = frame_data.get(key, np.asarray([]))
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        all_hist_arrays.append(arr)
            if not all_hist_arrays:
                print("[warn] Skipping property histogram animation: no property samples collected.")
            else:
                global_min = min(arr.min() for arr in all_hist_arrays if arr.size > 0)
                global_max = max(arr.max() for arr in all_hist_arrays if arr.size > 0)
                if not np.isfinite(global_min) or not np.isfinite(global_max):
                    global_min, global_max = -1.0, 1.0
                if global_min == global_max:
                    global_max = global_min + 1e-3
                hist_bins = np.linspace(global_min, global_max, 41)
                bin_lefts = hist_bins[:-1]
                bin_widths = np.diff(hist_bins)

                counts_history = []
                max_hist_height = 0.0
                for frame_data in property_hist_history:
                    step_idx = int(frame_data.get("step", 0))
                    frame_counts = {}
                    for key in ("noise", "data"):
                        values = frame_data.get(key, np.asarray([]))
                        if isinstance(values, np.ndarray) and values.size > 0:
                            weights = np.ones_like(values) / values.size
                            counts, _ = np.histogram(values, bins=hist_bins, weights=weights)
                        else:
                            counts = np.zeros_like(bin_lefts)
                        if counts.size > 0:
                            max_hist_height = max(max_hist_height, float(counts.max()))
                        frame_counts[key] = counts
                    counts_history.append((step_idx, frame_counts))
                if max_hist_height <= 0.0:
                    max_hist_height = 1.0
                counts_history.sort(key=lambda item: item[0])

                if len(counts_history) == 0:
                    print("[warn] Skipping property histogram animation: no histogram history recorded.")
                else:
                    fps_target = max(property_hist_animation_fps, 1.0)
                    steps_per_sec = max(property_hist_animation_steps_per_sec, 1.0)
                    interval_ms = max(1.0, float(property_hist_animation_interval_ms))
                    if len(counts_history) == 1:
                        frame_indices = np.array([0], dtype=int)
                    else:
                        total_steps_recorded = counts_history[-1][0]
                        if total_steps_recorded <= 0:
                            frame_indices = np.arange(len(counts_history))
                        else:
                            duration_seconds = total_steps_recorded / steps_per_sec
                            desired_frame_count = int(np.ceil(duration_seconds * fps_target))
                            desired_frame_count = max(2, desired_frame_count)
                            desired_frame_count = min(desired_frame_count, len(counts_history))
                            frame_indices = np.linspace(
                                0,
                                len(counts_history) - 1,
                                desired_frame_count,
                                dtype=int,
                            )
                            frame_indices = np.unique(
                                np.concatenate(([0], frame_indices, [len(counts_history) - 1]))
                            )
                    selected_counts = [counts_history[idx] for idx in frame_indices]

                    fig_width = 4.0 * len(hist_cols) + 1.0
                    fig_anim, axes_anim = plt.subplots(
                        1,
                        len(hist_cols),
                        figsize=(fig_width, 3.6),
                        sharey=True,
                    )
                    if isinstance(axes_anim, np.ndarray):
                        axes_list = axes_anim.flatten().tolist()
                    else:
                        axes_list = [axes_anim]
                    bar_containers = {}
                    for idx, (ax, (key, title, color)) in enumerate(zip(axes_list, hist_cols)):
                        initial_counts = selected_counts[0][1].get(key, np.zeros_like(bin_lefts))
                        bars = ax.bar(
                            bin_lefts,
                            initial_counts,
                            align="edge",
                            width=bin_widths,
                            color=color,
                            alpha=0.7,
                            edgecolor="black",
                        )
                        bar_containers[key] = bars
                        if not np.isnan(property_target_scalar):
                            ax.axvline(
                                property_target_scalar,
                                color="black",
                                linestyle="--",
                                linewidth=1.2,
                                label="target",
                            )
                        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
                        ax.set_xlim(hist_bins[0], hist_bins[-1])
                        ax.set_ylim(0.0, max_hist_height * 1.05)
                        if idx == 0:
                            ax.set_ylabel("Frequency")
                        else:
                            ax.set_ylabel("")
                        ax.set_xlabel(f"{conditioner.property_name}")
                        ax.set_title(title)

                    step_text = fig_anim.suptitle("", fontsize=12)

                    def _update_hist(frame_idx: int):
                        step_value, counts_dict = selected_counts[frame_idx]
                        updated_artists = []
                        for key, bars in bar_containers.items():
                            counts = counts_dict.get(key, np.zeros_like(bin_lefts))
                            for bar, height in zip(bars, counts):
                                bar.set_height(float(height))
                                updated_artists.append(bar)
                        step_text.set_text(f"MCMC step {step_value}")
                        updated_artists.append(step_text)
                        return updated_artists

                    ani = animation.FuncAnimation(
                        fig_anim,
                        _update_hist,
                        frames=len(selected_counts),
                        interval=interval_ms,
                        repeat=False,
                        blit=False,
                    )
                    animation_path = Path(property_hist_animation_path)
                    animation_format = animation_path.suffix.lstrip(".").lower()
                    if not animation_format:
                        animation_format = "gif"
                    fps = max(1, int(round(fps_target)))
                    try:
                        if animation_format in {"gif"}:
                            ani.save(
                                str(animation_path),
                                writer=animation.PillowWriter(fps=fps),
                                dpi=property_hist_animation_dpi,
                            )
                        else:
                            ani.save(
                                str(animation_path),
                                writer=animation_format,
                                dpi=property_hist_animation_dpi,
                                fps=fps,
                            )
                        print(f"[info] Saved property histogram animation to {animation_path}")
                    except Exception as exc:
                        print(f"[warn] Failed to save property histogram animation ({animation_format}): {exc}")
                    plt.close(fig_anim)

        if save_csv:
            import csv

            with open(csv_path, "w", newline="") as fp:
                writer = csv.writer(fp)
                validity_csv_key = validity_label_primary
                header = [
                    "step",
                    "noise_mean",
                    "noise_std",
                    "data_mean",
                    "data_std",
                    "data_band_mean",
                    "data_band_std",
                    validity_csv_key,
                    "uniqueness",
                    "novelty",
                    f"{validity_csv_key}_noise",
                    "uniqueness_noise",
                    "novelty_noise",
                    f"{validity_csv_key}_data",
                    "uniqueness_data",
                    "novelty_data",
                ]
                if property_enabled:
                    header.extend(
                        [
                            "property_target",
                            "property_noise_mean",
                            "property_data_mean",
                        ]
                    )
                writer.writerow(header)
                for i in range(history_len):
                    if plot_metrics:
                        combined_metrics = metric_groups["combined"]
                        noise_metrics = metric_groups["noise"]
                        data_metrics = metric_groups["data"]
                        v_i = float(combined_metrics["validity"][i])
                        u_i = float(combined_metrics["uniqueness"][i])
                        n_i = float(combined_metrics["novelty"][i])
                        v_noise_i = float(noise_metrics["validity"][i])
                        u_noise_i = float(noise_metrics["uniqueness"][i])
                        n_noise_i = float(noise_metrics["novelty"][i])
                        v_data_i = float(data_metrics["validity"][i])
                        u_data_i = float(data_metrics["uniqueness"][i])
                        n_data_i = float(data_metrics["novelty"][i])
                    else:
                        v_i = float("nan")
                        u_i = float("nan")
                        n_i = float("nan")
                        v_noise_i = float("nan")
                        u_noise_i = float("nan")
                        n_noise_i = float("nan")
                        v_data_i = float("nan")
                        u_data_i = float("nan")
                        n_data_i = float("nan")
                    row = [
                        int(i),
                        float(noise_mean[i]),
                        float(noise_std[i]),
                        float(data_start_mean[i]),
                        float(data_start_std[i]),
                        float(data_mean),
                        float(data_std),
                        v_i,
                        u_i,
                        n_i,
                        v_noise_i,
                        u_noise_i,
                        n_noise_i,
                        v_data_i,
                        u_data_i,
                        n_data_i,
                    ]
                    if property_enabled:
                        row.extend(
                            [
                                property_target_scalar,
                                float(property_noise[i]),
                                float(property_data[i]),
                            ]
                        )
                    writer.writerow(row)
            print(f"[info] Saved CSV to {csv_path}")
    finally:
        register_property_conditioner(None)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="gem_ebm_fm_moses_ver3",
)
def main(cfg: DictConfig):
    run_viz(cfg)


if __name__ == "__main__":
    main()
