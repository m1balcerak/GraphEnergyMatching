"""MOSES checkpoint sampling using the release sampler and built-in metadata."""

from pathlib import Path
import csv

import torch
from hydra import compose, initialize_config_dir

from gem import sampler
from gem.checkpoint_utils import load_model_checkpoint
from gem.datasets import moses_dataset
from gem.metrics_over_time import _molecule_records, _run_mcmc, _set_seed
from gem.models.extra_features import ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures
from gem.models.transformer_model import GraphTransformer


TRANSPORT_STEPS = 225
CSV_FIELDS = (
    "step", "index", "valid", "connected", "valid_connected",
    "num_fragments", "smiles", "error",
)


def load_sampling_config(config_dir: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        cfg = compose(config_name="gem_metrics_over_time_moses")
    cfg.release_sampler.chain_warmup.steps = TRANSPORT_STEPS
    cfg.metrics_run.compute_fcd = False
    cfg.dataset.compute_fcd = False
    return cfg


def build_sampling_model(cfg, device):
    # Inference needs only the shipped MOSES distributions, not processed data
    # or statistics files from the caller's working directory.
    infos = moses_dataset.MOSESinfos(None, cfg, load_cached_statistics=False)
    extra = ExtraFeatures(cfg.model.extra_features, cfg.model.rrwp_steps, infos)
    domain = ExtraMolecularFeatures(infos)

    # Derive feature widths on a deterministic two-carbon graph, using the same
    # feature functions as training. The +1 global feature is diffusion time;
    # GraphTransformer adds its own timestep embedding internally.
    atom_classes, bond_classes = infos.num_atom_types, len(infos.edge_types)
    x = torch.zeros(1, 2, atom_classes)
    x[..., infos.atom_encoder["C"]] = 1
    e = torch.zeros(1, 2, 2, bond_classes)
    e[0, 0, 1, 1] = e[0, 1, 0, 1] = 1
    example = dict(X_t=x, E_t=e, y_t=torch.zeros(1, 0),
                   node_mask=torch.ones(1, 2, dtype=torch.bool))
    extra_dims, domain_dims = extra(example), domain(example)
    infos.output_dims = {"X": atom_classes, "E": bond_classes, "y": 0}
    infos.input_dims = {
        key: base + getattr(extra_dims, key).shape[-1] + getattr(domain_dims, key).shape[-1]
        for key, base in {"X": atom_classes, "E": bond_classes, "y": 1}.items()
    }
    activation = torch.nn.SiLU() if cfg.model.activation == "silu" else torch.nn.ReLU()
    torch.set_float32_matmul_precision("medium")
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=infos.output_dims,
        act_fn_in=activation, act_fn_out=activation, tf_activation=activation,
    ).to(device).eval()
    return model, infos, extra, domain


def generate_batches(
    *, model, dataset_infos, extra_features, domain_features, device, run_cfg,
    num_samples, mixing_steps, batch_size,
):
    if num_samples <= 0 or batch_size <= 0 or mixing_steps < 0:
        raise ValueError("Sample count and batch size must be positive; mixing steps must be nonnegative.")
    for offset in range(0, num_samples, batch_size):
        graphs = sampler.initialize_random_graphs(
            batch_size=min(batch_size, num_samples - offset),
            dataset_info=dataset_infos, device=device, transition="uniform",
        )
        nodes, edges = [g[0] for g in graphs], [g[1] for g in graphs]
        nodes, edges, _, _, _ = sampler.run_simple_v2_warmup_vectorized(
            model=model, dataset_info=dataset_infos,
            node_types_list=nodes, edge_types_list=edges,
            extra_features=extra_features, domain_features=domain_features,
            steps=TRANSPORT_STEPS, device=device, edits_per_step=1,
            amp_dtype=None, stop_when_unchanged=True, collect_stats=False,
        )
        if mixing_steps:
            # Keep the calibrated annealing schedule; only the number of mixing
            # steps changes. Graph-input gradients remain enabled for sampling.
            nodes, edges, _, _, _ = _run_mcmc(
                model=model, dataset_infos=dataset_infos,
                node_types=nodes, edge_types=edges,
                extra_features=extra_features, domain_features=domain_features,
                device=device, run_cfg=run_cfg, steps=mixing_steps, step_offset=0,
            )
        yield _molecule_records(
            nodes, edges, dataset_infos, step=mixing_steps, sample_offset=offset,
        )


def sample_checkpoint(
    checkpoint: Path, output: Path, *, num_samples: int, mixing_steps: int,
    batch_size: int, seed: int, config_dir: Path,
):
    cfg = load_sampling_config(config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sampling on {device}: {TRANSPORT_STEPS} transport steps + {mixing_steps} mixing steps.", flush=True)
    model, infos, extra, domain = build_sampling_model(cfg, device)
    load_model_checkpoint(model, str(checkpoint), map_location=device, use_ema=False)
    _set_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Retain all attempts, including invalid molecules, and flush each batch so
    # completed batches remain available if generation is interrupted.
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        count = 0
        for records in generate_batches(
            model=model, dataset_infos=infos, extra_features=extra,
            domain_features=domain, device=device, run_cfg=cfg.metrics_run,
            num_samples=num_samples, mixing_steps=mixing_steps, batch_size=batch_size,
        ):
            writer.writerows({key: row.get(key) for key in CSV_FIELDS} for row in records)
            handle.flush()
            count += len(records)
            print(f"Generated {count}/{num_samples} samples.", flush=True)
    print(f"Saved samples to {output}", flush=True)
