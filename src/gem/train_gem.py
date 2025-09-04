# train_gem.py, do not delete this line
import os
import sys
import time
from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from datasets import qm9_dataset
from metrics.molecular_metrics import SamplingMolecularMetrics
from models.transformer_model import GraphTransformer
from models.extra_features import ExtraFeatures
from models.extra_features_molecular import ExtraMolecularFeatures
from . import sampler
import utils


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="gem")
def main(cfg: DictConfig):
    """Train the GEM energy model with *iteration-based*, parallel CD-k."""
    pl.seed_everything(cfg.train.seed)

    # Data & dataset info
    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    dataset_smiles = qm9_dataset.get_smiles(
        cfg=cfg,
        datamodule=datamodule,
        dataset_infos=dataset_infos,
        evaluate_datasets=False,
    )

    # Features & dimensions
    extra_features = ExtraFeatures(
        cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos
    )
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    # Metrics & references
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, dataset_smiles, cfg)
    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )
    print("Reference metrics:", dataset_infos.ref_metrics)

    # Device & precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    # Model
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=dataset_infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=dataset_infos.output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    ).to(device)

    # Optional compile (guarded)
    if getattr(cfg.train, "torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=getattr(cfg.train, "compile_mode", "max-autotune"))
            print("[info] torch.compile enabled.")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}. Continuing without compile.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # AMP setup
    amp_dtype_str = (cfg.train.amp_dtype or "").lower() if getattr(cfg.train, "amp_dtype", None) else ""
    use_cuda = (device.type == "cuda")
    amp_ctx = nullcontext()
    scaler = None

    if use_cuda and amp_dtype_str in {"fp16", "float16"}:
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        print("[info] AMP fp16 enabled.")
    elif use_cuda and amp_dtype_str in {"bf16", "bfloat16"}:
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        print("[info] AMP bf16 enabled.")
    else:
        print("[info] AMP disabled.")

    # Data loader iterator (iteration-based training)
    train_loader = datamodule.train_dataloader()
    train_iter = iter(train_loader)

    model.train()

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(datamodule.train_dataloader())
            return next(train_iter)

    # ----------------------------- Training loop -----------------------------
    max_iters = int(cfg.train.max_iters)
    for it in range(max_iters):
        t0 = time.perf_counter()

        batch = _next_batch()

        # Convert to dense and split into graphs
        dense_data, node_mask = utils.to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        graphs = dense_data.mask(node_mask, collapse=True).split(node_mask)

        # Extract typed node/edge tensors per graph (on CPU; moved to device inside sampler)
        node_list = [g.X.long().cpu() for g in graphs]
        edge_list = [g.E.long().cpu() for g in graphs]
        B = len(node_list)

        # ----------------- Positive energies (with gradients) -----------------
        _sync_if_cuda(device)
        t_pos0 = time.perf_counter()
        with amp_ctx:
            pos_E = sampler.energy_batch(
                model=model,
                node_types_list=node_list,
                edge_types_list=edge_list,
                dataset_info=dataset_infos,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=False,
            )  # (B,)
        _sync_if_cuda(device)
        t_pos1 = time.perf_counter()

        # -------------------- Negative phase: batched CD-k --------------------
        _sync_if_cuda(device)
        t_mcmc0 = time.perf_counter()
        # Prepare initial states mixing data and random graphs according to gamma_train.
        init_nodes = [t.clone() for t in node_list]
        init_edges = [t.clone() for t in edge_list]
        gamma_train = float(getattr(cfg.train, "gamma_train", 0.0))
        n_rand = int(round(B * gamma_train))
        if n_rand > 0:
            rand_graphs = sampler.initialize_random_graphs(
                batch_size=n_rand,
                dataset_info=dataset_infos,
                device=device,
                transition=cfg.model.transition,
            )
            rand_nodes = [nt for (nt, _) in rand_graphs]
            rand_edges = [et for (_, et) in rand_graphs]
            replace_idx = torch.randperm(B)[:n_rand]
            for i, idx in enumerate(replace_idx):
                init_nodes[idx] = rand_nodes[i]
                init_edges[idx] = rand_edges[i]

        # NOTE: the sampler internally enables grad when needed (GWD).
        with torch.no_grad():
            neg_nodes, neg_edges, n_accepts, n_steps_total = sampler.mcmc_sample_batch(
                model=model,
                dataset_info=dataset_infos,
                node_types_list=init_nodes,
                edge_types_list=init_edges,
                extra_features=extra_features,
                domain_features=domain_features,
                steps=cfg.train.cd_steps,
                device=device,
                proposal=str(getattr(cfg.train, "proposal", "random")),
                gwd_tau=float(getattr(cfg.train, "gwd_tau", 1.0)),
            )
        _sync_if_cuda(device)
        t_mcmc1 = time.perf_counter()

        # ----------------- Negative energies (with gradients) -----------------
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
            loss = (pos_E - neg_E).mean()
        _sync_if_cuda(device)
        t_neg1 = time.perf_counter()

        # ------------------------------- Update -------------------------------
        _sync_if_cuda(device)
        t_back0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _sync_if_cuda(device)
        t_back1 = time.perf_counter()

        # ------------------------------- Logging ------------------------------
        t1 = time.perf_counter()

        acc_rate = (n_accepts / max(n_steps_total, 1)) if n_steps_total > 0 else 0.0
        iter_time = t1 - t0
        t_pos = t_pos1 - t_pos0
        t_mcmc = t_mcmc1 - t_mcmc0
        t_neg = t_neg1 - t_neg0
        t_back = t_back1 - t_back0
        throughput = (B / iter_time) if iter_time > 0 else float("inf")

        if (it + 1) % int(cfg.train.log_interval) == 0:
            print(
                f"[it {it+1}/{max_iters}] "
                f"loss={loss.item():.4f} | acc={acc_rate*100:.1f}% "
                f"| batch={B} | t_iter={iter_time:.3f}s "
                f"(pos={t_pos:.3f}s mcmc={t_mcmc:.3f}s neg={t_neg:.3f}s back={t_back:.3f}s) "
                f"| throughput={throughput:.1f} graphs/s"
            )

    # -------------------------------- Evaluate -------------------------------
    model.eval()

    eval_bs = int(getattr(cfg.sample, "eval_batch_size", cfg.train.batch_size))

    # Start from data graphs and replace a fraction with random graphs according to gamma_evaluate.
    init_nodes: List[torch.Tensor] = []
    init_edges: List[torch.Tensor] = []
    while len(init_nodes) < eval_bs:
        batch = _next_batch()
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

    # NOTE: the sampler internally enables grad when needed (GWD)
    with torch.no_grad():
        final_nodes, final_edges, eval_accepts, eval_steps_total = sampler.mcmc_sample_batch(
            model=model,
            dataset_info=dataset_infos,
            node_types_list=init_nodes,
            edge_types_list=init_edges,
            extra_features=extra_features,
            domain_features=domain_features,
            steps=cfg.sample.sample_steps,
            device=device,
            proposal=str(getattr(cfg.sample, "proposal", "random")),
            gwd_tau=float(getattr(cfg.sample, "gwd_tau", 1.0)),
        )

    eval_acc_rate = (eval_accepts / max(eval_steps_total, 1)) if eval_steps_total > 0 else 0.0
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

if __name__ == "__main__":
    main()
