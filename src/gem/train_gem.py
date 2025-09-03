import os
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import qm9_dataset
from metrics.molecular_metrics import SamplingMolecularMetrics
from models.transformer_model import GraphTransformer
from models.extra_features import ExtraFeatures
from models.extra_features_molecular import ExtraMolecularFeatures
from . import sampler
import utils


@hydra.main(version_base="1.3", config_path="../../configs", config_name="gem")
def main(cfg: DictConfig):
    """Train the GEM energy model with contrastive divergence."""
    pl.seed_everything(cfg.train.seed)

    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    dataset_smiles = qm9_dataset.get_smiles(
        cfg=cfg,
        datamodule=datamodule,
        dataset_infos=dataset_infos,
        evaluate_datasets=False,
    )

    extra_features = ExtraFeatures(
        cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos
    )
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, dataset_smiles, cfg)
    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )
    print("Reference metrics:", dataset_infos.ref_metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=dataset_infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=dataset_infos.output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    model.train()
    iteration = 0
    for epoch in range(cfg.train.n_epochs):
        total_accepts = 0
        total_steps = 0
        for batch in datamodule.train_dataloader():
            dense_data, node_mask = utils.to_dense(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            graphs = dense_data.mask(node_mask, collapse=True).split(node_mask)

            loss = 0.0
            batch_accepts = 0
            batch_steps = 0
            for graph in graphs:
                node_types = graph.X.to(device).long()
                edge_types = graph.E.to(device).long()

                pos_energy = sampler._energy(
                    model,
                    node_types,
                    edge_types,
                    dataset_infos,
                    device,
                    extra_features,
                    domain_features,
                    detach=False,
                )

                neg_nodes, neg_edges, n_acc, n_steps = sampler.mcmc_sample(
                    model,
                    dataset_infos,
                    node_types,
                    edge_types,
                    extra_features,
                    domain_features,
                    steps=cfg.train.cd_steps,
                    device=device,
                )
                total_accepts += n_acc
                total_steps += n_steps
                batch_accepts += n_acc
                batch_steps += n_steps

                neg_energy = sampler._energy(
                    model,
                    neg_nodes.to(device),
                    neg_edges.to(device),
                    dataset_infos,
                    device,
                    extra_features,
                    domain_features,
                    detach=False,
                )

                loss = loss + (pos_energy - neg_energy)

            loss = loss / max(len(graphs), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_rate = batch_accepts / max(batch_steps, 1)
            print(
                f"Iteration {iteration + 1} - Loss: {loss.item():.4f} - Acceptance: {acc_rate:.3f}"
            )
            iteration += 1

        epoch_acc_rate = total_accepts / max(total_steps, 1)
        print(
            f"Epoch {epoch + 1}/{cfg.train.n_epochs} - Loss: {loss.item():.4f} - Acceptance: {epoch_acc_rate:.3f}"
        )

    model.eval()

    init_graphs = sampler.initialize_random_graphs(
        batch_size=cfg.train.batch_size,
        dataset_info=dataset_infos,
        device=device,
        transition=cfg.model.transition,
    )

    molecules = []
    for node_types, edge_types in init_graphs:
        sampled_nodes, sampled_edges, _, _ = sampler.mcmc_sample(
            model,
            dataset_infos,
            node_types,
            edge_types,
            extra_features,
            domain_features,
            steps=cfg.sample.sample_steps,
            device=device,
        )
        molecules.append((sampled_nodes, sampled_edges))

    sampling_metrics(
        molecules=molecules,
        ref_metrics=dataset_infos.ref_metrics,
        name="GEM",
        current_epoch=0,
        val_counter=0,
        local_rank=0,
        test=True,
    )
    print("Reference metrics:", dataset_infos.ref_metrics)


if __name__ == "__main__":
    main()
