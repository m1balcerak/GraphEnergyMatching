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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Placeholder training loop for the GEM energy model."""
    pl.seed_everything(cfg.train.seed)

    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    dataset_smiles = qm9_dataset.get_smiles(
        cfg=cfg, datamodule=datamodule, dataset_infos=dataset_infos, evaluate_datasets=False
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

    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=dataset_infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=dataset_infos.output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )
    model.eval()
    device = torch.device("cpu")

    init_graphs = sampler.initialize_random_graphs(
        batch_size=cfg.train.batch_size,
        dataset_info=dataset_infos,
        device=device,
        transition=cfg.model.transition,
    )

    molecules = []
    for node_types, edge_types in init_graphs:
        sampled_nodes, sampled_edges = sampler.mcmc_sample(
            model,
            dataset_infos,
            node_types,
            edge_types,
            extra_features,
            domain_features,
            steps=10,
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
