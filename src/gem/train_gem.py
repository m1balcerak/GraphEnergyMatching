import os
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import utils
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

    data = datamodule.train_dataset[0]
    dense, node_mask = utils.to_dense(
        data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long)
    )
    n = node_mask.sum().item()
    node_types = torch.argmax(dense.X[0, :n], dim=-1)
    edge_types = torch.argmax(dense.E[0, :n, :n], dim=-1)

    sampled_nodes, sampled_edges = sampler.mcmc_sample(
        model, dataset_infos, node_types, edge_types, steps=10, device=torch.device("cpu")
    )

    molecules = [(sampled_nodes, sampled_edges)]
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
