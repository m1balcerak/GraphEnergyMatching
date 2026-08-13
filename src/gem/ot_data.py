"""Data utilities for OT diagnostics: dataset context, graph collection, and noise init."""

from __future__ import annotations

from typing import List, Tuple

import torch

from gem import utils
from omegaconf import DictConfig

from gem.datasets.dataset_context import build_dataset_context
from gem.models.extra_features import DummyExtraFeatures, ExtraFeatures
from gem.models.extra_features_molecular import ExtraMolecularFeatures
from gem.flow_matching.noise_distribution import NoiseDistribution
from gem.flow_matching import flow_matching_utils


def build_data_context(cfg: DictConfig):
    """Create datamodule, dataset_infos, and feature builders similar to train/eval."""
    datamodule, dataset_infos, dataset_smiles = build_dataset_context(cfg)

    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=dataset_infos,
    )
    domain_features = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if dataset_smiles is not None
        else DummyExtraFeatures()
    )

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    return datamodule, dataset_infos


def collect_graphs_from_data(datamodule, M: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Collect at least M graphs from the training loader as discrete label tensors.

    Returns two lists of length >= M:
      - node label tensors of shape (n,)
      - edge type tensors of shape (n, n)
    """
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


def train_node_count_distribution(datamodule) -> torch.Tensor:
    """Compute distribution of number of nodes per graph over the training dataset."""
    max_nodes = 0
    for batch in datamodule.train_dataloader():
        _, counts = torch.unique(batch.batch, return_counts=True)
        if counts.numel() > 0:
            max_nodes = max(max_nodes, int(counts.max().item()))
    if max_nodes == 0:
        return torch.tensor([1.0])

    counts_hist = torch.zeros(max_nodes + 1, dtype=torch.float32)
    total = 0.0
    for batch in datamodule.train_dataloader():
        _, counts = torch.unique(batch.batch, return_counts=True)
        for c in counts:
            n = int(c.item())
            counts_hist[n] += 1.0
            total += 1.0
    if total > 0:
        counts_hist = counts_hist / total
    return counts_hist


def initialize_random_graphs_with_counts(
    counts: List[int],
    dataset_info,
    device: torch.device = torch.device("cpu"),
    transition: str = "marginal",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Sample random graphs with exact node counts per graph using the noise distribution."""
    if not counts:
        return []
    B = len(counts)
    n_max = max(int(c) for c in counts)
    if n_max <= 0:
        return []

    n_nodes = torch.tensor(counts, device=device, dtype=torch.long)
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(B, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    noise_dist = NoiseDistribution(transition, dataset_info)
    limit_dist = noise_dist.get_limit_dist()
    z_T = flow_matching_utils.sample_discrete_feature_noise(
        limit_dist=limit_dist, node_mask=node_mask
    )

    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(B):
        n = int(counts[i])
        if n <= 0:
            graphs.append((torch.empty(0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)))
            continue
        node_types = torch.argmax(z_T.X[i, :n], dim=-1)
        edge_types = torch.argmax(z_T.E[i, :n, :n], dim=-1)
        graphs.append((node_types.cpu(), edge_types.cpu()))
    return graphs
