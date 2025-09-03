import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F


def _energy(
    model,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    dataset_info,
    device: torch.device,
) -> float:
    """Compute scalar energy of a graph using the transformer model."""
    n = node_types.shape[0]
    max_n = dataset_info.max_n_nodes
    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]

    X = F.one_hot(node_types, num_classes=num_node_types).float()
    E = F.one_hot(edge_types, num_classes=num_edge_types).float()

    X_pad = torch.zeros((1, max_n, num_node_types), device=device)
    E_pad = torch.zeros((1, max_n, max_n, num_edge_types), device=device)
    X_pad[0, :n] = X
    E_pad[0, :n, :n] = E

    y = torch.zeros((1, dataset_info.input_dims["y"]), device=device)
    node_mask = torch.zeros((1, max_n), device=device)
    node_mask[0, :n] = 1

    _, energy = model(X_pad, E_pad, y, node_mask, return_energy=True)
    return energy.item()


def _local_proposal(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    num_node_types: int,
    num_edge_types: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propose a uniformly random single edit on nodes or edges."""
    node_types = node_types.clone()
    edge_types = edge_types.clone()
    n = node_types.size(0)

    if random.random() < 0.5:
        idx = random.randrange(n)
        node_types[idx] = random.randrange(num_node_types)
    else:
        i = random.randrange(n)
        j = random.randrange(n)
        while j == i:
            j = random.randrange(n)
        val = random.randrange(num_edge_types)
        edge_types[i, j] = val
        edge_types[j, i] = val
    return node_types, edge_types


def mcmc_sample(
    model,
    dataset_info,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    steps: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """Run a simple MCMC chain starting from the provided graph."""
    node_types = node_types.to(device)
    edge_types = edge_types.to(device)
    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]

    current_energy = _energy(model, node_types, edge_types, dataset_info, device)
    for _ in range(steps):
        prop_node, prop_edge = _local_proposal(
            node_types, edge_types, num_node_types, num_edge_types
        )
        prop_energy = _energy(model, prop_node, prop_edge, dataset_info, device)
        if math.log(random.random()) < current_energy - prop_energy:
            node_types, edge_types = prop_node, prop_edge
            current_energy = prop_energy

    return node_types.cpu(), edge_types.cpu()
