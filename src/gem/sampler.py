import math
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching import flow_matching_utils


def _energy(
    model,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
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

    y = torch.zeros((1, dataset_info.output_dims["y"]), device=device)
    node_mask = torch.zeros((1, max_n), device=device)
    node_mask[0, :n] = 1
    t = torch.zeros((1, 1), device=device)

    noisy_data = {
        "X_t": X_pad,
        "E_t": E_pad,
        "y_t": y,
        "node_mask": node_mask,
        "t": t,
    }

    extra_feat = extra_features(noisy_data)
    extra_mol_feat = domain_features(noisy_data)

    extra_X = torch.cat((extra_feat.X, extra_mol_feat.X), dim=-1)
    extra_E = torch.cat((extra_feat.E, extra_mol_feat.E), dim=-1)
    extra_y = torch.cat((extra_feat.y, extra_mol_feat.y), dim=-1)
    extra_y = torch.cat((extra_y, t), dim=1)

    X_input = torch.cat((X_pad, extra_X), dim=2)
    E_input = torch.cat((E_pad, extra_E), dim=3)
    y_input = torch.cat((y, extra_y), dim=1)

    if X_input.shape[-1] < dataset_info.input_dims["X"]:
        pad = dataset_info.input_dims["X"] - X_input.shape[-1]
        X_input = torch.cat(
            (X_input, torch.zeros(1, max_n, pad, device=device)), dim=-1
        )
    if E_input.shape[-1] < dataset_info.input_dims["E"]:
        pad = dataset_info.input_dims["E"] - E_input.shape[-1]
        E_input = torch.cat(
            (E_input, torch.zeros(1, max_n, max_n, pad, device=device)), dim=-1
        )
    if y_input.shape[-1] < dataset_info.input_dims["y"]:
        pad = dataset_info.input_dims["y"] - y_input.shape[-1]
        y_input = torch.cat((y_input, torch.zeros(1, pad, device=device)), dim=-1)

    _, energy = model(X_input, E_input, y_input, node_mask, return_energy=True)
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

    if n == 0:
        return node_types, edge_types

    if n > 1 and random.random() >= 0.5:
        i, j = random.sample(range(n), 2)
        val = random.randrange(num_edge_types)
        edge_types[i, j] = edge_types[j, i] = val
    else:
        idx = random.randrange(n)
        node_types[idx] = random.randrange(num_node_types)
    return node_types, edge_types


def initialize_random_graphs(
    batch_size: int,
    dataset_info,
    device: torch.device = torch.device("cpu"),
    transition: str = "marginal",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Sample a batch of random graphs used as MCMC initial states.

    The number of nodes is drawn from the empirical training-set distribution
    and node/edge types are sampled from the reference (noise) distribution
    used in DEFoG.
    """

    # Sample number of nodes from empirical distribution
    n_nodes = dataset_info.nodes_dist.sample_n(batch_size, device)
    n_max = torch.max(n_nodes).item()
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    # Sample node and edge types from reference noise distribution
    noise_dist = NoiseDistribution(transition, dataset_info)
    limit_dist = noise_dist.get_limit_dist()
    z_T = flow_matching_utils.sample_discrete_feature_noise(
        limit_dist=limit_dist, node_mask=node_mask
    )

    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(batch_size):
        n = n_nodes[i].item()
        node_types = torch.argmax(z_T.X[i, :n], dim=-1)
        edge_types = torch.argmax(z_T.E[i, :n, :n], dim=-1)
        graphs.append((node_types.cpu(), edge_types.cpu()))

    return graphs


def mcmc_sample(
    model,
    dataset_info,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    extra_features,
    domain_features,
    steps: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """Run a simple MCMC chain starting from the provided graph."""
    node_types = node_types.to(device)
    edge_types = edge_types.to(device)
    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]

    current_energy = _energy(
        model,
        node_types,
        edge_types,
        dataset_info,
        device,
        extra_features,
        domain_features,
    )
    for _ in range(steps):
        prop_node, prop_edge = _local_proposal(
            node_types, edge_types, num_node_types, num_edge_types
        )
        prop_energy = _energy(
            model,
            prop_node,
            prop_edge,
            dataset_info,
            device,
            extra_features,
            domain_features,
        )
        if math.log(random.random()) < current_energy - prop_energy:
            node_types, edge_types = prop_node, prop_edge
            current_energy = prop_energy

    return node_types.cpu(), edge_types.cpu()
