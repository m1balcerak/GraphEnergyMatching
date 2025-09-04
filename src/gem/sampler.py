# sampler.py, do not delete this line
from typing import List, Sequence, Tuple, Optional
import torch

from flow_matching.noise_distribution import NoiseDistribution
from flow_matching import flow_matching_utils

# Re-export energy utilities for external use
from .sampler_energy import (
    build_batched_inputs,
    energy_batch,
    energy_and_grads_batch,
)

# Proposals
from .proposals.base import Proposal
from .proposals.random_proposal import RandomProposal
from .proposals.gwd_proposal import GWDProposal


def make_proposal(method: str, **kwargs) -> Proposal:
    """Factory for proposal mechanisms."""
    m = (method or "").lower()
    if m == "random":
        return RandomProposal()
    if m == "gwd":
        tau = float(kwargs.get("gwd_tau", 1.0))
        return GWDProposal(tau=tau)
    raise ValueError(f"Unknown proposal '{method}'. Supported: ['random', 'gwd'].")


def initialize_random_graphs(
    batch_size: int,
    dataset_info,
    device: torch.device = torch.device("cpu"),
    transition: str = "marginal",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Sample a batch of random graphs used as MCMC initial states."""
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
        n = int(n_nodes[i].item())
        node_types = torch.argmax(z_T.X[i, :n], dim=-1)
        edge_types = torch.argmax(z_T.E[i, :n, :n], dim=-1)
        graphs.append((node_types.cpu(), edge_types.cpu()))
    return graphs


def mcmc_sample_batch(
    model,
    dataset_info,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    extra_features,
    domain_features,
    steps: int = 10,
    device: torch.device = torch.device("cpu"),
    proposal: str = "random",
    gwd_tau: float = 1.0,
):
    """Run parallel MCMC chains (one per graph) starting from the provided batch.

    Returns
    -------
    tuple
        (node_types_list, edge_types_list, n_accept_total, n_steps_total)
    """
    assert len(node_types_list) == len(edge_types_list)
    B = len(node_types_list)
    if B == 0:
        return [], [], 0, 0

    # Current energies for all graphs (detached for acceptance decisions)
    current_E = energy_batch(
        model=model,
        node_types_list=node_types_list,
        edge_types_list=edge_types_list,
        dataset_info=dataset_info,
        device=device,
        extra_features=extra_features,
        domain_features=domain_features,
        detach=True,
    )  # (B,)

    prop_impl: Proposal = make_proposal(proposal, gwd_tau=gwd_tau)
    total_accepts = 0

    for _ in range(steps):
        # === Propose ===
        prop_result = prop_impl.propose(
            model=model,
            dataset_info=dataset_info,
            node_types_list=node_types_list,
            edge_types_list=edge_types_list,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
        )

        # === Score proposals if needed by the proposal ===
        prop_E: Optional[torch.Tensor] = None
        if prop_impl.needs_proposed_energy():
            prop_E = energy_batch(
                model=model,
                node_types_list=prop_result.prop_nodes,
                edge_types_list=prop_result.prop_edges,
                dataset_info=dataset_info,
                device=device,
                extra_features=extra_features,
                domain_features=domain_features,
                detach=True,
            )  # (B,)

        # === Accept/Reject ===
        accept_mask, prop_E_from_accept = prop_impl.accept(
            model=model,
            dataset_info=dataset_info,
            current_nodes=node_types_list,
            current_edges=edge_types_list,
            prop_result=prop_result,
            current_E=current_E,
            prop_E=prop_E,
            extra_features=extra_features,
            domain_features=domain_features,
            device=device,
        )

        # Update states and energies where accepted
        n_acc = int(accept_mask.sum().item())
        total_accepts += n_acc
        if n_acc > 0:
            for i, acc in enumerate(accept_mask):
                if acc:
                    node_types_list[i] = prop_result.prop_nodes[i]
                    edge_types_list[i] = prop_result.prop_edges[i]

            # Prefer fused prop_E returned by accept() when available
            eff_prop_E = prop_E if prop_E is not None else prop_E_from_accept
            assert eff_prop_E is not None, "Proposal did not provide proposed energies."
            mask_dev = accept_mask.to(current_E.device)
            current_E[mask_dev] = eff_prop_E[mask_dev]

    return node_types_list, edge_types_list, total_accepts, steps * B
