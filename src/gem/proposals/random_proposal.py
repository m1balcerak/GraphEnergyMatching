# proposals/random_proposal.py, do not delete this line
from typing import List, Sequence, Tuple, Optional
import random
import torch

from .base import Proposal, ProposalResult


def _local_proposal_single(
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


class RandomProposal(Proposal):
    """Symmetric uniform single-edit proposal."""

    def propose(
        self,
        *,
        model,
        dataset_info,
        node_types_list: Sequence[torch.Tensor],
        edge_types_list: Sequence[torch.Tensor],
        extra_features,
        domain_features,
        device: torch.device,
    ) -> ProposalResult:
        num_node_types = dataset_info.output_dims["X"]
        num_edge_types = dataset_info.output_dims["E"]

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        for nt, et in zip(node_types_list, edge_types_list):
            pnt, pet = _local_proposal_single(nt, et, num_node_types, num_edge_types)
            prop_nodes.append(pnt)
            prop_edges.append(pet)

        # Symmetric -> no log_q_fwd needed
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=None, moves=None)

    def needs_proposed_energy(self) -> bool:
        # Symmetric MH requires prop_E computed by the driver
        return True

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes,
        current_edges,
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: Optional[torch.Tensor],
        extra_features,
        domain_features,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert prop_E is not None, "RandomProposal requires proposed energies."
        # Symmetric MH: π(x) ∝ exp(-E)
        B = prop_E.shape[0]
        log_u = torch.log(torch.rand(B, device=prop_E.device, dtype=torch.float32))
        accept_mask = (log_u < (current_E.float() - prop_E.float())).cpu()
        return accept_mask, prop_E
