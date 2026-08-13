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
    """Propose a uniformly random single edit over all candidates.

    - Uniform over the union of node and edge edits (no 50/50 split).
    - Excludes no-ops by never proposing the current type.
    - If no valid edits exist (e.g., K=1 everywhere or n<2 for edges),
      returns the inputs unchanged.
    """
    node_types = node_types.clone()
    edge_types = edge_types.clone()
    n = node_types.size(0)

    if n == 0:
        return node_types, edge_types

    # Count candidates
    node_k = max(0, num_node_types - 1)
    edge_k = max(0, num_edge_types - 1)
    P = (n * (n - 1)) // 2 if n > 1 else 0
    node_cands = n * node_k
    edge_cands = P * edge_k
    total = node_cands + edge_cands

    if total == 0:
        # Nothing to change (only one type available everywhere)
        return node_types, edge_types

    flat_idx = random.randrange(total)

    # Node edit region [0, node_cands)
    if flat_idx < node_cands:
        local = flat_idx
        i = local // node_k
        type_idx = local % node_k  # in [0, node_k)
        cur = int(node_types[i].item())
        # Map type_idx to actual type in [0, num_node_types) \ {cur}
        new_t = type_idx if type_idx < cur else type_idx + 1
        node_types[i] = new_t
        return node_types, edge_types

    # Edge edit region [node_cands, total)
    local = flat_idx - node_cands
    # Decode pair index and type index
    if edge_cands == 0:
        return node_types, edge_types
    pair_idx = local // edge_k
    type_idx = local % edge_k

    # Build (i,j) list once per call (n is typically small)
    pairs: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i, j = pairs[pair_idx]
    cur = int(edge_types[i, j].item())
    new_t = type_idx if type_idx < cur else type_idx + 1
    edge_types[i, j] = edge_types[j, i] = new_t
    return node_types, edge_types


class RandomProposal(Proposal):
    """Symmetric uniform single-edit proposal (no-ops avoided, no 50/50).

    Samples uniformly from all valid node and edge edits (excluding current
    types), keeping the proposal symmetric without favoring node vs edge.
    """

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
        amp_dtype: Optional[str] = None,
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
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert prop_E is not None, "RandomProposal requires proposed energies."
        # Symmetric MH: π(x) ∝ exp(-E)
        B = prop_E.shape[0]
        log_u = torch.log(torch.rand(B, device=prop_E.device, dtype=torch.float32))
        accept_mask = (log_u < (current_E.float() - prop_E.float())).cpu()
        return accept_mask, prop_E
