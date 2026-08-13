from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch


@dataclass
class ProposalResult:
    prop_nodes: List[torch.Tensor]
    prop_edges: List[torch.Tensor]
    # Optional fields for asymmetric proposals
    log_q_fwd: Optional[torch.Tensor] = None  # (B,)
    moves: Optional[List[Tuple]] = None       # per-graph description of move


class Proposal:
    """Abstract proposal interface."""

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
        raise NotImplementedError

    def on_step_start(self, step_idx: int) -> None:
        """
        Optional hook invoked at the start of each MCMC step.

        Implementations can use this to update internal schedules (e.g., temperature annealing).
        """
        return None

    def needs_proposed_energy(self) -> bool:
        """Whether mcmc driver should compute E' (prop_E) before accept()."""
        return True

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        prop_result: ProposalResult,
        current_E: torch.Tensor,                 # (B,)
        prop_E: Optional[torch.Tensor],          # (B,) or None if proposal will compute it
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (accept_mask on CPU, prop_E on device or None)."""
        raise NotImplementedError
