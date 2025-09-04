# proposals/gwd_proposal.py, do not delete this line
from typing import List, Sequence, Tuple
import torch

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch


def _gwd_build_candidates(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    grad_X: torch.Tensor,
    grad_E: torch.Tensor,
    num_node_types: int,
    num_edge_types: int,
    tau: float,
) -> Tuple[List[Tuple], torch.Tensor]:
    """Enumerate all single-edit moves and their logits."""
    device = grad_X.device
    n = int(node_types.shape[0])
    moves: List[Tuple] = []
    logits_list: List[torch.Tensor] = []

    # Node edits
    for i in range(n):
        old = int(node_types[i].item())
        for new_t in range(num_node_types):
            if new_t == old:
                continue
            dE_approx = grad_X[i, new_t] - grad_X[i, old]
            logit = (-dE_approx / float(tau))
            moves.append(("node", i, new_t, old))
            logits_list.append(logit)

    # Edge edits (undirected: i<j)
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                old = int(edge_types[i, j].item())
                for new_t in range(num_edge_types):
                    if new_t == old:
                        continue
                    dE_approx = (grad_E[i, j, new_t] - grad_E[i, j, old]) + \
                                (grad_E[j, i, new_t] - grad_E[j, i, old])
                    logit = (-dE_approx / float(tau))
                    moves.append(("edge", i, j, new_t, old))
                    logits_list.append(logit)

    if len(logits_list) == 0:
        return moves, torch.empty((0,), device=device)

    logits = torch.stack(logits_list, dim=0).to(torch.float32)
    return moves, logits


def _gwd_apply_move(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    move: Tuple,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return new (node_types, edge_types) after applying a single move."""
    kind = move[0]
    node_types_new = node_types.clone()
    edge_types_new = edge_types.clone()
    if kind == "node":
        _, i, new_t, _old = move
        node_types_new[i] = int(new_t)
    else:
        _, i, j, new_t, _old = move
        edge_types_new[i, j] = int(new_t)
        edge_types_new[j, i] = int(new_t)
    return node_types_new, edge_types_new


def _gwd_log_prob_of_move(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    grad_X: torch.Tensor,
    grad_E: torch.Tensor,
    num_node_types: int,
    num_edge_types: int,
    tau: float,
    target_move: Tuple,
) -> torch.Tensor:
    """Compute log q(move | state) by rebuilding all candidate logits and extracting the target."""
    moves, logits = _gwd_build_candidates(node_types, edge_types, grad_X, grad_E, num_node_types, num_edge_types, tau)
    if len(moves) == 0:
        return torch.tensor(0.0, device=grad_X.device)
    log_probs = logits - torch.logsumexp(logits, dim=0)
    for idx, mv in enumerate(moves):
        if mv == target_move:
            return log_probs[idx]
    raise RuntimeError("Target move not found among candidate moves for GWD.")


class GWDProposal(Proposal):
    """Gradient-weighted single-edit proposal (asymmetric)."""

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

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
        B = len(node_types_list)

        # Gradients at current states (re-enable grad explicitly)
        with torch.enable_grad():
            _, grad_X_list, grad_E_list = energy_and_grads_batch(
                model, node_types_list, edge_types_list, dataset_info, device, extra_features, domain_features
            )

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        moves: List[Tuple] = []
        log_q_fwd_list: List[torch.Tensor] = []

        for b in range(B):
            nt = node_types_list[b]
            et = edge_types_list[b]
            gX = grad_X_list[b]
            gE = grad_E_list[b]

            mv_candidates, logits = _gwd_build_candidates(nt, et, gX, gE, num_node_types, num_edge_types, self.tau)
            if len(mv_candidates) == 0:
                # Fallback: no-op
                prop_nodes.append(nt.clone())
                prop_edges.append(et.clone())
                moves.append(("node", 0, int(nt[0].item()), int(nt[0].item())))  # dummy
                log_q_fwd_list.append(torch.tensor(0.0, device=device))
                continue

            probs = torch.softmax(logits, dim=0)
            sel = torch.multinomial(probs, num_samples=1).item()
            move = mv_candidates[sel]
            log_q_fwd = (logits[sel] - torch.logsumexp(logits, dim=0))

            new_nt, new_et = _gwd_apply_move(nt, et, move)

            prop_nodes.append(new_nt)
            prop_edges.append(new_et)
            moves.append(move)
            log_q_fwd_list.append(log_q_fwd.detach())

        log_q_fwd = torch.stack(log_q_fwd_list, dim=0) if len(log_q_fwd_list) > 0 else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd, moves=moves)

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes,
        current_edges,
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: torch.Tensor,
        extra_features,
        domain_features,
        device: torch.device,
    ) -> torch.Tensor:
        """Exact MH with asymmetric proposal q (needs reverse log-probs at x')."""
        assert prop_result.moves is not None and prop_result.log_q_fwd is not None
        B = prop_E.shape[0]
        num_node_types = dataset_info.output_dims["X"]
        num_edge_types = dataset_info.output_dims["E"]

        # Compute reverse probabilities at proposed states
        with torch.enable_grad():
            _, grad_X_prop_list, grad_E_prop_list = energy_and_grads_batch(
                model,
                prop_result.prop_nodes,
                prop_result.prop_edges,
                dataset_info,
                device,
                extra_features,
                domain_features,
            )

        log_q_rev_list: List[torch.Tensor] = []
        for b in range(B):
            mv = prop_result.moves[b]
            # Reverse move: same site, type goes back to previous
            if mv[0] == "node":
                _, i, _new_t, old_t = mv
                rev_move = ("node", i, int(old_t), int(prop_result.prop_nodes[b][i].item()))
            else:
                _, i, j, _new_t, old_t = mv
                rev_move = ("edge", i, j, int(old_t), int(prop_result.prop_edges[b][i, j].item()))
            log_q_rev_b = _gwd_log_prob_of_move(
                node_types=prop_result.prop_nodes[b],
                edge_types=prop_result.prop_edges[b],
                grad_X=grad_X_prop_list[b],
                grad_E=grad_E_prop_list[b],
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                tau=self.tau,
                target_move=rev_move,
            )
            log_q_rev_list.append(log_q_rev_b.detach())

        log_q_rev = torch.stack(log_q_rev_list, dim=0)
        log_q_fwd = prop_result.log_q_fwd.to(prop_E.device)

        # MH: log u < -(E' - E) + log q_rev - log q_fwd
        log_u = torch.log(torch.rand(B, device=prop_E.device))
        mh_term = (-(prop_E - current_E)) + (log_q_rev - log_q_fwd)
        accept_mask = (log_u < mh_term).cpu()
        return accept_mask
