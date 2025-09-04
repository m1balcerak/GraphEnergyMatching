# proposals/gwd_proposal.py, do not delete this line
from typing import List, Sequence, Tuple, Optional
from contextlib import nullcontext
import torch

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch


def _amp_autocast_ctx(device: torch.device):
    """Autocast context for CUDA; no-op on CPU."""
    if device.type == "cuda":
        # Prefer bf16 if supported, else fp16
        use_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _node_logits(nt: torch.Tensor, gX: torch.Tensor, tau: float) -> torch.Tensor:
    """Return (n, X_types) logits for node edits, masked at current types."""
    n, X = gX.shape
    old = nt.to(gX.device).view(-1)  # (n,)
    dE = gX - gX.gather(1, old[:, None])  # (n, X)
    logits = (-dE / float(tau)).to(torch.float32)
    logits.scatter_(1, old[:, None], float("-inf"))
    return logits  # (n, X)


def _edge_logits(et: torch.Tensor, gE: torch.Tensor, tau: float):
    """Return ((P, E_types) logits, pairs indices) for undirected i<j edits."""
    n = et.size(0)
    E = gE.size(-1)
    if n <= 1:
        pairs = torch.empty((2, 0), dtype=torch.long, device=gE.device)
        return torch.empty((0, E), device=gE.device, dtype=torch.float32), pairs

    pairs = torch.triu_indices(n, n, 1, device=gE.device)  # (2, P)
    i, j = pairs[0], pairs[1]
    gE_sym = gE + gE.transpose(0, 1)        # (n, n, E)
    gE_pairs = gE_sym[i, j, :]              # (P, E)
    old_e = et.to(gE.device)[i, j]          # (P,)
    dE = gE_pairs - gE_pairs.gather(1, old_e[:, None])  # (P, E)
    logits = (-dE / float(tau)).to(torch.float32)
    logits.scatter_(1, old_e[:, None], float("-inf"))    # exclude current type
    return logits, pairs  # (P, E), (2, P)


def _apply_move(nt: torch.Tensor, et: torch.Tensor, move: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a single move to (nt, et)."""
    kind = move[0]
    nt2 = nt.clone()
    et2 = et.clone()
    if kind == "node":
        _, i, new_t, _old = move
        nt2[i] = int(new_t)
    else:
        _, i, j, new_t, _old = move
        et2[i, j] = int(new_t)
        et2[j, i] = int(new_t)
    return nt2, et2


def _pairs_index(i: int, j: int, n: int) -> int:
    """Index of pair (i,j) in the triu list with i<j."""
    return int(i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1))


def _log_prob_of_move_vectorized(
    nt: torch.Tensor,
    et: torch.Tensor,
    gX: torch.Tensor,
    gE: torch.Tensor,
    tau: float,
    target_move: Tuple,
) -> torch.Tensor:
    """Compute log q(move | state) via vectorized logits."""
    device = gX.device
    n = int(nt.shape[0])
    X_types = gX.shape[1]
    E_types = gE.shape[2]

    node_logits = _node_logits(nt, gX, tau) if n > 0 else torch.empty((0, X_types), device=device, dtype=torch.float32)
    edge_logits, _pairs = _edge_logits(et, gE, tau)

    flat_nodes = node_logits.reshape(-1) if node_logits.numel() else node_logits
    flat_edges = edge_logits.reshape(-1) if edge_logits.numel() else edge_logits
    logits_all = flat_nodes
    if flat_edges.numel():
        logits_all = torch.cat([logits_all, flat_edges], dim=0)

    if logits_all.numel() == 0 or (~torch.isfinite(logits_all)).all():
        # Degenerate: by convention return log(1)
        return torch.tensor(0.0, device=device)

    logZ = torch.logsumexp(logits_all, dim=0)

    if target_move[0] == "node":
        _, i, new_t, _old = target_move
        idx = int(i) * X_types + int(new_t)
    else:
        _, i, j, new_t, _old = target_move
        pair_idx = _pairs_index(int(i), int(j), n)
        idx = n * X_types + pair_idx * E_types + int(new_t)

    return (logits_all[idx].float() - logZ.float())


class GWDProposal(Proposal):
    """Gradient-weighted single-edit proposal (asymmetric)."""

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

    # ------------------------- Proposal (vectorized/GPU) -------------------------

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

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        moves: List[Tuple] = []
        log_q_fwd_list: List[torch.Tensor] = []

        with torch.enable_grad():
            with _amp_autocast_ctx(device):
                # Grads at current states (model runs in AMP; features stay FP32 inside energy_and_grads_batch)
                _, grad_X_list, grad_E_list = energy_and_grads_batch(
                    model, node_types_list, edge_types_list, dataset_info, device, extra_features, domain_features
                )

        for b in range(B):
            nt = node_types_list[b].to(device, non_blocking=True)
            et = edge_types_list[b].to(device, non_blocking=True)
            gX = grad_X_list[b]  # (n, X)
            gE = grad_E_list[b]  # (n, n, E)
            n = int(nt.shape[0])

            node_logits = _node_logits(nt, gX, self.tau) if n > 0 else torch.empty((0, num_node_types), device=device, dtype=torch.float32)
            edge_logits, pairs = _edge_logits(et, gE, self.tau)

            flat_nodes = node_logits.reshape(-1) if node_logits.numel() else node_logits
            flat_edges = edge_logits.reshape(-1) if edge_logits.numel() else edge_logits
            if flat_nodes.numel() == 0 and flat_edges.numel() == 0:
                # Fallback no-op
                prop_nodes.append(nt.cpu())
                prop_edges.append(et.cpu())
                mv = ("node", 0, int(nt[0].item()) if n > 0 else 0, int(nt[0].item()) if n > 0 else 0)
                moves.append(mv)
                log_q_fwd_list.append(torch.tensor(0.0, device=device))
                continue

            logits_all = flat_nodes if flat_edges.numel() == 0 else torch.cat([flat_nodes, flat_edges], dim=0)
            logits_all = logits_all.to(torch.float32)

            if (~torch.isfinite(logits_all)).all():
                # Fallback if every candidate is masked
                prop_nodes.append(nt.cpu())
                prop_edges.append(et.cpu())
                mv = ("node", 0, int(nt[0].item()) if n > 0 else 0, int(nt[0].item()) if n > 0 else 0)
                moves.append(mv)
                log_q_fwd_list.append(torch.tensor(0.0, device=device))
                continue

            logZ = torch.logsumexp(logits_all, dim=0)
            dist = torch.distributions.Categorical(logits=logits_all)
            sel = dist.sample()
            log_q_fwd = (logits_all[sel] - logZ)

            # Decode selection
            X_types = num_node_types
            E_types = num_edge_types
            total_node_cands = (n * X_types) if n > 0 else 0
            if int(sel.item()) < total_node_cands:
                i = int(sel.item()) // X_types
                new_t = int(sel.item()) % X_types
                old_t = int(nt[i].item())
                move = ("node", i, new_t, old_t)
                new_nt, new_et = _apply_move(nt, et, move)
            else:
                offset = int(sel.item()) - total_node_cands
                pair_idx = offset // E_types
                new_t = offset % E_types
                if pairs.numel() == 0:
                    # Defensive: no edge pairs, fallback to node no-op
                    mv = ("node", 0, int(nt[0].item()) if n > 0 else 0, int(nt[0].item()) if n > 0 else 0)
                    move = mv
                    new_nt, new_et = nt.clone(), et.clone()
                else:
                    i = int(pairs[0, pair_idx].item())
                    j = int(pairs[1, pair_idx].item())
                    old_t = int(et[i, j].item())
                    move = ("edge", i, j, int(new_t), old_t)
                    new_nt, new_et = _apply_move(nt, et, move)

            prop_nodes.append(new_nt.detach().cpu())
            prop_edges.append(new_et.detach().cpu())
            moves.append(move)
            log_q_fwd_list.append(log_q_fwd.detach())

        log_q_fwd = torch.stack(log_q_fwd_list, dim=0) if len(log_q_fwd_list) > 0 else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd, moves=moves)

    # ------------------------- Acceptance (fused E'+reverse ∇) -------------------------

    def needs_proposed_energy(self) -> bool:
        # We compute E' together with reverse gradients inside accept()
        return False

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes,
        current_edges,
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: Optional[torch.Tensor],  # ignored; we recompute fused
        extra_features,
        domain_features,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact MH with asymmetric proposal q, with fused E' + reverse grads."""
        assert prop_result.moves is not None and prop_result.log_q_fwd is not None
        B = len(prop_result.prop_nodes)

        with torch.enable_grad():
            with _amp_autocast_ctx(device):
                # Fused: energies at x' + gradients for reverse q
                prop_E_fused, grad_X_prop_list, grad_E_prop_list = energy_and_grads_batch(
                    model,
                    prop_result.prop_nodes,
                    prop_result.prop_edges,
                    dataset_info,
                    device,
                    extra_features,
                    domain_features,
                )

        # Reverse log-probs
        log_q_rev_list: List[torch.Tensor] = []
        for b in range(B):
            mv = prop_result.moves[b]
            if mv[0] == "node":
                _, i, _new_t, old_t = mv
                rev_move = ("node", i, int(old_t), int(prop_result.prop_nodes[b][i].item()))
            else:
                _, i, j, _new_t, old_t = mv
                rev_move = ("edge", i, j, int(old_t), int(prop_result.prop_edges[b][i, j].item()))

            log_q_rev_b = _log_prob_of_move_vectorized(
                nt=prop_result.prop_nodes[b].to(device, non_blocking=True),
                et=prop_result.prop_edges[b].to(device, non_blocking=True),
                gX=grad_X_prop_list[b],
                gE=grad_E_prop_list[b],
                tau=self.tau,
                target_move=rev_move,
            )
            log_q_rev_list.append(log_q_rev_b.detach())

        log_q_rev = torch.stack(log_q_rev_list, dim=0).to(current_E.device, non_blocking=True)
        log_q_fwd = prop_result.log_q_fwd.to(current_E.device, non_blocking=True)

        # MH: log u < -(E' - E) + log q_rev - log q_fwd  (all FP32)
        prop_E_f32 = prop_E_fused.to(current_E.device, non_blocking=True).float()
        current_E_f32 = current_E.float()
        mh_term = (-(prop_E_f32 - current_E_f32)) + (log_q_rev.float() - log_q_fwd.float())
        log_u = torch.log(torch.rand(B, device=current_E.device, dtype=torch.float32))
        accept_mask = (log_u < mh_term).cpu()

        return accept_mask, prop_E_fused.to(current_E.device, non_blocking=True)
