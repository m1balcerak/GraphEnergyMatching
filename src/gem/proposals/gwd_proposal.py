from typing import List, Sequence, Tuple, Optional, Dict
from contextlib import nullcontext, contextmanager
import torch
import torch.nn.functional as F

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch


def _amp_autocast_ctx(device: torch.device, amp_dtype: Optional[str] = None):
    """Autocast context controlled by caller; no-op if amp_dtype is None."""
    if device.type != "cuda":
        return nullcontext()
    if not amp_dtype:
        return nullcontext()
    amp = amp_dtype.lower()
    if amp in {"bf16", "bfloat16"}:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if amp in {"fp16", "float16"}:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@contextmanager
def _model_eval_ctx(model: torch.nn.Module):
    """
    Temporarily set the model to eval() and restore the previous training mode.
    Used to guarantee dropout is disabled during energy/grad computations.
    """
    prev = model.training
    try:
        if prev:
            model.eval()
        yield
    finally:
        if prev:
            model.train()


def _node_logits(nt: torch.Tensor, gX: torch.Tensor, beta: float) -> torch.Tensor:
    """Return (n, X_types) logits for node edits, masked at current types.

    Uses inverse temperature beta = 1/tau.
    """
    n, X = gX.shape
    old = nt.to(gX.device).view(-1)  # (n,)
    dE = gX - gX.gather(1, old[:, None])  # (n, X) 
    logits = (-(float(beta)) * dE).to(torch.float32)
    logits.scatter_(1, old[:, None], float("-inf"))
    return logits  # (n, X)


def _edge_logits(et: torch.Tensor, gE: torch.Tensor, beta: float, pairs: Optional[torch.Tensor] = None):
    """Return ((P, E_types) logits, pairs indices) for undirected i<j edits.

    Uses inverse temperature beta = 1/tau.
    """
    n = int(et.size(0))
    E = int(gE.size(-1))
    if n <= 1:
        pairs = torch.empty((2, 0), dtype=torch.long, device=gE.device)
        return torch.empty((0, E), device=gE.device, dtype=torch.float32), pairs

    if pairs is None:
        pairs = torch.triu_indices(n, n, 1, device=gE.device)  # (2, P)
    i, j = pairs[0], pairs[1]
    # Symmetrize only for the sampled pairs to avoid full (n,n,E) materialization
    gE_pairs = 0.5 * (gE[i, j, :] + gE[j, i, :])    # (P, E)
    old_e = et[i, j]                                # (P,)
    dE = gE_pairs - gE_pairs.gather(1, old_e[:, None])  # (P, E)
    logits = (-(float(beta)) * dE).to(torch.float32)
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
    beta: float,
    target_move: Tuple,
    pairs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute log q(move | state) via vectorized logits."""
    device = gX.device
    n = int(nt.shape[0])
    X_types = gX.shape[1]
    E_types = gE.shape[2]

    node_logits = _node_logits(nt, gX, beta) if n > 0 else torch.empty((0, X_types), device=device, dtype=torch.float32)
    edge_logits, _pairs = _edge_logits(et, gE, beta, pairs)

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

    def __init__(self, beta: float = 1.0):
        # Inverse temperature beta = 1 / tau
        self.beta = float(beta)
        # Per-chain gradient cache: i -> (gX, gE) at the current state of chain i
        self._grad_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache of upper-triangular index tensors keyed by (device, n)
        self._pair_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def _pairs_for(self, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, int(device.index or -1), int(n))
        pairs = self._pair_cache.get(key)
        if pairs is None:
            pairs = torch.triu_indices(n, n, 1, device=device)
            self._pair_cache[key] = pairs
        return pairs

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
        amp_dtype: Optional[str] = None,
    ) -> ProposalResult:
        """
        Propose single-edit moves using gradient-weighted logits.

        Performance improvements:
        - Chain states and outputs remain on `device`.
        - Reuse cached per-chain gradients; compute grads only for chains with no cache
          (i.e., first step, or after an acceptance that changed the state).
        """
        num_node_types = dataset_info.output_dims["X"]
        num_edge_types = dataset_info.output_dims["E"]
        B = len(node_types_list)

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        moves: List[Tuple] = []
        log_q_fwd_list: List[torch.Tensor] = []

        # Determine which chains need fresh gradients
        need_idx = [i for i in range(B) if i not in self._grad_cache]

        # Compute grads only for missing entries
        if len(need_idx) > 0:
            sub_nodes = [node_types_list[i] for i in need_idx]
            sub_edges = [edge_types_list[i] for i in need_idx]
            with torch.enable_grad():
                # Disable dropout during gradient computation for MCMC
                with _model_eval_ctx(model):
                    with _amp_autocast_ctx(device, amp_dtype):
                        _, gx_sub, ge_sub = energy_and_grads_batch(
                            model, sub_nodes, sub_edges, dataset_info, device, extra_features, domain_features
                        )
            # Populate cache with freshly computed grads
            for k, i in enumerate(need_idx):
                self._grad_cache[i] = (gx_sub[k], ge_sub[k])

        # Assemble full grad lists in batch order from cache
        grad_X_list = [self._grad_cache[i][0] for i in range(B)]
        grad_E_list = [self._grad_cache[i][1] for i in range(B)]

        for b in range(B):
            # States are device-resident already in sampler; avoid redundant copies
            nt = node_types_list[b]
            et = edge_types_list[b]
            gX = grad_X_list[b]  # (n, X)
            gE = grad_E_list[b]  # (n, n, E)
            n = int(nt.shape[0])

            node_logits = _node_logits(nt, gX, self.beta) if n > 0 else torch.empty((0, num_node_types), device=device, dtype=torch.float32)
            pairs = self._pairs_for(n, device)
            edge_logits, pairs = _edge_logits(et, gE, self.beta, pairs)

            flat_nodes = node_logits.reshape(-1) if node_logits.numel() else node_logits
            flat_edges = edge_logits.reshape(-1) if edge_logits.numel() else edge_logits
            if flat_nodes.numel() == 0 and flat_edges.numel() == 0:
                # Fallback no-op
                prop_nodes.append(nt.clone().detach())
                prop_edges.append(et.clone().detach())
                mv = ("node", 0, int(nt[0].item()) if n > 0 else 0, int(nt[0].item()) if n > 0 else 0)
                moves.append(mv)
                log_q_fwd_list.append(torch.tensor(0.0, device=device))
                continue

            logits_all = flat_nodes if flat_edges.numel() == 0 else torch.cat([flat_nodes, flat_edges], dim=0)
            logits_all = logits_all.to(torch.float32)

            if (~torch.isfinite(logits_all)).all():
                # Fallback if every candidate is masked
                prop_nodes.append(nt.clone().detach())
                prop_edges.append(et.clone().detach())
                mv = ("node", 0, int(nt[0].item()) if n > 0 else 0, int(nt[0].item()) if n > 0 else 0)
                moves.append(mv)
                log_q_fwd_list.append(torch.tensor(0.0, device=device))
                continue

            # Sample via log_softmax + multinomial (fewer Python objects/kernels)
            logp_all = F.log_softmax(logits_all, dim=0)
            sel = torch.multinomial(logp_all.exp(), num_samples=1).squeeze(0)
            log_q_fwd = logp_all[sel]

            # Decode selection
            X_types = num_node_types
            E_types = num_edge_types
            total_node_cands = (n * X_types) if n > 0 else 0
            sel_i = int(sel.item())
            if sel_i < total_node_cands:
                i = sel_i // X_types
                new_t = sel_i % X_types
                old_t = int(nt[i].item())
                move = ("node", i, new_t, old_t)
                new_nt, new_et = _apply_move(nt, et, move)
            else:
                offset = sel_i - total_node_cands
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

            prop_nodes.append(new_nt.detach())
            prop_edges.append(new_et.detach())
            moves.append(move)
            log_q_fwd_list.append(log_q_fwd.detach())

        log_q_fwd = torch.stack(log_q_fwd_list, dim=0) if len(log_q_fwd_list) > 0 else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd, moves=moves)

    # ------------------------- Acceptance (with early reject & caching) -------------------------

    def needs_proposed_energy(self) -> bool:
        # We compute E' ourselves inside accept() (first forward-only for early reject, then fused for survivors).
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
        prop_E: Optional[torch.Tensor],  # ignored; we recompute
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Exact MH with asymmetric proposal q, without early-reject screening.
        Computes E' and reverse gradients for all proposals once, then performs
        the exact MH test and caches grads for accepted chains.
        """
        assert prop_result.moves is not None and prop_result.log_q_fwd is not None
        B = len(prop_result.prop_nodes)
        assert B == len(current_nodes) == len(current_edges) == int(current_E.shape[0])

        # Ensure everything resides on device
        log_q_fwd = prop_result.log_q_fwd.to(current_E.device, non_blocking=True).float()
        current_E_f32 = current_E.float()

        # Single pass: energies at x' and gradients for reverse q for ALL proposals
        with torch.enable_grad():
            with _model_eval_ctx(model):
                with _amp_autocast_ctx(device, amp_dtype):
                    prop_E_full, grad_X_prop_list, grad_E_prop_list = energy_and_grads_batch(
                        model=model,
                        node_types_list=prop_result.prop_nodes,
                        edge_types_list=prop_result.prop_edges,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                    )

        prop_E_full = prop_E_full.to(current_E.device, non_blocking=True).float()

        # Reverse log-probs for all proposals
        log_q_rev_list: List[torch.Tensor] = []
        for b, mv in enumerate(prop_result.moves):
            if mv[0] == "node":
                _, i, _new_t, old_t = mv
                rev_move = ("node", i, int(old_t), int(prop_result.prop_nodes[b][i].item()))
            else:
                _, i, j, _new_t, old_t = mv
                rev_move = ("edge", i, j, int(old_t), int(prop_result.prop_edges[b][i, j].item()))

            log_q_rev_b = _log_prob_of_move_vectorized(
                nt=prop_result.prop_nodes[b],
                et=prop_result.prop_edges[b],
                gX=grad_X_prop_list[b],
                gE=grad_E_prop_list[b],
                beta=self.beta,
                target_move=rev_move,
                pairs=self._pairs_for(int(prop_result.prop_nodes[b].shape[0]), device),
            )
            log_q_rev_list.append(log_q_rev_b.detach())

        log_q_rev = torch.stack(log_q_rev_list, dim=0).to(current_E.device, non_blocking=True).float()

        # Exact MH term for all proposals
        mh_term = (-(prop_E_full - current_E_f32)) + (log_q_rev - log_q_fwd)
        log_u = torch.log(torch.rand(B, device=current_E.device, dtype=torch.float32))
        accept_mask = (log_u < mh_term)

        # Cache grads at x' for ACCEPTED chains only
        if torch.any(accept_mask):
            acc_idx = torch.nonzero(accept_mask, as_tuple=False).flatten().tolist()
            for rel in acc_idx:
                self._grad_cache[int(rel)] = (grad_X_prop_list[rel].detach(), grad_E_prop_list[rel].detach())

        return accept_mask, prop_E_full.to(current_E.device, non_blocking=True)
