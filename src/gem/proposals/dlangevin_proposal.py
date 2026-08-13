from typing import List, Sequence, Tuple, Optional, Dict
from contextlib import nullcontext, contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch


# ----------------------------- Local helper contexts -----------------------------

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
def _model_eval_ctx(model: nn.Module):
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


# ----------------------------- Misc helpers -----------------------------

_CHANGE_EPS = 1e-12
_TEMP_DECAY_FACTOR = 0.9
_LOG_CUM_STAY_THRESHOLD = -50.0  # contributions beyond this log-prob are negligible
_MAX_REVERSE_ESCALATION_STEPS = 256
_MAX_RESAMPLE_TRIES = 20

def _factorized_log_prob(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    stay_nt: torch.Tensor,
    stay_edge: torch.Tensor,
    target_nt: torch.Tensor,
    target_edge: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Sum log-prob of target labels and stay event under factorized categoricals, plus change flag.
    """
    log_target = torch.zeros((), device=device)
    log_stay = torch.zeros((), device=device)
    change_possible = False

    if node_logits.numel():
        node_log_probs = F.log_softmax(node_logits, dim=-1)
        if target_nt.numel():
            log_target = log_target + node_log_probs.gather(1, target_nt.view(-1, 1)).sum()
        if stay_nt.numel():
            stay_lp = node_log_probs.gather(1, stay_nt.view(-1, 1)).squeeze(1)
            log_stay = log_stay + stay_lp.sum()
            if stay_lp.numel():
                stay_prob = torch.exp(stay_lp)
                if torch.any(stay_prob < 1.0 - _CHANGE_EPS):
                    change_possible = True

    if edge_logits.numel():
        edge_log_probs = F.log_softmax(edge_logits, dim=-1)
        if target_edge.numel():
            log_target = log_target + edge_log_probs.gather(1, target_edge.view(-1, 1)).sum()
        if stay_edge.numel():
            stay_lp_e = edge_log_probs.gather(1, stay_edge.view(-1, 1)).squeeze(1)
            log_stay = log_stay + stay_lp_e.sum()
            if stay_lp_e.numel():
                stay_prob_e = torch.exp(stay_lp_e)
                if torch.any(stay_prob_e < 1.0 - _CHANGE_EPS):
                    change_possible = True

    return log_target, log_stay, change_possible


# ----------------------------- Logit builders (with 'stay') -----------------------------

def _node_logits_dlang(
    nt: torch.Tensor,            # (n,)
    gX: torch.Tensor,            # (n, X)
    beta: float,
    lambda_X: float,
) -> torch.Tensor:
    """
    Return per-node logits (n, X) with 'stay' enabled and a change penalty (lambda_X).
    Logits implement: -beta * (gX - gX[current]) - lambda_X * 1{c != current}, so current class has 0.
    """
    device = gX.device
    n, X = gX.shape
    if n == 0:
        return torch.empty((0, X), device=device, dtype=torch.float32)

    current = nt.to(device).view(-1, 1)  # (n,1)
    gx_cur = gX.gather(1, current)       # (n,1)
    base = (-float(beta)) * (gX - gx_cur)  # (n, X); base[:, current] == 0 by construction

    # Apply change penalty to all non-current classes, keep 'stay' at 0
    logits = base - float(lambda_X)
    logits.scatter_(1, current, torch.zeros((n, 1), device=device, dtype=logits.dtype))
    return logits.to(torch.float32)


def _edge_logits_dlang(
    et: torch.Tensor,            # (n, n)
    gE: torch.Tensor,            # (n, n, E)
    beta: float,
    lambda_E: float,
    pairs: Optional[torch.Tensor] = None,  # (2, P) precomputed i<j indices
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-pair logits (P, E) with 'stay' enabled and a change penalty (lambda_E),
    along with the (2, P) index tensor of pairs (i<j).
    """
    device = gE.device
    n = int(et.size(0))
    E = int(gE.size(-1))
    if n <= 1:
        pairs = torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.empty((0, E), device=device, dtype=torch.float32), pairs

    if pairs is None:
        pairs = torch.triu_indices(n, n, 1, device=device)  # (2, P)
    i, j = pairs[0], pairs[1]

    # Symmetrize only on the sampled pairs to avoid building full (n,n,E)
    g_pairs = 0.5 * (gE[i, j, :] + gE[j, i, :])    # (P, E)
    old_e = et[i, j].view(-1, 1)                   # (P,1)

    g_cur = g_pairs.gather(1, old_e)               # (P,1)
    base = (-float(beta)) * (g_pairs - g_cur)      # (P, E); base[:, old] == 0

    logits = base - float(lambda_E)
    logits.scatter_(1, old_e, torch.zeros((old_e.size(0), 1), device=device, dtype=logits.dtype))
    return logits.to(torch.float32), pairs


# ----------------------------- Proposal implementation -----------------------------

class DLangevinProposal(Proposal):
    """
    Discrete Langevin-like factorized proposal with 'stay' and per-domain change penalties.

    q(x' | x) factorizes across nodes and undirected edge pairs:
    logits_node  = -beta * (gX - gX[current])  - lambda_X * 1{c != current}
    logits_edge  = -beta * (gE_sym - gE_sym[current]) - lambda_E * 1{c != current}
    """
    supports_property_lambda_override = True

    def __init__(
        self,
        beta: float = 1.0,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        self.beta = float(beta)
        self.lambda_X = float(lambda_X)
        self.lambda_E = float(lambda_E)
        self.property_lambda_override: Optional[float] = None
        # Per-chain gradient cache: i -> (gX, gE) at the current state of chain i
        self._grad_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache of upper-triangular index tensors keyed by (device, n)
        self._pair_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def set_property_lambda_override(self, value: Optional[float]) -> None:
        self.property_lambda_override = None if value is None else float(value)

    def _mh_beta(self) -> float:
        """Inverse temperature of the distribution targeted by MH acceptance."""
        return float(self.beta)

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
        Build factorized logits with 'stay' for all coordinates, sample them independently,
        return proposed states and log q_fwd (sum of per-coordinate log-probabilities).
        """
        B = len(node_types_list)
        if B == 0:
            return ProposalResult(prop_nodes=[], prop_edges=[], log_q_fwd=torch.empty((0,), device=device), moves=None)

        # --- Ensure gradient cache is complete for current states ---
        need_idx = [i for i in range(B) if i not in self._grad_cache]
        if len(need_idx) > 0:
            sub_nodes = [node_types_list[i] for i in need_idx]
            sub_edges = [edge_types_list[i] for i in need_idx]
            with torch.enable_grad():
                with _model_eval_ctx(model):
                    with _amp_autocast_ctx(device, amp_dtype):
                        _, gx_sub, ge_sub = energy_and_grads_batch(
                            model,
                            sub_nodes,
                            sub_edges,
                            dataset_info,
                            device,
                            extra_features,
                            domain_features,
                            property_lambda_override=self.property_lambda_override,
                        )
            for k, i in enumerate(need_idx):
                self._grad_cache[i] = (gx_sub[k], ge_sub[k])

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        log_q_fwd_list: List[torch.Tensor] = []

        # --- Build proposals chain-by-chain (states kept on device) ---
        for b in range(B):
            # States are already device-resident from the sampler; avoid redundant .to()
            nt = node_types_list[b].long()
            et = edge_types_list[b].long()
            gX, gE = self._grad_cache[b]

            n = int(nt.shape[0])

            # Prepare edge helpers
            pairs = self._pairs_for(int(et.shape[0]), device)
            if pairs.numel():
                i_idx, j_idx = pairs[0], pairs[1]
                et_pairs_current = et[i_idx, j_idx]
            else:
                i_idx = j_idx = None
                et_pairs_current = torch.empty((0,), device=device, dtype=torch.long)

            nt_new = nt.clone()
            e_pairs_new = et_pairs_current.clone()
            log_q_fwd = torch.tensor(0.0, device=device, dtype=torch.float32)
            temp_beta = float(self.beta)
            temp_lambda_X = float(self.lambda_X)
            temp_lambda_E = float(self.lambda_E)
            log_cum_stay = torch.zeros((), device=device, dtype=torch.float64)

            tries = 0
            while True:
                node_logits = _node_logits_dlang(nt, gX, temp_beta, temp_lambda_X) if n > 0 else torch.empty(
                    (0, dataset_info.output_dims["X"]), device=device, dtype=torch.float32
                )
                if node_logits.numel():
                    node_log_probs = F.log_softmax(node_logits, dim=-1)
                    stay_logp_nodes = node_log_probs.gather(1, nt.view(-1, 1)).squeeze(1)
                    node_stay_logp = stay_logp_nodes.sum()
                    stay_prob_nodes = stay_logp_nodes.exp()
                    node_can_change = bool(torch.any(stay_prob_nodes < 1.0 - _CHANGE_EPS).item())
                    node_probs = node_log_probs.exp()
                else:
                    node_log_probs = None
                    node_probs = None
                    node_stay_logp = torch.tensor(0.0, device=device)
                    node_can_change = False

                edge_logits_stage, _ = _edge_logits_dlang(et, gE, temp_beta, temp_lambda_E, pairs)
                if edge_logits_stage.numel():
                    edge_log_probs = F.log_softmax(edge_logits_stage, dim=-1)
                    if et_pairs_current.numel():
                        stay_logp_edges = edge_log_probs.gather(1, et_pairs_current.view(-1, 1)).squeeze(1)
                        edge_stay_logp = stay_logp_edges.sum()
                        stay_prob_edges = stay_logp_edges.exp()
                        edge_can_change = bool(torch.any(stay_prob_edges < 1.0 - _CHANGE_EPS).item())
                    else:
                        edge_stay_logp = torch.tensor(0.0, device=device)
                        edge_can_change = False
                    edge_probs = edge_log_probs.exp()
                else:
                    edge_log_probs = None
                    edge_probs = None
                    edge_stay_logp = torch.tensor(0.0, device=device)
                    edge_can_change = False

                log_stay_total = (node_stay_logp + edge_stay_logp).double()
                change_possible_stage = node_can_change or edge_can_change

                if not change_possible_stage:
                    log_q_fwd = (log_cum_stay + log_stay_total).to(torch.float32)
                    break

                if node_probs is not None:
                    nt_candidate = torch.multinomial(node_probs, num_samples=1).squeeze(1)
                else:
                    nt_candidate = nt.clone()

                if edge_probs is not None:
                    e_pairs_candidate = torch.multinomial(edge_probs, num_samples=1).squeeze(1)
                else:
                    e_pairs_candidate = et_pairs_current.clone()

                nodes_changed = node_probs is not None and not torch.equal(nt_candidate, nt)
                edges_changed = edge_probs is not None and not torch.equal(e_pairs_candidate, et_pairs_current)

                if nodes_changed or edges_changed:
                    nt_new = nt_candidate
                    e_pairs_new = e_pairs_candidate
                    node_logp = (
                        node_log_probs.gather(1, nt_new.view(-1, 1)).sum()
                        if node_log_probs is not None
                        else torch.tensor(0.0, device=device)
                    )
                    edge_logp = (
                        edge_log_probs.gather(1, e_pairs_new.view(-1, 1)).sum()
                        if edge_log_probs is not None
                        else torch.tensor(0.0, device=device)
                    )
                    log_q_fwd = (log_cum_stay + node_logp.double() + edge_logp.double()).to(torch.float32)
                    break

                log_cum_stay = log_cum_stay + log_stay_total
                temp_beta *= _TEMP_DECAY_FACTOR
                temp_lambda_X *= _TEMP_DECAY_FACTOR
                temp_lambda_E *= _TEMP_DECAY_FACTOR
                tries += 1
                if tries >= _MAX_RESAMPLE_TRIES:
                    log_q_fwd = log_cum_stay.to(torch.float32)
                    break

            # Assemble new edge matrix
            et_new = et.clone()
            if pairs.numel():
                et_new[i_idx, j_idx] = e_pairs_new
                et_new[j_idx, i_idx] = e_pairs_new

            prop_nodes.append(nt_new.detach())
            prop_edges.append(et_new.detach())
            log_q_fwd_list.append(log_q_fwd.detach())

        log_q_fwd = torch.stack(log_q_fwd_list, dim=0) if log_q_fwd_list else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd, moves=None)

    def _log_transition_prob_escalated(
        self,
        nt_current: torch.Tensor,
        et_current: torch.Tensor,
        nt_target: torch.Tensor,
        et_target: torch.Tensor,
        grad_X: torch.Tensor,
        grad_E: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute log probability of transitioning from the current state to the target state
        under repeated resampling with temperature escalation.
        """
        pairs = self._pairs_for(int(et_current.shape[0]), device)
        if pairs.numel():
            stay_edge = et_current[pairs[0], pairs[1]]
            target_edge = et_target[pairs[0], pairs[1]]
        else:
            stay_edge = torch.empty((0,), device=device, dtype=torch.long)
            target_edge = torch.empty((0,), device=device, dtype=torch.long)

        temp_beta = float(self.beta)
        temp_lambda_X = float(self.lambda_X)
        temp_lambda_E = float(self.lambda_E)
        log_total = torch.full((), float("-inf"), device=device, dtype=torch.float64)
        log_cum_stay = torch.zeros((), device=device, dtype=torch.float64)

        for _ in range(_MAX_REVERSE_ESCALATION_STEPS):
            node_logits = (
                _node_logits_dlang(nt_current, grad_X, temp_beta, temp_lambda_X)
                if nt_current.numel()
                else torch.empty((0, grad_X.shape[1]), device=device, dtype=torch.float32)
            )
            edge_logits, _ = _edge_logits_dlang(et_current, grad_E, temp_beta, temp_lambda_E, pairs)

            log_target, log_stay, change_possible = _factorized_log_prob(
                node_logits,
                edge_logits,
                nt_current,
                stay_edge,
                nt_target,
                target_edge,
                device,
            )

            if not change_possible:
                break

            stage_log = log_cum_stay + log_target.double()
            if torch.isfinite(log_total):
                log_total = torch.logaddexp(log_total, stage_log)
            else:
                log_total = stage_log

            if torch.isneginf(log_stay):
                break

            log_cum_stay = log_cum_stay + log_stay.double()
            if log_cum_stay.item() <= _LOG_CUM_STAY_THRESHOLD:
                break

            temp_beta *= _TEMP_DECAY_FACTOR
            temp_lambda_X *= _TEMP_DECAY_FACTOR
            temp_lambda_E *= _TEMP_DECAY_FACTOR

        return log_total.to(torch.float32)

    def _log_transition_prob_many(
        self,
        *,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        target_nodes: Sequence[torch.Tensor],
        target_edges: Sequence[torch.Tensor],
        grad_X_list: Sequence[torch.Tensor],
        grad_E_list: Sequence[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        log_q_rev: List[torch.Tensor] = []
        for idx in range(len(current_nodes)):
            log_q_rev.append(
                self._log_transition_prob_escalated(
                    nt_current=current_nodes[idx],
                    et_current=current_edges[idx],
                    nt_target=target_nodes[idx],
                    et_target=target_edges[idx],
                    grad_X=grad_X_list[idx],
                    grad_E=grad_E_list[idx],
                    device=device,
                ).detach()
            )
        return torch.stack(log_q_rev, dim=0) if log_q_rev else torch.empty((0,), device=device)

    # ------------------------- Acceptance (with caching) -------------------------

    def needs_proposed_energy(self) -> bool:
        # We compute E' and reverse q inside accept()
        return False

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: Optional[torch.Tensor],  # ignored
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Exact MH with asymmetric factorized proposal q. Computes E' and reverse grads for all proposals,
        then performs MH and caches grads for accepted chains.
        """
        B = len(prop_result.prop_nodes)
        assert B == len(current_nodes) == len(current_edges) == int(current_E.shape[0])

        same_chain_flags: List[bool] = []
        eval_indices: List[int] = []
        for idx in range(B):
            same_nodes = torch.equal(current_nodes[idx], prop_result.prop_nodes[idx])
            same_edges = torch.equal(current_edges[idx], prop_result.prop_edges[idx])
            same = bool(same_nodes and same_edges)
            same_chain_flags.append(same)
            if not same:
                eval_indices.append(idx)

        device_E = current_E.device
        current_E_f32 = current_E.to(device_E, non_blocking=True).float()

        if len(eval_indices) == 0:
            accept_mask = torch.ones(B, dtype=torch.bool, device=device_E)
            return accept_mask, current_E_f32

        prop_nodes_eval = [prop_result.prop_nodes[i] for i in eval_indices]
        prop_edges_eval = [prop_result.prop_edges[i] for i in eval_indices]

        with torch.enable_grad():
            with _model_eval_ctx(model):
                with _amp_autocast_ctx(device, amp_dtype):
                    prop_E_eval, grad_X_eval_list, grad_E_eval_list = energy_and_grads_batch(
                        model=model,
                        node_types_list=prop_nodes_eval,
                        edge_types_list=prop_edges_eval,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        property_lambda_override=self.property_lambda_override,
                    )

        idx_tensor = torch.tensor(eval_indices, device=prop_result.log_q_fwd.device, dtype=torch.long)
        log_q_fwd_eval = prop_result.log_q_fwd.index_select(0, idx_tensor)

        log_q_rev_tensor = self._log_transition_prob_many(
            current_nodes=[prop_result.prop_nodes[i].long() for i in eval_indices],
            current_edges=[prop_result.prop_edges[i].long() for i in eval_indices],
            target_nodes=[current_nodes[i].long() for i in eval_indices],
            target_edges=[current_edges[i].long() for i in eval_indices],
            grad_X_list=grad_X_eval_list,
            grad_E_list=grad_E_eval_list,
            device=device,
        ).to(device_E, non_blocking=True).float()
        log_q_fwd_tensor = log_q_fwd_eval.to(device_E, non_blocking=True).float()
        prop_E_eval_f32 = prop_E_eval.to(device_E, non_blocking=True).float()
        current_E_eval = current_E_f32.index_select(0, idx_tensor.to(device_E))

        mh_term = (-(self._mh_beta()) * (prop_E_eval_f32 - current_E_eval)) + (
            log_q_rev_tensor - log_q_fwd_tensor
        )
        log_u = torch.log(torch.rand(len(eval_indices), device=device_E, dtype=torch.float32))
        accept_eval = (log_u < mh_term)

        accept_mask = torch.zeros(B, dtype=torch.bool, device=device_E)
        prop_E_full = current_E_f32.clone()
        prop_E_full.index_copy_(0, idx_tensor.to(device_E), prop_E_eval_f32)

        if any(same_chain_flags):
            same_indices = torch.tensor(
                [i for i, same in enumerate(same_chain_flags) if same],
                device=device_E,
                dtype=torch.long,
            )
            if same_indices.numel():
                accept_mask.index_fill_(0, same_indices, True)

        accept_mask.index_copy_(0, idx_tensor.to(device_E), accept_eval)

        if torch.any(accept_eval):
            acc_local = torch.nonzero(accept_eval, as_tuple=False).flatten().tolist()
            for rel in acc_local:
                global_idx = eval_indices[rel]
                self._grad_cache[int(global_idx)] = (
                    grad_X_eval_list[rel].detach(),
                    grad_E_eval_list[rel].detach(),
                )

        return accept_mask, prop_E_full


class DLangevinVectorizedProposal(DLangevinProposal):
    """
    DLangevin variant with batched reverse transition probabilities.

    The forward proposal is inherited from DLangevinProposal to keep the sampled
    proposal distribution unchanged while we isolate the reverse-logq bottleneck.
    """

    def _log_transition_prob_many(
        self,
        *,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        target_nodes: Sequence[torch.Tensor],
        target_edges: Sequence[torch.Tensor],
        grad_X_list: Sequence[torch.Tensor],
        grad_E_list: Sequence[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        B = len(current_nodes)
        if B == 0:
            return torch.empty((0,), device=device, dtype=torch.float32)

        by_size: Dict[int, List[int]] = {}
        for idx, nt in enumerate(current_nodes):
            by_size.setdefault(int(nt.shape[0]), []).append(idx)

        out = torch.empty((B,), device=device, dtype=torch.float32)
        for idxs in by_size.values():
            vals = self._log_transition_prob_same_size_batch(
                current_nodes=[current_nodes[i] for i in idxs],
                current_edges=[current_edges[i] for i in idxs],
                target_nodes=[target_nodes[i] for i in idxs],
                target_edges=[target_edges[i] for i in idxs],
                grad_X_list=[grad_X_list[i] for i in idxs],
                grad_E_list=[grad_E_list[i] for i in idxs],
                device=device,
            )
            out.index_copy_(0, torch.tensor(idxs, device=device, dtype=torch.long), vals)
        return out

    def _log_transition_prob_same_size_batch(
        self,
        *,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        target_nodes: Sequence[torch.Tensor],
        target_edges: Sequence[torch.Tensor],
        grad_X_list: Sequence[torch.Tensor],
        grad_E_list: Sequence[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        B = len(current_nodes)
        if B == 0:
            return torch.empty((0,), device=device, dtype=torch.float32)

        nt_current = torch.stack([t.to(device, non_blocking=True).long().view(-1) for t in current_nodes], dim=0)
        et_current = torch.stack([t.to(device, non_blocking=True).long() for t in current_edges], dim=0)
        nt_target = torch.stack([t.to(device, non_blocking=True).long().view(-1) for t in target_nodes], dim=0)
        et_target = torch.stack([t.to(device, non_blocking=True).long() for t in target_edges], dim=0)
        grad_X = torch.stack([g.to(device, non_blocking=True) for g in grad_X_list], dim=0)
        grad_E = torch.stack([g.to(device, non_blocking=True) for g in grad_E_list], dim=0)

        n = int(nt_current.shape[1])
        pairs = self._pairs_for(n, device)
        if pairs.numel():
            pair_i, pair_j = pairs[0], pairs[1]
            edge_current = et_current[:, pair_i, pair_j]
            edge_target = et_target[:, pair_i, pair_j]
        else:
            pair_i = pair_j = None
            edge_current = torch.empty((B, 0), device=device, dtype=torch.long)
            edge_target = torch.empty((B, 0), device=device, dtype=torch.long)

        log_total = torch.full((B,), float("-inf"), device=device, dtype=torch.float64)
        log_cum_stay = torch.zeros((B,), device=device, dtype=torch.float64)
        active = torch.ones((B,), device=device, dtype=torch.bool)

        temp_beta = float(self.beta)
        temp_lambda_X = float(self.lambda_X)
        temp_lambda_E = float(self.lambda_E)

        for _ in range(_MAX_REVERSE_ESCALATION_STEPS):
            if not bool(torch.any(active).item()):
                break

            log_target = torch.zeros((B,), device=device, dtype=torch.float32)
            log_stay = torch.zeros((B,), device=device, dtype=torch.float32)
            change_possible = torch.zeros((B,), device=device, dtype=torch.bool)

            if n > 0:
                cur_node_idx = nt_current.unsqueeze(-1)
                node_cur_grad = grad_X.gather(2, cur_node_idx)
                node_logits = (-temp_beta) * (grad_X - node_cur_grad) - temp_lambda_X
                node_logits.scatter_(2, cur_node_idx, 0.0)
                node_log_probs = F.log_softmax(node_logits.to(torch.float32), dim=-1)

                node_target_lp = node_log_probs.gather(2, nt_target.unsqueeze(-1)).squeeze(-1)
                node_stay_lp = node_log_probs.gather(2, cur_node_idx).squeeze(-1)
                log_target = log_target + node_target_lp.sum(dim=1)
                log_stay = log_stay + node_stay_lp.sum(dim=1)
                change_possible = change_possible | torch.any(
                    node_stay_lp.exp() < 1.0 - _CHANGE_EPS,
                    dim=1,
                )

            if pairs.numel():
                assert pair_i is not None and pair_j is not None
                edge_pair_grad = 0.5 * (grad_E[:, pair_i, pair_j, :] + grad_E[:, pair_j, pair_i, :])
                cur_edge_idx = edge_current.unsqueeze(-1)
                edge_cur_grad = edge_pair_grad.gather(2, cur_edge_idx)
                edge_logits = (-temp_beta) * (edge_pair_grad - edge_cur_grad) - temp_lambda_E
                edge_logits.scatter_(2, cur_edge_idx, 0.0)
                edge_log_probs = F.log_softmax(edge_logits.to(torch.float32), dim=-1)

                edge_target_lp = edge_log_probs.gather(2, edge_target.unsqueeze(-1)).squeeze(-1)
                edge_stay_lp = edge_log_probs.gather(2, cur_edge_idx).squeeze(-1)
                log_target = log_target + edge_target_lp.sum(dim=1)
                log_stay = log_stay + edge_stay_lp.sum(dim=1)
                change_possible = change_possible | torch.any(
                    edge_stay_lp.exp() < 1.0 - _CHANGE_EPS,
                    dim=1,
                )

            update_mask = active & change_possible
            next_active = torch.zeros_like(active)
            if torch.any(update_mask):
                update_idx = torch.nonzero(update_mask, as_tuple=False).flatten()
                stage_log = log_cum_stay.index_select(0, update_idx) + log_target.index_select(0, update_idx).double()
                log_total.index_copy_(
                    0,
                    update_idx,
                    torch.logaddexp(log_total.index_select(0, update_idx), stage_log),
                )

                stay_update = log_stay.index_select(0, update_idx).double()
                finite_stay = ~torch.isneginf(stay_update)
                if torch.any(finite_stay):
                    finite_idx = update_idx.index_select(0, torch.nonzero(finite_stay, as_tuple=False).flatten())
                    new_cum = log_cum_stay.index_select(0, finite_idx) + stay_update[finite_stay]
                    log_cum_stay.index_copy_(0, finite_idx, new_cum)
                    next_active.index_copy_(0, finite_idx, new_cum > _LOG_CUM_STAY_THRESHOLD)

            active = next_active
            temp_beta *= _TEMP_DECAY_FACTOR
            temp_lambda_X *= _TEMP_DECAY_FACTOR
            temp_lambda_E *= _TEMP_DECAY_FACTOR

        return log_total.to(torch.float32)


class DLangevinMTProposal(DLangevinProposal):
    """
    Multiple-try Metropolis wrapper around DLangevin.

    Generates `num_tries` candidates per chain, scores them with weights
    proportional to pi(y) * q(y->x), samples one to consider, draws `num_tries-1`
    reverse proposals from q(y*), and runs the MTM accept/reject. Energies and
    gradients for all proposals are batched for efficiency.
    """
    supports_property_lambda_override = False

    def __init__(
        self,
        beta: float = 1.0,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
        num_tries: int = 4,
        energy_batch_size: int = 256,
    ):
        super().__init__(beta=beta, lambda_X=lambda_X, lambda_E=lambda_E)
        self.num_tries = max(int(num_tries), 1)
        self.energy_batch_size = max(int(energy_batch_size), 1)

    def needs_proposed_energy(self) -> bool:
        # Energies and grads for all tries are computed inside accept()
        return False

    def _sample_candidate(
        self,
        nt: torch.Tensor,
        et: torch.Tensor,
        gX: torch.Tensor,
        gE: torch.Tensor,
        pairs: torch.Tensor,
        dataset_info,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a single DLangevin proposal for a given state/gradient and return
        (nt_new, et_new, log_q_fwd).
        """
        n = int(nt.shape[0])
        if pairs.numel():
            i_idx, j_idx = pairs[0], pairs[1]
            et_pairs_current = et[i_idx, j_idx]
        else:
            i_idx = j_idx = None
            et_pairs_current = torch.empty((0,), device=device, dtype=torch.long)

        nt_new = nt.clone()
        e_pairs_new = et_pairs_current.clone()
        log_q_fwd = torch.tensor(0.0, device=device, dtype=torch.float32)
        temp_beta = float(self.beta)
        temp_lambda_X = float(self.lambda_X)
        temp_lambda_E = float(self.lambda_E)
        log_cum_stay = torch.zeros((), device=device, dtype=torch.float64)

        while True:
            node_logits = (
                _node_logits_dlang(nt, gX, temp_beta, temp_lambda_X)
                if n > 0
                else torch.empty((0, dataset_info.output_dims["X"]), device=device, dtype=torch.float32)
            )
            if node_logits.numel():
                node_log_probs = F.log_softmax(node_logits, dim=-1)
                stay_logp_nodes = node_log_probs.gather(1, nt.view(-1, 1)).squeeze(1)
                node_stay_logp = stay_logp_nodes.sum()
                stay_prob_nodes = stay_logp_nodes.exp()
                node_can_change = bool(torch.any(stay_prob_nodes < 1.0 - _CHANGE_EPS).item())
                node_probs = node_log_probs.exp()
            else:
                node_log_probs = None
                node_probs = None
                node_stay_logp = torch.tensor(0.0, device=device)
                node_can_change = False

            edge_logits_stage, _ = _edge_logits_dlang(et, gE, temp_beta, temp_lambda_E, pairs)
            if edge_logits_stage.numel():
                edge_log_probs = F.log_softmax(edge_logits_stage, dim=-1)
                if et_pairs_current.numel():
                    stay_logp_edges = edge_log_probs.gather(1, et_pairs_current.view(-1, 1)).squeeze(1)
                    edge_stay_logp = stay_logp_edges.sum()
                    stay_prob_edges = stay_logp_edges.exp()
                    edge_can_change = bool(torch.any(stay_prob_edges < 1.0 - _CHANGE_EPS).item())
                else:
                    edge_stay_logp = torch.tensor(0.0, device=device)
                    edge_can_change = False
                edge_probs = edge_log_probs.exp()
            else:
                edge_log_probs = None
                edge_probs = None
                edge_stay_logp = torch.tensor(0.0, device=device)
                edge_can_change = False

            log_stay_total = (node_stay_logp + edge_stay_logp).double()
            change_possible_stage = node_can_change or edge_can_change

            if not change_possible_stage:
                log_q_fwd = (log_cum_stay + log_stay_total).to(torch.float32)
                break

            if node_probs is not None:
                nt_candidate = torch.multinomial(node_probs, num_samples=1).squeeze(1)
            else:
                nt_candidate = nt.clone()

            if edge_probs is not None:
                e_pairs_candidate = torch.multinomial(edge_probs, num_samples=1).squeeze(1)
            else:
                e_pairs_candidate = et_pairs_current.clone()

            nodes_changed = node_probs is not None and not torch.equal(nt_candidate, nt)
            edges_changed = edge_probs is not None and not torch.equal(e_pairs_candidate, et_pairs_current)

            if nodes_changed or edges_changed:
                nt_new = nt_candidate
                e_pairs_new = e_pairs_candidate
                node_logp = (
                    node_log_probs.gather(1, nt_new.view(-1, 1)).sum()
                    if node_log_probs is not None
                    else torch.tensor(0.0, device=device)
                )
                edge_logp = (
                    edge_log_probs.gather(1, e_pairs_new.view(-1, 1)).sum()
                    if edge_log_probs is not None
                    else torch.tensor(0.0, device=device)
                )
                log_q_fwd = (log_cum_stay + node_logp.double() + edge_logp.double()).to(torch.float32)
                break

            log_cum_stay = log_cum_stay + log_stay_total
            temp_beta *= _TEMP_DECAY_FACTOR
            temp_lambda_X *= _TEMP_DECAY_FACTOR
            temp_lambda_E *= _TEMP_DECAY_FACTOR

        et_new = et.clone()
        if pairs.numel():
            et_new[i_idx, j_idx] = e_pairs_new
            et_new[j_idx, i_idx] = e_pairs_new

        return nt_new.detach(), et_new.detach(), log_q_fwd.detach()

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
        B = len(node_types_list)
        if B == 0:
            return ProposalResult(prop_nodes=[], prop_edges=[], log_q_fwd=torch.empty((0,), device=device), moves=None)

        need_idx = [i for i in range(B) if i not in self._grad_cache]
        if len(need_idx) > 0:
            sub_nodes = [node_types_list[i] for i in need_idx]
            sub_edges = [edge_types_list[i] for i in need_idx]
            with torch.enable_grad():
                with _model_eval_ctx(model):
                    with _amp_autocast_ctx(device, amp_dtype):
                        _, gx_sub, ge_sub = energy_and_grads_batch(
                            model, sub_nodes, sub_edges, dataset_info, device, extra_features, domain_features
                        )
            for k, i in enumerate(need_idx):
                self._grad_cache[i] = (gx_sub[k], ge_sub[k])

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        log_q_first: List[torch.Tensor] = []
        moves: List[Dict[str, List[Dict[str, torch.Tensor]]]] = []

        for b in range(B):
            nt = node_types_list[b].long()
            et = edge_types_list[b].long()
            gX, gE = self._grad_cache[b]
            pairs = self._pairs_for(int(et.shape[0]), device)

            cand_list: List[Dict[str, torch.Tensor]] = []
            for _ in range(self.num_tries):
                nt_new, et_new, log_q = self._sample_candidate(nt, et, gX, gE, pairs, dataset_info, device)
                cand_list.append({"nt": nt_new, "et": et_new, "log_q": log_q})

            prop_nodes.append(cand_list[0]["nt"])
            prop_edges.append(cand_list[0]["et"])
            log_q_first.append(cand_list[0]["log_q"])
            moves.append({"candidates": cand_list})

        log_q_fwd = torch.stack(log_q_first, dim=0) if log_q_first else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd.detach(), moves=moves)

    def _energy_grads_chunked(
        self,
        nodes: List[torch.Tensor],
        edges: List[torch.Tensor],
        *,
        model,
        dataset_info,
        device: torch.device,
        extra_features,
        domain_features,
        amp_dtype: Optional[str],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Compute energies+grads in manageable chunks to avoid OOM."""
        energies: List[torch.Tensor] = []
        grad_X_all: List[torch.Tensor] = []
        grad_E_all: List[torch.Tensor] = []
        chunk = max(int(self.energy_batch_size), 1)
        total = len(nodes)
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            sub_nodes = nodes[start:end]
            sub_edges = edges[start:end]
            with _amp_autocast_ctx(device, amp_dtype):
                e_sub, gx_sub, ge_sub = energy_and_grads_batch(
                    model=model,
                    node_types_list=sub_nodes,
                    edge_types_list=sub_edges,
                    dataset_info=dataset_info,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                )
            energies.append(e_sub)
            grad_X_all.extend(gx_sub)
            grad_E_all.extend(ge_sub)
        energy_cat = torch.cat(energies, dim=0) if energies else torch.empty((0,), device=device)
        return energy_cat, grad_X_all, grad_E_all

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: Optional[torch.Tensor],  # ignored
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = len(current_nodes)
        if B == 0:
            return torch.empty((0,), dtype=torch.bool, device=current_E.device), current_E

        moves = prop_result.moves or []
        if len(moves) != B:
            raise ValueError("DLangevinMTProposal expects per-chain candidate sets in prop_result.moves.")

        device_E = current_E.device
        current_E_f32 = current_E.to(device_E, non_blocking=True).float()

        # Flatten forward candidates for batched energy/grad computation
        f_nodes: List[torch.Tensor] = []
        f_edges: List[torch.Tensor] = []
        f_logq: List[torch.Tensor] = []
        f_meta: List[Tuple[int, int]] = []
        for chain_idx, info in enumerate(moves):
            cand_list = info.get("candidates", [])
            if len(cand_list) < self.num_tries:
                raise ValueError(f"Chain {chain_idx} has {len(cand_list)} candidates, expected {self.num_tries}.")
            for try_idx in range(self.num_tries):
                cand = cand_list[try_idx]
                f_nodes.append(cand["nt"])
                f_edges.append(cand["et"])
                f_logq.append(cand["log_q"])
                f_meta.append((chain_idx, try_idx))

        if len(f_nodes) == 0:
            accept_mask = torch.zeros(B, dtype=torch.bool, device=device_E)
            return accept_mask, current_E_f32

        with torch.enable_grad():
            with _model_eval_ctx(model):
                f_energy, f_grad_X, f_grad_E = self._energy_grads_chunked(
                    nodes=f_nodes,
                    edges=f_edges,
                    model=model,
                    dataset_info=dataset_info,
                    device=device,
                    extra_features=extra_features,
                    domain_features=domain_features,
                    amp_dtype=amp_dtype,
                )

        # Organize forward results per chain/try
        per_chain: List[List[Dict[str, torch.Tensor]]] = [
            [dict() for _ in range(self.num_tries)] for _ in range(B)
        ]
        for idx, (chain_idx, try_idx) in enumerate(f_meta):
            per_chain[chain_idx][try_idx] = dict(
                nt=f_nodes[idx],
                et=f_edges[idx],
                energy=f_energy[idx].to(device_E, non_blocking=True).float(),
                grad_X=f_grad_X[idx],
                grad_E=f_grad_E[idx],
                log_q_fwd=f_logq[idx].to(device_E, non_blocking=True).float(),
            )

        logsum_fwd: List[torch.Tensor] = []
        selected_idx: List[int] = []
        selected_records: List[Dict[str, torch.Tensor]] = []

        for chain_idx in range(B):
            cand_records = per_chain[chain_idx]
            log_w_list: List[torch.Tensor] = []
            for rec in cand_records:
                log_q_rev = self._log_transition_prob_escalated(
                    nt_current=rec["nt"],
                    et_current=rec["et"],
                    nt_target=current_nodes[chain_idx].long(),
                    et_target=current_edges[chain_idx].long(),
                    grad_X=rec["grad_X"],
                    grad_E=rec["grad_E"],
                    device=device,
                )
                rec["log_q_rev"] = log_q_rev.to(device_E, non_blocking=True).float()
                log_w_list.append((-(self.beta) * rec["energy"]) + rec["log_q_rev"])

            log_weights = torch.stack(log_w_list, dim=0)
            if not torch.isfinite(log_weights).any():
                log_weights = torch.zeros_like(log_weights)
            log_w_sum = torch.logsumexp(log_weights, dim=0)
            probs = torch.exp(log_weights - log_w_sum)
            sel = int(torch.multinomial(probs, num_samples=1).item())

            logsum_fwd.append(log_w_sum.detach())
            selected_idx.append(sel)
            selected_records.append(cand_records[sel])

        # Sample reverse proposals (excluding the "current" reverse element) in batch
        rev_nodes: List[torch.Tensor] = []
        rev_edges: List[torch.Tensor] = []
        rev_meta: List[Tuple[int, int]] = []
        for chain_idx, rec in enumerate(selected_records):
            nt_sel = rec["nt"]
            et_sel = rec["et"]
            gX_sel = rec["grad_X"]
            gE_sel = rec["grad_E"]
            pairs_sel = self._pairs_for(int(et_sel.shape[0]), device)
            for rev_idx in range(self.num_tries - 1):
                nt_r, et_r, _ = self._sample_candidate(
                    nt_sel, et_sel, gX_sel, gE_sel, pairs_sel, dataset_info, device
                )
                rev_nodes.append(nt_r)
                rev_edges.append(et_r)
                rev_meta.append((chain_idx, rev_idx))

        rev_energy: Optional[torch.Tensor] = None
        rev_grad_X: List[torch.Tensor] = []
        rev_grad_E: List[torch.Tensor] = []
        if len(rev_nodes) > 0:
            with torch.enable_grad():
                with _model_eval_ctx(model):
                    rev_energy, rev_grad_X, rev_grad_E = self._energy_grads_chunked(
                        nodes=rev_nodes,
                        edges=rev_edges,
                        model=model,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        amp_dtype=amp_dtype,
                    )

        # Organize reverse results per chain
        rev_per_chain: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(B)]
        if rev_energy is not None:
            for idx, (chain_idx, _) in enumerate(rev_meta):
                rev_per_chain[chain_idx].append(
                    dict(
                        nt=rev_nodes[idx],
                        et=rev_edges[idx],
                        energy=rev_energy[idx].to(device_E, non_blocking=True).float(),
                        grad_X=rev_grad_X[idx],
                        grad_E=rev_grad_E[idx],
                    )
                )

        accept_mask = torch.zeros(B, dtype=torch.bool, device=device_E)
        prop_E_full = current_E_f32.clone()

        for chain_idx in range(B):
            rec = selected_records[chain_idx]
            log_weights_rev: List[torch.Tensor] = []
            # Reverse weight for the current state
            log_weights_rev.append((-(self.beta) * current_E_f32[chain_idx]) + rec["log_q_fwd"])

            for rev_rec in rev_per_chain[chain_idx]:
                log_q_back = self._log_transition_prob_escalated(
                    nt_current=rev_rec["nt"],
                    et_current=rev_rec["et"],
                    nt_target=rec["nt"],
                    et_target=rec["et"],
                    grad_X=rev_rec["grad_X"],
                    grad_E=rev_rec["grad_E"],
                    device=device,
                ).to(device_E, non_blocking=True).float()
                log_weights_rev.append((-(self.beta) * rev_rec["energy"]) + log_q_back)

            log_weights_rev_tensor = torch.stack(log_weights_rev, dim=0)
            if not torch.isfinite(log_weights_rev_tensor).any():
                log_weights_rev_tensor = torch.zeros_like(log_weights_rev_tensor)
            logsum_rev = torch.logsumexp(log_weights_rev_tensor, dim=0)

            mh_term = logsum_fwd[chain_idx] - logsum_rev
            log_u = torch.log(torch.rand((), device=device_E, dtype=torch.float32))
            acc = bool(log_u < mh_term)
            accept_mask[chain_idx] = acc

            prop_result.prop_nodes[chain_idx] = rec["nt"].detach()
            prop_result.prop_edges[chain_idx] = rec["et"].detach()
            if prop_result.log_q_fwd is not None and chain_idx < len(prop_result.log_q_fwd):
                prop_result.log_q_fwd[chain_idx] = rec["log_q_fwd"].detach()

            if acc:
                prop_E_full[chain_idx] = rec["energy"]
                self._grad_cache[chain_idx] = (rec["grad_X"].detach(), rec["grad_E"].detach())

        return accept_mask, prop_E_full


class DLangevinAnnealingProposal(DLangevinProposal):
    """
    DLangevin variant with a two-phase temperature schedule: linear annealing
    from `beta_init` to `beta_final` for the first `anneal_steps` MCMC steps,
    then constant `beta_final` afterwards.
    """

    def __init__(
        self,
        beta_init: float = 1.0,
        beta_final: float = 1.0,
        anneal_steps: int = 0,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        super().__init__(
            beta=beta_init,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
        self.beta_init = float(beta_init)
        self.beta_final = float(beta_final)
        self.anneal_steps = max(int(anneal_steps), 0)

    def _beta_at_step(self, step_idx: int) -> float:
        if self.anneal_steps <= 0:
            return float(self.beta_final)
        s = max(int(step_idx), 0)
        if s >= self.anneal_steps:
            return float(self.beta_final)
        denom = max(self.anneal_steps - 1, 1)
        progress = float(s) / float(denom)
        return float(self.beta_init + (self.beta_final - self.beta_init) * progress)

    def on_step_start(self, step_idx: int) -> None:
        self.beta = self._beta_at_step(step_idx)


class DLangevinTwoBetasProposal(DLangevinProposal):
    """
    DLangevin variant with separate betas for proposal logits (beta_prop) and MH acceptance (beta_mh).
    Proposal temperature controls step sizes; MH temperature controls the target distribution.
    """
    supports_property_lambda_override = False

    def __init__(
        self,
        beta_prop: float,
        beta_mh: float,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        super().__init__(beta=beta_prop, lambda_X=lambda_X, lambda_E=lambda_E)
        self.beta_prop = float(beta_prop)
        self.beta_mh = float(beta_mh)

    def _log_transition_prob_escalated(
        self,
        nt_current: torch.Tensor,
        et_current: torch.Tensor,
        nt_target: torch.Tensor,
        et_target: torch.Tensor,
        grad_X: torch.Tensor,
        grad_E: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        temp_beta = float(self.beta_prop)
        temp_lambda_X = float(self.lambda_X)
        temp_lambda_E = float(self.lambda_E)
        pairs = self._pairs_for(int(et_current.shape[0]), device)
        if pairs.numel():
            stay_edge = et_current[pairs[0], pairs[1]]
            target_edge = et_target[pairs[0], pairs[1]]
        else:
            stay_edge = torch.empty((0,), device=device, dtype=torch.long)
            target_edge = torch.empty((0,), device=device, dtype=torch.long)

        log_total = torch.full((), float("-inf"), device=device, dtype=torch.float64)
        log_cum_stay = torch.zeros((), device=device, dtype=torch.float64)

        for _ in range(_MAX_REVERSE_ESCALATION_STEPS):
            node_logits = (
                _node_logits_dlang(nt_current, grad_X, temp_beta, temp_lambda_X)
                if nt_current.numel()
                else torch.empty((0, grad_X.shape[1]), device=device, dtype=torch.float32)
            )
            edge_logits, _ = _edge_logits_dlang(et_current, grad_E, temp_beta, temp_lambda_E, pairs)

            log_target, log_stay, change_possible = _factorized_log_prob(
                node_logits,
                edge_logits,
                nt_current,
                stay_edge,
                nt_target,
                target_edge,
                device,
            )

            if not change_possible:
                break

            stage_log = log_cum_stay + log_target.double()
            if torch.isfinite(log_total):
                log_total = torch.logaddexp(log_total, stage_log)
            else:
                log_total = stage_log

            if torch.isneginf(log_stay):
                break

            log_cum_stay = log_cum_stay + log_stay.double()
            if log_cum_stay.item() <= _LOG_CUM_STAY_THRESHOLD:
                break

            temp_beta *= _TEMP_DECAY_FACTOR
            temp_lambda_X *= _TEMP_DECAY_FACTOR
            temp_lambda_E *= _TEMP_DECAY_FACTOR

        return log_total.to(torch.float32)

    def accept(
        self,
        *,
        model,
        dataset_info,
        current_nodes: Sequence[torch.Tensor],
        current_edges: Sequence[torch.Tensor],
        prop_result: ProposalResult,
        current_E: torch.Tensor,
        prop_E: Optional[torch.Tensor],  # ignored
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Copy-paste of base accept with beta_mh in MH term and beta_prop in reverse q.
        B = len(prop_result.prop_nodes)
        assert B == len(current_nodes) == len(current_edges) == int(current_E.shape[0])

        same_chain_flags: List[bool] = []
        eval_indices: List[int] = []
        for idx in range(B):
            same_nodes = torch.equal(current_nodes[idx], prop_result.prop_nodes[idx])
            same_edges = torch.equal(current_edges[idx], prop_result.prop_edges[idx])
            same = bool(same_nodes and same_edges)
            same_chain_flags.append(same)
            if not same:
                eval_indices.append(idx)

        device_E = current_E.device
        current_E_f32 = current_E.to(device_E, non_blocking=True).float()

        if len(eval_indices) == 0:
            accept_mask = torch.ones(B, dtype=torch.bool, device=device_E)
            return accept_mask, current_E_f32

        prop_nodes_eval = [prop_result.prop_nodes[i] for i in eval_indices]
        prop_edges_eval = [prop_result.prop_edges[i] for i in eval_indices]

        with torch.enable_grad():
            with _model_eval_ctx(model):
                with _amp_autocast_ctx(device, amp_dtype):
                    prop_E_eval, grad_X_eval_list, grad_E_eval_list = energy_and_grads_batch(
                        model=model,
                        node_types_list=prop_nodes_eval,
                        edge_types_list=prop_edges_eval,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                    )

        log_q_rev_eval: List[torch.Tensor] = []
        for rel_idx, global_idx in enumerate(eval_indices):
            nt_old = current_nodes[global_idx].long()
            et_old = current_edges[global_idx].long()
            nt_new = prop_result.prop_nodes[global_idx].long()
            et_new = prop_result.prop_edges[global_idx].long()

            gX_p = grad_X_eval_list[rel_idx]
            gE_p = grad_E_eval_list[rel_idx]

            log_q_rev = self._log_transition_prob_escalated(
                nt_current=nt_new,
                et_current=et_new,
                nt_target=nt_old,
                et_target=et_old,
                grad_X=gX_p,
                grad_E=gE_p,
                device=device,
            )
            log_q_rev_eval.append(log_q_rev.detach())

        idx_tensor = torch.tensor(eval_indices, device=prop_result.log_q_fwd.device, dtype=torch.long)
        log_q_fwd_eval = prop_result.log_q_fwd.index_select(0, idx_tensor)

        log_q_rev_tensor = torch.stack(log_q_rev_eval, dim=0).to(device_E, non_blocking=True).float()
        log_q_fwd_tensor = log_q_fwd_eval.to(device_E, non_blocking=True).float()
        prop_E_eval_f32 = prop_E_eval.to(device_E, non_blocking=True).float()
        current_E_eval = current_E_f32.index_select(0, idx_tensor.to(device_E))

        mh_term = (-(self.beta_mh) * (prop_E_eval_f32 - current_E_eval)) + (log_q_rev_tensor - log_q_fwd_tensor)
        log_u = torch.log(torch.rand(len(eval_indices), device=device_E, dtype=torch.float32))
        accept_eval = (log_u < mh_term)

        accept_mask = torch.zeros(B, dtype=torch.bool, device=device_E)
        prop_E_full = current_E_f32.clone()
        prop_E_full.index_copy_(0, idx_tensor.to(device_E), prop_E_eval_f32)

        if any(same_chain_flags):
            same_indices = torch.tensor(
                [i for i, same in enumerate(same_chain_flags) if same],
                device=device_E,
                dtype=torch.long,
            )
            if same_indices.numel():
                accept_mask.index_fill_(0, same_indices, True)

        accept_mask.index_copy_(0, idx_tensor.to(device_E), accept_eval)

        if torch.any(accept_eval):
            acc_local = torch.nonzero(accept_eval, as_tuple=False).flatten().tolist()
            for rel in acc_local:
                global_idx = eval_indices[rel]
                self._grad_cache[int(global_idx)] = (
                    grad_X_eval_list[rel].detach(),
                    grad_E_eval_list[rel].detach(),
                )

        return accept_mask, prop_E_full


class DLangevinTwoBetasVectorizedProposal(DLangevinVectorizedProposal):
    """Two-beta DLangevin with batched reverse transition probabilities."""

    supports_property_lambda_override = False

    def __init__(
        self,
        beta_prop: float,
        beta_mh: float,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        super().__init__(beta=beta_prop, lambda_X=lambda_X, lambda_E=lambda_E)
        self.beta_prop = float(beta_prop)
        self.beta_mh = float(beta_mh)

    def _mh_beta(self) -> float:
        return float(self.beta_mh)


class DLangevinTwoBetasAnnealingProposal(DLangevinTwoBetasProposal):
    """
    Two-betas DLangevin with annealed MH temperature (beta_mh).
    """
    supports_property_lambda_override = False

    def __init__(
        self,
        beta_prop: float,
        beta_mh_init: float,
        beta_mh_final: float,
        beta_mh_anneal_steps: int,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        super().__init__(
            beta_prop=beta_prop,
            beta_mh=beta_mh_init,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
        self.beta_mh_init = float(beta_mh_init)
        self.beta_mh_final = float(beta_mh_final)
        self.beta_mh_anneal_steps = max(int(beta_mh_anneal_steps), 0)

    def _beta_mh_at_step(self, step_idx: int) -> float:
        if self.beta_mh_anneal_steps <= 0:
            return float(self.beta_mh_final)
        s = max(int(step_idx), 0)
        if s >= self.beta_mh_anneal_steps:
            return float(self.beta_mh_final)
        denom = max(self.beta_mh_anneal_steps - 1, 1)
        progress = float(s) / float(denom)
        return float(self.beta_mh_init + (self.beta_mh_final - self.beta_mh_init) * progress)

    def on_step_start(self, step_idx: int) -> None:
        self.beta_mh = self._beta_mh_at_step(step_idx)


class DLangevinTwoBetasAnnealingVectorizedProposal(DLangevinTwoBetasVectorizedProposal):
    """Two-beta MH annealing with batched reverse transition probabilities."""

    supports_property_lambda_override = False

    def __init__(
        self,
        beta_prop: float,
        beta_mh_init: float,
        beta_mh_final: float,
        beta_mh_anneal_steps: int,
        lambda_X: float = 2.0,
        lambda_E: float = 2.0,
    ):
        super().__init__(
            beta_prop=beta_prop,
            beta_mh=beta_mh_init,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
        self.beta_mh_init = float(beta_mh_init)
        self.beta_mh_final = float(beta_mh_final)
        self.beta_mh_anneal_steps = max(int(beta_mh_anneal_steps), 0)

    def _beta_mh_at_step(self, step_idx: int) -> float:
        if self.beta_mh_anneal_steps <= 0:
            return float(self.beta_mh_final)
        s = max(int(step_idx), 0)
        if s >= self.beta_mh_anneal_steps:
            return float(self.beta_mh_final)
        denom = max(self.beta_mh_anneal_steps - 1, 1)
        progress = float(s) / float(denom)
        return float(
            self.beta_mh_init
            + (self.beta_mh_final - self.beta_mh_init) * progress
        )

    def on_step_start(self, step_idx: int) -> None:
        self.beta_mh = self._beta_mh_at_step(step_idx)


class DLangevinTwoBetasAnnealingVectorizedNoOriginProposal(
    DLangevinTwoBetasAnnealingVectorizedProposal
):
    """Backward-compatible name for the unconstrained annealed kernel."""

    excludes_origin = False


class DLangevinNoMHProposal(DLangevinProposal):
    """
    Variant of DLangevin that deterministically accepts each proposal (no MH test).

    Keeps the gradient cache in sync by recomputing energies/grads for the chains
    that actually changed, but skips the Metropolis-Hastings ratio entirely.
    """
    supports_property_lambda_override = False

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
        B = len(prop_result.prop_nodes)
        if B == 0:
            return torch.empty(0, dtype=torch.bool, device=current_E.device), current_E

        eval_indices: List[int] = []
        for idx in range(B):
            same_nodes = torch.equal(current_nodes[idx], prop_result.prop_nodes[idx])
            same_edges = torch.equal(current_edges[idx], prop_result.prop_edges[idx])
            if not (same_nodes and same_edges):
                eval_indices.append(idx)

        device_E = current_E.device
        current_E_f32 = current_E.to(device_E, non_blocking=True).float()

        if len(eval_indices) == 0:
            accept_mask = torch.ones(B, dtype=torch.bool, device=device_E)
            return accept_mask, current_E_f32

        prop_nodes_eval = [prop_result.prop_nodes[i] for i in eval_indices]
        prop_edges_eval = [prop_result.prop_edges[i] for i in eval_indices]

        with torch.enable_grad():
            with _model_eval_ctx(model):
                with _amp_autocast_ctx(device, amp_dtype):
                    prop_E_eval, grad_X_eval_list, grad_E_eval_list = energy_and_grads_batch(
                        model=model,
                        node_types_list=prop_nodes_eval,
                        edge_types_list=prop_edges_eval,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                    )

        prop_E_eval_f32 = prop_E_eval.to(device_E, non_blocking=True).float()
        prop_E_full = current_E_f32.clone()
        idx_tensor = torch.tensor(eval_indices, device=device_E, dtype=torch.long)
        prop_E_full.index_copy_(0, idx_tensor, prop_E_eval_f32)

        accept_mask = torch.ones(B, dtype=torch.bool, device=device_E)

        for rel_idx, global_idx in enumerate(eval_indices):
            self._grad_cache[int(global_idx)] = (
                grad_X_eval_list[rel_idx].detach(),
                grad_E_eval_list[rel_idx].detach(),
            )

        return accept_mask, prop_E_full
