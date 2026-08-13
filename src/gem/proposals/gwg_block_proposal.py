from typing import List, Sequence, Tuple, Optional, Dict
from contextlib import nullcontext, contextmanager
import torch

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch
from .gwd_proposal import _node_logits, _edge_logits, _pairs_index


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
    """Temporarily switch the model to eval() for proposals."""
    prev = model.training
    try:
        if prev:
            model.eval()
        yield
    finally:
        if prev:
            model.train()


class GWGBlockProposal(Proposal):
    """Gibbs-with-Gradients block proposal (multi-move, single gradient eval).

    - Computes gradients once per chain at the current state.
    - Builds GWG logits for every node and undirected edge pair (excluding
      current labels) and samples up to ``max_moves`` coordinates without
      replacement.
    - All moves sampled within the block are applied deterministically to
      produce the proposed state. Acceptance uses MH with asymmetric q(x'|x).
    """

    def __init__(self, beta: float = 1.0, max_moves: Optional[int] = None):
        self.beta = float(beta)
        self.max_moves = int(max_moves) if max_moves is not None else None
        # Cache gradients per chain index (populated lazily)
        self._grad_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache i<j index tensors keyed by (device, device_index, n)
        self._pair_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ utils

    def _pairs_for(self, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, int(device.index or -1), int(n))
        pairs = self._pair_cache.get(key)
        if pairs is None:
            pairs = torch.triu_indices(n, n, 1, device=device)
            self._pair_cache[key] = pairs
        return pairs

    @staticmethod
    def _flatten_logits(node_logits: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
        flat_nodes = node_logits.reshape(-1) if node_logits.numel() else node_logits
        flat_edges = edge_logits.reshape(-1) if edge_logits.numel() else edge_logits
        if flat_nodes.numel() and flat_edges.numel():
            return torch.cat([flat_nodes, flat_edges], dim=0)
        if flat_nodes.numel():
            return flat_nodes
        return flat_edges

    @staticmethod
    def _coordinate_limit(n: int, pairs: torch.Tensor) -> int:
        # #nodes coordinates + #edge-pairs coordinates
        return int(n) + int(pairs.shape[1])

    @staticmethod
    def _index_from_move(move: Tuple, n: int, num_node_types: int, num_edge_types: int) -> int:
        if move[0] == "node":
            _, i, new_t, _old = move
            return int(i) * num_node_types + int(new_t)
        _, i, j, new_t, _old = move
        pair_idx = _pairs_index(int(i), int(j), int(n))
        offset = n * num_node_types
        return offset + pair_idx * num_edge_types + int(new_t)

    @staticmethod
    def _mask_coordinate(
        logits_vec: torch.Tensor,
        idx: int,
        n: int,
        num_node_types: int,
        num_edge_types: int,
    ) -> None:
        total_node_entries = n * num_node_types
        if idx < total_node_entries:
            node_idx = idx // num_node_types
            start = node_idx * num_node_types
            logits_vec[start : start + num_node_types] = float("-inf")
            return
        offset = idx - total_node_entries
        pair_idx = offset // num_edge_types
        start = total_node_entries + pair_idx * num_edge_types
        logits_vec[start : start + num_edge_types] = float("-inf")

    @staticmethod
    def _apply_move_inplace(
        nt: torch.Tensor,
        et: torch.Tensor,
        move: Tuple,
    ) -> None:
        kind = move[0]
        if kind == "node":
            _, i, new_t, _old = move
            nt[int(i)] = int(new_t)
            return
        _, i, j, new_t, _old = move
        i = int(i)
        j = int(j)
        new_t = int(new_t)
        et[i, j] = new_t
        et[j, i] = new_t

    def _log_prob_of_sequence(
        self,
        nt: torch.Tensor,
        et: torch.Tensor,
        gX: torch.Tensor,
        gE: torch.Tensor,
        moves: List[Tuple],
        num_node_types: int,
        num_edge_types: int,
    ) -> torch.Tensor:
        device = gX.device
        n = int(nt.shape[0])
        if len(moves) == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        node_logits = _node_logits(nt, gX, self.beta) if n > 0 else torch.empty((0, num_node_types), device=device, dtype=torch.float32)
        pairs = self._pairs_for(n, device)
        edge_logits, pairs = _edge_logits(et, gE, self.beta, pairs)
        logits = self._flatten_logits(node_logits, edge_logits).clone()
        if logits.numel() == 0:
            return torch.full((), float("-inf"), device=device, dtype=torch.float32)

        log_prob = torch.zeros((), device=device, dtype=torch.float32)
        for mv in moves:
            idx = self._index_from_move(mv, n, num_node_types, num_edge_types)
            if not torch.isfinite(logits[idx]):
                return torch.full((), float("-inf"), device=device, dtype=torch.float32)
            finite_mask = torch.isfinite(logits)
            log_den = torch.logsumexp(logits[finite_mask], dim=0)
            log_prob = log_prob + (logits[idx] - log_den)
            self._mask_coordinate(logits, idx, n, num_node_types, num_edge_types)
        return log_prob

    # ---------------------------------------------------------------- proposal

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
        B = len(node_types_list)
        if B == 0:
            empty = torch.empty((0,), device=device)
            return ProposalResult(prop_nodes=[], prop_edges=[], log_q_fwd=empty, moves=[])

        need_idx = [i for i in range(B) if i not in self._grad_cache]
        if need_idx:
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
                        )
            for cache_i, chain_idx in enumerate(need_idx):
                self._grad_cache[chain_idx] = (gx_sub[cache_i], ge_sub[cache_i])

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []
        log_q_fwd_list: List[torch.Tensor] = []
        moves_all: List[List[Tuple]] = []

        for chain_idx in range(B):
            nt = node_types_list[chain_idx].long()
            et = edge_types_list[chain_idx].long()
            gX, gE = self._grad_cache[chain_idx]
            n = int(nt.shape[0])
            pairs = self._pairs_for(n, device)

            node_logits = _node_logits(nt, gX, self.beta) if n > 0 else torch.empty((0, num_node_types), device=device, dtype=torch.float32)
            edge_logits, pairs = _edge_logits(et, gE, self.beta, pairs)
            logits_vec = self._flatten_logits(node_logits, edge_logits)

            if logits_vec.numel() == 0 or (~torch.isfinite(logits_vec)).all():
                prop_nodes.append(nt.clone())
                prop_edges.append(et.clone())
                log_q_fwd_list.append(torch.zeros((), device=device, dtype=torch.float32))
                moves_all.append([])
                continue

            nt_new = nt.clone()
            et_new = et.clone()
            logits_work = logits_vec.clone()
            moves_chain: List[Tuple] = []
            log_terms: List[torch.Tensor] = []

            coord_limit = self._coordinate_limit(n, pairs)
            max_steps = coord_limit if self.max_moves is None else min(coord_limit, self.max_moves)
            steps = int(max_steps)

            for _ in range(steps):
                finite_mask = torch.isfinite(logits_work)
                if not finite_mask.any():
                    break
                logits_active = logits_work[finite_mask]
                log_den = torch.logsumexp(logits_active, dim=0)
                log_probs = logits_active - log_den
                probs = torch.exp(log_probs)
                sel_local = torch.multinomial(probs, num_samples=1).item()
                active_indices = torch.nonzero(finite_mask, as_tuple=False).view(-1)
                idx = int(active_indices[sel_local].item())

                if idx < n * num_node_types:
                    node_idx = idx // num_node_types
                    new_t = idx % num_node_types
                    old_t = int(nt_new[node_idx].item())
                    move = ("node", int(node_idx), int(new_t), old_t)
                else:
                    offset = idx - n * num_node_types
                    pair_idx = offset // num_edge_types
                    if pair_idx >= pairs.shape[1]:
                        break
                    new_t = offset % num_edge_types
                    i = int(pairs[0, pair_idx].item())
                    j = int(pairs[1, pair_idx].item())
                    old_t = int(et_new[i, j].item())
                    move = ("edge", i, j, int(new_t), old_t)

                if move[0] == "node" and move[2] == move[3]:
                    self._mask_coordinate(logits_work, idx, n, num_node_types, num_edge_types)
                    continue
                if move[0] == "edge" and move[3] == move[4]:
                    self._mask_coordinate(logits_work, idx, n, num_node_types, num_edge_types)
                    continue

                moves_chain.append(move)
                log_terms.append(logits_work[idx] - log_den)
                self._apply_move_inplace(nt_new, et_new, move)
                self._mask_coordinate(logits_work, idx, n, num_node_types, num_edge_types)

            log_q = torch.stack(log_terms, dim=0).sum() if log_terms else torch.zeros((), device=device, dtype=torch.float32)
            prop_nodes.append(nt_new.detach())
            prop_edges.append(et_new.detach())
            log_q_fwd_list.append(log_q.detach())
            moves_all.append(moves_chain)

        log_q_fwd = torch.stack(log_q_fwd_list, dim=0) if log_q_fwd_list else torch.empty((0,), device=device)
        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=log_q_fwd, moves=moves_all)

    # ---------------------------------------------------------------- acceptance

    def needs_proposed_energy(self) -> bool:
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
        prop_E: Optional[torch.Tensor],
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert prop_result.log_q_fwd is not None and prop_result.moves is not None
        B = len(prop_result.prop_nodes)
        assert B == len(current_nodes) == len(current_edges) == int(current_E.shape[0])

        same_flags: List[bool] = []
        eval_indices: List[int] = []
        for idx in range(B):
            same_nodes = torch.equal(current_nodes[idx], prop_result.prop_nodes[idx])
            same_edges = torch.equal(current_edges[idx], prop_result.prop_edges[idx])
            same = bool(same_nodes and same_edges)
            same_flags.append(same)
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
                    prop_E_eval, grad_X_eval, grad_E_eval = energy_and_grads_batch(
                        model=model,
                        node_types_list=prop_nodes_eval,
                        edge_types_list=prop_edges_eval,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                    )

        prop_E_eval_f32 = prop_E_eval.to(device_E, non_blocking=True).float()
        idx_tensor = torch.tensor(eval_indices, device=device_E, dtype=torch.long)
        log_q_fwd_eval = prop_result.log_q_fwd.index_select(0, idx_tensor.to(prop_result.log_q_fwd.device))
        log_q_rev_list: List[torch.Tensor] = []

        num_node_types = dataset_info.output_dims["X"]
        num_edge_types = dataset_info.output_dims["E"]

        for rel_idx, global_idx in enumerate(eval_indices):
            moves_forward = prop_result.moves[global_idx]
            reverse_moves = []
            for mv in reversed(moves_forward):
                if mv[0] == "node":
                    reverse_moves.append(("node", mv[1], mv[3], mv[2]))
                else:
                    reverse_moves.append(("edge", mv[1], mv[2], mv[4], mv[3]))

            log_q_rev = self._log_prob_of_sequence(
                nt=prop_result.prop_nodes[global_idx],
                et=prop_result.prop_edges[global_idx],
                gX=grad_X_eval[rel_idx],
                gE=grad_E_eval[rel_idx],
                moves=reverse_moves,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
            )
            log_q_rev_list.append(log_q_rev.detach())

        log_q_rev_tensor = torch.stack(log_q_rev_list, dim=0).to(device_E, non_blocking=True).float()
        log_q_fwd_tensor = log_q_fwd_eval.to(device_E, non_blocking=True).float()
        current_E_eval = current_E_f32.index_select(0, idx_tensor)

        mh_term = (-(prop_E_eval_f32 - current_E_eval)) + (log_q_rev_tensor - log_q_fwd_tensor)
        log_u = torch.log(torch.rand(len(eval_indices), device=device_E, dtype=torch.float32))
        accept_eval = log_u < mh_term

        accept_mask = torch.zeros(B, dtype=torch.bool, device=device_E)
        accept_mask.index_copy_(0, idx_tensor, accept_eval)

        same_indices = [i for i, same in enumerate(same_flags) if same]
        if same_indices:
            accept_mask.index_fill_(0, torch.tensor(same_indices, device=device_E, dtype=torch.long), True)

        prop_E_full = current_E_f32.clone()
        prop_E_full.index_copy_(0, idx_tensor, prop_E_eval_f32)

        if torch.any(accept_eval):
            acc_local = torch.nonzero(accept_eval, as_tuple=False).flatten().tolist()
            for rel in acc_local:
                global_idx = eval_indices[rel]
                self._grad_cache[int(global_idx)] = (
                    grad_X_eval[rel].detach(),
                    grad_E_eval[rel].detach(),
                )

        return accept_mask, prop_E_full
