from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional
from contextlib import contextmanager, nullcontext
import torch

from .base import Proposal, ProposalResult
from ..sampler_energy import energy_and_grads_batch, energy_and_grads_dense


def _amp_autocast_ctx(device: torch.device, amp_dtype: Optional[str] = None):
    """Match the autocast helper used by other proposals."""
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
    prev = model.training
    try:
        if prev:
            model.eval()
        yield
    finally:
        if prev:
            model.train()


class SimpleProposal(Proposal):
    """Greedy multi-edit proposal that follows the steepest negative gradients."""

    def __init__(
        self,
        edits_per_step: int = 5,
        grad_eps: float = 0.0,
        random_fill: bool = True,
        use_conditioner: bool = True,
    ):
        self.edits_per_step = max(int(edits_per_step), 0)
        self.grad_eps = float(grad_eps)
        self.random_fill = bool(random_fill)
        self.use_conditioner = bool(use_conditioner)

    def _apply_vectorized_edits(
        self,
        nt: torch.Tensor,
        et: torch.Tensor,
        gX: torch.Tensor,
        gE: torch.Tensor,
        num_node_types: int,
        num_edge_types: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply greedy edits using vectorized top-k over node/edge candidates."""
        n = int(nt.shape[0])
        if n == 0 or self.edits_per_step <= 0:
            return nt.long().clone(), et.long().clone()

        nt_new = nt.long().clone()
        et_new = et.long().clone()
        device = nt_new.device
        threshold = -self.grad_eps

        # --- Node candidates ---
        node_best_delta = torch.full((n,), float("inf"), device=device)
        node_best_type = torch.zeros((n,), dtype=torch.long, device=device)
        node_valid = torch.zeros((n,), dtype=torch.bool, device=device)
        if num_node_types > 1:
            cur_idx = nt_new.view(-1)
            grad_cur = gX.gather(1, cur_idx.unsqueeze(1)).squeeze(1)
            delta = gX - grad_cur.unsqueeze(1)
            delta = delta.clone()
            delta.scatter_(1, cur_idx.unsqueeze(1), float("inf"))
            node_best_delta, node_best_type = delta.min(dim=1)
            node_valid = node_best_delta < threshold

        # --- Edge candidates (upper triangle) ---
        edge_best_delta_flat = torch.zeros((0,), device=device)
        edge_best_type_flat = torch.zeros((0,), dtype=torch.long, device=device)
        edge_valid_flat = torch.zeros((0,), dtype=torch.bool, device=device)
        upper_idx = None
        if num_edge_types > 1 and n > 1 and et_new.numel() > 0:
            g_sym = 0.5 * (gE + gE.transpose(0, 1))
            flat = g_sym.reshape(-1, num_edge_types)
            cur_flat = et_new.reshape(-1)
            grad_cur = flat.gather(1, cur_flat.unsqueeze(1)).squeeze(1)
            delta = flat - grad_cur.unsqueeze(1)
            delta = delta.clone()
            delta.scatter_(1, cur_flat.unsqueeze(1), float("inf"))
            best_delta_edge, best_type_edge = delta.min(dim=1)
            best_delta_edge = best_delta_edge.view(n, n)
            best_type_edge = best_type_edge.view(n, n)
            upper_idx = torch.triu_indices(n, n, 1, device=device)
            edge_best_delta_flat = best_delta_edge[upper_idx[0], upper_idx[1]]
            edge_best_type_flat = best_type_edge[upper_idx[0], upper_idx[1]]
            edge_valid_flat = edge_best_delta_flat < threshold

        # --- Select top-k across node + edge candidates ---
        if edge_best_delta_flat.numel() > 0:
            all_deltas = torch.cat((node_best_delta, edge_best_delta_flat), dim=0)
            all_valid = torch.cat((node_valid, edge_valid_flat), dim=0)
        else:
            all_deltas = node_best_delta
            all_valid = node_valid

        valid_count = int(all_valid.sum().item())
        k = min(self.edits_per_step, valid_count)
        if k > 0:
            masked = all_deltas.clone()
            masked[~all_valid] = float("inf")
            top_idx = torch.topk(-masked, k=k, largest=True).indices
            node_mask = top_idx < n

            if node_mask.any():
                idx_nodes = top_idx[node_mask]
                nt_new[idx_nodes] = node_best_type[idx_nodes]

            if edge_best_delta_flat.numel() > 0:
                edge_idx = top_idx[~node_mask] - n
                if edge_idx.numel() > 0:
                    if upper_idx is None:
                        upper_idx = torch.triu_indices(n, n, 1, device=device)
                    i = upper_idx[0, edge_idx]
                    j = upper_idx[1, edge_idx]
                    new_t = edge_best_type_flat[edge_idx]
                    et_new[i, j] = new_t
                    et_new[j, i] = new_t

        remaining = self.edits_per_step - k
        if self.random_fill and remaining > 0:
            nt_new, et_new = self._random_fill_vectorized(
                nt_new,
                et_new,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                num_edits=remaining,
                upper_idx=upper_idx,
            )
        return nt_new, et_new

    def _random_fill_vectorized(
        self,
        nt: torch.Tensor,
        et: torch.Tensor,
        *,
        num_node_types: int,
        num_edge_types: int,
        num_edits: int,
        upper_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized uniform random edits (no Python loops)."""
        n = int(nt.shape[0])
        if num_edits <= 0 or n == 0:
            return nt, et

        node_k = max(int(num_node_types) - 1, 0)
        edge_k = max(int(num_edge_types) - 1, 0)
        P = (n * (n - 1)) // 2 if n > 1 else 0
        node_cands = n * node_k
        edge_cands = P * edge_k
        total = node_cands + edge_cands
        if total <= 0:
            return nt, et

        device = nt.device
        idx = torch.randint(0, total, (num_edits,), device=device)

        if node_cands > 0:
            node_mask = idx < node_cands
            if node_mask.any():
                local = idx[node_mask]
                i = torch.div(local, node_k, rounding_mode="floor")
                type_idx = local % node_k
                cur = nt[i]
                new_t = type_idx + (type_idx >= cur).long()
                nt[i] = new_t

        if edge_cands > 0:
            edge_mask = idx >= node_cands
            if edge_mask.any():
                local = idx[edge_mask] - node_cands
                pair_idx = torch.div(local, edge_k, rounding_mode="floor")
                type_idx = local % edge_k
                if upper_idx is None:
                    upper_idx = torch.triu_indices(n, n, 1, device=device)
                i = upper_idx[0, pair_idx]
                j = upper_idx[1, pair_idx]
                cur = et[i, j]
                new_t = type_idx + (type_idx >= cur).long()
                et[i, j] = new_t
                et[j, i] = new_t

        return nt, et

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
            return ProposalResult(prop_nodes=[], prop_edges=[], log_q_fwd=None, moves=None)

        # Gradients drive the greedy edits (one eval per batch).
        with torch.enable_grad():
            with _model_eval_ctx(model):
                with _amp_autocast_ctx(device, amp_dtype):
                    _, grad_X_list, grad_E_list = energy_and_grads_batch(
                        model=model,
                        node_types_list=node_types_list,
                        edge_types_list=edge_types_list,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        apply_property_conditioner=self.use_conditioner,
                        property_lambda_override=getattr(self, "property_lambda_override", None),
                    )

        num_node_types = dataset_info.output_dims["X"]
        num_edge_types = dataset_info.output_dims["E"]

        prop_nodes: List[torch.Tensor] = []
        prop_edges: List[torch.Tensor] = []

        for nt, et, gX, gE in zip(node_types_list, edge_types_list, grad_X_list, grad_E_list):
            nt_new, et_new = self._apply_vectorized_edits(
                nt,
                et,
                gX,
                gE,
                num_node_types,
                num_edge_types,
            )
            prop_nodes.append(nt_new.detach())
            prop_edges.append(et_new.detach())

        return ProposalResult(prop_nodes=prop_nodes, prop_edges=prop_edges, log_q_fwd=None, moves=None)

    def needs_proposed_energy(self) -> bool:
        # Always accept without scoring; no proposed energies needed.
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
        prop_E: Optional[torch.Tensor],
        extra_features,
        domain_features,
        device: torch.device,
        amp_dtype: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Always accept all proposals; keep energies unchanged to avoid recomputation.
        if len(prop_result.prop_nodes) == 0:
            return torch.empty(0, dtype=torch.bool), current_E
        accept_mask = torch.ones(len(prop_result.prop_nodes), dtype=torch.bool)
        return accept_mask, current_E


class SimpleProposalV2(SimpleProposal):
    """Variant without random fill; stops once no decreasing moves remain."""

    def __init__(self, edits_per_step: int = 5, grad_eps: float = 0.0, use_conditioner: bool = True):
        super().__init__(
            edits_per_step=edits_per_step,
            grad_eps=grad_eps,
            random_fill=False,
            use_conditioner=use_conditioner,
        )


def _pack_graph_types(
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    *,
    n_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Pack variable-size categorical graphs once for device-resident warmup."""
    if len(node_types_list) != len(edge_types_list):
        raise ValueError("Node and edge graph batches must have equal length.")

    batch_size = len(node_types_list)
    node_types = torch.zeros((batch_size, n_max), device=device, dtype=torch.long)
    edge_types = torch.zeros(
        (batch_size, n_max, n_max),
        device=device,
        dtype=torch.long,
    )
    node_mask = torch.zeros((batch_size, n_max), device=device, dtype=torch.bool)
    node_counts: List[int] = []

    for batch_idx, (nodes, edges) in enumerate(zip(node_types_list, edge_types_list)):
        n = int(nodes.shape[0])
        if n > n_max:
            raise ValueError(f"Graph with {n} nodes exceeds padded size {n_max}.")
        node_counts.append(n)
        if n == 0:
            continue
        node_types[batch_idx, :n].copy_(
            nodes.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        )
        edge_types[batch_idx, :n, :n].copy_(
            edges.to(device=device, dtype=torch.long, non_blocking=True).view(n, n)
        )
        node_mask[batch_idx, :n] = True

    return node_types, edge_types, node_mask, node_counts


def _unpack_graph_types(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    node_counts: Sequence[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    nodes_out = [
        node_types[idx, :n].detach().cpu()
        for idx, n in enumerate(node_counts)
    ]
    edges_out = [
        edge_types[idx, :n, :n].detach().cpu()
        for idx, n in enumerate(node_counts)
    ]
    return nodes_out, edges_out


def apply_simple_v2_edits_batched(
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    node_mask: torch.Tensor,
    grad_X: torch.Tensor,
    grad_E: torch.Tensor,
    *,
    edits_per_step: int = 1,
    grad_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the existing simple-v2 edit rule across a padded graph batch."""
    batch_size, n_max = node_types.shape
    edits = max(int(edits_per_step), 0)
    if batch_size == 0 or n_max == 0 or edits == 0:
        return (
            node_types.clone(),
            edge_types.clone(),
            torch.zeros(batch_size, device=node_types.device, dtype=torch.bool),
        )

    device = node_types.device
    threshold = -float(grad_eps)
    inf = torch.tensor(float("inf"), device=device, dtype=grad_X.dtype)
    node_mask = node_mask.to(device=device, dtype=torch.bool)

    num_node_types = int(grad_X.shape[-1])
    if num_node_types > 1:
        current_node = node_types.unsqueeze(-1)
        current_node_grad = grad_X.gather(-1, current_node)
        node_delta = grad_X - current_node_grad
        node_delta = node_delta.scatter(
            -1,
            current_node,
            torch.full_like(current_node, float("inf"), dtype=grad_X.dtype),
        )
        node_best_delta, node_best_type = node_delta.min(dim=-1)
        node_valid = node_mask & (node_best_delta < threshold)
    else:
        node_best_delta = torch.full(
            (batch_size, n_max),
            float("inf"),
            device=device,
            dtype=grad_X.dtype,
        )
        node_best_type = torch.zeros_like(node_types)
        node_valid = torch.zeros_like(node_mask)

    num_edge_types = int(grad_E.shape[-1])
    if num_edge_types > 1 and n_max > 1:
        grad_E_sym = 0.5 * (grad_E + grad_E.transpose(1, 2))
        current_edge = edge_types.unsqueeze(-1)
        current_edge_grad = grad_E_sym.gather(-1, current_edge)
        edge_delta = grad_E_sym - current_edge_grad
        edge_delta = edge_delta.scatter(
            -1,
            current_edge,
            torch.full_like(current_edge, float("inf"), dtype=grad_E.dtype),
        )
        edge_best_delta, edge_best_type = edge_delta.min(dim=-1)
        upper_mask = torch.triu(
            torch.ones((n_max, n_max), device=device, dtype=torch.bool),
            diagonal=1,
        )
        edge_position_mask = (
            node_mask.unsqueeze(2)
            & node_mask.unsqueeze(1)
            & upper_mask.unsqueeze(0)
        )
        edge_valid = edge_position_mask & (edge_best_delta < threshold)
    else:
        edge_best_delta = torch.full(
            (batch_size, n_max, n_max),
            float("inf"),
            device=device,
            dtype=grad_E.dtype,
        )
        edge_best_type = torch.zeros_like(edge_types)
        edge_valid = torch.zeros(
            (batch_size, n_max, n_max),
            device=device,
            dtype=torch.bool,
        )

    location_delta = torch.cat(
        (node_best_delta, edge_best_delta.flatten(start_dim=1)),
        dim=1,
    )
    location_valid = torch.cat(
        (node_valid, edge_valid.flatten(start_dim=1)),
        dim=1,
    )
    masked_delta = torch.where(location_valid, location_delta, inf)
    k = min(edits, int(masked_delta.shape[1]))
    selected = torch.topk(-masked_delta, k=k, dim=1, largest=True).indices
    selected_valid = location_valid.gather(1, selected)
    changed = selected_valid.any(dim=1)

    nodes_new = node_types.clone()
    edges_new = edge_types.clone()
    batch_index = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(selected)

    selected_nodes = selected < n_max
    apply_nodes = selected_valid & selected_nodes
    node_batch = batch_index[apply_nodes]
    node_position = selected[apply_nodes]
    nodes_new[node_batch, node_position] = node_best_type[
        node_batch,
        node_position,
    ]

    apply_edges = selected_valid & ~selected_nodes
    edge_batch = batch_index[apply_edges]
    edge_flat = selected[apply_edges] - n_max
    edge_i = torch.div(edge_flat, n_max, rounding_mode="floor")
    edge_j = edge_flat % n_max
    new_edge_type = edge_best_type[edge_batch, edge_i, edge_j]
    edges_new[edge_batch, edge_i, edge_j] = new_edge_type
    edges_new[edge_batch, edge_j, edge_i] = new_edge_type

    return nodes_new, edges_new, changed


def run_simple_v2_warmup_vectorized(
    *,
    model,
    dataset_info,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    extra_features,
    domain_features,
    steps: int,
    device: torch.device,
    edits_per_step: int = 1,
    grad_eps: float = 0.0,
    amp_dtype: Optional[str] = None,
    stop_when_unchanged: bool = True,
    collect_stats: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int, Dict[str, object]]:
    """Run simple-v2 warmup with dense states and batch-wide early stopping."""
    batch_size = len(node_types_list)
    max_steps = max(int(steps), 0)
    if batch_size == 0:
        return [], [], 0, 0, {
            "steps_executed": 0,
            "stop_reason": "disabled",
            "nontrivial_moves": 0,
        }
    if max_steps == 0:
        return (
            [nodes.detach().cpu().clone() for nodes in node_types_list],
            [edges.detach().cpu().clone() for edges in edge_types_list],
            0,
            0,
            {
                "steps_executed": 0,
                "stop_reason": "disabled",
                "nontrivial_moves": 0,
            },
        )

    n_max = int(
        getattr(
            dataset_info,
            "max_n_nodes",
            max(int(nodes.shape[0]) for nodes in node_types_list),
        )
    )
    node_types, edge_types, node_mask, node_counts = _pack_graph_types(
        node_types_list,
        edge_types_list,
        n_max=n_max,
        device=device,
    )
    if collect_stats:
        origin_nodes = node_types.clone()
        origin_edges = edge_types.clone()
        edge_position_mask = (
            node_mask.unsqueeze(2)
            & node_mask.unsqueeze(1)
            & torch.triu(
                torch.ones((n_max, n_max), device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0)
        )
        sampler_stats: Dict[str, float] = {
            "total_proposals": 0.0,
            "total_accepted": 0.0,
            "nontriv_any": 0.0,
            "nontriv_node": 0.0,
            "nontriv_edge": 0.0,
            "acc_nontriv_any": 0.0,
            "acc_nontriv_node": 0.0,
            "acc_nontriv_edge": 0.0,
            "prop_dist_nodes_sum": 0.0,
            "prop_dist_edges_sum": 0.0,
            "acc_dist_nodes_sum": 0.0,
            "acc_dist_edges_sum": 0.0,
            "step_prop_nodes_sum": 0.0,
            "step_prop_edges_sum": 0.0,
            "step_acc_nodes_sum": 0.0,
            "step_acc_edges_sum": 0.0,
            "distance_total_nodes": 0.0,
            "distance_total_edges": 0.0,
            "distance_total": 0.0,
        }
    else:
        origin_nodes = None
        origin_edges = None
        edge_position_mask = None
        sampler_stats = {}

    total_moves = 0
    total_attempts = 0
    steps_executed = 0
    stop_reason = "max"

    with torch.enable_grad():
        with _model_eval_ctx(model):
            for _ in range(max_steps):
                with _amp_autocast_ctx(device, amp_dtype):
                    _, grad_X, grad_E = energy_and_grads_dense(
                        model=model,
                        node_types=node_types,
                        edge_types=edge_types,
                        node_mask=node_mask,
                        dataset_info=dataset_info,
                        device=device,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        apply_property_conditioner=False,
                    )
                next_nodes, next_edges, changed = apply_simple_v2_edits_batched(
                    node_types,
                    edge_types,
                    node_mask,
                    grad_X,
                    grad_E,
                    edits_per_step=edits_per_step,
                    grad_eps=grad_eps,
                )
                if collect_stats:
                    assert origin_nodes is not None
                    assert origin_edges is not None
                    assert edge_position_mask is not None
                    node_step = (next_nodes != node_types) & node_mask
                    edge_step = (next_edges != edge_types) & edge_position_mask
                    node_step_count = node_step.sum(dim=1)
                    edge_step_count = edge_step.sum(dim=(1, 2))
                    nontriv_node = node_step_count > 0
                    nontriv_edge = edge_step_count > 0
                    nontriv_any = nontriv_node | nontriv_edge
                    node_origin_dist = (
                        ((next_nodes != origin_nodes) & node_mask)
                        .sum(dim=1)
                    )
                    edge_origin_dist = (
                        ((next_edges != origin_edges) & edge_position_mask)
                        .sum(dim=(1, 2))
                    )

                    proposals_this_step = float(batch_size)
                    nontriv_any_total = float(nontriv_any.sum().item())
                    nontriv_node_total = float(nontriv_node.sum().item())
                    nontriv_edge_total = float(nontriv_edge.sum().item())
                    node_origin_total = float(node_origin_dist.sum().item())
                    edge_origin_total = float(edge_origin_dist.sum().item())
                    node_step_total = float(node_step_count.sum().item())
                    edge_step_total = float(edge_step_count.sum().item())
                    sampler_stats["total_proposals"] += proposals_this_step
                    sampler_stats["total_accepted"] += proposals_this_step
                    sampler_stats["nontriv_any"] += nontriv_any_total
                    sampler_stats["nontriv_node"] += nontriv_node_total
                    sampler_stats["nontriv_edge"] += nontriv_edge_total
                    sampler_stats["acc_nontriv_any"] += nontriv_any_total
                    sampler_stats["acc_nontriv_node"] += nontriv_node_total
                    sampler_stats["acc_nontriv_edge"] += nontriv_edge_total
                    sampler_stats["prop_dist_nodes_sum"] += node_origin_total
                    sampler_stats["prop_dist_edges_sum"] += edge_origin_total
                    sampler_stats["acc_dist_nodes_sum"] += node_origin_total
                    sampler_stats["acc_dist_edges_sum"] += edge_origin_total
                    sampler_stats["step_prop_nodes_sum"] += node_step_total
                    sampler_stats["step_prop_edges_sum"] += edge_step_total
                    sampler_stats["step_acc_nodes_sum"] += node_step_total
                    sampler_stats["step_acc_edges_sum"] += edge_step_total
                    sampler_stats["distance_total_nodes"] += node_step_total
                    sampler_stats["distance_total_edges"] += edge_step_total
                    sampler_stats["distance_total"] += node_step_total + edge_step_total

                node_types = next_nodes
                edge_types = next_edges
                moves_this_step = int(changed.sum().item())
                total_moves += moves_this_step
                total_attempts += batch_size
                steps_executed += 1
                if stop_when_unchanged and moves_this_step == 0:
                    stop_reason = "stuck"
                    break

    nodes_out, edges_out = _unpack_graph_types(
        node_types,
        edge_types,
        node_counts,
    )
    stats: Dict[str, object] = {
        "steps_executed": steps_executed,
        "stop_reason": stop_reason,
        "nontrivial_moves": total_moves,
        "total_proposals": total_attempts,
    }
    if collect_stats:
        stats.update(sampler_stats)
    return nodes_out, edges_out, total_moves, total_attempts, stats


class SimpleProposalV2GuidedStrong(SimpleProposalV2):
    """Guided V2 proposal that requires a property lambda override."""

    supports_property_lambda_override = True

    def __init__(
        self,
        edits_per_step: int = 5,
        grad_eps: float = 0.0,
        require_lambda_override: bool = True,
    ):
        super().__init__(edits_per_step=edits_per_step, grad_eps=grad_eps, use_conditioner=True)
        self.property_lambda_override: Optional[float] = None
        self.require_lambda_override = bool(require_lambda_override)

    def set_property_lambda_override(self, value: Optional[float]) -> None:
        self.property_lambda_override = None if value is None else float(value)

    def propose(self, *args, **kwargs) -> ProposalResult:
        if self.require_lambda_override and self.property_lambda_override is None:
            raise ValueError(
                "simple_ver2_guided_strong requires a property lambda override. "
                "Set chain_warmup.property_lambda_prop when using this proposal."
            )
        return super().propose(*args, **kwargs)
