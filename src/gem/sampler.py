from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import torch

from gem.flow_matching.noise_distribution import NoiseDistribution
from gem.flow_matching import flow_matching_utils

# Re-export energy utilities for external use
from .sampler_energy import (
    energy_and_grads_batch,
    energy_batch,
    register_property_time_override,
)

# Proposals
from .proposals.base import Proposal
from .proposals.random_proposal import RandomProposal
from .proposals.gwd_proposal import GWDProposal
from .proposals.gwg_block_proposal import GWGBlockProposal
from .proposals.dlangevin_proposal import (
    DLangevinProposal,
    DLangevinVectorizedProposal,
    DLangevinNoMHProposal,
    DLangevinMTProposal,
    DLangevinAnnealingProposal,
    DLangevinTwoBetasProposal,
    DLangevinTwoBetasVectorizedProposal,
    DLangevinTwoBetasAnnealingProposal,
    DLangevinTwoBetasAnnealingVectorizedProposal,
    DLangevinTwoBetasAnnealingVectorizedNoOriginProposal,
)
from .proposals.simple_proposal import (
    SimpleProposal,
    SimpleProposalV2,
    SimpleProposalV2GuidedStrong,
    run_simple_v2_warmup_vectorized,
)
from .dlangevin_utils import (
    TWO_BETA_PROPOSALS,
    TWO_BETA_SCALAR_PROPOSALS,
    TWO_BETA_VECTOR_PROPOSALS,
    TWO_BETA_ANNEALING_PROPOSALS,
    TWO_BETA_ANNEALING_VECTOR_PROPOSALS,
)

_VECTORIZED_SIMPLE_WARMUP_PROPOSALS = frozenset({"simple_ver2", "simple_v2"})


@dataclass(frozen=True)
class MCMCStepEvent:
    """Read-only views of an MCMC step for diagnostics."""

    step: int
    nodes: Sequence[torch.Tensor]
    edges: Sequence[torch.Tensor]
    energies: torch.Tensor
    accepted: torch.Tensor
    proposed_nodes: Sequence[torch.Tensor]
    proposed_edges: Sequence[torch.Tensor]
    proposed_energies: Optional[torch.Tensor]


def _amp_autocast_ctx(device: torch.device, amp_dtype: Optional[str] = None):
    """Autocast context controlled by desired AMP dtype.

    If `amp_dtype` is None or empty, returns a no-op context to keep FP32.
    Otherwise, uses the requested dtype ("fp16"/"float16" or "bf16"/"bfloat16").
    """
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


def should_vectorize_simple_warmup(
    proposal: str,
    *,
    vectorized: bool = True,
) -> bool:
    """Return whether a configured warmup can use the dense simple-v2 runner."""
    return bool(vectorized) and str(proposal or "").strip().lower() in (
        _VECTORIZED_SIMPLE_WARMUP_PROPOSALS
    )


def make_proposal(method: str, **kwargs) -> Proposal:
    """Factory for proposal mechanisms."""
    m = (method or "").lower()
    if m == "random":
        return RandomProposal()
    if m == "gwd":
        beta = float(kwargs.get("gwd_beta", 1.0))
        return GWDProposal(beta=beta)
    if m in {"gwg", "gwg_block", "gibbs_grad"}:
        beta = float(kwargs.get("gwd_beta", 1.0))
        return GWGBlockProposal(beta=beta)
    if m in {"dlangevin", "dlang", "dl"}:
        beta = float(kwargs.get("dl_beta", 1.0))
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinProposal(
            beta=beta,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in {"dlangevin_vec", "dlangevin_vectorized", "dlang_vec", "dl_vec"}:
        beta = float(kwargs.get("dl_beta", 1.0))
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinVectorizedProposal(
            beta=beta,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in {"dlangevin_mt", "dlangevinmt", "dlang_mt", "dl_mt", "dlangevin_multi"}:
        beta = float(kwargs.get("dl_beta", 1.0))
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        num_tries_raw = (
            kwargs.get("dl_num_tries", None)
            or kwargs.get("dl_mt_k", None)
            or kwargs.get("dl_k", None)
        )
        energy_batch_size_raw = (
            kwargs.get("dl_mt_energy_batch", None)
            or kwargs.get("dl_energy_chunk", None)
        )
        try:
            num_tries = int(num_tries_raw) if num_tries_raw is not None else 1
        except (TypeError, ValueError):
            num_tries = 1
        try:
            energy_batch_size = int(energy_batch_size_raw) if energy_batch_size_raw is not None else 256
        except (TypeError, ValueError):
            energy_batch_size = 256
        return DLangevinMTProposal(
            beta=beta,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
            num_tries=num_tries,
            energy_batch_size=energy_batch_size,
        )
    if m in TWO_BETA_SCALAR_PROPOSALS:
        beta_prop = kwargs.get("dl_beta_prop")
        beta_mh = kwargs.get("dl_beta_mh")
        if beta_prop is None or beta_mh is None:
            raise ValueError("dlangevintwobetas requires dl_beta_prop and dl_beta_mh to be set.")
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinTwoBetasProposal(
            beta_prop=float(beta_prop),
            beta_mh=float(beta_mh),
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in TWO_BETA_VECTOR_PROPOSALS:
        beta_prop = kwargs.get("dl_beta_prop")
        beta_mh = kwargs.get("dl_beta_mh")
        if beta_prop is None or beta_mh is None:
            raise ValueError(
                "dlangevin_two_betas_vec requires dl_beta_prop and dl_beta_mh to be set."
            )
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinTwoBetasVectorizedProposal(
            beta_prop=float(beta_prop),
            beta_mh=float(beta_mh),
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in {"dlangevin_annealing", "dlang_annealing", "dl_annealing"}:
        beta_init = float(kwargs.get("dl_beta_init", kwargs.get("dl_beta", 1.0)))
        beta_final = float(kwargs.get("dl_beta_final", beta_init))
        anneal_steps_raw = kwargs.get("dl_beta_anneal_steps", kwargs.get("dl_anneal_steps", 0))
        try:
            anneal_steps = max(int(anneal_steps_raw), 0)
        except (TypeError, ValueError):
            anneal_steps = 0
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinAnnealingProposal(
            beta_init=beta_init,
            beta_final=beta_final,
            anneal_steps=anneal_steps,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in TWO_BETA_ANNEALING_PROPOSALS:
        beta_prop = kwargs.get("dl_beta_prop")
        beta_mh_init = kwargs.get("dl_beta_mh_init")
        beta_mh_final = kwargs.get("dl_beta_mh_final")
        beta_mh_anneal_steps = kwargs.get("dl_beta_mh_anneal_steps")
        missing = [
            k
            for k, v in [
                ("dl_beta_prop", beta_prop),
                ("dl_beta_mh_init", beta_mh_init),
                ("dl_beta_mh_final", beta_mh_final),
                ("dl_beta_mh_anneal_steps", beta_mh_anneal_steps),
            ]
            if v is None
        ]
        if missing:
            raise ValueError(f"dlangevintwobetas_annealing requires {', '.join(missing)}.")
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        if m == "dlangevin_two_betas_annealing_vec_no_origin":
            proposal_cls = DLangevinTwoBetasAnnealingVectorizedNoOriginProposal
        elif m in TWO_BETA_ANNEALING_VECTOR_PROPOSALS:
            proposal_cls = DLangevinTwoBetasAnnealingVectorizedProposal
        else:
            proposal_cls = DLangevinTwoBetasAnnealingProposal
        return proposal_cls(
            beta_prop=float(beta_prop),
            beta_mh_init=float(beta_mh_init),
            beta_mh_final=float(beta_mh_final),
            beta_mh_anneal_steps=int(beta_mh_anneal_steps),
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m in {"dlangevin_nomh", "dlang_nomh", "dl_nomh"}:
        beta = float(kwargs.get("dl_beta", 1.0))
        lambda_X = float(kwargs.get("dl_lambda_X", 1.0))
        lambda_E = float(kwargs.get("dl_lambda_E", 1.0))
        return DLangevinNoMHProposal(
            beta=beta,
            lambda_X=lambda_X,
            lambda_E=lambda_E,
        )
    if m == "simple":
        edits = kwargs.get("simple_n_edits", 5)
        edits_int = 5 if edits is None else int(edits)
        return SimpleProposal(edits_per_step=edits_int, use_conditioner=True)
    if m in {"simple_ver2", "simple_v2"}:
        edits = kwargs.get("simple_n_edits", 5)
        edits_int = 5 if edits is None else int(edits)
        return SimpleProposalV2(edits_per_step=edits_int, use_conditioner=False)
    if m in {"simple_ver2_guided", "simple_v2_guided"}:
        edits = kwargs.get("simple_n_edits", 5)
        edits_int = 5 if edits is None else int(edits)
        return SimpleProposalV2(edits_per_step=edits_int, use_conditioner=True)
    if m in {"simple_ver2_guided_strong", "simple_v2_guided_strong"}:
        edits = kwargs.get("simple_n_edits", 5)
        edits_int = 5 if edits is None else int(edits)
        return SimpleProposalV2GuidedStrong(edits_per_step=edits_int)
    raise ValueError(
        f"Unknown proposal '{method}'. Supported: ['random', 'gwd', 'gwg_block', 'dlangevin', 'dlangevin_vec', 'dlangevin_mt', 'dlangevin_annealing', 'dlangevintwobetas', 'dlangevin_two_betas_vec', 'dlangevintwobetas_annealing', 'dlangevin_two_betas_annealing_vec', 'dlangevin_two_betas_annealing_vec_no_origin', 'dlangevin_noMH', 'simple', 'simple_ver2', 'simple_ver2_guided', 'simple_ver2_guided_strong']."
    )


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


@contextmanager
def _model_eval_ctx(model: torch.nn.Module):
    """
    Temporarily set the model to eval() and restore the previous training mode.
    Used to disable dropout for the entire MCMC section.
    """
    prev = model.training
    try:
        if prev:
            model.eval()
        yield
    finally:
        if prev:
            model.train()


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
    gwd_beta: float = 1.0,
    dl_beta: float = 1.0,
    dl_beta_init: Optional[float] = None,
    dl_beta_final: Optional[float] = None,
    dl_beta_anneal_steps: Optional[int] = None,
    dl_beta_prop: Optional[float] = None,
    dl_beta_mh: Optional[float] = None,
    dl_beta_mh_init: Optional[float] = None,
    dl_beta_mh_final: Optional[float] = None,
    dl_beta_mh_anneal_steps: Optional[int] = None,
    dl_lambda_X: float = 1.0,
    dl_lambda_E: float = 1.0,
    dl_num_tries: Optional[int] = None,
    dl_mt_energy_batch: Optional[int] = None,
    simple_n_edits: Optional[int] = None,
    energy_split_threshold: Optional[float] = None,
    dl_beta_near: Optional[float] = None,
    dl_lambda_X_near: Optional[float] = None,
    dl_lambda_E_near: Optional[float] = None,
    dl_beta_far: Optional[float] = None,
    dl_lambda_X_far: Optional[float] = None,
    dl_lambda_E_far: Optional[float] = None,
    amp_dtype: Optional[str] = None,
    collect_stats: bool = False,
    step_offset: int = 0,
    property_lambda_prop: Optional[float] = None,
    property_t_override: Optional[float] = None,
    property_t_schedule: Optional[Dict[str, Any]] = None,
    step_callback: Optional[Callable[[MCMCStepEvent], None]] = None,
):
    """Run parallel MCMC chains (one per graph) starting from the provided batch.

    Returns
    -------
    tuple
        (node_types_list, edge_types_list, n_accept_total, n_steps_total)

    Notes on performance:
    - Chain states are moved to `device` once here and kept device-resident
      throughout the loop. We only move them back to CPU on return.
    - Model is put into eval() for the entire MCMC to disable dropout.
    - If provided, `step_callback` is called after each state update with
      device-resident accepted and proposed states. Callers must clone data
      they retain.
    """
    assert len(node_types_list) == len(edge_types_list)
    B = len(node_types_list)
    if B == 0:
        return [], [], 0, 0, {}

    try:
        step_offset_int = int(step_offset)
    except (TypeError, ValueError):
        step_offset_int = 0

    # --- Move chain states to device ONCE, keep them there during the loop ---
    node_types_list = [t.to(device, non_blocking=True) for t in node_types_list]
    edge_types_list = [t.to(device, non_blocking=True) for t in edge_types_list]

    # Keep a copy of chain origins for distance-to-origin stats
    if collect_stats:
        origin_nodes = [t.clone() for t in node_types_list]
        origin_edges = [t.clone() for t in edge_types_list]
    else:
        origin_nodes = []
        origin_edges = []

    dual_params = (
        energy_split_threshold is not None
        and dl_beta_near is not None
        and dl_lambda_X_near is not None
        and dl_lambda_E_near is not None
        and dl_beta_far is not None
        and dl_lambda_X_far is not None
        and dl_lambda_E_far is not None
    )
    if dual_params and proposal.lower() not in {"dlangevin", "dlang", "dl"}:
        raise ValueError(
            "Dual DLangevin parameters require proposal='dlangevin'."
        )
    if dual_params and step_callback is not None:
        raise ValueError(
            "step_callback is not supported with dual DLangevin parameters."
        )
    if proposal.lower() in TWO_BETA_PROPOSALS:
        if dl_beta_prop is None or dl_beta_mh is None:
            raise ValueError(
                f"proposal='{proposal}' requires dl_beta_prop and dl_beta_mh."
            )
    if proposal.lower() in TWO_BETA_ANNEALING_PROPOSALS:
        missing = [
            k
            for k, v in [
                ("dl_beta_prop", dl_beta_prop),
                ("dl_beta_mh_init", dl_beta_mh_init),
                ("dl_beta_mh_final", dl_beta_mh_final),
                ("dl_beta_mh_anneal_steps", dl_beta_mh_anneal_steps),
            ]
            if v is None
        ]
        if missing:
            raise ValueError(f"dlangevintwobetas_annealing requires {', '.join(missing)}.")

    total_accepts = 0

    # Stats accumulators for single-parameter mode
    stats: Dict[str, Any] = {}
    if collect_stats and not dual_params:
        stats.update(
            dict(
                total_proposals=0,
                total_accepted=0,
                nontriv_any=0,
                nontriv_node=0,
                nontriv_edge=0,
                acc_nontriv_any=0,
                acc_nontriv_node=0,
                acc_nontriv_edge=0,
                # Distances to origin accumulators (proposed and accepted)
                prop_dist_nodes_sum=0.0,
                prop_dist_edges_sum=0.0,
                acc_dist_nodes_sum=0.0,
                acc_dist_edges_sum=0.0,
                # Step size accumulators (current -> proposed)
                step_prop_nodes_sum=0.0,
                step_prop_edges_sum=0.0,
                step_acc_nodes_sum=0.0,
                step_acc_edges_sum=0.0,
                step_acc_total_sq_sum=0.0,
            )
        )

    track_distance = collect_stats
    total_distance_nodes = 0.0
    total_distance_edges = 0.0
    B_initial = B

    prop_impl: Optional[Proposal] = None
    if not dual_params:
        beta_init = dl_beta if dl_beta_init is None else float(dl_beta_init)
        beta_final = dl_beta if dl_beta_final is None else float(dl_beta_final)
        try:
            anneal_steps_val = None if dl_beta_anneal_steps is None else int(dl_beta_anneal_steps)
        except (TypeError, ValueError):
            anneal_steps_val = None
        prop_impl = make_proposal(
            proposal,
            gwd_beta=gwd_beta,
            dl_beta=dl_beta,
            dl_beta_init=beta_init,
            dl_beta_final=beta_final,
            dl_beta_anneal_steps=anneal_steps_val,
            dl_beta_prop=dl_beta_prop,
            dl_beta_mh=dl_beta_mh,
            dl_beta_mh_init=dl_beta_mh_init,
            dl_beta_mh_final=dl_beta_mh_final,
            dl_beta_mh_anneal_steps=dl_beta_mh_anneal_steps,
            dl_lambda_X=dl_lambda_X,
            dl_lambda_E=dl_lambda_E,
            dl_num_tries=dl_num_tries,
            dl_mt_energy_batch=dl_mt_energy_batch,
            simple_n_edits=simple_n_edits,
        )
        if (
            property_lambda_prop is not None
            and prop_impl is not None
            and bool(getattr(prop_impl, "supports_property_lambda_override", False))
        ):
            prop_impl.set_property_lambda_override(property_lambda_prop)
    def _resolve_property_t(step_idx: int) -> Optional[float]:
        if property_t_override is not None:
            try:
                return float(property_t_override)
            except (TypeError, ValueError):
                return None
        if not property_t_schedule:
            return None
        mode = str(property_t_schedule.get("mode", "linear")).strip().lower()
        if mode not in {"linear", "lin"}:
            return None
        try:
            steps_total = int(property_t_schedule.get("steps", 0) or 0)
            t_start = float(property_t_schedule.get("t_start", 0.0))
            t_end = float(property_t_schedule.get("t_end", 1.0))
        except (TypeError, ValueError):
            return None
        if steps_total <= 1:
            return t_end
        max_idx = max(steps_total - 1, 1)
        s = min(max(int(step_idx), 0), max_idx)
        ratio = float(s) / float(max_idx)
        return float(t_start + (t_end - t_start) * ratio)

    def _set_property_t(step_idx: int) -> None:
        register_property_time_override(_resolve_property_t(step_idx))

    # Disable dropout for the entire MCMC run (restored on exit)
    with _model_eval_ctx(model):
        try:
            _set_property_t(0)
            # Current energies for all graphs (detached for acceptance decisions)
            # Use autocast only if requested by caller
            with _amp_autocast_ctx(device, amp_dtype):
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
            if not dual_params:
                assert prop_impl is not None
                for step_idx in range(steps):
                    _set_property_t(step_idx)
                    prop_impl.on_step_start(step_offset_int + step_idx)
                    prev_nodes = [t.clone() for t in node_types_list] if track_distance else None
                    prev_edges = [t.clone() for t in edge_types_list] if track_distance else None
                    # === Propose ===
                    prop_result = prop_impl.propose(
                        model=model,
                        dataset_info=dataset_info,
                        node_types_list=node_types_list,
                        edge_types_list=edge_types_list,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        device=device,
                        amp_dtype=amp_dtype,
                    )

                    # === Score proposals if needed by the proposal ===
                    prop_E: Optional[torch.Tensor] = None
                    if prop_impl.needs_proposed_energy():
                        with _amp_autocast_ctx(device, amp_dtype):
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

                    # === Accept/Reject (early-reject + caching inside) ===
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
                        amp_dtype=amp_dtype,
                    )

                    # ---- Stats accumulation (use pre-update current state) ----
                    if collect_stats:
                        B_local = len(node_types_list)
                        stats["total_proposals"] += B_local

                        # Loop per-chain due to variable sizes
                        for b in range(B_local):
                            nt_cur = node_types_list[b]
                            et_cur = edge_types_list[b]
                            nt_prop = prop_result.prop_nodes[b]
                            et_prop = prop_result.prop_edges[b]

                            # Step change (current -> proposed)
                            # Nodes
                            h_node_step = int((nt_prop != nt_cur).sum().item())
                            # Edges: count only i<j to avoid double count
                            if et_prop.numel() == 0:
                                h_edge_step = 0
                            else:
                                diff = et_prop != et_cur
                                h_edge_step = int(torch.triu(diff, diagonal=1).sum().item())

                            nontriv_node = (h_node_step > 0)
                            nontriv_edge = (h_edge_step > 0)
                            nontriv_any = (nontriv_node or nontriv_edge)

                            stats["nontriv_node"] += int(nontriv_node)
                            stats["nontriv_edge"] += int(nontriv_edge)
                            stats["nontriv_any"] += int(nontriv_any)

                            acc_b = bool(accept_mask[b].item())
                            if acc_b:
                                stats["total_accepted"] += 1
                            if acc_b and nontriv_any:
                                stats["acc_nontriv_any"] += 1
                            if acc_b and nontriv_node:
                                stats["acc_nontriv_node"] += 1
                            if acc_b and nontriv_edge:
                                stats["acc_nontriv_edge"] += 1

                            # Accumulate step sizes (current -> proposed)
                            stats["step_prop_nodes_sum"] += float(h_node_step)
                            stats["step_prop_edges_sum"] += float(h_edge_step)
                            if acc_b:
                                stats["step_acc_nodes_sum"] += float(h_node_step)
                                stats["step_acc_edges_sum"] += float(h_edge_step)
                                stats["step_acc_total_sq_sum"] += float(
                                    (h_node_step + h_edge_step) ** 2
                                )

                            # Distances to origin
                            nt0 = origin_nodes[b]
                            et0 = origin_edges[b]
                            prop_dist_nodes = int((nt_prop != nt0).sum().item())
                            if et_prop.numel() == 0:
                                prop_dist_edges = 0
                            else:
                                diff0 = et_prop != et0
                                prop_dist_edges = int(torch.triu(diff0, diagonal=1).sum().item())
                            stats["prop_dist_nodes_sum"] += float(prop_dist_nodes)
                            stats["prop_dist_edges_sum"] += float(prop_dist_edges)

                            if acc_b:
                                stats["acc_dist_nodes_sum"] += float(prop_dist_nodes)
                                stats["acc_dist_edges_sum"] += float(prop_dist_edges)

                    # Update states and energies where accepted (keep everything on device)
                    # Ensure mask device matches energies for indexing
                    accept_mask_dev = accept_mask.to(current_E.device)
                    n_acc = int(accept_mask.sum().item())
                    total_accepts += n_acc
                    if n_acc > 0:
                        acc_idx = torch.nonzero(accept_mask, as_tuple=False).flatten().tolist()
                        for i in acc_idx:
                            node_types_list[i] = prop_result.prop_nodes[i]
                            edge_types_list[i] = prop_result.prop_edges[i]

                        eff_prop_E = prop_E if prop_E is not None else prop_E_from_accept
                        assert eff_prop_E is not None, "Proposal did not provide proposed energies."
                        current_E[accept_mask_dev] = eff_prop_E[accept_mask_dev]
                    if track_distance and prev_nodes is not None and prev_edges is not None:
                        for prev_nt, prev_et, cur_nt, cur_et in zip(prev_nodes, prev_edges, node_types_list, edge_types_list):
                            dist_nodes = int((cur_nt != prev_nt).sum().item())
                            if cur_et.numel() == 0:
                                dist_edges = 0
                            else:
                                diff_step = cur_et != prev_et
                                dist_edges = int(torch.triu(diff_step, diagonal=1).sum().item())
                            total_distance_nodes += float(dist_nodes)
                            total_distance_edges += float(dist_edges)
                    if step_callback is not None:
                        step_callback(
                            MCMCStepEvent(
                                step=step_offset_int + step_idx,
                                nodes=node_types_list,
                                edges=edge_types_list,
                                energies=current_E,
                                accepted=accept_mask,
                                proposed_nodes=prop_result.prop_nodes,
                                proposed_edges=prop_result.prop_edges,
                                proposed_energies=(
                                    prop_E
                                    if prop_E is not None
                                    else prop_E_from_accept
                                ),
                            )
                        )
            else:
                threshold_val = float(energy_split_threshold)

                def _run_subset(
                    mask: torch.Tensor,
                    beta_val: float,
                    lambda_X_val: float,
                    lambda_E_val: float,
                ) -> int:
                    nonlocal current_E, node_types_list, edge_types_list

                    if mask is None:
                        return 0
                    idx = torch.nonzero(mask, as_tuple=False).flatten()
                    if idx.numel() == 0:
                        return 0

                    idx_list = idx.tolist()
                    sub_nodes = [node_types_list[i] for i in idx_list]
                    sub_edges = [edge_types_list[i] for i in idx_list]

                    prop_local = DLangevinProposal(
                        beta=beta_val,
                        lambda_X=lambda_X_val,
                        lambda_E=lambda_E_val,
                    )

                    prop_res = prop_local.propose(
                        model=model,
                        dataset_info=dataset_info,
                        node_types_list=sub_nodes,
                        edge_types_list=sub_edges,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        device=device,
                        amp_dtype=amp_dtype,
                    )

                    prop_E_local: Optional[torch.Tensor] = None
                    if prop_local.needs_proposed_energy():
                        with _amp_autocast_ctx(device, amp_dtype):
                            prop_E_local = energy_batch(
                                model=model,
                                node_types_list=prop_res.prop_nodes,
                                edge_types_list=prop_res.prop_edges,
                                dataset_info=dataset_info,
                                device=device,
                                extra_features=extra_features,
                                domain_features=domain_features,
                                detach=True,
                            )

                    sub_idx = idx.to(current_E.device)
                    current_E_subset = current_E.index_select(0, sub_idx)
                    accept_mask_local, prop_E_updated = prop_local.accept(
                        model=model,
                        dataset_info=dataset_info,
                        current_nodes=sub_nodes,
                        current_edges=sub_edges,
                        prop_result=prop_res,
                        current_E=current_E_subset,
                        prop_E=prop_E_local,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        device=device,
                        amp_dtype=amp_dtype,
                    )

                    eff_prop_E_local = prop_E_local if prop_E_local is not None else prop_E_updated
                    assert eff_prop_E_local is not None, "Proposal did not provide updated energies."

                    acc_idx_local = torch.nonzero(accept_mask_local, as_tuple=False).flatten().tolist()
                    for rel_idx in acc_idx_local:
                        global_idx = idx_list[rel_idx]
                        node_types_list[global_idx] = prop_res.prop_nodes[rel_idx]
                        edge_types_list[global_idx] = prop_res.prop_edges[rel_idx]

                    current_E.index_copy_(0, sub_idx, eff_prop_E_local)
                    return int(accept_mask_local.sum().item())

                for step_idx in range(steps):
                    _set_property_t(step_idx)
                    prev_nodes = [t.clone() for t in node_types_list] if track_distance else None
                    prev_edges = [t.clone() for t in edge_types_list] if track_distance else None
                    mask_near = current_E < threshold_val
                    mask_far = ~mask_near

                    total_accepts += _run_subset(
                        mask_near,
                        float(dl_beta_near),
                        float(dl_lambda_X_near),
                        float(dl_lambda_E_near),
                    )
                    total_accepts += _run_subset(
                        mask_far,
                        float(dl_beta_far),
                        float(dl_lambda_X_far),
                        float(dl_lambda_E_far),
                    )
                    if track_distance and prev_nodes is not None and prev_edges is not None:
                        for prev_nt, prev_et, cur_nt, cur_et in zip(prev_nodes, prev_edges, node_types_list, edge_types_list):
                            dist_nodes = int((cur_nt != prev_nt).sum().item())
                            if cur_et.numel() == 0:
                                dist_edges = 0
                            else:
                                diff_step = cur_et != prev_et
                                dist_edges = int(torch.triu(diff_step, diagonal=1).sum().item())
                            total_distance_nodes += float(dist_nodes)
                            total_distance_edges += float(dist_edges)
        finally:
            register_property_time_override(None)

    # Return CPU tensors to keep external API stable
    # Finalize stats means
    if collect_stats and stats.get("total_proposals", 0) > 0:
        total_props = max(int(stats["total_proposals"]), 1)
        total_acc = max(int(stats["total_accepted"]), 0)
        n_any = max(int(stats["nontriv_any"]), 0)
        n_node = max(int(stats["nontriv_node"]), 0)
        n_edge = max(int(stats["nontriv_edge"]), 0)
        mean_step_acc_total = (
            (stats["step_acc_nodes_sum"] + stats["step_acc_edges_sum"])
            / max(total_acc, 1)
            if total_acc > 0
            else 0.0
        )
        variance_step_acc_total = (
            max(
                stats["step_acc_total_sq_sum"] / total_acc
                - mean_step_acc_total**2,
                0.0,
            )
            if total_acc > 0
            else 0.0
        )

        stats.update(
            dict(
                overall_accept=(stats["total_accepted"] / total_props),
                accept_nontrivial_any=(stats["acc_nontriv_any"] / n_any) if n_any > 0 else 0.0,
                accept_nontrivial_node=(stats["acc_nontriv_node"] / n_node) if n_node > 0 else 0.0,
                accept_nontrivial_edge=(stats["acc_nontriv_edge"] / n_edge) if n_edge > 0 else 0.0,
                mean_prop_distance_nodes=(stats["prop_dist_nodes_sum"] / total_props),
                mean_prop_distance_edges=(stats["prop_dist_edges_sum"] / total_props),
                mean_prop_distance_total=((stats["prop_dist_nodes_sum"] + stats["prop_dist_edges_sum"]) / total_props),
                mean_acc_distance_nodes=(stats["acc_dist_nodes_sum"] / max(total_acc, 1)) if total_acc > 0 else 0.0,
                mean_acc_distance_edges=(stats["acc_dist_edges_sum"] / max(total_acc, 1)) if total_acc > 0 else 0.0,
                mean_acc_distance_total=((stats["acc_dist_nodes_sum"] + stats["acc_dist_edges_sum"]) / max(total_acc, 1)) if total_acc > 0 else 0.0,
                # Mean step sizes per proposal and per accepted
                mean_step_distance_nodes=(stats["step_prop_nodes_sum"] / total_props),
                mean_step_distance_edges=(stats["step_prop_edges_sum"] / total_props),
                mean_step_distance_total=((stats["step_prop_nodes_sum"] + stats["step_prop_edges_sum"]) / total_props),
                mean_step_acc_distance_nodes=(stats["step_acc_nodes_sum"] / max(total_acc, 1)) if total_acc > 0 else 0.0,
                mean_step_acc_distance_edges=(stats["step_acc_edges_sum"] / max(total_acc, 1)) if total_acc > 0 else 0.0,
                mean_step_acc_distance_total=mean_step_acc_total,
                std_step_acc_distance_total=variance_step_acc_total**0.5,
            )
        )

    distance_total_nodes = float(total_distance_nodes)
    distance_total_edges = float(total_distance_edges)
    distance_total = distance_total_nodes + distance_total_edges
    steps_effective = max(int(steps), 1)
    chains_effective = max(int(B_initial), 1)
    distance_per_step = distance_total / steps_effective if steps_effective > 0 else 0.0
    denom = steps_effective * chains_effective
    distance_per_chain_step = distance_total / denom if denom > 0 else 0.0
    distance_summary = dict(
        distance_total=distance_total,
        distance_total_nodes=distance_total_nodes,
        distance_total_edges=distance_total_edges,
        distance_per_step=distance_per_step,
        distance_per_chain_step=distance_per_chain_step,
    )
    stats.update(distance_summary)

    node_types_cpu = [t.detach().cpu() for t in node_types_list]
    edge_types_cpu = [t.detach().cpu() for t in edge_types_list]
    return node_types_cpu, edge_types_cpu, total_accepts, steps * B, (stats if collect_stats else {})
