"""
Flow-matching and interpolation helpers to keep scripts lean.

Contains utilities for sampling interpolated graphs, computing gradient
strengths, mean/std helpers, and evaluating energies along an interpolation
path using the model.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


def sample_interpolated_graph(
    nt_noise: torch.Tensor,
    et_noise: torch.Tensor,
    nt_data: torch.Tensor,
    et_data: torch.Tensor,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a discrete graph along the interpolation path at time tau.

    - tau=0 => pure noise graph
    - tau=1 => pure data graph
    - in-between => per-node/edge Bernoulli(tau) pick from data vs noise

    Ensures edge symmetry and copies diagonal from data (usually zeros).
    """
    assert nt_noise.shape == nt_data.shape
    assert et_noise.shape == et_data.shape
    n = int(nt_noise.shape[0])
    if n == 0:
        return nt_noise.clone(), et_noise.clone()

    device = nt_noise.device

    # Nodes: independent Bernoulli per node
    node_mask = (torch.rand(n, device=device) < tau)
    nt = torch.where(node_mask, nt_data, nt_noise)

    # Edges: sample only upper triangle, then mirror; diagonal from data
    et = et_noise.clone()
    if et.numel() > 0:
        triu_i, triu_j = torch.triu_indices(n, n, offset=1, device=device)
        m = triu_i.numel()
        edge_mask = (torch.rand(m, device=device) < tau)
        chosen = torch.where(edge_mask, et_data[triu_i, triu_j], et_noise[triu_i, triu_j])
        et[triu_i, triu_j] = chosen
        et[triu_j, triu_i] = chosen
        # diagonal from data (usually zeros)
        diag = torch.arange(n, device=device)
        et[diag, diag] = et_data[diag, diag]

    return nt, et


def grad_scalar_strength(gX: torch.Tensor, gE: torch.Tensor) -> float:
    """Compute a scalar gradient strength from grads wrt one-hot nodes/edges.

    Uses per-element L2, averaged over all active nodes and i<j edges.
    Returns a Python float detached from graph.
    """
    n = int(gX.shape[0])
    # Nodes: norm over class dim, then sum
    node_norms = torch.linalg.vector_norm(gX, ord=2, dim=-1)  # (n,)
    node_sum = float(node_norms.sum().item())
    node_cnt = n

    # Edges: norm over class dim, keep upper triangle
    if gE.numel() > 0 and n > 1:
        e_norm = torch.linalg.vector_norm(gE, ord=2, dim=-1)  # (n,n)
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=e_norm.device), diagonal=1)
        e_vals = e_norm[mask]
        edge_sum = float(e_vals.sum().item())
        edge_cnt = int(e_vals.numel())
    else:
        edge_sum = 0.0
        edge_cnt = 0

    denom = max(node_cnt + edge_cnt, 1)
    return (node_sum + edge_sum) / float(denom)


def mean_std(x: torch.Tensor) -> Tuple[float, float]:
    if x.numel() == 0:
        return 0.0, 0.0
    mu = float(x.mean().item())
    std = float(x.std(unbiased=False).item())
    return mu, std


def evaluate_interpolation_path(
    noise_nodes: Sequence[torch.Tensor],
    noise_edges: Sequence[torch.Tensor],
    data_nodes: Sequence[torch.Tensor],
    data_edges: Sequence[torch.Tensor],
    taus: Iterable[float],
    *,
    model,
    dataset_info,
    extra_features,
    domain_features,
    device: torch.device,
    log_progress: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Evaluate model energies and grad strengths along an interpolation path.

    Note: This function computes input gradients; avoid wrapping it in
    a global no_grad/inference context so autograd can build graphs.
    Returns four lists (energy_means, energy_stds, grad_means, grad_stds).
    """
    from gem import sampler  # lazy import to avoid circularity at module load

    assert len(noise_nodes) == len(noise_edges) == len(data_nodes) == len(data_edges)
    pairs = list(zip(noise_nodes, noise_edges, data_nodes, data_edges))

    taus_list = list(taus)
    n_points = len(taus_list)

    energy_means: List[float] = []
    energy_stds: List[float] = []
    grad_means: List[float] = []
    grad_stds: List[float] = []

    for idx, tau in enumerate(taus_list):
        it_nodes: List[torch.Tensor] = []
        it_edges: List[torch.Tensor] = []
        for nt_n, et_n, nt_d, et_d in pairs:
            nt_i, et_i = sample_interpolated_graph(nt_n, et_n, nt_d, et_d, float(tau))
            it_nodes.append(nt_i)
            it_edges.append(et_i)

        # Evaluate energy and grads (detached; no autograd graph)
        E_tau, gX_list, gE_list = sampler.energy_and_grads_batch(
            model=model,
            node_types_list=it_nodes,
            edge_types_list=it_edges,
            dataset_info=dataset_info,
            device=device,
            extra_features=extra_features,
            domain_features=domain_features,
            create_graph=False,
            detach_energies=True,
            detach_grads=True,
        )

        mu_t, sd_t = mean_std(E_tau)
        energy_means.append(mu_t)
        energy_stds.append(sd_t)

        strengths = []
        for gX, gE in zip(gX_list, gE_list):
            strengths.append(grad_scalar_strength(-gX, -gE))
        if strengths:
            s_t = torch.tensor(strengths, dtype=torch.float32)
            grad_means.append(float(s_t.mean().item()))
            grad_stds.append(float(s_t.std(unbiased=False).item()))
        else:
            grad_means.append(0.0)
            grad_stds.append(0.0)

        if log_progress and n_points > 0:
            every = max(1, n_points // 10)
            if (idx + 1) % every == 0:
                print(
                    f"  tau[{idx+1}/{n_points}]={float(tau):.3f} | E mean={energy_means[-1]:.4f} std={energy_stds[-1]:.4f} | "
                    f"||-grad E|| mean={grad_means[-1]:.4f} std={grad_stds[-1]:.4f}"
                )

    return energy_means, energy_stds, grad_means, grad_stds
