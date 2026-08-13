"""Matching and OT pairing utilities for OT diagnostics.

Includes Hungarian solvers (square and rectangular), cost builders, and pairing
methods. Relies on gem.ot_utils for timing and low-level distances.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from gem.ot_utils import (
    t_now as _t,
    add_time as _add_time,
    one_hot_distance_pair as _one_hot_distance_pair,
)

# Prefer POT if available for node-level Hungarian (square LAP); SciPy as fallback.
try:  # pragma: no cover - optional path
    import ot as _pot  # type: ignore
    _HAVE_POT = True
except Exception:  # pragma: no cover - optional path
    _pot = None
    _HAVE_POT = False

from scipy.optimize import linear_sum_assignment as _hungarian_scipy


# --------------------------------------
# Node-level Hungarian (square LAP)
# --------------------------------------

def _hungarian_match(cost: torch.Tensor) -> torch.Tensor:
    """Solve the linear assignment on a 2D square cost matrix using Hungarian.

    Returns a permutation vector `perm` such that column `perm[i]` is assigned to row `i`.
    Uses POT if available, otherwise SciPy's reference implementation.
    """
    assert cost.dim() == 2 and cost.shape[0] == cost.shape[1]
    C = cost.detach().cpu().to(torch.float64).numpy()
    try:
        if _HAVE_POT and hasattr(_pot, "optim") and hasattr(_pot.optim, "hungarian"):
            row_ind, col_ind = _pot.optim.hungarian(C)  # type: ignore[attr-defined]
        else:
            raise AttributeError
    except Exception:
        row_ind, col_ind = _hungarian_scipy(C)
    order = row_ind.argsort()
    col_ind = col_ind[order]
    perm = torch.tensor(col_ind, dtype=torch.long)
    return perm


def pairwise_stats_node_matching(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    dx: int,
    de: int,
):
    """Compute distances after node-matching permutation on B per pair (aligned by index)."""
    assert len(A_nodes) == len(A_edges)
    assert len(B_nodes) == len(B_edges)
    n_pairs = min(len(A_nodes), len(B_nodes))

    totals: List[float] = []
    nodes: List[float] = []
    edges: List[float] = []

    for i in range(n_pairs):
        nt_a = A_nodes[i]
        nt_b = B_nodes[i]
        et_a = A_edges[i]
        et_b = B_edges[i]

        n_a = int(nt_a.shape[0])
        n_b = int(nt_b.shape[0])
        if n_a != n_b or n_a == 0:
            continue

        n = n_a
        # Primary cost: 0 if labels equal, 1 otherwise
        eq = (nt_a.view(-1, 1).expand(n, n) == nt_b.view(1, -1).expand(n, n))
        cost = (~eq).to(torch.float32)

        # Tie-breaker using edge-structure histograms
        num_edge_classes = int(de)

        def edge_histograms(
            E: torch.Tensor,
            edge_classes: int = num_edge_classes,
        ) -> torch.Tensor:
            nloc = int(E.shape[0])
            counts = torch.stack(
                [(E == t).to(torch.float32).sum(dim=1) for t in range(edge_classes)],
                dim=1,
            )
            # remove diagonal contribution
            diag = torch.diag(E)
            diag_oh = F.one_hot(diag, num_classes=edge_classes).to(torch.float32)
            counts = counts - diag_oh
            denom = max(nloc - 1, 1)
            return counts / float(denom)

        hist_a = edge_histograms(et_a)
        hist_b = edge_histograms(et_b)
        edges_l1 = torch.cdist(hist_a, hist_b, p=1)
        cost = cost + 1e-5 * edges_l1

        # deterministic tiny jitter
        cols = torch.arange(n, dtype=torch.float32).view(1, n).expand(n, n)
        rows = torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, n)
        cost = cost + (1e-9 * cols) + (1e-11 * rows)

        perm = _hungarian_match(cost)
        nt_b_perm = nt_b[perm]
        et_b_perm = et_b[perm][:, perm]

        t, dn, de_ = _one_hot_distance_pair(nt_a, et_a, nt_b_perm, et_b_perm, dx, de)
        totals.append(t)
        nodes.append(dn)
        edges.append(de_)

    return totals, nodes, edges


# ------------------------------------------
# Minibatch OT matching between graphs
# ------------------------------------------

def _bucket_by_size(N_list: List[torch.Tensor]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for idx, nt in enumerate(N_list):
        n = int(nt.shape[0])
        if n not in buckets:
            buckets[n] = []
        buckets[n].append(idx)
    return buckets


def _pair_index_map(dx: int) -> torch.Tensor:
    """Create a (dx, dx) map giving the index of the upper-triangular (a<=b) pair (a,b)."""
    idx_map = torch.full((dx, dx), -1, dtype=torch.long)
    running = 0
    for a in range(dx):
        for b in range(a, dx):
            idx_map[a, b] = running
            running += 1
    return idx_map


def _graph_signature_hist(
    nt: torch.Tensor,
    et: torch.Tensor,
    dx: int,
    de: int,
    idx_map: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    n = int(nt.shape[0])
    if n <= 0:
        pair_classes = dx * (dx + 1) // 2
        D = dx + de + pair_classes * de
        return torch.zeros(D, dtype=torch.float32)

    # Node histogram
    node_hist = torch.bincount(nt, minlength=dx).to(torch.float32) / max(n, 1)

    # Edge histogram over i<j
    triu_i, triu_j = torch.triu_indices(n, n, offset=1)
    et_ij = et[triu_i, triu_j]
    m = et_ij.numel()
    edge_hist = torch.bincount(et_ij, minlength=de).to(torch.float32) / max(m, 1)

    # Pair-edge histogram: symmetrize label pairs (min,max), count per edge type
    li = nt[triu_i]
    lj = nt[triu_j]
    a = torch.minimum(li, lj)
    b = torch.maximum(li, lj)
    pair_idx = idx_map[a, b]
    pair_classes = int(idx_map.max().item()) + 1
    triple_idx = pair_idx * de + et_ij
    pair_edge_hist = torch.bincount(triple_idx, minlength=pair_classes * de).to(torch.float32) / max(m, 1)

    return torch.cat([alpha * node_hist, beta * edge_hist, gamma * pair_edge_hist], dim=0)


def _bucket_signatures(
    nodes: List[torch.Tensor],
    edges: List[torch.Tensor],
    idxs: List[int],
    dx: int,
    de: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    idx_map = _pair_index_map(dx)
    sigs = []
    for k in idxs:
        sig = _graph_signature_hist(nodes[k], edges[k], dx, de, idx_map, alpha, beta, gamma)
        sigs.append(sig)
    if not sigs:
        return torch.zeros((0, dx + de + (dx * (dx + 1) // 2) * de), dtype=torch.float32)
    return torch.stack(sigs, dim=0)


def _build_cost_matrix_hist(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    A_idxs: List[int],
    B_idxs: List[int],
    dx: int,
    de: int,
    alpha: float,
    beta: float,
    gamma: float,
    t_acc: Dict[str, float] | None = None,
) -> torch.Tensor:
    t0 = _t()
    SA = _bucket_signatures(A_nodes, A_edges, A_idxs, dx, de, alpha, beta, gamma)
    _add_time(t_acc, "build_cost.sig_A", _t() - t0)

    t1 = _t()
    SB = _bucket_signatures(B_nodes, B_edges, B_idxs, dx, de, alpha, beta, gamma)
    _add_time(t_acc, "build_cost.sig_B", _t() - t1)

    if SA.numel() == 0 or SB.numel() == 0:
        return torch.zeros((SA.shape[0], SB.shape[0]), dtype=torch.float32)

    t2 = _t()
    C = torch.cdist(SA, SB, p=1).to(torch.float32)
    _add_time(t_acc, "build_cost.cdist", _t() - t2)
    return C


def _node_match_distance_single(
    nt_a: torch.Tensor, et_a: torch.Tensor, nt_b: torch.Tensor, et_b: torch.Tensor, dx: int, de: int
) -> float:
    n = int(nt_a.shape[0])
    eq = (nt_a.view(-1, 1).expand(n, n) == nt_b.view(1, -1).expand(n, n))
    cost = (~eq).to(torch.float32)

    def edge_hist(E: torch.Tensor) -> torch.Tensor:
        nloc = int(E.shape[0])
        counts = torch.stack([(E == t).to(torch.float32).sum(dim=1) for t in range(de)], dim=1)
        diag = torch.diag(E)
        diag_oh = F.one_hot(diag, num_classes=de).to(torch.float32)
        counts = counts - diag_oh
        denom = max(nloc - 1, 1)
        return counts / float(denom)

    cost = cost + 1e-5 * torch.cdist(edge_hist(et_a), edge_hist(et_b), p=1)
    cost = cost + 1e-9 * torch.arange(n, dtype=torch.float32).view(1, n).expand(n, n) \
                 + 1e-11 * torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, n)

    perm = _hungarian_match(cost)
    nt_b_perm = nt_b[perm]
    et_b_perm = et_b[perm][:, perm]
    t, _, _ = _one_hot_distance_pair(nt_a, et_a, nt_b_perm, et_b_perm, dx, de)
    return float(t)


def _build_cost_matrix_exact_naive(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    A_idxs: List[int],
    B_idxs: List[int],
    dx: int,
    de: int,
    t_acc: Dict[str, float] | None = None,
) -> torch.Tensor:
    m = len(A_idxs)
    n = len(B_idxs)
    C = torch.zeros((m, n), dtype=torch.float32)
    t0 = _t()
    for i, ia in enumerate(A_idxs):
        nt_a = A_nodes[ia]; et_a = A_edges[ia]
        for j, jb in enumerate(B_idxs):
            nt_b = B_nodes[jb]; et_b = B_edges[jb]
            if nt_a.shape[0] != nt_b.shape[0]:
                C[i, j] = float("inf")
                continue
            t, _, _ = _one_hot_distance_pair(nt_a, et_a, nt_b, et_b, dx, de)
            C[i, j] = float(t)
    _add_time(t_acc, "build_cost.exact_naive", _t() - t0)
    return C


def _build_cost_matrix_exact_node_match(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    A_idxs: List[int],
    B_idxs: List[int],
    dx: int,
    de: int,
    t_acc: Dict[str, float] | None = None,
) -> torch.Tensor:
    m = len(A_idxs)
    n = len(B_idxs)
    C = torch.zeros((m, n), dtype=torch.float32)
    t0 = _t()
    for i, ia in enumerate(A_idxs):
        nt_a = A_nodes[ia]; et_a = A_edges[ia]
        for j, jb in enumerate(B_idxs):
            nt_b = B_nodes[jb]; et_b = B_edges[jb]
            if nt_a.shape[0] != nt_b.shape[0] or nt_a.shape[0] == 0:
                C[i, j] = float("inf")
                continue
            C[i, j] = _node_match_distance_single(nt_a, et_a, nt_b, et_b, dx, de)
    _add_time(t_acc, "build_cost.exact_node_match", _t() - t0)
    return C


def _hungarian_assignment_rect(C: torch.Tensor, t_acc: Dict[str, float] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    t0 = _t()
    C_np = C.detach().cpu().to(torch.float64).numpy()
    row_ind, col_ind = _hungarian_scipy(C_np)
    _add_time(t_acc, "hungarian", _t() - t0)
    return torch.tensor(row_ind, dtype=torch.long), torch.tensor(col_ind, dtype=torch.long)


def minibatch_ot_pairs(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    dx: int,
    de: int,
    cost_mode: str = "hist",   # "hist" | "naive" | "node_match"
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    verbose: bool = True,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    A_buckets = _bucket_by_size(A_nodes)
    B_buckets = _bucket_by_size(B_nodes)

    pairs: List[Tuple[int, int]] = []
    sizes = sorted(set(A_buckets.keys()) & set(B_buckets.keys()))
    timing: Dict[str, float] = {}

    for n in sizes:
        A_idxs = A_buckets[n]
        B_idxs = B_buckets[n]
        if not A_idxs or not B_idxs:
            continue

        if cost_mode == "hist":
            C = _build_cost_matrix_hist(
                A_nodes, A_edges, B_nodes, B_edges, A_idxs, B_idxs, dx, de,
                alpha, beta, gamma, t_acc=timing
            )
        elif cost_mode == "naive":
            C = _build_cost_matrix_exact_naive(
                A_nodes, A_edges, B_nodes, B_edges, A_idxs, B_idxs, dx, de, t_acc=timing
            )
        elif cost_mode == "node_match":
            C = _build_cost_matrix_exact_node_match(
                A_nodes, A_edges, B_nodes, B_edges, A_idxs, B_idxs, dx, de, t_acc=timing
            )
        else:
            raise ValueError(f"Unknown OT cost_mode '{cost_mode}'")

        if C.numel() == 0:
            continue

        row_ind, col_ind = _hungarian_assignment_rect(C, t_acc=timing)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            pairs.append((A_idxs[r], B_idxs[c]))

        if verbose:
            print(
                f"[ot] size={n} | matched {len(row_ind)} pairs | "
                f"A_bucket={len(A_idxs)}, B_bucket={len(B_idxs)}"
            )

    return pairs, timing


def evaluate_matched_pairs(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    pairs: List[Tuple[int, int]],
    dx: int,
    de: int,
    variant: str = "naive",  # "naive" | "node_match"
) -> Tuple[List[float], List[float], List[float], float]:
    t0 = _t()
    totals: List[float] = []
    nodes: List[float] = []
    edges: List[float] = []
    for ia, jb in pairs:
        nt_a, et_a = A_nodes[ia], A_edges[ia]
        nt_b, et_b = B_nodes[jb], B_edges[jb]
        if nt_a.shape[0] != nt_b.shape[0] or nt_a.shape[0] == 0:
            continue

        if variant == "naive":
            t, dn, de_ = _one_hot_distance_pair(nt_a, et_a, nt_b, et_b, dx, de)
        elif variant == "node_match":
            n = int(nt_a.shape[0])
            eq = (nt_a.view(-1, 1).expand(n, n) == nt_b.view(1, -1).expand(n, n))
            cost = (~eq).to(torch.float32)

            def edge_hist(E: torch.Tensor) -> torch.Tensor:
                nloc = int(E.shape[0])
                counts = torch.stack([(E == t2).to(torch.float32).sum(dim=1) for t2 in range(de)], dim=1)
                diag = torch.diag(E)
                diag_oh = F.one_hot(diag, num_classes=de).to(torch.float32)
                counts = counts - diag_oh
                denom = max(nloc - 1, 1)
                return counts / float(denom)

            cost = cost + 1e-5 * torch.cdist(edge_hist(et_a), edge_hist(et_b), p=1)
            cost = cost + 1e-9 * torch.arange(n, dtype=torch.float32).view(1, n).expand(n, n) \
                         + 1e-11 * torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, n)
            perm = _hungarian_match(cost)
            nt_b_perm = nt_b[perm]
            et_b_perm = et_b[perm][:, perm]
            t, dn, de_ = _one_hot_distance_pair(nt_a, et_a, nt_b_perm, et_b_perm, dx, de)
        else:
            raise ValueError(f"Unknown variant '{variant}'")

        totals.append(float(t))
        nodes.append(float(dn))
        edges.append(float(de_))
    return totals, nodes, edges, (_t() - t0)


def random_pairs_by_bucket(
    A_nodes: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    t0 = _t()
    A_b = _bucket_by_size(A_nodes)
    B_b = _bucket_by_size(B_nodes)
    sizes = sorted(set(A_b.keys()) & set(B_b.keys()))
    pairs: List[Tuple[int, int]] = []
    for n in sizes:
        A_idxs = A_b[n]
        B_idxs = B_b[n]
        if not A_idxs or not B_idxs:
            continue
        k = min(len(A_idxs), len(B_idxs))
        a_sel = random.sample(A_idxs, k)
        b_sel = random.sample(B_idxs, k)
        for ai, bi in zip(a_sel, b_sel):
            pairs.append((ai, bi))
    return pairs, {"bucket_rand.pairing": (_t() - t0)}
