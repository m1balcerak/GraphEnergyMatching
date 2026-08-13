"""Utilities for OT diagnostics: timing, distances, and simple pairwise stats."""

from __future__ import annotations

from typing import Dict, List, Tuple

import time
import torch
import torch.nn.functional as F


# ----------------------------
# Small timing utilities
# ----------------------------

def t_now() -> float:
    return time.perf_counter()


def add_time(dst: Dict[str, float] | None, key: str, dt: float) -> None:
    if dst is None:
        return
    dst[key] = dst.get(key, 0.0) + float(dt)


def print_timing_summary(title: str, tdict: Dict[str, float]):
    if not tdict:
        return
    print(f"\n[timing] {title}")
    keys = sorted(tdict.keys())
    total = 0.0
    for k in keys:
        v = float(tdict[k])
        total += v
        unit = "s"
        if v < 1e-3:
            v *= 1e6
            unit = "µs"
        elif v < 1.0:
            v *= 1e3
            unit = "ms"
        print(f"  {k:<35s} {v:9.3f} {unit}")
    print(f"  {'total':<35s} {total:9.3f} s\n")


# ----------------------------
# Distances within a graph pair
# ----------------------------

def one_hot_distance_pair(
    nt_a: torch.Tensor,
    et_a: torch.Tensor,
    nt_b: torch.Tensor,
    et_b: torch.Tensor,
    dx: int,
    de: int,
) -> Tuple[float, float, float]:
    """Compute naive one-hot distance between two graphs (no permutations).

    Pads to the larger n with a special 'pad' class to make one-hot comparable.
    Counts edges only for i<j (upper triangle) to avoid double counting.
    Returns (total_distance, node_distance, edge_distance).
    """
    assert nt_a.dim() == 1 and nt_b.dim() == 1
    assert et_a.dim() == 2 and et_b.dim() == 2

    device = nt_a.device
    n1 = int(nt_a.shape[0]); n2 = int(nt_b.shape[0])
    n = max(n1, n2)

    # Nodes
    pad_node = dx
    nt_a_pad = torch.full((n,), pad_node, dtype=torch.long, device=device)
    nt_b_pad = torch.full((n,), pad_node, dtype=torch.long, device=device)
    nt_a_pad[:n1] = nt_a
    nt_b_pad[:n2] = nt_b
    Xa = F.one_hot(nt_a_pad, num_classes=dx + 1).to(torch.float32)
    Xb = F.one_hot(nt_b_pad, num_classes=dx + 1).to(torch.float32)
    node_dist = float((Xa - Xb).abs().sum().item() / 2.0)

    # Edges (upper triangle only, ignore diagonal)
    pad_edge = de
    Ea_idx = torch.full((n, n), pad_edge, dtype=torch.long, device=device)
    Eb_idx = torch.full((n, n), pad_edge, dtype=torch.long, device=device)
    Ea_idx[:n1, :n1] = et_a
    Eb_idx[:n2, :n2] = et_b
    Ea = F.one_hot(Ea_idx, num_classes=de + 1).to(torch.float32)
    Eb = F.one_hot(Eb_idx, num_classes=de + 1).to(torch.float32)
    triu = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    diff_edges = (Ea - Eb).abs().sum(dim=-1)
    edge_dist = float(diff_edges[triu].sum().item() / 2.0)

    total = node_dist + edge_dist
    return total, node_dist, edge_dist


def _stats(distances: List[float]) -> Tuple[float, float]:
    if not distances:
        return 0.0, 0.0
    t = torch.tensor(distances, dtype=torch.float32)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def print_stats(label: str, totals: List[float], nodes: List[float], edges: List[float]):
    m_total, s_total = _stats(totals)
    m_nodes, s_nodes = _stats(nodes)
    m_edges, s_edges = _stats(edges)
    print(
        f"[{label}] total: {m_total:.4f} ± {s_total:.4f} | "
        f"nodes: {m_nodes:.4f} ± {s_nodes:.4f} | "
        f"edges: {m_edges:.4f} ± {s_edges:.4f}"
    )


# -----------------------------
# Pairwise (index-aligned) util
# -----------------------------

def pairwise_stats(
    A_nodes: List[torch.Tensor],
    A_edges: List[torch.Tensor],
    B_nodes: List[torch.Tensor],
    B_edges: List[torch.Tensor],
    dx: int,
    de: int,
):
    """Compute pairwise distances between A[i] and B[i] only.

    Only compares pairs with the same number of nodes; mismatched pairs are skipped.
    """
    assert len(A_nodes) == len(A_edges)
    assert len(B_nodes) == len(B_edges)
    n = min(len(A_nodes), len(B_nodes))

    totals: List[float] = []
    nodes: List[float] = []
    edges: List[float] = []
    for i in range(n):
        if int(A_nodes[i].shape[0]) != int(B_nodes[i].shape[0]):
            continue
        t, dn, de_ = one_hot_distance_pair(
            A_nodes[i], A_edges[i], B_nodes[i], B_edges[i], dx, de
        )
        totals.append(t)
        nodes.append(dn)
        edges.append(de_)
    return totals, nodes, edges
