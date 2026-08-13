"""
Plotting helpers for OT energy evaluation and FM diagnostics.

All functions are side-effect free besides writing image files. They raise on
errors, so callers can handle exceptions (keeping compatibility with existing
try/except blocks in scripts).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _to_numpy1d(x: Iterable) -> "object":
    """Convert a 1D-like iterable (list/torch/tensor/numpy) to a numpy array."""
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required for plotting helpers") from e

    # Torch tensor support without importing torch at module level
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    # Numpy array or array-like
    try:
        return np.asarray(list(x), dtype=float)
    except Exception:
        # Best-effort fallback
        return np.array(list(x), dtype=float)


def plot_fm_loss(
    iterations: Sequence[int],
    loss_mean: Sequence[float],
    loss_std: Sequence[float],
    *,
    out_path: str,
    show_std: bool = False,
    sigma_k: float = 3.0,
    dpi: int = 150,
    title: str = "FM training loss vs. iteration",
):
    import matplotlib.pyplot as plt

    its = list(iterations)
    lmean = _to_numpy1d(loss_mean)
    lstd = _to_numpy1d(loss_std)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (mean per selected pair)")
    ax.plot(its, lmean, color="C0", label="loss")
    if show_std:
        ax.fill_between(its, lmean - sigma_k * lstd, lmean + sigma_k * lstd, color="C0", alpha=0.2, label=f"±{sigma_k}σ")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_fm_ediff(
    iterations: Sequence[int],
    ediff_mean: Sequence[float],
    ediff_std: Sequence[float],
    *,
    out_path: str,
    sigma_k: float = 3.0,
    dpi: int = 150,
    title: str = "E_data - E_noise vs. iteration",
):
    import matplotlib.pyplot as plt

    its = list(iterations)
    edmean = _to_numpy1d(ediff_mean)
    edstd = _to_numpy1d(ediff_std)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy difference (mean per selected pair)")
    ax.plot(its, edmean, color="C1", label="ΔE mean")
    ax.fill_between(its, edmean - sigma_k * edstd, edmean + sigma_k * edstd, color="C1", alpha=0.2, label=f"±{sigma_k}σ")
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_interpolation_energy(
    taus: Iterable[float],
    energy_means: Sequence[float],
    energy_stds: Sequence[float],
    *,
    mu_noise: float,
    sd_noise: float,
    mu_data: float,
    sd_data: float,
    out_path: str,
    sigma_k: float = 3.0,
    dpi: int = 150,
    title: str = "GEM energy vs. interpolation time (t=0 in model)",
):
    import matplotlib.pyplot as plt

    t_np = _to_numpy1d(taus)
    m_np = _to_numpy1d(energy_means)
    s_np = _to_numpy1d(energy_stds)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_xlabel("interpolation time (0=noise, 1=data)")
    ax.set_ylabel("energy")

    ax.plot(t_np, m_np, label="interp mean", color="C0")
    ax.fill_between(t_np, m_np - sigma_k * s_np, m_np + sigma_k * s_np, color="C0", alpha=0.2, label=f"interp ±{sigma_k}σ")

    # Noise/data reference bands
    ax.axhline(mu_noise, color="C1", linestyle="--", label="noise mean")
    ax.fill_between([0, 1], mu_noise - sigma_k * sd_noise, mu_noise + sigma_k * sd_noise, color="C1", alpha=0.15, label=f"noise ±{sigma_k}σ")

    ax.axhline(mu_data, color="C2", linestyle="--", label="data mean")
    ax.fill_between([0, 1], mu_data - sigma_k * sd_data, mu_data + sigma_k * sd_data, color="C2", alpha=0.15, label=f"data ±{sigma_k}σ")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_interpolation_grad(
    taus: Iterable[float],
    grad_means: Sequence[float],
    grad_stds: Sequence[float],
    *,
    out_path: str,
    sigma_k: float = 3.0,
    dpi: int = 150,
    title: str = "||-∇_x E|| vs. interpolation time (t=0 in model)",
):
    import matplotlib.pyplot as plt

    t_np = _to_numpy1d(taus)
    gm_np = _to_numpy1d(grad_means)
    gs_np = _to_numpy1d(grad_stds)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_xlabel("interpolation time (0=noise, 1=data)")
    ax.set_ylabel("gradient strength (mean per node/edge)")

    ax.plot(t_np, gm_np, label="||-∇E|| mean", color="C3")
    ax.fill_between(t_np, gm_np - sigma_k * gs_np, gm_np + sigma_k * gs_np, color="C3", alpha=0.2, label=f"±{sigma_k}σ")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

