# sampler_energy.py, do not delete this line
from typing import List, Sequence, Tuple
from contextlib import nullcontext
import torch
import torch.nn.functional as F


def _no_autocast_ctx(device: torch.device):
    # Ensure numerically sensitive feature engineering stays in FP32
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return nullcontext()


def build_batched_inputs(
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
):
    """Build padded batched inputs for a list of graphs (variable #nodes)."""
    assert len(node_types_list) == len(edge_types_list)
    B = len(node_types_list)
    if B == 0:
        raise ValueError("Empty batch for energy computation.")

    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]
    y_dim = dataset_info.output_dims["y"]

    n_max = max(int(nt.shape[0]) for nt in node_types_list)

    X_pad = torch.zeros((B, n_max, num_node_types), device=device, dtype=torch.float32)
    E_pad = torch.zeros((B, n_max, n_max, num_edge_types), device=device, dtype=torch.float32)
    node_mask = torch.zeros((B, n_max), device=device, dtype=torch.float32)

    # Pack graphs
    for b, (nt, et) in enumerate(zip(node_types_list, edge_types_list)):
        n = int(nt.shape[0])
        if n == 0:
            continue
        nt = nt.to(device, non_blocking=True).long().view(-1)
        et = et.to(device, non_blocking=True).long().view(n, n)

        X_pad[b, :n] = F.one_hot(nt, num_classes=num_node_types).to(torch.float32)
        E_pad[b, :n, :n] = F.one_hot(et, num_classes=num_edge_types).to(torch.float32)
        node_mask[b, :n] = 1.0

    y = torch.zeros((B, y_dim), device=device)
    t = torch.zeros((B, 1), device=device)

    noisy_data = {
        "X_t": X_pad,
        "E_t": E_pad,
        "y_t": y,
        "node_mask": node_mask,
        "t": t,
    }

    # Extra/domain features in **FP32** regardless of surrounding autocast
    with _no_autocast_ctx(device):
        extra_feat = extra_features(noisy_data)
        extra_mol_feat = domain_features(noisy_data)

    extra_X = torch.cat((extra_feat.X, extra_mol_feat.X), dim=-1)
    extra_E = torch.cat((extra_feat.E, extra_mol_feat.E), dim=-1)
    extra_y = torch.cat((extra_feat.y, extra_mol_feat.y), dim=-1)
    extra_y = torch.cat((extra_y, t), dim=1)

    # Final inputs (pad to expected dims if needed)
    X_input = torch.cat((X_pad, extra_X), dim=2)
    E_input = torch.cat((E_pad, extra_E), dim=3)
    y_input = torch.cat((y, extra_y), dim=1)

    if X_input.shape[-1] < dataset_info.input_dims["X"]:
        pad = dataset_info.input_dims["X"] - X_input.shape[-1]
        X_input = torch.cat((X_input, torch.zeros(B, n_max, pad, device=device)), dim=-1)
    if E_input.shape[-1] < dataset_info.input_dims["E"]:
        pad = dataset_info.input_dims["E"] - E_input.shape[-1]
        E_input = torch.cat((E_input, torch.zeros(B, n_max, n_max, pad, device=device)), dim=-1)
    if y_input.shape[-1] < dataset_info.input_dims["y"]:
        pad = dataset_info.input_dims["y"] - y_input.shape[-1]
        y_input = torch.cat((y_input, torch.zeros(B, pad, device=device)), dim=-1)

    return X_input, E_input, y_input, node_mask


def energy_batch(
    model,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
    detach: bool = True,
) -> torch.Tensor:
    """Batched energy for a list of graphs."""
    X_input, E_input, y_input, node_mask = build_batched_inputs(
        node_types_list, edge_types_list, dataset_info, device, extra_features, domain_features
    )
    _, energy = model(X_input, E_input, y_input, node_mask, return_energy=True)  # (B,) or (B,1)
    energy = energy.view(-1)
    return energy.detach() if detach else energy


def energy_and_grads_batch(
    model,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Compute energies and one-hot input gradients.

    Returns
    -------
    energies : (B,) float tensor (detached)
    grad_X_list : list of (n_b, num_node_types) tensors (on device)
    grad_E_list : list of (n_b, n_b, num_edge_types) tensors (on device)
    """
    assert len(node_types_list) == len(edge_types_list)
    B = len(node_types_list)
    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]
    y_dim = dataset_info.output_dims["y"]

    n_max = max(int(nt.shape[0]) for nt in node_types_list)

    X_pad = torch.zeros((B, n_max, num_node_types), device=device, dtype=torch.float32)
    E_pad = torch.zeros((B, n_max, n_max, num_edge_types), device=device, dtype=torch.float32)
    node_mask = torch.zeros((B, n_max), device=device, dtype=torch.float32)

    n_list = []
    for b, (nt, et) in enumerate(zip(node_types_list, edge_types_list)):
        n = int(nt.shape[0])
        n_list.append(n)
        if n == 0:
            continue
        nt = nt.to(device, non_blocking=True).long().view(-1)
        et = et.to(device, non_blocking=True).long().view(n, n)
        X_pad[b, :n].copy_(F.one_hot(nt, num_classes=num_node_types).to(torch.float32))
        E_pad[b, :n, :n].copy_(F.one_hot(et, num_classes=num_edge_types).to(torch.float32))
        node_mask[b, :n] = 1.0

    # Require grads on one-hot bases only
    X_pad.requires_grad_(True)
    E_pad.requires_grad_(True)

    y = torch.zeros((B, y_dim), device=device)
    t = torch.zeros((B, 1), device=device)

    # Extra features built from no-grad copies in **FP32**
    noisy_data_ng = {
        "X_t": X_pad.detach(),
        "E_t": E_pad.detach(),
        "y_t": y,
        "node_mask": node_mask,
        "t": t,
    }
    with _no_autocast_ctx(device):
        extra_feat = extra_features(noisy_data_ng)
        extra_mol_feat = domain_features(noisy_data_ng)

    extra_X = torch.cat((extra_feat.X, extra_mol_feat.X), dim=-1).detach()
    extra_E = torch.cat((extra_feat.E, extra_mol_feat.E), dim=-1).detach()
    extra_y = torch.cat((extra_feat.y, extra_mol_feat.y), dim=-1).detach()
    extra_y = torch.cat((extra_y, t), dim=1)

    X_input = torch.cat((X_pad, extra_X), dim=2)
    E_input = torch.cat((E_pad, extra_E), dim=3)
    y_input = torch.cat((y, extra_y), dim=1)

    if X_input.shape[-1] < dataset_info.input_dims["X"]:
        pad = dataset_info.input_dims["X"] - X_input.shape[-1]
        X_input = torch.cat((X_input, torch.zeros(B, n_max, pad, device=device)), dim=-1)
    if E_input.shape[-1] < dataset_info.input_dims["E"]:
        pad = dataset_info.input_dims["E"] - E_input.shape[-1]
        E_input = torch.cat((E_input, torch.zeros(B, n_max, n_max, pad, device=device)), dim=-1)
    if y_input.shape[-1] < dataset_info.input_dims["y"]:
        pad = dataset_info.input_dims["y"] - y_input.shape[-1]
        y_input = torch.cat((y_input, torch.zeros(B, pad, device=device)), dim=-1)

    _, energy = model(X_input, E_input, y_input, node_mask, return_energy=True)  # (B,)
    energy = energy.view(-1)

    gX, gE = torch.autograd.grad(energy.sum(), (X_pad, E_pad), retain_graph=False, create_graph=False)

    energies = energy.detach()
    grad_X_list: List[torch.Tensor] = []
    grad_E_list: List[torch.Tensor] = []
    for b, n in enumerate(n_list):
        grad_X_list.append(gX[b, :n, :].detach())
        grad_E_list.append(gE[b, :n, :n, :].detach())
    return energies, grad_X_list, grad_E_list
