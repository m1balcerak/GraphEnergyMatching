from typing import Any, List, Sequence, Tuple
from contextlib import nullcontext
import torch
import torch.nn.functional as F

_PROPERTY_TIME_OVERRIDE: float | None = None


def register_property_conditioner(conditioner: Any | None) -> None:
    if conditioner is not None:
        raise NotImplementedError(
            "Property conditioning is not included in this slim MOSES release."
        )


def get_property_conditioner() -> None:
    return None


def register_property_time_override(t_value: float | None) -> None:
    global _PROPERTY_TIME_OVERRIDE
    _PROPERTY_TIME_OVERRIDE = None if t_value is None else float(t_value)


def get_property_time_override() -> float | None:
    return _PROPERTY_TIME_OVERRIDE


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

    # Use a static maximum number of nodes to stabilize compiled graphs
    n_max = int(getattr(dataset_info, "max_n_nodes", max(int(nt.shape[0]) for nt in node_types_list)))

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
    apply_property_conditioner: bool = True,
    t: torch.Tensor | float | None = None,
    property_lambda_override: float | None = None,
) -> torch.Tensor:
    """Batched energy for a list of graphs.

    Set `apply_property_conditioner=False` to obtain the base/prior energy without
    any conditional penalties (useful for diagnostics/visualizations).
    """
    X_input, E_input, y_input, node_mask = build_batched_inputs(
        node_types_list, edge_types_list, dataset_info, device, extra_features, domain_features
    )
    # Ensure grad mode is consistent regardless of outer no_grad contexts
    with torch.set_grad_enabled(not detach):
        _, energy = model(X_input, E_input, y_input, node_mask, return_energy=True)  # (B,) or (B,1)
    energy = energy.view(-1)

    if apply_property_conditioner:
        conditioner = get_property_conditioner()
        if conditioner is not None:
            if t is None:
                t = get_property_time_override()
            penalty = conditioner.penalty_from_graphs(
                node_types_list,
                edge_types_list,
                device=device,
                requires_grad=not detach,
                t=t,
                lambda_override=property_lambda_override,
            )
            energy = conditioner.lambda_energy * energy + penalty

    return energy.detach() if detach else energy


def _energy_and_grads_one_hot(
    model,
    X_pad: torch.Tensor,
    E_pad: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
    *,
    t: torch.Tensor | float | None = None,
    apply_property_conditioner: bool = True,
    property_lambda_override: float | None = None,
    create_graph: bool = False,
    detach_energies: bool = True,
    detach_grads: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute energies and dense gradients from padded one-hot graph states."""
    B, n_max = X_pad.shape[:2]
    y_dim = dataset_info.output_dims["y"]

    # Require gradients on fresh one-hot leaves only.
    X_pad = X_pad.detach().requires_grad_(True)
    E_pad = E_pad.detach().requires_grad_(True)
    node_mask = node_mask.to(device=device, dtype=torch.float32)

    y = torch.zeros((B, y_dim), device=device)
    # Preserve the existing sampler behavior: energy gradients are evaluated at t=0.
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

    with torch.set_grad_enabled(True):
        _, energy = model(X_input, E_input, y_input, node_mask, return_energy=True)  # (B,)
    energy = energy.view(-1)

    conditioner = get_property_conditioner() if apply_property_conditioner else None
    if conditioner is not None:
        if t is None:
            t = get_property_time_override()
        penalty = conditioner.penalty_from_one_hot(
            X_pad,
            E_pad,
            node_mask,
            requires_grad=True,
            t=t,
            lambda_override=property_lambda_override,
        )
        total_energy = conditioner.lambda_energy * energy + penalty
    else:
        total_energy = energy

    gX, gE = torch.autograd.grad(
        total_energy.sum(), (X_pad, E_pad), retain_graph=create_graph, create_graph=create_graph
    )

    energies = total_energy.detach() if detach_energies else total_energy
    if detach_grads:
        gX = gX.detach()
        gE = gE.detach()
    return energies, gX, gE


def energy_and_grads_dense(
    model,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
    *,
    t: torch.Tensor | float | None = None,
    apply_property_conditioner: bool = True,
    property_lambda_override: float | None = None,
    create_graph: bool = False,
    detach_energies: bool = True,
    detach_grads: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute energies and gradients for padded categorical graph tensors."""
    num_node_types = int(dataset_info.output_dims["X"])
    num_edge_types = int(dataset_info.output_dims["E"])
    node_types = node_types.to(device=device, dtype=torch.long, non_blocking=True)
    edge_types = edge_types.to(device=device, dtype=torch.long, non_blocking=True)
    node_mask = node_mask.to(device=device, dtype=torch.float32, non_blocking=True)

    X_pad = F.one_hot(node_types, num_classes=num_node_types).to(torch.float32)
    X_pad = X_pad * node_mask.unsqueeze(-1)
    edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
    E_pad = F.one_hot(edge_types, num_classes=num_edge_types).to(torch.float32)
    E_pad = E_pad * edge_mask.unsqueeze(-1)

    return _energy_and_grads_one_hot(
        model=model,
        X_pad=X_pad,
        E_pad=E_pad,
        node_mask=node_mask,
        dataset_info=dataset_info,
        device=device,
        extra_features=extra_features,
        domain_features=domain_features,
        t=t,
        apply_property_conditioner=apply_property_conditioner,
        property_lambda_override=property_lambda_override,
        create_graph=create_graph,
        detach_energies=detach_energies,
        detach_grads=detach_grads,
    )


def energy_and_grads_batch(
    model,
    node_types_list: Sequence[torch.Tensor],
    edge_types_list: Sequence[torch.Tensor],
    dataset_info,
    device: torch.device,
    extra_features,
    domain_features,
    *,
    t: torch.Tensor | float | None = None,
    apply_property_conditioner: bool = True,
    property_lambda_override: float | None = None,
    create_graph: bool = False,
    detach_energies: bool = True,
    detach_grads: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Compute energies and one-hot input gradients for variable-size graphs."""
    assert len(node_types_list) == len(edge_types_list)
    B = len(node_types_list)
    num_node_types = dataset_info.output_dims["X"]
    num_edge_types = dataset_info.output_dims["E"]
    n_max = int(
        getattr(
            dataset_info,
            "max_n_nodes",
            max(int(nt.shape[0]) for nt in node_types_list),
        )
    )

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

    energies, gX, gE = _energy_and_grads_one_hot(
        model=model,
        X_pad=X_pad,
        E_pad=E_pad,
        node_mask=node_mask,
        dataset_info=dataset_info,
        device=device,
        extra_features=extra_features,
        domain_features=domain_features,
        t=t,
        apply_property_conditioner=apply_property_conditioner,
        property_lambda_override=property_lambda_override,
        create_graph=create_graph,
        detach_energies=detach_energies,
        detach_grads=detach_grads,
    )

    grad_X_list: List[torch.Tensor] = []
    grad_E_list: List[torch.Tensor] = []
    for b, n in enumerate(n_list):
        gx_b = gX[b, :n, :]
        ge_b = gE[b, :n, :n, :]
        grad_X_list.append(gx_b)
        grad_E_list.append(ge_b)
    return energies, grad_X_list, grad_E_list
