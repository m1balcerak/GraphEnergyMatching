from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def model_state_from_checkpoint(ckpt: Any, *, use_ema: bool = False) -> Any:
    """Extract online or EMA model weights from supported checkpoint formats."""
    if use_ema:
        if not isinstance(ckpt, Mapping):
            raise KeyError("EMA weights require a wrapped checkpoint.")
        ema_payload = ckpt.get("ema")
        if isinstance(ema_payload, Mapping):
            state = ema_payload.get("state_dict")
            if isinstance(state, Mapping):
                return state
        ema_state = ckpt.get("ema_model")
        if isinstance(ema_state, Mapping):
            return ema_state
        raise KeyError("Checkpoint does not contain EMA weights.")

    if (
        isinstance(ckpt, Mapping)
        and "state_dict" in ckpt
        and not any(str(key).startswith("model.") for key in ckpt.keys())
    ):
        return ckpt["state_dict"]
    if isinstance(ckpt, Mapping) and ckpt and all(
        str(key).startswith("model.") for key in ckpt.keys()
    ):
        return {
            str(key)[len("model.") :]: value
            for key, value in ckpt.items()
            if str(key).startswith("model.")
        }
    if isinstance(ckpt, Mapping) and any(
        key in ckpt for key in ("model", "net", "weights")
    ):
        return ckpt.get("model", ckpt.get("net", ckpt.get("weights", {})))
    return ckpt


def load_checkpoint(path: str, *, map_location: torch.device | str) -> Any:
    """Load a trusted GEM checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_checkpoint(
    model: nn.Module,
    path: str,
    *,
    map_location: torch.device | str,
    use_ema: bool = False,
) -> Any:
    """Load a checkpoint and require an exact match with ``model``."""
    checkpoint = load_checkpoint(path, map_location=map_location)
    state = model_state_from_checkpoint(checkpoint, use_ema=use_ema)
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"Checkpoint '{path}' does not contain a model state dictionary.")
    model.load_state_dict(state, strict=True)
    return checkpoint
