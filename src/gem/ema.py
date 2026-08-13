from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Device-resident exponential moving average of a module state."""

    def __init__(self, module: nn.Module, decay: float):
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}.")
        self.decay = decay
        self.num_updates = 0
        self.shadow = self._clone_state(module.state_dict())

    @staticmethod
    def _clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in state.items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        current = module.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("EMA and model state keys do not match.")
        one_minus_decay = 1.0 - self.decay
        for name, value in current.items():
            shadow_value = self.shadow[name]
            value = value.detach().to(
                device=shadow_value.device,
                dtype=shadow_value.dtype,
            )
            if torch.is_floating_point(shadow_value) or torch.is_complex(shadow_value):
                shadow_value.mul_(self.decay).add_(value, alpha=one_minus_decay)
            else:
                shadow_value.copy_(value)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        module.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": float(self.decay),
            "num_updates": int(self.num_updates),
            "state_dict": self._clone_state(self.shadow),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("EMA checkpoint is missing state_dict.")
        if state.keys() != self.shadow.keys():
            raise RuntimeError("EMA checkpoint and model state keys do not match.")
        for name, value in state.items():
            if not torch.is_tensor(value):
                raise TypeError(f"EMA state '{name}' is not a tensor.")
            self.shadow[name].copy_(
                value.detach().to(
                    device=self.shadow[name].device,
                    dtype=self.shadow[name].dtype,
                )
            )
        self.num_updates = int(payload.get("num_updates", 0) or 0)

    @contextmanager
    def average_parameters(self, module: nn.Module) -> Iterator[None]:
        online_state = self._clone_state(module.state_dict())
        self.copy_to(module)
        try:
            yield
        finally:
            module.load_state_dict(online_state, strict=True)
