from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from gem.checkpoint_utils import model_state_from_checkpoint
from gem.ema import ExponentialMovingAverage


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.register_buffer("counter", torch.tensor(1, dtype=torch.long))


class ExponentialMovingAverageTest(unittest.TestCase):
    def _model(self, weight: float = 1.0) -> TinyModel:
        model = TinyModel()
        with torch.no_grad():
            model.linear.weight.fill_(weight)
        return model

    def test_update_averages_floating_state_and_copies_integer_buffers(self):
        model = self._model(1.0)
        ema = ExponentialMovingAverage(model, decay=0.5)

        with torch.no_grad():
            model.linear.weight.fill_(3.0)
            model.counter.fill_(7)
        ema.update(model)

        torch.testing.assert_close(
            ema.shadow["linear.weight"],
            torch.full_like(model.linear.weight, 2.0),
        )
        self.assertEqual(int(ema.shadow["counter"].item()), 7)
        self.assertEqual(ema.num_updates, 1)

    def test_state_round_trip_preserves_shadow_and_update_count(self):
        source = self._model(1.0)
        ema = ExponentialMovingAverage(source, decay=0.9)
        with torch.no_grad():
            source.linear.weight.fill_(5.0)
        ema.update(source)

        restored = ExponentialMovingAverage(self._model(-3.0), decay=0.9)
        restored.load_state_dict(ema.state_dict())

        self.assertEqual(restored.num_updates, 1)
        for name in ema.shadow:
            torch.testing.assert_close(restored.shadow[name], ema.shadow[name])

    def test_average_parameters_restores_online_weights(self):
        model = self._model(1.0)
        ema = ExponentialMovingAverage(model, decay=0.5)
        with torch.no_grad():
            model.linear.weight.fill_(3.0)
        ema.update(model)

        with ema.average_parameters(model):
            torch.testing.assert_close(
                model.linear.weight,
                torch.full_like(model.linear.weight, 2.0),
            )
        torch.testing.assert_close(
            model.linear.weight,
            torch.full_like(model.linear.weight, 3.0),
        )

    def test_checkpoint_selector_requires_explicit_ema_weights(self):
        online = self._model(1.0).state_dict()
        averaged = self._model(2.0).state_dict()
        checkpoint = {
            "model": online,
            "ema": {
                "decay": 0.99,
                "num_updates": 10,
                "state_dict": averaged,
            },
        }

        self.assertIs(model_state_from_checkpoint(checkpoint), online)
        self.assertIs(
            model_state_from_checkpoint(checkpoint, use_ema=True),
            averaged,
        )
        with self.assertRaisesRegex(KeyError, "does not contain EMA"):
            model_state_from_checkpoint({"model": online}, use_ema=True)


if __name__ == "__main__":
    unittest.main()
