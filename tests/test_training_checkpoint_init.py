from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from gem.train_gem_ebm_fm import (
    _checkpoint_rng_state_for_rank,
    _load_model_weights,
)


class TinyModel(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(weight)


class TrainingCheckpointInitializationTest(unittest.TestCase):
    def test_init_can_select_ema_weights(self):
        checkpoint = {
            "model": TinyModel(1.0).state_dict(),
            "ema": {
                "decay": 0.99,
                "num_updates": 10,
                "state_dict": TinyModel(2.0).state_dict(),
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.pt"
            torch.save(checkpoint, path)
            target = TinyModel(-1.0)
            loaded = _load_model_weights(
                target,
                torch.device("cpu"),
                str(path),
                use_ema=True,
                log=False,
            )

        self.assertEqual(loaded["ema"]["num_updates"], 10)
        torch.testing.assert_close(
            target.linear.weight,
            torch.full_like(target.linear.weight, 2.0),
        )

    def test_init_rejects_partial_state_dict(self):
        checkpoint = {"model": {}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "partial.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "does not contain a model state"):
                _load_model_weights(
                    TinyModel(-1.0),
                    torch.device("cpu"),
                    str(path),
                    log=False,
                )

    def test_init_rejects_incompatible_state_dict(self):
        checkpoint = {"model": {"unexpected": torch.ones(1)}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "incompatible.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "Could not load online checkpoint"):
                _load_model_weights(
                    TinyModel(-1.0),
                    torch.device("cpu"),
                    str(path),
                    log=False,
                )

    def test_distributed_resume_selects_rank_specific_rng_state(self):
        rank_states = [{"rank": 0}, {"rank": 1}]
        checkpoint = {
            "rng_state": rank_states[0],
            "rng_state_by_rank": rank_states,
        }

        self.assertIs(
            _checkpoint_rng_state_for_rank(
                checkpoint,
                rank=1,
                is_distributed=True,
            ),
            rank_states[1],
        )

    def test_legacy_distributed_resume_does_not_duplicate_rank_zero_rng(self):
        legacy_state = {"rank": 0}
        checkpoint = {"rng_state": legacy_state}

        self.assertIs(
            _checkpoint_rng_state_for_rank(
                checkpoint,
                rank=0,
                is_distributed=True,
            ),
            legacy_state,
        )
        self.assertIsNone(
            _checkpoint_rng_state_for_rank(
                checkpoint,
                rank=1,
                is_distributed=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
