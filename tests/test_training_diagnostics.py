from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gem.train_gem_ebm_fm import (
    _plot_energy_gap,
    _plot_energy_trajectories,
    _plot_loss_curves,
    _plot_transport_gradient_strength,
)


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_transport_history_plots_are_written_without_cl_metrics(self):
        history = {
            "iter": [1, 2, 3],
            "loss_fm": [0.3, 0.2, 0.1],
            "loss_cl": [float("nan")] * 3,
            "loss_total": [0.3, 0.2, 0.1],
            "fm_ediff_mu": [1.0, 0.5, -0.2],
            "fm_ediff_sd": [0.2, 0.2, 0.1],
            "fm_grad_mu": [0.1, 0.2, 0.3],
            "fm_grad_sd": [0.01, 0.02, 0.03],
            "energy_data_mean": [-1.0, -2.0, -3.0],
            "energy_data_std": [0.2, 0.2, 0.3],
            "energy_noise_mean": [1.0, 0.0, -1.0],
            "energy_noise_std": [0.4, 0.3, 0.2],
        }
        plotters = {
            "losses.png": _plot_loss_curves,
            "gap.png": _plot_energy_gap,
            "energies.png": _plot_energy_trajectories,
            "gradients.png": _plot_transport_gradient_strength,
        }

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            for filename, plotter in plotters.items():
                path = Path(tmp_dir) / filename
                plotter(history, out_path=str(path))
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
