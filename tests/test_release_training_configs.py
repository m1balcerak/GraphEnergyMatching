import inspect
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from gem import sampler
from gem.train_gem_ebm_fm import _validate_required_initialization


PHASE_1_OVERRIDES = [
    "general.gpus=4",
    "train.distributed.enabled=true",
    "train.max_iters=400000",
    "train.lr=1e-4",
    "train.lambda_cl=0.0",
    "train.cl_steps=0",
    "train.save_every=25000",
]

PHASE_2_OVERRIDES = [
    "general.gpus=4",
    "train.distributed.enabled=true",
    "train.init_ckpt=/tmp/transport-checkpoint.pt",
]


class ReleaseTrainingConfigsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())

    def _compose(self, name, overrides=None):
        with initialize_config_dir(version_base=None, config_dir=self.config_dir):
            return compose(config_name=name, overrides=overrides or [])

    def test_readme_phase1_is_online_transport_only(self):
        cfg = self._compose("gem_ebm_fm_moses_ver3", PHASE_1_OVERRIDES)

        self.assertFalse(cfg.dataset.filter)
        self.assertEqual(cfg.train.max_iters, 400000)
        self.assertEqual(cfg.train.lambda_fm, 1.0)
        self.assertEqual(cfg.train.lambda_cl, 0.0)
        self.assertEqual(cfg.train.cl_steps, 0)
        self.assertEqual(cfg.train.ema_decay, 0.0)
        self.assertFalse(cfg.train.ema_use_for_eval)
        self.assertFalse(cfg.train.init_use_ema)

    def test_readme_phase2_uses_no_origin_recipe(self):
        cfg = self._compose("gem_ebm_fm_moses_ver3", PHASE_2_OVERRIDES)

        self.assertFalse(cfg.dataset.filter)
        self.assertEqual(cfg.train.max_iters, 500)
        self.assertEqual(
            cfg.train.proposal,
            "dlangevin_two_betas_annealing_vec_no_origin",
        )
        self.assertEqual(cfg.train.cl_steps, 1000)
        self.assertEqual(cfg.train.cl_bad_negative_weight, 10.0)
        self.assertEqual(cfg.train.cl_clip_mode, "paired_connected")
        self.assertEqual(cfg.train.chain_warmup.steps, 225)
        self.assertEqual(cfg.train.chain_warmup.simple_n_edits, 1)
        self.assertEqual(cfg.train.ema_decay, 0.0)
        self.assertFalse(cfg.train.ema_use_for_eval)
        self.assertFalse(cfg.train.init_use_ema)
        self.assertFalse(cfg.sample.evaluate)
        self.assertFalse(cfg.viz.enabled)
        self.assertFalse(cfg.viz.run_during_training)

    def test_animation_uses_release_sampler_and_online_weights(self):
        phase2 = self._compose("gem_ebm_fm_moses_ver3", PHASE_2_OVERRIDES)
        animation = self._compose("gem_ebm_animation_moses_public")

        self.assertEqual(animation.animation.steps, phase2.train.cl_steps)
        self.assertEqual(animation.animation.proposal, phase2.train.proposal)
        self.assertEqual(animation.animation.dl_beta_prop, phase2.train.dl_beta_prop)
        self.assertEqual(
            animation.animation.dl_beta_mh_init,
            phase2.train.dl_beta_mh_init,
        )
        self.assertEqual(
            animation.animation.dl_beta_mh_final,
            phase2.train.dl_beta_mh_final,
        )
        self.assertEqual(
            animation.animation.dl_beta_mh_anneal_steps,
            phase2.train.dl_beta_mh_anneal_steps,
        )
        self.assertEqual(animation.animation.dl_lambda_X, phase2.train.dl_lambda_X)
        self.assertEqual(animation.animation.dl_lambda_E, phase2.train.dl_lambda_E)
        self.assertFalse(animation.dataset.filter)
        self.assertFalse(animation.animation.use_ema)

    def test_evaluation_uses_full_raw_official_scaffold_split(self):
        phase2 = self._compose("gem_ebm_fm_moses_ver3", PHASE_2_OVERRIDES)
        metrics = self._compose("gem_metrics_over_time_moses")

        self.assertFalse(metrics.dataset.filter)
        self.assertEqual(metrics.metrics_run.steps, [phase2.train.cl_steps])
        self.assertEqual(metrics.metrics_run.proposal, phase2.train.proposal)
        self.assertEqual(metrics.metrics_run.dl_beta_prop, phase2.train.dl_beta_prop)
        self.assertEqual(
            metrics.metrics_run.dl_beta_mh_init,
            phase2.train.dl_beta_mh_init,
        )
        self.assertEqual(
            metrics.metrics_run.dl_beta_mh_final,
            phase2.train.dl_beta_mh_final,
        )
        self.assertEqual(
            metrics.metrics_run.dl_beta_mh_anneal_steps,
            phase2.train.dl_beta_mh_anneal_steps,
        )
        self.assertEqual(metrics.metrics_run.chain_warmup, phase2.train.chain_warmup)
        self.assertEqual(metrics.metrics_run.fcd_reference_split, "val")
        self.assertFalse(metrics.metrics_run.fcd_reference_filter)
        self.assertEqual(metrics.metrics_run.fcd_reference_size, 0)
        self.assertEqual(metrics.metrics_run.fcd_reference_expected_size, 176225)
        self.assertFalse(metrics.metrics_run.fcd_reference_canonicalize)
        self.assertTrue(metrics.metrics_run.fcd_generated_largest_fragment)
        self.assertFalse(metrics.metrics_run.use_ema)

    def test_release_checkpoint_is_iteration_500_online_weights(self):
        metrics = self._compose("gem_metrics_over_time_moses")

        self.assertEqual(
            metrics.checkpoint.pretrained.name,
            "gem_moses_pretrained_it400000.pt",
        )
        self.assertEqual(metrics.checkpoint.pretrained.iteration, 400000)
        self.assertEqual(
            metrics.checkpoint.pretrained.sha256,
            "bc25ecc18af06396e9d19211cf5300ab98e80ba295be9b93b1ff99651ef9b874",
        )
        self.assertEqual(metrics.checkpoint.name, "gem_moses_it000500.pt")
        self.assertEqual(metrics.checkpoint.iteration, 500)
        self.assertFalse(metrics.checkpoint.use_ema)
        self.assertEqual(
            metrics.checkpoint.sha256,
            "6360e1fb2c1bfd0e9a6ff8c5d5b83c8baf07211f81c1727581c01e7da0be50a0",
        )
        self.assertEqual(metrics.checkpoint.evaluation.mcmc_steps, 1000)
        self.assertAlmostEqual(metrics.checkpoint.evaluation.vun, 0.94032)
        self.assertAlmostEqual(metrics.checkpoint.evaluation.fcd, 1.466017749760752)

    def test_origin_exclusion_is_not_exposed(self):
        parameters = inspect.signature(sampler.mcmc_sample_batch).parameters
        self.assertNotIn("exclude_origin", parameters)

    def test_required_initialization_accepts_checkpoint_or_resume(self):
        with self.assertRaisesRegex(ValueError, "requires train.init_ckpt"):
            _validate_required_initialization(
                required=True,
                resume_path="",
                init_ckpt_path="",
            )

        _validate_required_initialization(
            required=True,
            resume_path="checkpoint.pt",
            init_ckpt_path="",
        )
        _validate_required_initialization(
            required=True,
            resume_path="",
            init_ckpt_path="phase1.pt",
        )


if __name__ == "__main__":
    unittest.main()
