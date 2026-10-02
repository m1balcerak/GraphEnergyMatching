import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from gem import sample_moses
from gem.datasets import moses_dataset
from gem import sampler


ROOT = Path(__file__).resolve().parents[1]


class SampleMosesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = sample_moses.load_sampling_config(ROOT / 'configs')

    def run_config(self):
        return OmegaConf.create(OmegaConf.to_container(self.cfg.metrics_run, resolve=True))

    def test_release_sampler_configuration(self):
        cfg = self.run_config()
        self.assertEqual(cfg.proposal, 'dlangevin_two_betas_annealing_vec_no_origin')
        self.assertAlmostEqual(cfg.dl_beta_prop, 2.360255480670657)
        self.assertAlmostEqual(cfg.dl_beta_mh_init, 0.7765126196192521)
        self.assertAlmostEqual(cfg.dl_beta_mh_final, 17.619552679597852)
        self.assertEqual(cfg.dl_beta_mh_anneal_steps, 500)
        self.assertAlmostEqual(cfg.dl_lambda_X, 4.002714439035672)
        self.assertAlmostEqual(cfg.dl_lambda_E, 4.834996084900104)
        self.assertTrue(cfg.chain_warmup.enabled)
        self.assertEqual(cfg.chain_warmup.steps, 225)
        self.assertEqual(cfg.chain_warmup.proposal, 'simple_ver2')
        self.assertEqual(cfg.chain_warmup.simple_n_edits, 1)
        self.assertTrue(cfg.chain_warmup.vectorized)

    def test_model_build_uses_builtin_metadata_without_dataset_or_cwd_files(self):
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('n_counts', 'atom_types', 'edge_types', 'valencies'):
                (Path(directory) / ('MOSES_' + suffix + '.txt')).write_text('poisoned statistics\n')
            try:
                os.chdir(directory)
                with patch.object(moses_dataset, 'MosesDataModule',
                                  side_effect=AssertionError('Dataset construction is forbidden')), \
                        patch.object(moses_dataset.np, 'loadtxt',
                                     side_effect=AssertionError('CWD metadata loading is forbidden')):
                    model, infos, extra, molecular = sample_moses.build_sampling_model(
                        self.cfg, torch.device('cpu'))
            finally:
                os.chdir(previous_cwd)
        self.assertEqual(infos.input_dims, {'X': 30, 'E': 25, 'y': 7})
        self.assertEqual(infos.output_dims, {'X': 8, 'E': 5, 'y': 0})
        self.assertEqual(infos.max_n_nodes, 27)
        self.assertEqual(infos.atom_decoder, ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'])
        self.assertEqual(model.mlp_in_y[0].in_features, 71)
        self.assertEqual(model.mlp_in_X[0].in_features, 30)
        self.assertEqual(model.mlp_in_E[0].in_features, 25)
        self.assertEqual(len(model.tf_layers), 12)
        self.assertFalse(model.training)
        self.assertIsNotNone(extra)
        self.assertIsNotNone(molecular)

    def generated_with_mocks(self, mixing_steps):
        def initialize(*, batch_size, **kwargs):
            return [(torch.tensor([0]), torch.zeros((1, 1), dtype=torch.long))
                    for _ in range(batch_size)]

        def evolve(**kwargs):
            return (kwargs['node_types_list'], kwargs['edge_types_list'], 0, 0, {})

        def records(nodes, edges, infos, *, step, sample_offset):
            return [dict(step=step, index=sample_offset + i,
                         valid=(sample_offset + i) % 2 == 0,
                         connected=(sample_offset + i) % 2 == 0,
                         valid_connected=(sample_offset + i) % 2 == 0,
                         num_fragments=1 if (sample_offset + i) % 2 == 0 else None,
                         smiles='C' if (sample_offset + i) % 2 == 0 else None)
                    for i in range(len(nodes))]

        with patch.object(sampler, 'initialize_random_graphs', side_effect=initialize) as noise, \
                patch.object(sampler, 'run_simple_v2_warmup_vectorized', side_effect=evolve) as warmup, \
                patch.object(sampler, 'mcmc_sample_batch', side_effect=evolve) as mixing, \
                patch.object(sample_moses, '_molecule_records', side_effect=records):
            batches = list(sample_moses.generate_batches(
                model=object(), dataset_infos=object(), extra_features=object(),
                domain_features=object(), device=torch.device('cpu'),
                run_cfg=self.run_config(), num_samples=5, mixing_steps=mixing_steps,
                batch_size=2))
        return batches, noise, warmup, mixing

    def test_zero_mixing_keeps_warmup_partial_batch_and_invalid_attempts(self):
        batches, noise, warmup, mixing = self.generated_with_mocks(0)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        rows = [row for batch in batches for row in batch]
        self.assertEqual([row['index'] for row in rows], list(range(5)))
        self.assertEqual(sum(not row['valid'] for row in rows), 2)
        self.assertTrue(all(row['step'] == 0 for row in rows))
        self.assertEqual([call.kwargs['batch_size'] for call in noise.call_args_list], [2, 2, 1])
        self.assertTrue(all(call.kwargs['transition'] == 'uniform' for call in noise.call_args_list))
        self.assertEqual(warmup.call_count, 3)
        for call in warmup.call_args_list:
            self.assertEqual(call.kwargs['steps'], 225)
            self.assertEqual(call.kwargs['edits_per_step'], 1)
            self.assertTrue(call.kwargs['stop_when_unchanged'])
        mixing.assert_not_called()

    def test_mixing_uses_requested_steps_and_fixed_calibration_for_each_batch(self):
        batches, _, warmup, mixing = self.generated_with_mocks(37)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(warmup.call_count, 3)
        self.assertEqual(mixing.call_count, 3)
        for call in mixing.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs['steps'], 37)
            self.assertEqual(kwargs['step_offset'], 0)
            self.assertEqual(kwargs['proposal'], 'dlangevin_two_betas_annealing_vec_no_origin')
            self.assertEqual(kwargs['dl_beta_mh_anneal_steps'], 500)
            self.assertAlmostEqual(kwargs['dl_beta_prop'], 2.360255480670657)
            self.assertAlmostEqual(kwargs['dl_beta_mh_init'], 0.7765126196192521)
            self.assertAlmostEqual(kwargs['dl_beta_mh_final'], 17.619552679597852)
            self.assertAlmostEqual(kwargs['dl_lambda_X'], 4.002714439035672)
            self.assertAlmostEqual(kwargs['dl_lambda_E'], 4.834996084900104)
        self.assertTrue(all(row['step'] == 37 for batch in batches for row in batch))


class SampleMosesCommandLineTest(unittest.TestCase):
    def invoke(self, directory, *args):
        # -S removes site packages: argument handling must work without torch,
        # RDKit, or a dataset, and from a directory outside the repository.
        return subprocess.run([sys.executable, '-S', str(ROOT / 'sample_MOSES.py')] + list(args),
                              cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=20)

    def test_help_does_not_import_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(directory, '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--checkpoint', '--num-samples', '--mixing-steps', '--batch-size', '--seed', '--output'):
            self.assertIn(flag, result.stdout)

    def test_invalid_ranges_fail_before_loading_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pt'
            checkpoint.write_bytes(b'not a checkpoint')
            for flag, value in (('--num-samples', '0'), ('--batch-size', '0'), ('--mixing-steps', '-1')):
                with self.subTest(flag=flag):
                    result = self.invoke(directory, '--checkpoint', str(checkpoint), flag, value)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(flag, result.stderr)
                    self.assertNotIn('Traceback', result.stderr)

    def test_missing_checkpoint_has_a_clear_early_error(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'missing.pt'
            result = self.invoke(directory, '--checkpoint', str(checkpoint))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('missing.pt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
