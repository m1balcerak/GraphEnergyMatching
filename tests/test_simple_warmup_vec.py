from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from gem import sampler
from gem.proposals.simple_proposal import (
    SimpleProposalV2,
    _pack_graph_types,
    apply_simple_v2_edits_batched,
    run_simple_v2_warmup_vectorized,
)
from gem.sampler_energy import energy_and_grads_batch, energy_and_grads_dense


def _graph(n: int, x_classes: int, e_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    nodes = torch.randint(x_classes, (n,), generator=generator)
    upper = torch.randint(e_classes, (n, n), generator=generator)
    upper = torch.triu(upper, diagonal=1)
    edges = upper + upper.transpose(0, 1)
    return nodes, edges


def _mixed_graphs():
    cases = [
        _graph(2, 4, 3, 3),
        _graph(4, 4, 3, 5),
        _graph(5, 4, 3, 7),
    ]
    return [case[0] for case in cases], [case[1] for case in cases]


def _padded_gradients(node_counts, n_max, x_classes, e_classes):
    generator = torch.Generator().manual_seed(101)
    grad_X = torch.randn((len(node_counts), n_max, x_classes), generator=generator)
    grad_E = torch.randn(
        (len(node_counts), n_max, n_max, e_classes),
        generator=generator,
    )
    # Remove accidental ties while retaining arbitrary asymmetric edge gradients.
    grad_X += torch.arange(n_max).view(1, n_max, 1) * 1.0e-3
    grad_E += torch.arange(n_max * n_max).view(1, n_max, n_max, 1) * 1.0e-4
    return grad_X, grad_E


class _ZeroFeatures:
    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        return SimpleNamespace(
            X=X.new_zeros((*X.shape[:2], 0)),
            E=E.new_zeros((*E.shape[:3], 0)),
            y=X.new_zeros((X.shape[0], 0)),
        )


class _LinearEnergy(nn.Module):
    def __init__(self, node_weights, edge_weights):
        super().__init__()
        self.register_buffer("node_weights", torch.as_tensor(node_weights).float())
        self.register_buffer("edge_weights", torch.as_tensor(edge_weights).float())

    def forward(self, X, E, y, node_mask, return_energy=False):
        node_dim = self.node_weights.numel()
        edge_dim = self.edge_weights.numel()
        energy = (X[..., :node_dim] * self.node_weights).sum(dim=(1, 2))
        energy = energy + (E[..., :edge_dim] * self.edge_weights).sum(
            dim=(1, 2, 3)
        )
        return None, energy


class SimpleWarmupVectorizedTest(unittest.TestCase):
    def setUp(self):
        self.nodes, self.edges = _mixed_graphs()
        self.n_max = 5
        self.dataset_info = SimpleNamespace(
            max_n_nodes=self.n_max,
            output_dims={"X": 4, "E": 3, "y": 0},
            input_dims={"X": 4, "E": 3, "y": 1},
        )

    def _compare_one_step(self, edits_per_step: int):
        dense_nodes, dense_edges, node_mask, node_counts = _pack_graph_types(
            self.nodes,
            self.edges,
            n_max=self.n_max,
            device=torch.device("cpu"),
        )
        grad_X, grad_E = _padded_gradients(
            node_counts,
            self.n_max,
            self.dataset_info.output_dims["X"],
            self.dataset_info.output_dims["E"],
        )

        vector_nodes, vector_edges, vector_changed = apply_simple_v2_edits_batched(
            dense_nodes,
            dense_edges,
            node_mask,
            grad_X,
            grad_E,
            edits_per_step=edits_per_step,
        )

        reference = SimpleProposalV2(edits_per_step=edits_per_step)
        for idx, n in enumerate(node_counts):
            expected_nodes, expected_edges = reference._apply_vectorized_edits(
                self.nodes[idx],
                self.edges[idx],
                grad_X[idx, :n],
                grad_E[idx, :n, :n],
                self.dataset_info.output_dims["X"],
                self.dataset_info.output_dims["E"],
            )
            self.assertTrue(torch.equal(vector_nodes[idx, :n], expected_nodes))
            self.assertTrue(torch.equal(vector_edges[idx, :n, :n], expected_edges))
            expected_changed = not (
                torch.equal(expected_nodes, self.nodes[idx])
                and torch.equal(expected_edges, self.edges[idx])
            )
            self.assertEqual(bool(vector_changed[idx]), expected_changed)

    def test_one_edit_matches_reference_across_graph_sizes(self):
        self._compare_one_step(edits_per_step=1)

    def test_multiple_edits_match_reference_across_graph_sizes(self):
        self._compare_one_step(edits_per_step=3)

    def test_dense_and_list_gradient_paths_match(self):
        model = _LinearEnergy(
            node_weights=[0.0, 0.5, 1.0, 1.5],
            edge_weights=[0.0, 0.25, 0.75],
        )
        features = _ZeroFeatures()
        dense_nodes, dense_edges, node_mask, node_counts = _pack_graph_types(
            self.nodes,
            self.edges,
            n_max=self.n_max,
            device=torch.device("cpu"),
        )

        list_energy, list_grad_X, list_grad_E = energy_and_grads_batch(
            model,
            self.nodes,
            self.edges,
            self.dataset_info,
            torch.device("cpu"),
            features,
            features,
            apply_property_conditioner=False,
        )
        dense_energy, dense_grad_X, dense_grad_E = energy_and_grads_dense(
            model,
            dense_nodes,
            dense_edges,
            node_mask,
            self.dataset_info,
            torch.device("cpu"),
            features,
            features,
            apply_property_conditioner=False,
        )

        torch.testing.assert_close(list_energy, dense_energy, rtol=0.0, atol=0.0)
        for idx, n in enumerate(node_counts):
            torch.testing.assert_close(
                list_grad_X[idx],
                dense_grad_X[idx, :n],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                list_grad_E[idx],
                dense_grad_E[idx, :n, :n],
                rtol=0.0,
                atol=0.0,
            )

    def test_multistep_states_and_raw_stats_match_scalar_sampler(self):
        model = _LinearEnergy(
            node_weights=[0.0, 0.5, 1.0, 1.5],
            edge_weights=[0.0, 0.25, 0.75],
        )
        features = _ZeroFeatures()
        scalar = sampler.mcmc_sample_batch(
            model=model,
            dataset_info=self.dataset_info,
            node_types_list=self.nodes,
            edge_types_list=self.edges,
            extra_features=features,
            domain_features=features,
            steps=6,
            device=torch.device("cpu"),
            proposal="simple_ver2",
            simple_n_edits=3,
            collect_stats=True,
        )
        vectorized = run_simple_v2_warmup_vectorized(
            model=model,
            dataset_info=self.dataset_info,
            node_types_list=self.nodes,
            edge_types_list=self.edges,
            extra_features=features,
            domain_features=features,
            steps=6,
            device=torch.device("cpu"),
            edits_per_step=3,
            stop_when_unchanged=False,
            collect_stats=True,
        )

        for expected, actual in zip(scalar[0], vectorized[0]):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(scalar[1], vectorized[1]):
            self.assertTrue(torch.equal(expected, actual))

        raw_stat_keys = {
            "total_proposals",
            "total_accepted",
            "nontriv_any",
            "nontriv_node",
            "nontriv_edge",
            "acc_nontriv_any",
            "acc_nontriv_node",
            "acc_nontriv_edge",
            "prop_dist_nodes_sum",
            "prop_dist_edges_sum",
            "acc_dist_nodes_sum",
            "acc_dist_edges_sum",
            "step_prop_nodes_sum",
            "step_prop_edges_sum",
            "step_acc_nodes_sum",
            "step_acc_edges_sum",
            "distance_total_nodes",
            "distance_total_edges",
            "distance_total",
        }
        for key in raw_stat_keys:
            self.assertEqual(scalar[4][key], vectorized[4][key], key)

    def test_vectorized_warmup_dispatch_is_explicit(self):
        self.assertTrue(
            sampler.should_vectorize_simple_warmup(
                "simple_ver2",
                vectorized=True,
            )
        )
        self.assertTrue(
            sampler.should_vectorize_simple_warmup(
                "SIMPLE_V2",
                vectorized=True,
            )
        )
        self.assertFalse(
            sampler.should_vectorize_simple_warmup(
                "simple_ver2",
                vectorized=False,
            )
        )
        self.assertFalse(
            sampler.should_vectorize_simple_warmup(
                "dlangevin_vec",
                vectorized=True,
            )
        )

    def _deterministic_gradients(self, node_types, edge_types, node_mask, **kwargs):
        batch_size, n_max = node_types.shape
        node_weights = torch.tensor(
            [0.0, 2.0, 5.0, 9.0],
            device=node_types.device,
        )
        edge_weights = torch.tensor(
            [0.0, 1.0, 4.0],
            device=edge_types.device,
        )
        node_position = torch.arange(
            n_max,
            device=node_types.device,
        ).view(1, n_max, 1)
        edge_position = torch.arange(
            n_max * n_max,
            device=edge_types.device,
        ).view(1, n_max, n_max, 1)
        grad_X = node_weights.view(1, 1, -1).expand(batch_size, n_max, -1)
        grad_X = grad_X + node_position * 1.0e-3
        grad_E = edge_weights.view(1, 1, 1, -1).expand(
            batch_size,
            n_max,
            n_max,
            -1,
        )
        grad_E = grad_E + edge_position * 1.0e-4
        return torch.zeros(batch_size), grad_X, grad_E

    def _reference_warmup(self, max_steps: int):
        current_nodes = [nodes.clone() for nodes in self.nodes]
        current_edges = [edges.clone() for edges in self.edges]
        total_moves = 0
        attempts = 0
        steps_done = 0
        stop_reason = "max"
        proposal = SimpleProposalV2(edits_per_step=1)

        for _ in range(max_steps):
            dense_nodes, dense_edges, node_mask, node_counts = _pack_graph_types(
                current_nodes,
                current_edges,
                n_max=self.n_max,
                device=torch.device("cpu"),
            )
            _, grad_X, grad_E = self._deterministic_gradients(
                dense_nodes,
                dense_edges,
                node_mask,
            )
            next_nodes = []
            next_edges = []
            moves = 0
            for idx, n in enumerate(node_counts):
                nodes, edges = proposal._apply_vectorized_edits(
                    current_nodes[idx],
                    current_edges[idx],
                    grad_X[idx, :n],
                    grad_E[idx, :n, :n],
                    self.dataset_info.output_dims["X"],
                    self.dataset_info.output_dims["E"],
                )
                moves += int(
                    not (
                        torch.equal(nodes, current_nodes[idx])
                        and torch.equal(edges, current_edges[idx])
                    )
                )
                next_nodes.append(nodes)
                next_edges.append(edges)
            current_nodes = next_nodes
            current_edges = next_edges
            total_moves += moves
            attempts += len(current_nodes)
            steps_done += 1
            if moves == 0:
                stop_reason = "stuck"
                break

        return (
            current_nodes,
            current_edges,
            total_moves,
            attempts,
            steps_done,
            stop_reason,
        )

    def _run_vectorized(self, max_steps: int):
        with patch(
            "gem.proposals.simple_proposal.energy_and_grads_dense",
            side_effect=self._deterministic_gradients,
        ):
            return run_simple_v2_warmup_vectorized(
                model=nn.Identity(),
                dataset_info=self.dataset_info,
                node_types_list=self.nodes,
                edge_types_list=self.edges,
                extra_features=None,
                domain_features=None,
                steps=max_steps,
                device=torch.device("cpu"),
                edits_per_step=1,
                stop_when_unchanged=True,
            )

    def _assert_runs_match(self, max_steps: int):
        reference = self._reference_warmup(max_steps)
        vectorized = self._run_vectorized(max_steps)
        for expected, actual in zip(reference[0], vectorized[0]):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(reference[1], vectorized[1]):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(reference[2], vectorized[2])
        self.assertEqual(reference[3], vectorized[3])
        self.assertEqual(reference[4], vectorized[4]["steps_executed"])
        self.assertEqual(reference[5], vectorized[4]["stop_reason"])

    def test_multistep_trajectory_and_early_stop_match_reference(self):
        self._assert_runs_match(max_steps=100)

    def test_cap_limited_trajectory_matches_reference(self):
        self._assert_runs_match(max_steps=2)

    def test_noop_terminal_step_counts_attempts_but_not_moves(self):
        self.nodes = [torch.zeros_like(nodes) for nodes in self.nodes]
        self.edges = [torch.zeros_like(edges) for edges in self.edges]
        _, _, moves, attempts, stats = self._run_vectorized(max_steps=225)
        self.assertEqual(moves, 0)
        self.assertEqual(attempts, len(self.nodes))
        self.assertEqual(stats["steps_executed"], 1)
        self.assertEqual(stats["stop_reason"], "stuck")


if __name__ == "__main__":
    unittest.main()
