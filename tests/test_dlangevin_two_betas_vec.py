from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gem import sampler
from gem.dlangevin_utils import (
    resolve_two_beta_annealing_kwargs,
    resolve_two_beta_kwargs,
)
from gem.proposals.base import ProposalResult
from gem.proposals.dlangevin_proposal import (
    DLangevinTwoBetasProposal,
    DLangevinTwoBetasAnnealingProposal,
    DLangevinTwoBetasAnnealingVectorizedNoOriginProposal,
    DLangevinTwoBetasAnnealingVectorizedProposal,
    DLangevinTwoBetasVectorizedProposal,
)


def _graph_case(n: int, x_classes: int, e_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    nodes = torch.randint(x_classes, (n,), generator=generator)
    upper = torch.randint(e_classes, (n, n), generator=generator)
    upper = torch.triu(upper, diagonal=1)
    edges = upper + upper.transpose(0, 1)

    target_nodes = (nodes + 1) % x_classes
    target_edges = edges.clone()
    if n > 1:
        target_type = int((target_edges[0, 1].item() + 1) % e_classes)
        target_edges[0, 1] = target_type
        target_edges[1, 0] = target_type

    grad_nodes = torch.randn((n, x_classes), generator=generator)
    grad_edges = torch.randn((n, n, e_classes), generator=generator)
    return nodes, edges, target_nodes, target_edges, grad_nodes, grad_edges


class DLangevinTwoBetasVectorizedTest(unittest.TestCase):
    def setUp(self):
        self.beta_prop = 5.25
        self.beta_mh = 1.0
        self.lambda_x = 0.15
        self.lambda_e = 3.5

    def _make_pair(self):
        kwargs = dict(
            beta_prop=self.beta_prop,
            beta_mh=self.beta_mh,
            lambda_X=self.lambda_x,
            lambda_E=self.lambda_e,
        )
        return (
            DLangevinTwoBetasProposal(**kwargs),
            DLangevinTwoBetasVectorizedProposal(**kwargs),
        )

    def test_factory_and_config_resolution(self):
        proposal = sampler.make_proposal(
            "dlangevin_two_betas_vec",
            dl_beta_prop=self.beta_prop,
            dl_beta_mh=self.beta_mh,
            dl_lambda_X=self.lambda_x,
            dl_lambda_E=self.lambda_e,
        )
        self.assertIsInstance(proposal, DLangevinTwoBetasVectorizedProposal)
        self.assertEqual(proposal.beta_prop, self.beta_prop)
        self.assertEqual(proposal.beta_mh, self.beta_mh)

        resolved = resolve_two_beta_kwargs(
            "dlangevin_two_betas_vec",
            {"dl_beta_prop": "5.25", "dl_beta_mh": 1},
        )
        self.assertEqual(resolved, {"dl_beta_prop": 5.25, "dl_beta_mh": 1.0})
        with self.assertRaisesRegex(ValueError, "dl_beta_mh"):
            resolve_two_beta_kwargs(
                "dlangevin_two_betas_vec",
                {"dl_beta_prop": 5.25},
            )

    def test_forward_proposals_match_for_fixed_seed(self):
        scalar, vectorized = self._make_pair()
        cases = [_graph_case(3, 4, 3, 11), _graph_case(5, 4, 3, 17)]
        nodes = [case[0] for case in cases]
        edges = [case[1] for case in cases]
        scalar._grad_cache = {
            idx: (case[4].clone(), case[5].clone()) for idx, case in enumerate(cases)
        }
        vectorized._grad_cache = {
            idx: (case[4].clone(), case[5].clone()) for idx, case in enumerate(cases)
        }
        dataset_info = SimpleNamespace(output_dims={"X": 4, "E": 3})

        torch.manual_seed(1234)
        scalar_result = scalar.propose(
            model=None,
            dataset_info=dataset_info,
            node_types_list=nodes,
            edge_types_list=edges,
            extra_features=None,
            domain_features=None,
            device=torch.device("cpu"),
        )
        torch.manual_seed(1234)
        vectorized_result = vectorized.propose(
            model=None,
            dataset_info=dataset_info,
            node_types_list=nodes,
            edge_types_list=edges,
            extra_features=None,
            domain_features=None,
            device=torch.device("cpu"),
        )

        for scalar_nodes, vectorized_nodes in zip(
            scalar_result.prop_nodes, vectorized_result.prop_nodes
        ):
            self.assertTrue(torch.equal(scalar_nodes, vectorized_nodes))
        for scalar_edges, vectorized_edges in zip(
            scalar_result.prop_edges, vectorized_result.prop_edges
        ):
            self.assertTrue(torch.equal(scalar_edges, vectorized_edges))
        torch.testing.assert_close(
            scalar_result.log_q_fwd,
            vectorized_result.log_q_fwd,
            rtol=0.0,
            atol=0.0,
        )

    def test_annealing_factory_resolution_and_schedule(self):
        kwargs = dict(
            dl_beta_prop=self.beta_prop,
            dl_beta_mh_init=0.5,
            dl_beta_mh_final=10.0,
            dl_beta_mh_anneal_steps=200,
            dl_lambda_X=self.lambda_x,
            dl_lambda_E=self.lambda_e,
        )
        vectorized = sampler.make_proposal(
            "dlangevin_two_betas_annealing_vec",
            **kwargs,
        )
        scalar = DLangevinTwoBetasAnnealingProposal(
            beta_prop=self.beta_prop,
            beta_mh_init=0.5,
            beta_mh_final=10.0,
            beta_mh_anneal_steps=200,
            lambda_X=self.lambda_x,
            lambda_E=self.lambda_e,
        )
        self.assertIsInstance(
            vectorized,
            DLangevinTwoBetasAnnealingVectorizedProposal,
        )

        resolved = resolve_two_beta_annealing_kwargs(
            "dlangevin_two_betas_annealing_vec",
            kwargs,
        )
        self.assertEqual(
            resolved,
            {
                "dl_beta_prop": self.beta_prop,
                "dl_beta_mh_init": 0.5,
                "dl_beta_mh_final": 10.0,
                "dl_beta_mh_anneal_steps": 200,
            },
        )

        for step in (0, 1, 100, 199, 200, 250):
            scalar.on_step_start(step)
            vectorized.on_step_start(step)
            self.assertAlmostEqual(scalar.beta_mh, vectorized.beta_mh)
            self.assertAlmostEqual(vectorized._mh_beta(), vectorized.beta_mh)
        vectorized.on_step_start(0)
        self.assertEqual(vectorized.beta_mh, 0.5)
        vectorized.on_step_start(199)
        self.assertEqual(vectorized.beta_mh, 10.0)
        vectorized.on_step_start(200)
        self.assertEqual(vectorized.beta_mh, 10.0)

    def test_no_origin_factory_and_proposal_are_unconstrained(self):
        kwargs = dict(
            dl_beta_prop=self.beta_prop,
            dl_beta_mh_init=0.5,
            dl_beta_mh_final=10.0,
            dl_beta_mh_anneal_steps=200,
            dl_lambda_X=self.lambda_x,
            dl_lambda_E=self.lambda_e,
        )
        proposal = sampler.make_proposal(
            "dlangevin_two_betas_annealing_vec_no_origin",
            **kwargs,
        )
        self.assertIsInstance(
            proposal,
            DLangevinTwoBetasAnnealingVectorizedNoOriginProposal,
        )
        self.assertFalse(proposal.excludes_origin)
        self.assertEqual(
            resolve_two_beta_annealing_kwargs(
                "dlangevin_two_betas_annealing_vec_no_origin",
                kwargs,
            ),
            {
                "dl_beta_prop": self.beta_prop,
                "dl_beta_mh_init": 0.5,
                "dl_beta_mh_final": 10.0,
                "dl_beta_mh_anneal_steps": 200,
            },
        )

        case = _graph_case(4, 4, 3, 31)
        nodes = [case[0]]
        edges = [case[1]]
        self.assertFalse(hasattr(proposal, "set_origins"))

        reference = DLangevinTwoBetasAnnealingVectorizedProposal(
            beta_prop=self.beta_prop,
            beta_mh_init=0.5,
            beta_mh_final=10.0,
            beta_mh_anneal_steps=200,
            lambda_X=self.lambda_x,
            lambda_E=self.lambda_e,
        )
        proposal._grad_cache = {0: (case[4].clone(), case[5].clone())}
        reference._grad_cache = {0: (case[4].clone(), case[5].clone())}
        dataset_info = SimpleNamespace(output_dims={"X": 4, "E": 3})

        torch.manual_seed(2468)
        result = proposal.propose(
            model=None,
            dataset_info=dataset_info,
            node_types_list=nodes,
            edge_types_list=edges,
            extra_features=None,
            domain_features=None,
            device=torch.device("cpu"),
        )
        torch.manual_seed(2468)
        reference_result = reference.propose(
            model=None,
            dataset_info=dataset_info,
            node_types_list=nodes,
            edge_types_list=edges,
            extra_features=None,
            domain_features=None,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(result.prop_nodes[0], reference_result.prop_nodes[0]))
        self.assertTrue(torch.equal(result.prop_edges[0], reference_result.prop_edges[0]))
        torch.testing.assert_close(result.log_q_fwd, reference_result.log_q_fwd)

    def test_reverse_log_probabilities_match_across_sizes(self):
        scalar, vectorized = self._make_pair()
        cases = [
            _graph_case(2, 4, 3, 3),
            _graph_case(4, 4, 3, 5),
            _graph_case(4, 4, 3, 7),
            _graph_case(5, 4, 3, 9),
        ]
        kwargs = dict(
            current_nodes=[case[0] for case in cases],
            current_edges=[case[1] for case in cases],
            target_nodes=[case[2] for case in cases],
            target_edges=[case[3] for case in cases],
            grad_X_list=[case[4] for case in cases],
            grad_E_list=[case[5] for case in cases],
            device=torch.device("cpu"),
        )

        scalar_log_q = scalar._log_transition_prob_many(**kwargs)
        vectorized_log_q = vectorized._log_transition_prob_many(**kwargs)
        self.assertTrue(torch.isfinite(scalar_log_q).all())
        self.assertTrue(torch.isfinite(vectorized_log_q).all())
        torch.testing.assert_close(
            scalar_log_q,
            vectorized_log_q,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_acceptance_uses_mh_beta(self):
        proposal = DLangevinTwoBetasVectorizedProposal(
            beta_prop=10.0,
            beta_mh=0.1,
            lambda_X=1.0,
            lambda_E=1.0,
        )
        current_nodes = [torch.tensor([0, 0], dtype=torch.long)]
        current_edges = [torch.zeros((2, 2), dtype=torch.long)]
        proposed_nodes = [torch.tensor([1, 0], dtype=torch.long)]
        proposed_edges = [torch.zeros((2, 2), dtype=torch.long)]
        prop_result = ProposalResult(
            prop_nodes=proposed_nodes,
            prop_edges=proposed_edges,
            log_q_fwd=torch.zeros(1),
        )

        def equal_reverse_log_q(self, **kwargs):
            return torch.zeros(len(kwargs["current_nodes"]), device=kwargs["device"])

        proposal._log_transition_prob_many = types.MethodType(
            equal_reverse_log_q, proposal
        )

        def fake_energy_and_grads_batch(
            model,
            node_types_list,
            edge_types_list,
            dataset_info,
            device,
            extra_features,
            domain_features,
            **kwargs,
        ):
            energies = torch.ones(len(node_types_list), device=device)
            grad_nodes = [torch.zeros((len(nodes), 2), device=device) for nodes in node_types_list]
            grad_edges = [
                torch.zeros((len(nodes), len(nodes), 2), device=device)
                for nodes in node_types_list
            ]
            return energies, grad_nodes, grad_edges

        torch.manual_seed(0)
        with patch(
            "gem.proposals.dlangevin_proposal.energy_and_grads_batch",
            side_effect=fake_energy_and_grads_batch,
        ):
            accept_mask, _ = proposal.accept(
                model=torch.nn.Identity(),
                dataset_info=SimpleNamespace(),
                current_nodes=current_nodes,
                current_edges=current_edges,
                prop_result=prop_result,
                current_E=torch.zeros(1),
                prop_E=None,
                extra_features=None,
                domain_features=None,
                device=torch.device("cpu"),
            )

        # seed=0 gives log(U) around -0.70: beta_mh=0.1 accepts a +1 energy
        # move, whereas accidentally using beta_prop=10 would reject it.
        self.assertTrue(bool(accept_mask.item()))

    def test_mcmc_driver_matches_scalar_kernel(self):
        cases = [
            _graph_case(3, 4, 3, 21),
            _graph_case(4, 4, 3, 23),
            _graph_case(4, 4, 3, 25),
        ]
        initial_nodes = [case[0] for case in cases]
        initial_edges = [case[1] for case in cases]
        dataset_info = SimpleNamespace(output_dims={"X": 4, "E": 3})
        node_weights = torch.tensor([0.0, 0.25, 0.75, 1.25])
        edge_weights = torch.tensor([0.0, 0.2, 0.6])

        def synthetic_energy_batch(
            model,
            node_types_list,
            edge_types_list,
            dataset_info,
            device,
            extra_features,
            domain_features,
            **kwargs,
        ):
            energies = []
            for nodes, edges in zip(node_types_list, edge_types_list):
                node_energy = node_weights.to(device)[nodes.long()].sum()
                pair_idx = torch.triu_indices(len(nodes), len(nodes), 1, device=device)
                edge_energy = edge_weights.to(device)[
                    edges.long()[pair_idx[0], pair_idx[1]]
                ].sum()
                energies.append(node_energy + edge_energy)
            return torch.stack(energies)

        def synthetic_energy_and_grads_batch(
            model,
            node_types_list,
            edge_types_list,
            dataset_info,
            device,
            extra_features,
            domain_features,
            **kwargs,
        ):
            energies = synthetic_energy_batch(
                model,
                node_types_list,
                edge_types_list,
                dataset_info,
                device,
                extra_features,
                domain_features,
            )
            grad_nodes = [
                node_weights.to(device).expand(len(nodes), -1).clone()
                for nodes in node_types_list
            ]
            grad_edges = [
                edge_weights.to(device)
                .expand(len(nodes), len(nodes), -1)
                .clone()
                for nodes in node_types_list
            ]
            return energies, grad_nodes, grad_edges

        common_kwargs = dict(
            model=torch.nn.Identity(),
            dataset_info=dataset_info,
            node_types_list=initial_nodes,
            edge_types_list=initial_edges,
            extra_features=None,
            domain_features=None,
            steps=6,
            device=torch.device("cpu"),
            dl_beta_prop=self.beta_prop,
            dl_beta_mh=self.beta_mh,
            dl_lambda_X=self.lambda_x,
            dl_lambda_E=self.lambda_e,
            collect_stats=True,
        )

        callback_states = []

        def record_step(event):
            callback_states.append(
                (
                    event.step,
                    [node.clone() for node in event.nodes],
                    [edge.clone() for edge in event.edges],
                    event.energies.clone(),
                    event.accepted.clone(),
                    [node.clone() for node in event.proposed_nodes],
                    [edge.clone() for edge in event.proposed_edges],
                )
            )

        with patch("gem.sampler.energy_batch", side_effect=synthetic_energy_batch), patch(
            "gem.proposals.dlangevin_proposal.energy_and_grads_batch",
            side_effect=synthetic_energy_and_grads_batch,
        ):
            torch.manual_seed(4321)
            scalar_result = sampler.mcmc_sample_batch(
                proposal="dlangevintwobetas",
                step_callback=record_step,
                **common_kwargs,
            )
            torch.manual_seed(4321)
            vectorized_result = sampler.mcmc_sample_batch(
                proposal="dlangevin_two_betas_vec", **common_kwargs
            )

            anneal_kwargs = dict(common_kwargs)
            anneal_kwargs.pop("dl_beta_mh")
            anneal_kwargs.update(
                dl_beta_mh_init=0.5,
                dl_beta_mh_final=4.0,
                dl_beta_mh_anneal_steps=4,
            )
            torch.manual_seed(9876)
            scalar_annealed_result = sampler.mcmc_sample_batch(
                proposal="dlangevintwobetas_annealing",
                **anneal_kwargs,
            )
            torch.manual_seed(9876)
            vectorized_annealed_result = sampler.mcmc_sample_batch(
                proposal="dlangevin_two_betas_annealing_vec",
                **anneal_kwargs,
            )

        self.assertEqual([row[0] for row in callback_states], list(range(6)))
        self.assertTrue(
            torch.equal(callback_states[-1][1][0], scalar_result[0][0])
        )
        self.assertTrue(
            torch.equal(callback_states[-1][2][0], scalar_result[1][0])
        )
        expected_batch_shape = torch.Size([len(initial_nodes)])
        self.assertEqual(callback_states[-1][3].shape, expected_batch_shape)
        self.assertEqual(callback_states[-1][4].shape, expected_batch_shape)
        self.assertEqual(len(callback_states[-1][5]), len(initial_nodes))
        self.assertEqual(len(callback_states[-1][6]), len(initial_edges))

        for scalar_nodes, vectorized_nodes in zip(
            scalar_result[0], vectorized_result[0]
        ):
            self.assertTrue(torch.equal(scalar_nodes, vectorized_nodes))
        for scalar_edges, vectorized_edges in zip(
            scalar_result[1], vectorized_result[1]
        ):
            self.assertTrue(torch.equal(scalar_edges, vectorized_edges))
        self.assertEqual(scalar_result[2], vectorized_result[2])
        self.assertEqual(scalar_result[3], vectorized_result[3])
        self.assertEqual(
            scalar_result[4]["total_accepted"],
            vectorized_result[4]["total_accepted"],
        )
        for key in (
            "step_acc_total_sq_sum",
            "mean_step_acc_distance_total",
            "std_step_acc_distance_total",
        ):
            self.assertAlmostEqual(
                scalar_result[4][key],
                vectorized_result[4][key],
                places=7,
            )
        self.assertGreaterEqual(
            scalar_result[4]["std_step_acc_distance_total"],
            0.0,
        )
        for scalar_nodes, vectorized_nodes in zip(
            scalar_annealed_result[0], vectorized_annealed_result[0]
        ):
            self.assertTrue(torch.equal(scalar_nodes, vectorized_nodes))
        for scalar_edges, vectorized_edges in zip(
            scalar_annealed_result[1], vectorized_annealed_result[1]
        ):
            self.assertTrue(torch.equal(scalar_edges, vectorized_edges))
        self.assertEqual(
            scalar_annealed_result[2],
            vectorized_annealed_result[2],
        )
        self.assertEqual(
            scalar_annealed_result[3],
            vectorized_annealed_result[3],
        )
        self.assertEqual(
            scalar_annealed_result[4]["total_accepted"],
            vectorized_annealed_result[4]["total_accepted"],
        )
        for key in (
            "step_acc_total_sq_sum",
            "mean_step_acc_distance_total",
            "std_step_acc_distance_total",
        ):
            self.assertAlmostEqual(
                scalar_annealed_result[4][key],
                vectorized_annealed_result[4][key],
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
