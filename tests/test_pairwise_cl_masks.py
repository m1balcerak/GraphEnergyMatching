import unittest

import torch

from gem.train_gem_ebm_fm import _pairwise_cl_masks


class PairwiseClMasksTest(unittest.TestCase):
    def test_paired_all_caps_every_pair_but_preserves_bad_negative_mask(self):
        good_mask, cap_mask = _pairwise_cl_masks(
            valid_mask=[True, True, False],
            connected_mask=[True, False, False],
            count=3,
            device=torch.device("cpu"),
            clip_mode="paired_all",
        )

        self.assertEqual(good_mask.tolist(), [True, False, False])
        self.assertEqual(cap_mask.tolist(), [True, True, True])

    def test_paired_connected_caps_only_valid_connected_pairs(self):
        good_mask, cap_mask = _pairwise_cl_masks(
            valid_mask=[True, True, False],
            connected_mask=[True, False, False],
            count=3,
            device=torch.device("cpu"),
            clip_mode="paired_connected",
        )

        self.assertEqual(good_mask.tolist(), [True, False, False])
        self.assertEqual(cap_mask.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
