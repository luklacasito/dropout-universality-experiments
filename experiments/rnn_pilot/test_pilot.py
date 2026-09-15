"""Protocol tests: python -m unittest discover -s experiments/rnn_pilot."""
import unittest
from unittest.mock import patch

import torch

from analyze import ARMS as ANALYSIS_ARMS
from run import (ARMS, LAYERS, StackedLSTM, dataset, locked_dropout, mask_generators,
                 probabilities)


class ProtocolTests(unittest.TestCase):
    def test_probability_budget_and_order(self):
        self.assertEqual(ARMS, ANALYSIS_ARMS)
        for arm in ARMS[1:]:
            self.assertEqual(len(probabilities(arm, .1)), 6)
            self.assertAlmostEqual(sum(probabilities(arm, .1)), .6)
        self.assertEqual(probabilities("early_3_3", .1), probabilities("late_3_3", .1)[::-1])
        self.assertEqual(sum(p > 0 for p in probabilities("early_2_4", .1)), 2)
        linear = probabilities("linear_decreasing", .1)
        for a, b in zip(linear, linear[1:]):
            self.assertAlmostEqual(a - b, .04)
        self.assertEqual(linear[-1], 0.)
        with self.assertRaises(ValueError):
            probabilities("early_2_4", 1 / 3)

    def test_six_sites_mask_inputs_to_all_six_layers(self):
        x, _ = dataset(4, 8, 101)
        model = StackedLSTM(16, probabilities("uniform", .1))
        self.assertEqual(len(model.layers), LAYERS)
        with patch("run.locked_dropout", wraps=locked_dropout) as dropout:
            model(x, mask_generators(0, torch.device("cpu")))
        self.assertEqual(dropout.call_count, 6)
        self.assertEqual([call.args[0].shape[-1] for call in dropout.call_args_list], [9, 16, 16, 16, 16, 16])

    def test_masks_locked_in_time_and_independent_across_examples(self):
        x = torch.ones(32, 12, 32)
        gen = torch.Generator().manual_seed(0)
        masked = locked_dropout(x, .2, gen, True)
        torch.testing.assert_close(masked[:, 0], masked[:, -1])
        self.assertFalse(torch.equal(masked[0], masked[1]))
        self.assertLess(abs(masked.mean().item() - 1), .06)
        self.assertIs(locked_dropout(x, .2, gen, False), x)

    def test_data_prefix_pairing_and_balanced_labels(self):
        x, y = dataset(64, 12, 101)
        long_x, long_y = dataset(64, 24, 101)
        torch.testing.assert_close(x, long_x[:, :12])
        torch.testing.assert_close(y, long_y)
        self.assertTrue(torch.equal(y.bincount(), torch.full((8,), 8)))
        self.assertEqual(x[:, 4:, -1].count_nonzero().item(), 0)
        other, _ = dataset(64, 12, 202)
        self.assertFalse(torch.equal(x, other))

    def test_arms_share_initial_parameters_and_eval_predictions(self):
        x, _ = dataset(4, 8, 101)
        first = None
        for arm in ARMS:
            torch.manual_seed(7)
            model = StackedLSTM(8, probabilities(arm, .1)).eval()
            pred = model(x, mask_generators(7, torch.device("cpu")))
            if first is None:
                first = pred
            else:
                torch.testing.assert_close(pred, first, rtol=0, atol=0)

    def test_all_arms_backpropagate_to_first_layer(self):
        x, y = dataset(8, 8, 101)
        for arm in ARMS:
            torch.manual_seed(7)
            model = StackedLSTM(8, probabilities(arm, .1))
            loss = torch.nn.functional.cross_entropy(
                model(x, mask_generators(7, torch.device("cpu"))), y)
            loss.backward()
            grad = model.layers[0].weight_ih_l0.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
