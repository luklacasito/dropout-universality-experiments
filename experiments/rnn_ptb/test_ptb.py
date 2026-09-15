import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from report import ARMS, CALIBRATION_RATES, select_calibration
from run import (LanguageModel, batchify, batches, detach_state, encode,
                 locked_dropout, pass_epoch, probabilities, vocabulary,
                 atomic_checkpoint, trial, main)


def generators():
    return [torch.Generator().manual_seed(100 + i) for i in range(6)]


class PTBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_only_vocabulary_maps_unseen_words_to_unk(self):
        vocab = vocabulary(["a", "a", "b", "<unk>", "<eos>"])
        self.assertNotIn("validation-only", vocab)
        self.assertEqual(encode(["validation-only"], vocab).item(), vocab["<unk>"])

    def test_next_word_targets_and_tail_are_correct(self):
        streams = batchify(torch.arange(23), 2, "cpu")
        chunks = list(batches(streams, 4))
        self.assertEqual([x.shape[1] for x, _ in chunks], [4, 4, 2])
        self.assertEqual(sum(y.numel() for _, y in chunks), 20)
        for x, y in chunks:
            torch.testing.assert_close(y, x + 1)
        # No wrap from the end of one stream into another row.
        self.assertEqual(chunks[-1][1].tolist(), [[9, 10], [20, 21]])

    def test_state_carry_matches_unsplit_forward_and_detaches(self):
        torch.manual_seed(2)
        m = LanguageModel(12, 8, [0.] * 6).eval()
        x = torch.arange(10).reshape(1, 10)
        full, _ = m(x, None, generators())
        first, state = m(x[:, :4], None, generators())
        detached = detach_state(state)
        self.assertTrue(all(not v.requires_grad for pair in detached for v in pair))
        second, _ = m(x[:, 4:], detached, generators())
        torch.testing.assert_close(torch.cat([first, second], dim=1), full)

    def test_rates_and_all_masks_act_after_embedding(self):
        for arm in ARMS[1:]:
            self.assertAlmostEqual(sum(probabilities(arm, .1)), .6)
        m = LanguageModel(12, 16, probabilities("uniform", .1))
        with patch("run.locked_dropout", wraps=locked_dropout) as masks:
            m(torch.ones(2, 7, dtype=torch.long), None, generators())
        self.assertEqual([c.args[0].shape[-1] for c in masks.call_args_list], [16] * 6)
        out = locked_dropout(torch.ones(20, 7, 16), .2, generators()[0], True)
        torch.testing.assert_close(out[:, 0], out[:, -1])

    def test_training_and_eval_are_finite_and_eval_does_not_mutate_weights(self):
        m = LanguageModel(12, 8, probabilities("early_2_4", .1))
        streams = batchify(torch.arange(50) % 12, 2, "cpu")
        opt = torch.optim.Adam(m.parameters())
        r = pass_epoch(m, streams, 5, generators(), opt)
        self.assertEqual(r["tokens"], 48)
        state = copy.deepcopy(m.state_dict())
        r = pass_epoch(m, streams, 5, generators())
        self.assertGreater(r["perplexity"], 1.)
        for name, value in state.items():
            torch.testing.assert_close(value, m.state_dict()[name], rtol=0, atol=0)

    def test_selection_uses_only_validation_and_keeps_zero_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for p in CALIBRATION_RATES:
                arm = "none" if p == 0 else "uniform"
                folder = root / "calibrate" / "seed-0" / f"{arm}-p{p:g}"
                folder.mkdir(parents=True)
                # Zero beats every positive rate; positive optimum is .05.
                loss = 1. if p == 0 else 2. + abs(p - .05)
                r = {"stage": "calibrate", "seed": 0, "arm": arm, "mean_p": p,
                     "test": None, "protocol": {"fixed": True}, "best_valid_loss": loss,
                     "best_valid_perplexity": 10 * loss, "fingerprint": str(p)}
                (folder / "result.json").write_text(json.dumps(r))
            with patch("builtins.print"):
                selected = select_calibration(root)
            self.assertEqual(selected["selected_mean_p"], .05)
            self.assertFalse(selected["dropout_helped_calibration_seed"])
            path = root / "calibrate" / "seed-0" / "none-p0" / "result.json"
            r = json.loads(path.read_text())
            r["test"] = {"perplexity": 1.}
            path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, "leakage"):
                select_calibration(root)

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = {"vocab": {str(i): i for i in range(12)},
                      "train": batchify(torch.arange(50) % 12, 2, "cpu"),
                      "valid": batchify(torch.arange(30) % 12, 2, "cpu")}
            args = SimpleNamespace(out=root / "resumed", stage="canary", hidden=8,
                                   device="cpu", lr=.001, epochs=2, bptt=5, max_batches=2)
            def interrupt_after_checkpoint(path, contents):
                atomic_checkpoint(path, contents)
                if path.name == "latest.pt" and contents["epoch"] == 1:
                    raise RuntimeError("simulated interruption")
            with patch("run.atomic_checkpoint", side_effect=interrupt_after_checkpoint), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    trial(args, "uniform", .1, 0, corpus, {"fixed": True})
            with patch("builtins.print"):
                resumed = trial(args, "uniform", .1, 0, corpus, {"fixed": True})
                args.out = root / "full"
                full = trial(args, "uniform", .1, 0, corpus, {"fixed": True})
            self.assertEqual(resumed["best_valid_loss"], full["best_valid_loss"])
            a = torch.load(root / "resumed/canary/seed-0/uniform-p0.1/latest.pt", weights_only=False)
            b = torch.load(root / "full/canary/seed-0/uniform-p0.1/latest.pt", weights_only=False)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)

    def test_array_dispatch_runs_only_the_requested_trial(self):
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(out=Path(d), data_root=Path(d), stage="calibrate",
                                   device="cpu", threads=1, max_batches=0,
                                   batch_size=64, seed=0, trial_index=3)
            words = ["<unk>", "<eos>"] + [str(i) for i in range(9998)]
            with patch("run.read_words", return_value=words), patch("run.protocol", return_value={}), \
                    patch("run.trial") as run_trial, patch("run.select_calibration") as select:
                main(args)
                self.assertEqual(run_trial.call_count, 1)
                self.assertEqual(run_trial.call_args.args[1:4], ("uniform", .05, 0))
                select.assert_not_called()
                (args.out / "selection.json").write_text(json.dumps({"protocol": {}, "selected_mean_p": .025}))
                args.stage, args.seed, args.trial_index = "confirm", 2, 4
                run_trial.reset_mock()
                main(args)
                self.assertEqual(run_trial.call_count, 1)
                self.assertEqual(run_trial.call_args.args[1:4], ("linear_decreasing", .025, 2))


if __name__ == "__main__":
    unittest.main()
