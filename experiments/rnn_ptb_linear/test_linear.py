import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import extend
from compare import analyze


class LinearTests(unittest.TestCase):
    def test_reversed_profile_runs_training_and_records_rates(self):
        base = extend.base
        base.torch.set_num_threads(1)
        rates = extend.probabilities("linear_increasing", .05)
        self.assertEqual(rates, list(reversed(extend.original_probabilities("linear_decreasing", .05))))
        self.assertAlmostEqual(sum(rates), .3)
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(out=Path(d), stage="canary", hidden=8, device="cpu",
                                   lr=.001, epochs=1, bptt=5, max_batches=2)
            corpus = {"vocab": {str(i): i for i in range(12)},
                      "train": base.batchify(base.torch.arange(50) % 12, 2, "cpu"),
                      "valid": base.batchify(base.torch.arange(30) % 12, 2, "cpu")}
            with patch.object(base, "probabilities", extend.probabilities), patch("builtins.print"):
                r = base.trial(args, "linear_increasing", .05, 1, corpus, {})
            self.assertEqual(r["rates"], rates)
            self.assertGreater(r["train_clean"]["perplexity"], 1)

    def test_protocol_and_comparison_reject_mismatched_experiments(self):
        proto = {"source_sha256": extend.BASE_SOURCE_SHA256, "epochs": 20}
        extend.validate_protocol(proto, dict(proto))
        with self.assertRaises(ValueError):
            extend.validate_protocol({**proto, "epochs": 1}, proto)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parent, out = root / "parent", root / "followup"
            parent.mkdir()
            out.mkdir()
            (parent / "selection.json").write_text(json.dumps({"protocol": proto, "selected_mean_p": .05}))
            ext_hash = hashlib.sha256(Path(extend.__file__).read_bytes()).hexdigest()
            for seed in (1, 2, 3):
                for arm in ("none", "uniform", "linear_decreasing", "linear_increasing"):
                    folder = (out if arm == "linear_increasing" else parent) / "confirm" / f"seed-{seed}" / f"{arm}-p0.05"
                    folder.mkdir(parents=True)
                    r = {"protocol": dict(proto), "arm": arm, "seed": seed, "mean_p": .05,
                         "stage": "confirm", "test": {"perplexity": 200 + seed},
                         "train_clean": {"perplexity": 150}, "best_valid_perplexity": 210}
                    if arm == "linear_increasing":
                        r["protocol"]["linear_extension_sha256"] = ext_hash
                    (folder / "result.json").write_text(json.dumps(r))
            with patch("builtins.print"):
                analyze(parent, out)
            self.assertIn("0.00 ± 0.00", (out / "summary.md").read_text())
            r["protocol"]["epochs"] = 1
            (folder / "result.json").write_text(json.dumps(r))
            with self.assertRaises(ValueError):
                analyze(parent, out)


if __name__ == "__main__":
    unittest.main()
