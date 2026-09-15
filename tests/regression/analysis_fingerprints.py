"""Compare aggregate outputs using identical controlled trial records.

Usage: python tests/regression/analysis_fingerprints.py SOURCE_ROOT OUTPUT.json
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
if (root / "experiments/benchmark/run.py").exists():
    import experiments.benchmark.run as main
    import experiments.benchmark.run_jannis_100 as jannis
    import experiments.benchmark.run_transformer_sidecar as side
else:
    from dropout_mft.experiments.benchmark import cli as main
    from dropout_mft.experiments.benchmark import jannis_100 as jannis
    from dropout_mft.experiments.benchmark import sidecar as side
from dropout_mft.experiments.benchmark import protocol as p  # noqa: E402
from dropout_mft.experiments.benchmark.jannis_100 import (  # noqa: E402
    jannis_100epoch_specs,
)
from dropout_mft.provenance import sha256_file  # noqa: E402
from dropout_mft.results import save_npz_result  # noqa: E402

results = {}
with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
    tmp = Path(tmp)

    def records(profiles):
        return [
            dict(
                cell=f"{dataset}/transformer",
                dataset=dataset,
                model_kind="transformer",
                profile_id=profile,
                seed=seed,
                learning_rate=1e-4,
                mean_dropout=0.1,
                test_loss=1.0 + seed * 0.001 - index * 0.02,
                test_accuracy=0.4 + index * 0.01,
                final_epoch_test_loss=1.1 + seed * 0.001 - index * 0.02,
                final_epoch_test_accuracy=0.38 + index * 0.01,
                validation_loss=0.9 + seed * 0.001 - index * 0.02,
                validation_accuracy=0.5 + index * 0.01,
                test_evaluated=True,
                split_hash="fixed",
                split_protocol="test",
            )
            for dataset in ["fi2010", "openml_jannis"]
            for index, profile in enumerate(profiles)
            for seed in p.CONFIRM_SEEDS
        ]

    main._collect = lambda *_: records(["uniform", "step_early", "big_step"])
    main.command_aggregate(SimpleNamespace(run_dir=tmp / "main", resamples=500))
    results["benchmark"] = json.loads(
        (tmp / "main/summary/confirm_summary.json").read_text()
    )
    side._collect_manifest = lambda *_: records(["linear_early", "none_tuned"])
    side._baseline_records = lambda *_: records(["uniform", p.CONTROL_PROFILE_ID])
    side.command_aggregate(
        SimpleNamespace(run_dir=tmp / "side", baseline_run_dir=tmp / "baseline")
    )
    results["sidecar"] = json.loads(
        (tmp / "side/summary/linear_sidecar_summary.json").read_text()
    )
    run = tmp / "jannis"
    specs = jannis_100epoch_specs()
    p.write_benchmark_manifest(run / "manifests/confirm.jsonl", specs, provenance={})
    for i, spec in enumerate(specs):
        output = p.trial_output_path(run, spec)
        cp = p.trial_checkpoint_path(run, spec)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(b"checkpoint")
        save_npz_result(
            output,
            {
                "trial": {"trial_id": spec.trial_id},
                "checkpoint": {"sha256": sha256_file(cp)},
                "test": {"loss": 1.0 - i * 0.001, "accuracy": 0.4 + i * 0.001},
                "selection": {"selected_epoch": 15 + i % 5},
            },
        )
    jannis.command_aggregate(SimpleNamespace(run_dir=run))
    results["jannis"] = json.loads((run / "summary/confirm_summary.json").read_text())
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
            for k, v in results.items()
        },
        sort_keys=True,
    )
    + "\n"
)
print(f"{len(results)} aggregate summaries written to {sys.argv[2]}")
