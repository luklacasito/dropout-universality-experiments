"""Run identical tiny CPU trials against a supplied checkout, without downloads.

Usage: python tests/regression/training_fingerprints.py SOURCE_ROOT OUTPUT.json
Run once for each checkout, then compare the JSON files. Timings and runtime
provenance are excluded; scientific result fields and model tensors are hashed.
Requires the same Python/PyTorch environment for both runs.
"""

import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dropout_mft.experiments.benchmark import protocol as b  # noqa: E402
from dropout_mft.experiments.benchmark.datasets import BENCHMARK_SPECS  # noqa: E402
from dropout_mft.experiments.scale_transfer import protocol as scale  # noqa: E402
from dropout_mft.training import DatasetBundle, synthetic_bundle  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def normalized(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(normalized(value), sort_keys=True).encode()
    ).hexdigest()


states = {}
for module in (b, scale):
    original = module.train_model

    def capture(model, *args, _original=original, **kwargs):
        result = _original(model, *args, **kwargs)
        states["last"] = digest(model.state_dict())
        return result

    module.train_model = capture


def scientific_result(result):
    result = dict(result)
    result.pop("provenance")  # source location/commit/runtime intentionally differ
    result["trial"] = dict(result["trial"])
    result["trial"].pop("duration_seconds")
    result["compute"] = dict(result["compute"])
    result["compute"].pop("wall_seconds")
    if "checkpoint" in result:
        result["checkpoint"] = dict(result["checkpoint"])
        result["checkpoint"].pop("path", None)
        # Torch zip container names contain the temporary PID; hash tensors below.
        result["checkpoint"].pop("sha256", None)
    return result


def benchmark_bundle(dataset, model_kind):
    shape = BENCHMARK_SPECS[dataset]
    generator = torch.Generator().manual_seed(321)

    def samples(count):
        if model_kind == "mlp":
            x = torch.randn(count, shape.mlp_input_dim, generator=generator)
        elif shape.vocab_size:
            x = torch.randint(
                shape.vocab_size, (count, shape.sequence_length), generator=generator
            )
        elif shape.image_channels:
            x = torch.randn(
                count,
                shape.image_channels,
                shape.image_size,
                shape.image_size,
                generator=generator,
            )
        else:
            x = torch.randn(
                count, shape.sequence_length, shape.input_features, generator=generator
            )
        return torch.utils.data.TensorDataset(x, torch.arange(count) % shape.classes)

    return DatasetBundle(
        samples(8),
        samples(4),
        samples(4),
        "fixed-split",
        dataset,
        "synthetic-regression",
        "fixed-test",
        "synthetic-regression",
        321,
    )


results = {}
with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)

    def record(name, runner, spec, bundle, **kwargs):
        output = temporary / f"{len(results)}.npz"
        first = runner(spec, bundle, output, device="cpu", **kwargs)
        fingerprint = {"result": digest(scientific_result(first))}
        fingerprint["model_state"] = states["last"]
        if "checkpoint_path" in kwargs:
            checkpoint = torch.load(
                kwargs["checkpoint_path"], map_location="cpu", weights_only=True
            )
            fingerprint["best_checkpoint"] = digest(checkpoint)
        # Changing RNG before resume must neither retrain nor change any results.
        torch.manual_seed(987)
        torch.randn(101)
        before = output.read_bytes()
        resumed = runner(spec, bundle, output, device="cpu", **kwargs)
        assert digest(scientific_result(resumed)) == fingerprint["result"], name
        assert before == output.read_bytes(), name
        results[name] = fingerprint

    # Every modality/architecture, plus all schedules and sealed-test execution.
    for dataset in BENCHMARK_SPECS:
        for model_kind in ("mlp", "transformer"):
            bundle = benchmark_bundle(dataset, model_kind)
            profiles = (
                b.ALL_PROFILE_IDS
                if dataset == "openml_jannis"
                else ("uniform", "step_early")
            )
            for profile in profiles:
                spec = b.BenchmarkTrialSpec(
                    stage="confirm",
                    dataset=dataset,
                    model_kind=model_kind,
                    profile_id=profile,
                    mean_dropout=0 if profile in b.ZERO_DROPOUT_PROFILE_IDS else 0.1,
                    learning_rate=0.001,
                    seed=100,
                    depth=3,
                    width=8,
                    heads=2,
                    epochs=2,
                    batch_size=4,
                    train_size=8,
                    validation_size=4,
                    test_size=4,
                    evaluate_test=True,
                    weight_decay=0,
                )
                record(
                    f"benchmark/{dataset}/{model_kind}/{profile}",
                    b.run_benchmark_trial,
                    spec,
                    bundle,
                    checkpoint_path=temporary / f"{len(results)}.pt",
                )
            record(
                f"benchmark/{dataset}/{model_kind}/sealed",
                b.run_benchmark_trial,
                replace(spec, stage="lr_search", evaluate_test=False),
                bundle,
            )

    for parameterization in ("sp", "mup"):
        for profile in scale.PROFILE_IDS:
            spec = scale.TrialSpec(
                phase="profile_pilot",
                model_kind="mlp",
                parameterization=parameterization,
                dataset="cifar10",
                activation="relu",
                profile_id=profile,
                mean_dropout=0 if profile == "none" else 0.1,
                max_dropout=0.2,
                depth=3,
                width=8,
                base_width=4,
                delta_width=8,
                learning_rate=0.001,
                seed=100,
                epochs=2,
                batch_size=4,
                train_size=8,
                validation_size=4,
                test_size=4,
            )
            bundle = synthetic_bundle(
                input_dim=3072,
                classes=10,
                train_size=8,
                validation_size=4,
                test_size=4,
                seed=321,
            )
            record(f"scale/{parameterization}/{profile}", scale.run_trial, spec, bundle)
        record(
            f"scale/{parameterization}/field",
            scale.run_trial,
            replace(spec, profile_id="linear_early", budget_space="reference_field"),
            bundle,
        )

    spec = replace(
        scale.vit_confirmation_specs()[0],
        depth=1,
        width=16,
        epochs=2,
        batch_size=4,
        train_size=8,
        validation_size=4,
        test_size=4,
    )
    bundle = synthetic_bundle(
        input_dim=3072,
        classes=100,
        train_size=8,
        validation_size=4,
        test_size=4,
        seed=321,
    )

    def images(data):
        x, y = data.tensors
        return torch.utils.data.TensorDataset(x.reshape(-1, 3, 32, 32), y)

    bundle = replace(
        bundle,
        dataset="cifar100",
        train=images(bundle.train),
        validation=images(bundle.validation),
        test=images(bundle.test),
    )
    record("scale/vit", scale.run_trial, spec, bundle)

Path(sys.argv[2]).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(
    f"{len(results)} deterministic training and resume cases written to {sys.argv[2]}"
)
