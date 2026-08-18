"""Exact, isolated rerun of the paper's original CIFAR-10 MLP recipe.

This module deliberately does not extend :mod:`dropout_mft.scale_transfer`.
That protocol has a validation split, independent named RNG streams, and a
different batch size.  Adding legacy switches to its ``TrialSpec`` would also
change the content hashes of already frozen manifests.  The comparison here is
therefore a separate, content-addressed cohort whose reference arms can be
checked against ``results/mlp/dropout_experiment_results.npz``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import MLPConfig, build_mlp
from .provenance import provenance_sha256
from .results import load_npz_result, save_npz_result
from .schedules import power_profile_layers, schedule_layers
from .training import seed_everything


LEGACY_SCHEMA_VERSION = 1
LEGACY_PHASE = "legacy_apples_to_apples"
LEGACY_DATA_PROTOCOL = "cifar10_randomstate0_train_then_test_unstratified_v1"
LEGACY_TRAIN_INDEX_HASH = (
    "6bc58f8926f80f0b7913320eede89a0b85492fbb91844e5f6fb3e84199ce1ca5"
)
LEGACY_TEST_INDEX_HASH = (
    "6fce1ad32c7b2b132d367364e713742bca4d46be2ea9b80180072113b2afae51"
)
LEGACY_PROFILE_IDS = (
    "uniform",
    "linear_early",
    "step_early",
    "quadratic_early",
    "quartic_early",
)
# This is deliberately an extension rather than a sixth arm of the frozen
# 125-trial cohort.  At \bar p=0.10 and L=6 it has p=(.30,.30,0,0,0,0), the
# historical ``big_step`` profile used in the h-sweep.  Its .30 peak therefore
# does *not* satisfy the .20 cap used for the new quadratic/quartic profiles.
# It is useful as a historical benchmark, but not a cap-matched primary test.
LEGACY_EXTENSION_PROFILE_IDS = ("big_step",)
ALL_LEGACY_PROFILE_IDS = LEGACY_PROFILE_IDS + LEGACY_EXTENSION_PROFILE_IDS
LEGACY_SEEDS = tuple(range(42, 67))


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _indices_hash(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class LegacyTrialSpec:
    """One trial in the exact-paper comparison cohort."""

    profile_id: str
    seed: int
    phase: str = LEGACY_PHASE
    dataset: str = "cifar10"
    activation: str = "relu"
    parameterization: str = "sp"
    mean_dropout: float = 0.10
    max_dropout: float = 0.20
    depth: int = 6
    width: int = 256
    epochs: int = 75
    batch_size: int = 100
    learning_rate: float = 1e-4
    lr_min: float = 1e-7
    weight_decay: float = 1e-7
    sigma_w_sq: float = 1.98
    sigma_b_sq: float = 0.02
    train_size: int = 5000
    test_size: int = 5000
    data_seed: int = 0

    def __post_init__(self) -> None:
        if self.profile_id not in ALL_LEGACY_PROFILE_IDS:
            raise ValueError(f"Unknown legacy profile: {self.profile_id!r}")
        if self.phase != LEGACY_PHASE:
            raise ValueError(f"phase must be {LEGACY_PHASE!r}")
        if self.dataset != "cifar10" or self.activation != "relu":
            raise ValueError("The exact legacy cohort is CIFAR-10/ReLU only")
        if self.parameterization != "sp":
            raise ValueError("The exact legacy cohort uses standard parameterization")
        positive = (
            self.depth,
            self.width,
            self.epochs,
            self.batch_size,
            self.train_size,
            self.test_size,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in positive):
            raise ValueError(
                "architecture, duration, batch, and data sizes must be positive"
            )
        if not 0 <= self.mean_dropout <= self.max_dropout < 1:
            raise ValueError("dropout budget must satisfy 0 <= mean <= max < 1")
        if self.learning_rate <= 0 or not 0 <= self.lr_min <= self.learning_rate:
            raise ValueError("learning rates must satisfy 0 <= lr_min <= learning_rate")
        if self.weight_decay < 0 or self.sigma_w_sq <= 0 or self.sigma_b_sq < 0:
            raise ValueError("invalid optimizer or initialization scale")

    @property
    def trial_id(self) -> str:
        return _sha256_payload(asdict(self))[:20]

    @property
    def config_hash(self) -> str:
        values = asdict(self)
        values.pop("seed")
        return _sha256_payload(values)[:20]


@dataclass(frozen=True)
class LegacyDatasetBundle:
    train: torch.utils.data.TensorDataset
    test: torch.utils.data.TensorDataset
    split_hash: str
    train_index_hash: str
    test_index_hash: str
    protocol: str = LEGACY_DATA_PROTOCOL


def legacy_trial_specs(
    profile_ids: tuple[str, ...] = LEGACY_PROFILE_IDS,
) -> list[LegacyTrialSpec]:
    """Return a prespecified paired legacy cohort for the requested profiles."""

    if not profile_ids or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("profile_ids must be a nonempty set of unique profiles")
    unknown = set(profile_ids) - set(ALL_LEGACY_PROFILE_IDS)
    if unknown:
        raise ValueError(f"Unknown legacy profiles: {sorted(unknown)!r}")

    return [
        LegacyTrialSpec(
            profile_id=profile_id,
            seed=seed,
            # Historical big step has 3\bar p in its first L/3 layers.
            max_dropout=0.30 if profile_id == "big_step" else 0.20,
        )
        for profile_id in profile_ids
        for seed in LEGACY_SEEDS
    ]


def legacy_profile_layers(spec: LegacyTrialSpec) -> list[float]:
    """Build the exact legacy references and exact-budget new power profiles."""

    if spec.profile_id == "uniform":
        values = schedule_layers(
            "constant", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "linear_early":
        # This is the notebook's ``reverse_linear`` profile, including its
        # endpoint samples 0.2,...,0.0 rather than cell-center samples.
        values = schedule_layers(
            "reverse_linear", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "step_early":
        values = schedule_layers(
            "reverse_step", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id in {"quadratic_early", "quartic_early"}:
        power = 2.0 if spec.profile_id == "quadratic_early" else 4.0
        values = power_profile_layers(
            spec.depth,
            spec.mean_dropout,
            power,
            orientation="early",
            h_max=spec.max_dropout,
        )
    elif spec.profile_id == "big_step":
        values = schedule_layers(
            "big_step", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    else:  # Defensive guard for future profile additions.
        raise ValueError(f"Unknown legacy profile: {spec.profile_id!r}")
    if not math.isclose(
        float(np.mean(values)), spec.mean_dropout, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("legacy profile does not preserve its discrete budget")
    if max(values, default=0.0) > spec.max_dropout + 1e-12:
        raise RuntimeError("legacy profile exceeds its dropout cap")
    return [float(value) for value in values]


def legacy_cifar10_indices(
    *,
    train_population_size: int = 50_000,
    test_population_size: int = 10_000,
    train_size: int = 5_000,
    test_size: int = 5_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the notebook's two sequential ``RandomState`` draws."""

    sizes = (
        train_population_size,
        test_population_size,
        train_size,
        test_size,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in sizes
    ):
        raise ValueError("population and subset sizes must be integers")
    if not 0 < train_size <= train_population_size:
        raise ValueError("train_size must lie within the training population")
    if not 0 < test_size <= test_population_size:
        raise ValueError("test_size must lie within the test population")
    rng = np.random.RandomState(seed)
    train_indices = np.asarray(
        rng.choice(train_population_size, train_size, replace=False), dtype=np.int64
    )
    test_indices = np.asarray(
        rng.choice(test_population_size, test_size, replace=False), dtype=np.int64
    )
    return train_indices, test_indices


def _tensor_dataset(
    images: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
) -> torch.utils.data.TensorDataset:
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    inputs = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).float().div_(255)
    inputs = (inputs - mean) / std
    labels = torch.as_tensor(np.asarray(targets)[indices], dtype=torch.long)
    return torch.utils.data.TensorDataset(inputs, labels)


def load_legacy_cifar10(
    root: str | Path,
    *,
    download: bool = False,
) -> LegacyDatasetBundle:
    """Load exactly the 5,000/5,000 unstratified subsets used in the paper."""

    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("The legacy CIFAR-10 rerun requires torchvision") from exc

    train_source = datasets.CIFAR10(root=str(root), train=True, download=download)
    test_source = datasets.CIFAR10(root=str(root), train=False, download=download)
    train_indices, test_indices = legacy_cifar10_indices(
        train_population_size=len(train_source.data),
        test_population_size=len(test_source.data),
    )
    train_hash = _indices_hash(train_indices)
    test_hash = _indices_hash(test_indices)
    if train_hash != LEGACY_TRAIN_INDEX_HASH or test_hash != LEGACY_TEST_INDEX_HASH:
        raise RuntimeError(
            "CIFAR-10 legacy subset indices do not match the frozen hashes"
        )
    split_hash = _sha256_payload(
        {
            "protocol": LEGACY_DATA_PROTOCOL,
            "dataset": "cifar10",
            "train_index_hash": train_hash,
            "test_index_hash": test_hash,
        }
    )
    return LegacyDatasetBundle(
        train=_tensor_dataset(
            train_source.data, np.asarray(train_source.targets), train_indices
        ),
        test=_tensor_dataset(
            test_source.data, np.asarray(test_source.targets), test_indices
        ),
        split_hash=split_hash,
        train_index_hash=train_hash,
        test_index_hash=test_hash,
    )


def manifest_content(specs: list[LegacyTrialSpec], provenance: dict) -> str:
    """Serialize a deterministic, provenance-bound legacy manifest."""

    provenance_hash = provenance_sha256(provenance)
    rows = []
    for spec in sorted(specs, key=lambda item: item.trial_id):
        rows.append(
            {
                "schema_version": LEGACY_SCHEMA_VERSION,
                "artifact_type": "legacy_apples_to_apples_trial_plan",
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "factors": asdict(spec),
                "dropout_probabilities": legacy_profile_layers(spec),
                "provenance_sha256": provenance_hash,
            }
        )
    return "".join(canonical_json(row) + "\n" for row in rows)


def write_legacy_manifest(
    path: str | Path, specs: list[LegacyTrialSpec], *, provenance: dict
) -> Path:
    path = Path(path)
    content = manifest_content(specs, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite a different manifest: {path}")
        return path
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
    return path


def read_legacy_manifest(
    path: str | Path, *, provenance: dict | None = None
) -> list[LegacyTrialSpec]:
    path = Path(path)
    expected_provenance = provenance_sha256(provenance) if provenance else None
    specs: list[LegacyTrialSpec] = []
    seen: set[str] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"Legacy manifest is absent: {path}") from exc
    for line in lines:
        try:
            row = json.loads(line)
            spec = LegacyTrialSpec(**row["factors"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid legacy manifest row in {path}") from exc
        valid = (
            row.get("schema_version") == LEGACY_SCHEMA_VERSION
            and row.get("artifact_type") == "legacy_apples_to_apples_trial_plan"
            and row.get("trial_id") == spec.trial_id
            and row.get("config_hash") == spec.config_hash
            and np.allclose(
                row.get("dropout_probabilities", []),
                legacy_profile_layers(spec),
                rtol=0.0,
                atol=1e-15,
            )
            and (
                expected_provenance is None
                or row.get("provenance_sha256") == expected_provenance
            )
        )
        if not valid or spec.trial_id in seen:
            raise ValueError(f"Legacy manifest row failed hash validation: {path}")
        seen.add(spec.trial_id)
        specs.append(spec)
    if not specs:
        raise ValueError(f"Legacy manifest contains no trials: {path}")
    return sorted(specs, key=lambda item: item.trial_id)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _iterate_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
):
    count = inputs.shape[0]
    indices = (
        torch.randperm(count, device=inputs.device)
        if shuffle
        else torch.arange(count, device=inputs.device)
    )
    for start in range(0, count, batch_size):
        selected = indices[start : start + batch_size]
        yield inputs[selected], targets[selected]


def _autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def _evaluate_legacy(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch_inputs, batch_targets in _iterate_batches(
        inputs, targets, batch_size, shuffle=False
    ):
        with _autocast(device):
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
        loss_sum += float(loss) * batch_inputs.shape[0]
        correct += int((logits.argmax(dim=1) == batch_targets).sum())
        count += batch_inputs.shape[0]
    return loss_sum / count, 100.0 * correct / count


def _train_exact_legacy(
    spec: LegacyTrialSpec,
    bundle: LegacyDatasetBundle,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # The notebook places its complete subset tensors on the accelerator before
    # entering the seed loop. Tensor transfers do not consume RNG state.
    train_inputs, train_targets = (
        tensor.to(resolved) for tensor in bundle.train.tensors
    )
    test_inputs, test_targets = (tensor.to(resolved) for tensor in bundle.test.tensors)

    # One global stream is intentional: this exactly reproduces the notebook's
    # torch.manual_seed(seed), model construction, randperm, and dropout order.
    seed_everything(spec.seed)
    model = build_mlp(
        MLPConfig(
            width=spec.width,
            depth=spec.depth,
            activation="relu",
            sigma_w_sq=spec.sigma_w_sq,
            sigma_b_sq=spec.sigma_b_sq,
            output_dim=10,
            zero_readout=False,
        ),
        legacy_profile_layers(spec),
        parameterization="sp",
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=spec.epochs, eta_min=spec.lr_min
    )
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "learning_rate_used": [],
    }
    optimizer_steps = 0
    for _ in range(spec.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        history["learning_rate_used"].append(float(optimizer.param_groups[0]["lr"]))
        for batch_inputs, batch_targets in _iterate_batches(
            train_inputs, train_targets, spec.batch_size, shuffle=True
        ):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(resolved):
                logits = model(batch_inputs)
                loss = criterion(logits, batch_targets)
            # The original notebook creates a GradScaler but deliberately uses
            # plain backward/step; on H100 it uses bfloat16 autocast.
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * batch_inputs.shape[0]
            correct += int((logits.detach().argmax(dim=1) == batch_targets).sum())
            count += batch_inputs.shape[0]
        scheduler.step()
        test_loss, test_accuracy = _evaluate_legacy(
            model,
            test_inputs,
            test_targets,
            criterion,
            spec.batch_size,
            resolved,
        )
        history["train_loss"].append(loss_sum / count)
        history["train_accuracy"].append(100.0 * correct / count)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    curves = {key: np.asarray(value) for key, value in history.items()}
    runtime = {
        "device": str(resolved),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None
        ),
        "amp_dtype": (
            "bfloat16"
            if resolved.type == "cuda" and torch.cuda.is_bf16_supported()
            else "float16"
            if resolved.type == "cuda"
            else None
        ),
        "tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if resolved.type == "cuda"
            else False
        ),
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark) if resolved.type == "cuda" else False
        ),
    }
    compute = {
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_count,
        "examples_seen": spec.train_size * spec.epochs,
        "estimated_training_flops": 6 * parameter_count * spec.train_size * spec.epochs,
        "flop_model": "dense_train_6x_parameters_per_example_v1",
    }
    return curves, {"runtime": runtime, "compute": compute}


def run_legacy_trial(
    spec: LegacyTrialSpec,
    bundle: LegacyDatasetBundle,
    output_path: str | Path,
    *,
    provenance: dict,
    device: str = "auto",
    force: bool = False,
) -> dict:
    """Run one exact legacy trial with atomic, provenance-checked resume."""

    output_path = Path(output_path)
    provenance_hash = provenance_sha256(provenance)
    if output_path.exists() and not force:
        existing = load_npz_result(output_path)
        valid = (
            existing.get("schema_version") == LEGACY_SCHEMA_VERSION
            and existing.get("artifact_type") == "legacy_apples_to_apples_trial"
            and existing.get("trial", {}).get("trial_id") == spec.trial_id
            and existing.get("trial", {}).get("config_hash") == spec.config_hash
            and existing.get("trial", {}).get("status") == "complete"
            and existing.get("data", {}).get("split_hash") == bundle.split_hash
            and existing.get("provenance", {}).get("source_provenance_sha256")
            == provenance_hash
        )
        if valid:
            return existing
        raise ValueError(
            f"Existing legacy trial is corrupt or mismatched: {output_path}"
        )

    start = time.perf_counter()
    curves, metadata = _train_exact_legacy(spec, bundle, device=device)
    duration = time.perf_counter() - start
    result = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "artifact_type": "legacy_apples_to_apples_trial",
        "trial": {
            "trial_id": spec.trial_id,
            "config_hash": spec.config_hash,
            "phase": spec.phase,
            "status": "complete",
            "seed": spec.seed,
            "duration_seconds": duration,
        },
        "factors": asdict(spec),
        "schedule": {
            "profile_id": spec.profile_id,
            "dropout_probabilities": np.asarray(legacy_profile_layers(spec)),
            "mean_dropout_probability": spec.mean_dropout,
            "max_dropout_probability": spec.max_dropout,
            "budget_rule": "exact_discrete_probability_mean_with_cap",
        },
        "curves": {"epoch": np.arange(spec.epochs), **curves},
        "test": {
            "evaluated_every_epoch": True,
            "selected_epoch": spec.epochs - 1,
            "selection_rule": "final_epoch_only",
            "loss": float(curves["test_loss"][-1]),
            "accuracy_percent": float(curves["test_accuracy"][-1]),
        },
        "compute": {**metadata["compute"], "wall_seconds": duration},
        "data": {
            "dataset": "cifar10",
            "split_protocol": bundle.protocol,
            "split_hash": bundle.split_hash,
            "train_index_hash": bundle.train_index_hash,
            "test_index_hash": bundle.test_index_hash,
            "train_size": len(bundle.train),
            "validation_size": 0,
            "test_size": len(bundle.test),
        },
        "protocol": {
            "reference_notebook": "notebooks/mlp/mlp_dropout_scheduling_overfit.ipynb",
            "global_rng_stream": True,
            "test_evaluation_cadence": "every_epoch",
            "primary_endpoint": "final_epoch_test_cross_entropy",
            "scheduler": "CosineAnnealingLR_Tmax75_eta_min1e-7",
            "autocast_plain_backward": True,
        },
        "runtime": metadata["runtime"],
        "provenance": {"source_provenance_sha256": provenance_hash},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp.npz")
    save_npz_result(temporary, result)
    os.replace(temporary, output_path)
    return result
