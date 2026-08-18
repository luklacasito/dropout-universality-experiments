"""Run specifications and execution helpers for the scale-transfer study."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dropout_mft.fields import (
    dropout_field,
    field_matched_power_profile,
    propagated_field_profile,
    reference_field_profile,
)
from dropout_mft.models import MLPConfig, TinyViT, build_mlp, make_optimizer
from dropout_mft.provenance import provenance_sha256
from dropout_mft.results import load_npz_result, save_npz_result
from dropout_mft.schedules import field_damage, power_profile_layers, schedule_layers
from dropout_mft.training import DatasetBundle, TrainingConfig, seed_everything, train_model


SCHEMA_VERSION = 2
PROFILE_IDS = (
    "none",
    "uniform",
    "linear_early",
    "linear_late",
    "quadratic_early",
    "quadratic_late",
    "quartic_early",
    "quartic_late",
    "step_early",
    "step_late",
)
LR_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
TRANSFER_WIDTHS = (256, 512, 1024, 2048)


@dataclass(frozen=True)
class TrialSpec:
    phase: str
    model_kind: Literal["mlp", "vit"]
    parameterization: Literal["sp", "mup"]
    dataset: Literal["cifar10", "cifar100"]
    activation: Literal["relu", "gelu"]
    profile_id: str
    mean_dropout: float
    max_dropout: float
    depth: int
    width: int
    learning_rate: float
    seed: int
    budget_space: Literal[
        "dropout_probability", "reference_field"
    ] = "dropout_probability"
    zero_readout: bool = False
    evaluate_test: bool = True
    epochs: int = 75
    batch_size: int = 75
    sigma_w_sq: float = 1.98
    sigma_b_sq: float = 0.02
    base_width: int = 64
    delta_width: int = 128
    weight_decay: float = 1e-7
    lr_floor_ratio: float = 1e-3
    gradient_clip_norm: float | None = None
    train_size: int = 4000
    validation_size: int = 1000
    test_size: int = 5000
    split_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_IDS:
            raise ValueError(f"Unknown profile_id: {self.profile_id!r}")
        if self.model_kind == "vit" and self.parameterization != "sp":
            raise ValueError("The ViT confirmation is intentionally SP-only")
        if self.parameterization == "mup" and self.activation != "relu":
            raise ValueError("The first muP transfer study is restricted to ReLU")
        if self.budget_space not in {"dropout_probability", "reference_field"}:
            raise ValueError(f"Unknown budget_space: {self.budget_space!r}")
        if self.budget_space == "reference_field" and self.profile_id.startswith(
            "step_"
        ):
            raise ValueError(
                "Reference-field matching is not defined for step profiles"
            )
        if (
            self.depth <= 0
            or self.width <= 0
            or self.epochs <= 0
            or self.batch_size <= 0
        ):
            raise ValueError("depth, width, epochs, and batch_size must be positive")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("train, validation, and test sizes must be positive")
        if not 0 <= self.mean_dropout < 1 or not 0 < self.max_dropout < 1:
            raise ValueError("dropout probabilities must lie in [0, 1)")
        if self.profile_id != "none" and self.mean_dropout > self.max_dropout:
            raise ValueError("mean_dropout cannot exceed max_dropout")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")

    @property
    def trial_id(self) -> str:
        payload = canonical_json(asdict(self)).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        payload.pop("seed")
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:20]


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _derived_seed(stream: str, payload: object) -> int:
    digest = hashlib.sha256(f"{stream}\0{canonical_json(payload)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _profile_pair_family(profile_id: str) -> str:
    for suffix in ("_early", "_late"):
        if profile_id.endswith(suffix):
            return profile_id[: -len(suffix)]
    return profile_id


def seed_streams(spec: TrialSpec) -> dict:
    """Return independent seeds, with dropout CRNs shared by reversal pairs."""

    base = {"base_seed": spec.seed}
    pair_payload = asdict(spec)
    pair_payload["profile_id"] = _profile_pair_family(spec.profile_id)
    pair_key = hashlib.sha256(canonical_json(pair_payload).encode()).hexdigest()[:20]
    return {
        "scheme": "sha256_named_streams_v1",
        "base_seed": spec.seed,
        "initialization_seed": _derived_seed("initialization", base),
        "minibatch_seed": _derived_seed("minibatch", base),
        "dropout_seed": _derived_seed(
            "dropout_common_random_numbers", {"pair_key": pair_key}
        ),
        "dropout_crn_group": pair_key,
    }


def profile_layers(spec: TrialSpec) -> list[float]:
    profile = spec.profile_id
    if profile == "none":
        return [0.0] * spec.depth
    if profile == "uniform":
        power = 0.0
        orientation = "early"
    elif profile.startswith("linear_"):
        power = 1.0
    elif profile.startswith("quadratic_"):
        power = 2.0
    elif profile.startswith("quartic_"):
        power = 4.0
    elif profile == "step_early":
        return schedule_layers(
            "reverse_step", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif profile == "step_late":
        return schedule_layers("step", spec.depth, spec.mean_dropout, spec.max_dropout)
    else:  # pragma: no cover - TrialSpec validation protects callers
        raise ValueError(profile)
    if profile != "uniform":
        orientation = "early" if profile.endswith("_early") else "late"
    if spec.budget_space == "reference_field":
        target_field = dropout_field(
            spec.mean_dropout,
            variance=1.0,
            activation=spec.activation,
            sigma_w_sq=spec.sigma_w_sq,
            sigma_b_sq=spec.sigma_b_sq,
        )
        return field_matched_power_profile(
            depth=spec.depth,
            target_mean_field=target_field,
            power=power,
            orientation=orientation,
            p_max=spec.max_dropout,
            activation=spec.activation,
            reference_variance=1.0,
            sigma_w_sq=spec.sigma_w_sq,
            sigma_b_sq=spec.sigma_b_sq,
        )
    return power_profile_layers(
        spec.depth,
        spec.mean_dropout,
        power,
        orientation=orientation,
        h_max=spec.max_dropout,
    )


def schedule_metadata(spec: TrialSpec) -> dict:
    probabilities = profile_layers(spec)
    fields = reference_field_profile(
        probabilities,
        activation=spec.activation,
        sigma_w_sq=spec.sigma_w_sq,
        sigma_b_sq=spec.sigma_b_sq,
    )
    local_fields, variances = propagated_field_profile(
        probabilities,
        activation=spec.activation,
        sigma_w_sq=spec.sigma_w_sq,
        sigma_b_sq=spec.sigma_b_sq,
    )
    activation_class = "kinked" if spec.activation == "relu" else "smooth"
    damage = field_damage(fields, activation_class)
    return {
        "profile_id": spec.profile_id,
        "budget_space": spec.budget_space,
        "dropout_probabilities": np.asarray(probabilities),
        "mean_dropout_probability": float(np.mean(probabilities)),
        "max_dropout_probability": float(max(probabilities, default=0.0)),
        "field_mapping": (
            "exact_mlp_one_step_at_reference_variance_q1"
            if spec.model_kind == "mlp"
            else "mlp_one_step_proxy_at_reference_variance_q1"
        ),
        "interpretation": (
            "mlp_forward_field"
            if spec.model_kind == "mlp"
            else "mlp_inspired_profile_proxy_not_a_vit_recursion"
        ),
        "mean_reference_field": float(np.mean(fields)),
        "reference_fields": np.asarray(fields),
        "propagated_local_fields": np.asarray(local_fields),
        "variance_trajectory": np.asarray(variances),
        "field_damage_mean": damage,
        "field_damage_total": damage * spec.depth,
        "xi_proxy": float("inf") if damage == 0 else 1.0 / damage,
    }


def profile_pilot_specs() -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for profile in PROFILE_IDS:
        for seed in range(3):
            specs.append(
                TrialSpec(
                    phase="profile_pilot",
                    model_kind="mlp",
                    parameterization="sp",
                    dataset="cifar10",
                    activation="relu",
                    profile_id=profile,
                    mean_dropout=0.0 if profile == "none" else 0.10,
                    max_dropout=0.20,
                    depth=6,
                    width=256,
                    learning_rate=1e-4,
                    seed=seed,
                    evaluate_test=False,
                )
            )
    return specs


def profile_confirmation_specs() -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    profiles = (
        "uniform",
        "quadratic_early",
        "quadratic_late",
        "step_early",
        "step_late",
    )
    for activation in ("relu", "gelu"):
        sigma_w_sq, sigma_b_sq = (1.98, 0.02) if activation == "relu" else (2.20, 0.0)
        for profile in profiles:
            for seed in range(100, 110):
                specs.append(
                    TrialSpec(
                        phase="profile_confirmation",
                        model_kind="mlp",
                        parameterization="sp",
                        dataset="cifar10",
                        activation=activation,
                        profile_id=profile,
                        mean_dropout=0.10,
                        max_dropout=0.20,
                        depth=6,
                        width=256,
                        learning_rate=1e-4,
                        seed=seed,
                        sigma_w_sq=sigma_w_sq,
                        sigma_b_sq=sigma_b_sq,
                    )
                )
    return specs


def mup_tune_specs() -> list[TrialSpec]:
    return [
        TrialSpec(
            phase="lr_proxy",
            model_kind="mlp",
            parameterization=parameterization,
            dataset="cifar10",
            activation="relu",
            profile_id="uniform",
            mean_dropout=0.10,
            max_dropout=0.20,
            depth=6,
            width=128,
            learning_rate=learning_rate,
            seed=seed,
            zero_readout=True,
            evaluate_test=False,
        )
        for parameterization in ("sp", "mup")
        for learning_rate in LR_GRID
        for seed in range(3)
    ]


def mup_tune_extension_specs() -> list[TrialSpec]:
    """Conditional validation-only LR grid used only after an unstable base gate."""

    return [
        TrialSpec(
            phase="lr_proxy_extension",
            model_kind="mlp",
            parameterization=parameterization,
            dataset="cifar10",
            activation="relu",
            profile_id="uniform",
            mean_dropout=0.10,
            max_dropout=0.20,
            depth=6,
            width=128,
            learning_rate=learning_rate,
            seed=seed,
            zero_readout=True,
            evaluate_test=False,
        )
        for parameterization in ("sp", "mup")
        for learning_rate in LR_GRID
        for seed in range(3, 6)
    ]


def mup_transfer_specs(selected_lrs: dict[str, float]) -> list[TrialSpec]:
    profiles = (
        "uniform",
        "quadratic_early",
        "quadratic_late",
        "step_early",
        "step_late",
    )
    missing = {"sp", "mup"} - set(selected_lrs)
    if missing:
        raise ValueError(f"Missing selected learning rates for: {sorted(missing)}")
    return [
        TrialSpec(
            phase="width_transfer",
            model_kind="mlp",
            parameterization=parameterization,
            dataset="cifar10",
            activation="relu",
            profile_id=profile,
            mean_dropout=0.10,
            max_dropout=0.20,
            depth=6,
            width=width,
            learning_rate=float(selected_lrs[parameterization]),
            seed=seed,
            zero_readout=True,
        )
        for parameterization in ("sp", "mup")
        for width in TRANSFER_WIDTHS
        for profile in profiles
        for seed in range(100, 110)
    ]


def oracle_specs() -> list[TrialSpec]:
    return [
        TrialSpec(
            phase="target_oracle",
            model_kind="mlp",
            parameterization=parameterization,
            dataset="cifar10",
            activation="relu",
            profile_id="uniform",
            mean_dropout=0.10,
            max_dropout=0.20,
            depth=6,
            width=width,
            learning_rate=learning_rate,
            seed=seed,
            zero_readout=True,
            evaluate_test=False,
        )
        for parameterization in ("sp", "mup")
        for width in TRANSFER_WIDTHS
        for learning_rate in LR_GRID
        for seed in range(3)
    ]


def vit_confirmation_specs() -> list[TrialSpec]:
    """Residual-block ViT bridge using the original CIFAR-100 ViT recipe."""

    return [
        TrialSpec(
            phase="vit_confirmation",
            model_kind="vit",
            parameterization="sp",
            dataset="cifar100",
            activation="relu",
            profile_id=profile,
            mean_dropout=0.10,
            max_dropout=0.20,
            depth=10,
            width=128,
            learning_rate=5e-4,
            seed=seed,
            batch_size=250,
            sigma_w_sq=1.98,
            sigma_b_sq=0.02,
            weight_decay=0.0,
            lr_floor_ratio=0.01,
            gradient_clip_norm=1.0,
            train_size=1600,
            validation_size=400,
            test_size=5000,
        )
        for profile in (
            "uniform",
            "quadratic_early",
            "quadratic_late",
            "step_early",
            "step_late",
        )
        for seed in range(200, 206)
    ]


def write_manifest(
    path: str | Path,
    specs: list[TrialSpec],
    *,
    provenance: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(specs, key=lambda spec: spec.trial_id)
    provenance_hash = provenance_sha256(provenance)
    content = "".join(
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "provenance": provenance,
                "provenance_sha256": provenance_hash,
                "randomization": seed_streams(spec),
                "dropout_probabilities": profile_layers(spec),
                **asdict(spec),
            }
        )
        + "\n"
        for spec in ordered
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def read_manifest(path: str | Path) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.pop("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported manifest schema")
        trial_id = row.pop("trial_id")
        config_hash = row.pop("config_hash")
        provenance = row.pop("provenance")
        if row.pop("provenance_sha256") != provenance_sha256(provenance):
            raise ValueError("Manifest provenance hash does not match its contents")
        expected_randomization = row.pop("randomization")
        expected_probabilities = row.pop("dropout_probabilities")
        spec = TrialSpec(**row)
        if trial_id != spec.trial_id:
            raise ValueError("Manifest trial hash does not match its contents")
        if config_hash != spec.config_hash:
            raise ValueError("Manifest config hash does not match its contents")
        if expected_randomization != seed_streams(spec):
            raise ValueError("Manifest randomization does not match its contents")
        if not np.allclose(
            expected_probabilities, profile_layers(spec), rtol=0.0, atol=1e-14
        ):
            raise ValueError("Manifest schedule does not match its contents")
        specs.append(spec)
    return specs


def read_manifest_provenance(path: str | Path) -> dict:
    """Return the single frozen provenance record bound to every manifest row."""

    records: list[dict] = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        record = row.get("provenance")
        if not isinstance(record, dict) or row.get(
            "provenance_sha256"
        ) != provenance_sha256(record):
            raise ValueError("Manifest has invalid frozen provenance")
        records.append(record)
    if not records:
        raise ValueError("Manifest is empty")
    if any(record != records[0] for record in records[1:]):
        raise ValueError("Manifest rows do not share one frozen provenance record")
    return records[0]


def _git_commit() -> str:
    """Read HEAD without forking after PyTorch has started worker threads."""

    git_entry = Path(__file__).resolve().parents[2] / ".git"
    try:
        if git_entry.is_file():
            pointer = git_entry.read_text().strip()
            if not pointer.startswith("gitdir:"):
                return "unknown"
            git_dir = (git_entry.parent / pointer.split(":", 1)[1].strip()).resolve()
        else:
            git_dir = git_entry
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        reference = head.split(":", 1)[1].strip()
        loose_ref = git_dir / reference
        if loose_ref.exists():
            return loose_ref.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    except (OSError, ValueError):
        pass
    return "unknown"


def _provenance(device: str, source: dict | None) -> dict:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CLUSTER_NAME",
        "SLURMD_NODENAME",
    )
    runtime = {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": device,
        "slurm": {
            key.lower(): os.environ[key] for key in slurm_keys if key in os.environ
        },
    }
    return {
        "source": source,
        "source_provenance_sha256": (
            provenance_sha256(source) if source is not None else None
        ),
        "runtime": runtime,
    }


def run_trial(
    spec: TrialSpec,
    bundle: DatasetBundle,
    output_path: str | Path,
    *,
    device: str = "auto",
    force: bool = False,
    source_provenance: dict | None = None,
) -> dict:
    """Execute one manifest row and atomically persist its result."""

    output_path = Path(output_path)
    randomization = seed_streams(spec)
    expected_source_hash = (
        provenance_sha256(source_provenance) if source_provenance is not None else None
    )
    if output_path.exists() and not force:
        existing = load_npz_result(output_path)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("trial", {}).get("trial_id") == spec.trial_id
            and existing.get("trial", {}).get("config_hash") == spec.config_hash
            and existing.get("trial", {}).get("status") == "complete"
            and existing.get("data", {}).get("split_hash") == bundle.split_hash
            and existing.get("randomization") == randomization
            and existing.get("provenance", {}).get("source_provenance_sha256")
            == expected_source_hash
        ):
            return existing
        raise ValueError(f"Existing trial is corrupt or mismatched: {output_path}")
    if bundle.dataset != spec.dataset:
        raise ValueError("Dataset bundle does not match trial specification")

    # Reset before model construction so initialization is invariant to
    # manifest order, sharding, and resume state.  The same seed pairs initial
    # weights across profile comparisons.
    seed_everything(randomization["initialization_seed"])
    dropout_layers = profile_layers(spec)
    if spec.model_kind == "mlp":
        config = MLPConfig(
            width=spec.width,
            depth=spec.depth,
            activation=spec.activation,
            sigma_w_sq=spec.sigma_w_sq,
            sigma_b_sq=spec.sigma_b_sq,
            output_dim=10,
            zero_readout=spec.zero_readout,
        )
        model = build_mlp(
            config,
            dropout_layers,
            parameterization=spec.parameterization,
            base_width=spec.base_width,
            delta_width=spec.delta_width,
        )
        optimizer = make_optimizer(
            model,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
    else:
        model = TinyViT(dropout_layers, dimension=spec.width, output_dim=100)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )

    training_config = TrainingConfig(
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        lr_floor_ratio=spec.lr_floor_ratio,
        weight_decay=spec.weight_decay,
        gradient_clip_norm=spec.gradient_clip_norm,
        seed=randomization["minibatch_seed"],
        # Exact early/late reversal pairs share this dropout stream. Model
        # initialization and minibatch order use separate named streams above.
        stochastic_seed=randomization["dropout_seed"],
        evaluate_test=spec.evaluate_test,
        device=device,
    )
    start = time.perf_counter()
    trained = train_model(model, optimizer, bundle, training_config)
    duration = time.perf_counter() - start
    history = trained.pop("history")
    result = {
        "schema_version": SCHEMA_VERSION,
        "trial": {
            "trial_id": spec.trial_id,
            "config_hash": spec.config_hash,
            "phase": spec.phase,
            "status": "complete",
            "seed": spec.seed,
            "duration_seconds": duration,
        },
        "factors": asdict(spec),
        "randomization": randomization,
        "schedule": schedule_metadata(spec),
        "curves": {
            "epoch": np.arange(spec.epochs),
            **history,
        },
        "test": {
            "evaluated": spec.evaluate_test,
            "selected_epoch": spec.epochs - 1,
            "loss": trained.pop("final_test_loss"),
            "accuracy": trained.pop("final_test_accuracy"),
        },
        "compute": {
            "optimizer_steps": trained.pop("optimizer_steps"),
            "parameter_count": trained.pop("parameter_count"),
            "estimated_training_flops": trained.pop("estimated_training_flops"),
            "examples_seen": trained.pop("examples_seen"),
            "flop_model": "dense_train_6x_parameters_per_example_v1",
            "wall_seconds": duration,
        },
        "data": {
            "dataset": trained.pop("dataset"),
            "split_hash": trained.pop("split_hash"),
            "split_protocol": bundle.split_protocol,
            "test_subset_hash": bundle.test_subset_hash,
            "test_subset_protocol": bundle.test_subset_protocol,
            "test_subset_seed": bundle.test_subset_seed,
            "split_seed": spec.split_seed,
            "train_size": spec.train_size,
            "validation_size": spec.validation_size,
            "test_size": spec.test_size,
        },
        "provenance": _provenance(trained.pop("device"), source_provenance),
    }
    if trained:
        result["training"] = trained

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    save_npz_result(temporary, result)
    os.replace(temporary, output_path)
    return result
