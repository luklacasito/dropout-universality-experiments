"""Multi-modality confirmation of the front-loaded dropout result.

The paper establishes uniform-versus-front-loaded at fixed dropout budget on
CIFAR-10/100.  This module runs the same comparison across five tasks spanning
finance, vision, text, audio, and tabular data, for both an MLP and a
transformer.

Like :mod:`dropout_mft.experiments.legacy.protocol`, this is a separate
content-addressed cohort rather than an extension of
:class:`dropout_mft.experiments.scale_transfer.protocol.TrialSpec`.
Adding fields to that dataclass would change the content hashes of already
frozen manifests.  The trial-execution shape, named RNG streams, atomic result
writes, and resume semantics are deliberately identical.

Protocol, three staged passes per (task, architecture) cell:

``lr_search``
    Five log-spaced learning rates at the middle budget, one seed, *per
    profile*.  Tuning each profile separately is what keeps the uniform arm from
    being a strawman: the profiles differ in effective regularization strength,
    so a shared learning rate silently favours whichever one the grid was
    centred on.
``budget_search``
    The mean-dropout grid at each profile's selected learning rate, three
    seeds.  Doubles as the dropout-response curve.
``confirm``
    Five *fresh* seeds, disjoint from the tuning seeds, at the selected
    ``(learning_rate, mean_dropout)`` per profile, plus a no-dropout control.
    Only this stage touches the test set.

Selection reads validation loss only.  Test metrics are evaluated once, at the
end, and never enter a selection decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dropout_mft.experiments.benchmark.datasets import (
    BENCHMARK_SPECS,
    BenchmarkDataSpec,
    load_benchmark_bundle,
)
from dropout_mft.experiments.scale_transfer.protocol import (
    _provenance,
    canonical_json,
    seed_streams,
)
from dropout_mft.fields import propagated_field_profile, reference_field_profile
from dropout_mft.models import MLPConfig, SequenceTransformer, TinyViT, build_mlp, make_optimizer
from dropout_mft.provenance import provenance_sha256, sha256_file
from dropout_mft.results import load_npz_result, save_npz_result
from dropout_mft.schedules import field_damage, schedule_layers
from dropout_mft.training import BenchmarkDatasetName, DatasetBundle, TrainingConfig, train_model
from dropout_mft.training import seed_everything


# v2 restored the minimum-validation-loss checkpoint before test evaluation.
# v3 adds an explicit cohort identifier to the immutable trial specification so
# a corrected dataset/source cohort cannot reuse an older W&B run ID.  The
# schema bump makes pre-cohort manifests fail loudly under this source snapshot.
BENCHMARK_SCHEMA_VERSION = 3

# ``uniform`` is the control.  ``step_early`` is the paper's cap-matched
# saturated front-loaded profile.  ``big_step`` is the historical h-sweep
# profile: at \bar p=0.10, L=6 it is (.30,.30,0,0,0,0), so its peak exceeds the
# .20 cap and it is reported as an extension, not a cap-matched primary arm.
BENCHMARK_PROFILE_IDS = ("uniform", "step_early", "big_step")
CONTROL_PROFILE_ID = "none"
LINEAR_PROFILE_IDS = ("linear_early", "linear_late")
TUNED_CONTROL_PROFILE_ID = "none_tuned"
SIDECAR_PROFILE_IDS = (*LINEAR_PROFILE_IDS, TUNED_CONTROL_PROFILE_ID)
ALL_PROFILE_IDS = (*BENCHMARK_PROFILE_IDS, CONTROL_PROFILE_ID, *SIDECAR_PROFILE_IDS)
ZERO_DROPOUT_PROFILE_IDS = frozenset({CONTROL_PROFILE_ID, TUNED_CONTROL_PROFILE_ID})
CAP_EXEMPT_PROFILES = frozenset({"big_step"})

STAGES = ("lr_search", "budget_search", "confirm")

LR_SEARCH_SEEDS = (0,)
BUDGET_SEARCH_SEEDS = (0, 1, 2)
# Disjoint from the tuning seeds so the reported effect is not measured on the
# same noise draw that selected the hyperparameters.
CONFIRM_SEEDS = (100, 101, 102, 103, 104)

LR_GRIDS: dict[str, tuple[float, ...]] = {
    "mlp": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    "transformer": (5e-5, 1e-4, 3e-4, 1e-3, 3e-3),
}
MEAN_DROPOUT_GRID = (0.05, 0.10, 0.15, 0.20)
LR_SEARCH_MEAN_DROPOUT = 0.10

ModelKind = Literal["mlp", "transformer"]


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    """Per-task training budget, held fixed across every profile it compares."""

    dataset: BenchmarkDatasetName
    epochs: int
    batch_size: int
    # Transformers here use the ViT recipe's clipping; the MLP arm does not.
    transformer_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")


TASK_CONFIGS: dict[BenchmarkDatasetName, BenchmarkTaskConfig] = {
    "fi2010": BenchmarkTaskConfig(dataset="fi2010", epochs=50, batch_size=128),
    "tiny_imagenet": BenchmarkTaskConfig(
        dataset="tiny_imagenet", epochs=75, batch_size=128
    ),
    "amazon_reviews": BenchmarkTaskConfig(
        dataset="amazon_reviews", epochs=50, batch_size=128
    ),
    "speech_commands": BenchmarkTaskConfig(
        dataset="speech_commands", epochs=50, batch_size=128
    ),
    "openml_jannis": BenchmarkTaskConfig(
        dataset="openml_jannis", epochs=50, batch_size=128
    ),
}


@dataclass(frozen=True)
class BenchmarkTrialSpec:
    """One row of the multi-modality cohort."""

    stage: str
    dataset: BenchmarkDatasetName
    model_kind: ModelKind
    profile_id: str
    mean_dropout: float
    learning_rate: float
    seed: int
    max_dropout: float = 0.20
    depth: int = 6
    width: int = 256
    heads: int = 8
    mlp_ratio: float = 4.0
    activation: Literal["relu", "gelu"] = "relu"
    sigma_w_sq: float = 1.98
    sigma_b_sq: float = 0.02
    epochs: int = 50
    batch_size: int = 128
    weight_decay: float = 1e-7
    lr_floor_ratio: float = 1e-3
    gradient_clip_norm: float | None = None
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0
    split_seed: int = 20260812
    # Only the confirmation stage is allowed to look at the test set.
    evaluate_test: bool = False
    # Kept for parity with the published cohort's metadata helpers.
    budget_space: Literal["dropout_probability"] = "dropout_probability"
    parameterization: Literal["sp"] = "sp"
    # Included in trial/config hashes.  A new source/data cohort must use a new
    # value even when every scientific hyperparameter is otherwise identical.
    cohort_id: str = "benchmark-suite-v3"

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage: {self.stage!r}")
        if self.dataset not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown dataset: {self.dataset!r}")
        if self.model_kind not in {"mlp", "transformer"}:
            raise ValueError(f"Unknown model_kind: {self.model_kind!r}")
        if self.profile_id not in ALL_PROFILE_IDS:
            raise ValueError(f"Unknown profile_id: {self.profile_id!r}")
        if self.profile_id in ZERO_DROPOUT_PROFILE_IDS and self.mean_dropout != 0.0:
            raise ValueError("The no-dropout control must have mean_dropout 0")
        if self.profile_id not in ZERO_DROPOUT_PROFILE_IDS and self.mean_dropout <= 0:
            raise ValueError("Dropout profiles require a positive mean_dropout")
        if not 0 <= self.mean_dropout < 1 or not 0 < self.max_dropout < 1:
            raise ValueError("Dropout probabilities must lie in [0, 1)")
        if (
            self.profile_id not in CAP_EXEMPT_PROFILES
            and self.profile_id not in ZERO_DROPOUT_PROFILE_IDS
            and self.mean_dropout > self.max_dropout
        ):
            # A cap-matched profile cannot spend a budget above its own cap.
            raise ValueError("mean_dropout cannot exceed max_dropout")
        if self.depth <= 0 or self.width <= 0:
            raise ValueError("depth and width must be positive")
        if self.model_kind == "transformer" and self.width % self.heads:
            raise ValueError("Transformer width must be divisible by heads")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not self.cohort_id.strip():
            raise ValueError("cohort_id must not be empty")
        if self.evaluate_test and self.stage != "confirm":
            raise ValueError("Only the confirmation stage may evaluate the test set")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")

    @property
    def trial_id(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()[:20]

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        payload.pop("seed")
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:20]

    @property
    def cell(self) -> str:
        """The (task, architecture) cell this trial tunes or confirms within."""

        return f"{self.dataset}/{self.model_kind}"

    @property
    def data_view(self) -> str:
        return "mlp" if self.model_kind == "mlp" else "sequence"


def benchmark_profile_layers(spec: BenchmarkTrialSpec) -> list[float]:
    """Return the exact-budget layerwise dropout probabilities for one trial."""

    if spec.profile_id in ZERO_DROPOUT_PROFILE_IDS:
        return [0.0] * spec.depth
    if spec.profile_id == "uniform":
        values = schedule_layers(
            "constant", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "step_early":
        values = schedule_layers(
            "reverse_step", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "big_step":
        values = schedule_layers(
            "big_step", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "linear_early":
        values = schedule_layers(
            "reverse_linear", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    elif spec.profile_id == "linear_late":
        values = schedule_layers(
            "linear", spec.depth, spec.mean_dropout, spec.max_dropout
        )
    else:  # pragma: no cover - __post_init__ protects callers
        raise ValueError(f"Unknown profile: {spec.profile_id!r}")

    # The comparison is only meaningful if every arm spends the same budget.
    if not math.isclose(
        float(np.mean(values)), spec.mean_dropout, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Profile {spec.profile_id!r} does not preserve its discrete budget"
        )
    if spec.profile_id not in CAP_EXEMPT_PROFILES and (
        max(values, default=0.0) > spec.max_dropout + 1e-12
    ):
        raise RuntimeError(f"Profile {spec.profile_id!r} exceeds its dropout cap")
    if any(not 0 <= value < 1 for value in values):
        raise RuntimeError("Dropout probabilities must lie in [0, 1)")
    return [float(value) for value in values]


def _spec_defaults(dataset: BenchmarkDatasetName, model_kind: ModelKind) -> dict:
    task = TASK_CONFIGS[dataset]
    data = BENCHMARK_SPECS[dataset]
    return {
        "dataset": dataset,
        "model_kind": model_kind,
        "epochs": task.epochs,
        "batch_size": task.batch_size,
        "gradient_clip_norm": (
            task.transformer_clip_norm if model_kind == "transformer" else None
        ),
        "train_size": data.train_size,
        "validation_size": data.validation_size,
        "test_size": data.test_size,
    }


def lr_search_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    *,
    profile_ids: tuple[str, ...] = BENCHMARK_PROFILE_IDS,
    depth: int = 6,
) -> list[BenchmarkTrialSpec]:
    defaults = _spec_defaults(dataset, model_kind)
    return [
        BenchmarkTrialSpec(
            stage="lr_search",
            profile_id=profile_id,
            mean_dropout=LR_SEARCH_MEAN_DROPOUT,
            max_dropout=0.30 if profile_id in CAP_EXEMPT_PROFILES else 0.20,
            learning_rate=learning_rate,
            seed=seed,
            depth=depth,
            evaluate_test=False,
            **defaults,
        )
        for profile_id in profile_ids
        for learning_rate in LR_GRIDS[model_kind]
        for seed in LR_SEARCH_SEEDS
    ]


def budget_search_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
    *,
    profile_ids: tuple[str, ...] = BENCHMARK_PROFILE_IDS,
    depth: int = 6,
) -> list[BenchmarkTrialSpec]:
    defaults = _spec_defaults(dataset, model_kind)
    missing = set(profile_ids) - set(selected_learning_rates)
    if missing:
        raise ValueError(f"No selected learning rate for profiles: {sorted(missing)!r}")
    return [
        BenchmarkTrialSpec(
            stage="budget_search",
            profile_id=profile_id,
            mean_dropout=mean_dropout,
            max_dropout=0.30 if profile_id in CAP_EXEMPT_PROFILES else 0.20,
            learning_rate=selected_learning_rates[profile_id],
            seed=seed,
            depth=depth,
            evaluate_test=False,
            **defaults,
        )
        for profile_id in profile_ids
        for mean_dropout in MEAN_DROPOUT_GRID
        for seed in BUDGET_SEARCH_SEEDS
    ]


def confirm_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    selected: dict[str, dict],
    *,
    profile_ids: tuple[str, ...] = BENCHMARK_PROFILE_IDS,
    include_control: bool = True,
    depth: int = 6,
) -> list[BenchmarkTrialSpec]:
    """Final paired arms at fresh seeds, plus the no-dropout control."""

    defaults = _spec_defaults(dataset, model_kind)
    specs: list[BenchmarkTrialSpec] = []
    for profile_id in profile_ids:
        if profile_id not in selected:
            raise ValueError(f"No selected configuration for profile {profile_id!r}")
        choice = selected[profile_id]
        for seed in CONFIRM_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=float(choice["mean_dropout"]),
                    max_dropout=0.30 if profile_id in CAP_EXEMPT_PROFILES else 0.20,
                    learning_rate=float(choice["learning_rate"]),
                    seed=seed,
                    depth=depth,
                    evaluate_test=True,
                    **defaults,
                )
            )
    if include_control:
        # The control shares the uniform arm's learning rate so the only
        # difference is the presence of dropout itself.
        control_lr = float(selected["uniform"]["learning_rate"])
        for seed in CONFIRM_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=CONTROL_PROFILE_ID,
                    mean_dropout=0.0,
                    learning_rate=control_lr,
                    seed=seed,
                    depth=depth,
                    evaluate_test=True,
                    **defaults,
                )
            )
    return specs


def build_benchmark_model(spec: BenchmarkTrialSpec) -> torch.nn.Module:
    """Construct the architecture for one trial from its data specification."""

    data: BenchmarkDataSpec = BENCHMARK_SPECS[spec.dataset]
    dropout_layers = benchmark_profile_layers(spec)
    if spec.model_kind == "mlp":
        config = MLPConfig(
            input_dim=data.mlp_input_dim,
            width=spec.width,
            output_dim=data.classes,
            depth=spec.depth,
            activation=spec.activation,
            sigma_w_sq=spec.sigma_w_sq,
            sigma_b_sq=spec.sigma_b_sq,
        )
        return build_mlp(config, dropout_layers, parameterization="sp")

    if data.image_channels is not None:
        return TinyViT(
            dropout_layers,
            dimension=spec.width,
            heads=spec.heads,
            mlp_ratio=spec.mlp_ratio,
            output_dim=data.classes,
            patch_size=data.patch_size,
            in_channels=data.image_channels,
            image_size=data.image_size,
        )
    return SequenceTransformer(
        dropout_layers,
        sequence_length=data.sequence_length,
        dimension=spec.width,
        heads=spec.heads,
        mlp_ratio=spec.mlp_ratio,
        output_dim=data.classes,
        input_features=data.input_features,
        vocab_size=data.vocab_size,
        padding_index=0 if data.vocab_size is not None else None,
    )


def bundle_for(spec: BenchmarkTrialSpec, *, root: str | Path = "data") -> DatasetBundle:
    # Data-regime cohorts use separate run directories, so their ordinary
    # ``dataset/model`` cell remains unambiguous.  The cohort identifier pins
    # the one extra loading rule needed for Amazon: all train-size regimes draw
    # prefixes from the same 20k reservoir and share held-out examples.
    nested_train_max_size = None
    if spec.dataset == "amazon_reviews" and spec.cohort_id.startswith(
        "data-regime-scaling-v1/amazon_n"
    ):
        nested_train_max_size = 20_000
    return load_benchmark_bundle(
        spec.dataset,
        root=root,
        view=spec.data_view,
        train_size=spec.train_size,
        validation_size=spec.validation_size,
        test_size=spec.test_size,
        split_seed=spec.split_seed,
        nested_train_max_size=nested_train_max_size,
    )


def _schedule_record(spec: BenchmarkTrialSpec) -> dict:
    """Field diagnostics for one trial, matching the published record's keys.

    This deliberately does not call ``scale_transfer.schedule_metadata``: that
    helper rebuilds the probabilities from its own ``PROFILE_IDS`` table, which
    has no ``big_step`` entry.  Computing the diagnostics from the already
    validated probability vector keeps the two cohorts consistent while letting
    this one carry the historical profile.
    """

    probabilities = benchmark_profile_layers(spec)
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
    is_mlp = spec.model_kind == "mlp"
    return {
        "profile_id": spec.profile_id,
        "budget_space": spec.budget_space,
        "dropout_probabilities": np.asarray(probabilities),
        "mean_dropout_probability": float(np.mean(probabilities)),
        "max_dropout_probability": float(max(probabilities, default=0.0)),
        "field_mapping": (
            "exact_mlp_one_step_at_reference_variance_q1"
            if is_mlp
            else "mlp_one_step_proxy_at_reference_variance_q1"
        ),
        "interpretation": (
            "mlp_forward_field"
            if is_mlp
            else "mlp_inspired_profile_proxy_not_a_transformer_recursion"
        ),
        "criticality_status": (
            "near_critical_relu_mlp_fixed_across_dropout_profiles"
            if is_mlp
            else "pre_layernorm_transformer_not_criticality_calibrated"
        ),
        "mean_reference_field": float(np.mean(fields)),
        "reference_fields": np.asarray(fields),
        "propagated_local_fields": np.asarray(local_fields),
        "variance_trajectory": np.asarray(variances),
        "field_damage_mean": damage,
        "field_damage_total": damage * spec.depth,
        "xi_proxy": float("inf") if damage == 0 else 1.0 / damage,
    }


def run_benchmark_trial(
    spec: BenchmarkTrialSpec,
    bundle: DatasetBundle,
    output_path: str | Path,
    *,
    device: str = "auto",
    force: bool = False,
    source_provenance: dict | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Execute one manifest row and atomically persist its result."""

    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    randomization = seed_streams(spec)
    expected_source_hash = (
        provenance_sha256(source_provenance) if source_provenance is not None else None
    )
    if output_path.exists() and not force:
        existing = load_npz_result(output_path)
        valid_result = (
            existing.get("schema_version") == BENCHMARK_SCHEMA_VERSION
            and existing.get("trial", {}).get("trial_id") == spec.trial_id
            and existing.get("trial", {}).get("config_hash") == spec.config_hash
            and existing.get("trial", {}).get("status") == "complete"
            and existing.get("data", {}).get("split_hash") == bundle.split_hash
            and existing.get("randomization") == randomization
            and existing.get("provenance", {}).get("source_provenance_sha256")
            == expected_source_hash
        )
        if valid_result and checkpoint_path is not None:
            checkpoint = existing.get("checkpoint", {})
            valid_result = (
                checkpoint.get("saved") is True
                and checkpoint.get("format") == "torch_state_dict_v1"
                and checkpoint_path.is_file()
                and checkpoint.get("sha256") == sha256_file(checkpoint_path)
            )
        if valid_result:
            return existing
        raise ValueError(f"Existing trial is corrupt or mismatched: {output_path}")
    if bundle.dataset != spec.dataset:
        raise ValueError("Dataset bundle does not match trial specification")

    # Reset before construction so initialization is invariant to manifest
    # order, sharding, and resume state.  Paired profiles share this seed.
    seed_everything(randomization["initialization_seed"])
    model = build_benchmark_model(spec)
    if spec.model_kind == "mlp":
        optimizer = make_optimizer(
            model,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
    else:
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
        stochastic_seed=randomization["dropout_seed"],
        evaluate_test=spec.evaluate_test,
        # Selection reads the minimum-validation-loss epoch, so the test set has
        # to be read at that same epoch.  These tasks are deliberately sized to
        # overfit, and how far a run degrades after its own selection point
        # differs by dropout profile, which is the effect under test.
        restore_best_validation=True,
        device=device,
    )
    start = time.perf_counter()
    trained = train_model(
        model,
        optimizer,
        bundle,
        training_config,
        return_best_state=checkpoint_path is not None,
    )
    duration = time.perf_counter() - start
    history = trained.pop("history")

    validation_loss = np.asarray(history["validation_loss"], dtype=float)
    validation_accuracy = np.asarray(history["validation_accuracy"], dtype=float)
    selected_epoch = int(np.argmin(validation_loss))
    test_epoch = int(trained.pop("test_epoch"))
    test_protocol = str(trained.pop("test_protocol"))
    if test_epoch != selected_epoch:
        raise RuntimeError(
            f"Test was evaluated at epoch {test_epoch} but selection scored "
            f"epoch {selected_epoch}; the two must agree"
        )
    selected_test_loss = trained.pop("final_test_loss")
    selected_test_accuracy = trained.pop("final_test_accuracy")
    fixed_final_test_loss = trained.pop("final_epoch_test_loss")
    fixed_final_test_accuracy = trained.pop("final_epoch_test_accuracy")
    fixed_final_test_epoch = int(trained.pop("final_epoch_test_epoch"))
    fixed_final_test_protocol = str(trained.pop("final_epoch_test_protocol"))
    checkpoint_record = {"saved": False}
    if checkpoint_path is not None:
        best_state = trained.pop("best_state_dict")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        torch.save(
            {
                "format": "torch_state_dict_v1",
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "selected_epoch": selected_epoch,
                "model_state_dict": best_state,
            },
            checkpoint_temporary,
        )
        os.replace(checkpoint_temporary, checkpoint_path)
        checkpoint_record = {
            "saved": True,
            "format": "torch_state_dict_v1",
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selected_epoch": selected_epoch,
        }
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "trial": {
            "trial_id": spec.trial_id,
            "config_hash": spec.config_hash,
            "stage": spec.stage,
            "cell": spec.cell,
            "status": "complete",
            "seed": spec.seed,
            "duration_seconds": duration,
        },
        "factors": asdict(spec),
        "randomization": randomization,
        "schedule": _schedule_record(spec),
        "curves": {"epoch": np.arange(spec.epochs), **history},
        "selection": {
            "criterion": "minimum_validation_loss_v1",
            "selected_epoch": selected_epoch,
            "validation_loss": float(validation_loss[selected_epoch]),
            "validation_accuracy": float(validation_accuracy[selected_epoch]),
            "final_validation_loss": float(validation_loss[-1]),
            "final_validation_accuracy": float(validation_accuracy[-1]),
        },
        "test": {
            "evaluated": spec.evaluate_test,
            # Backward-compatible aliases: these remain the metrics at the
            # checkpoint selected by validation loss.
            "protocol": test_protocol,
            "epoch": test_epoch,
            "loss": selected_test_loss,
            "accuracy": selected_test_accuracy,
            "best_validation_checkpoint": {
                "evaluated": spec.evaluate_test,
                "protocol": test_protocol,
                "epoch": test_epoch,
                "loss": selected_test_loss,
                "accuracy": selected_test_accuracy,
            },
            "fixed_final_epoch": {
                "evaluated": spec.evaluate_test,
                "protocol": fixed_final_test_protocol,
                "epoch": fixed_final_test_epoch,
                "loss": fixed_final_test_loss,
                "accuracy": fixed_final_test_accuracy,
            },
        },
        "checkpoint": checkpoint_record,
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
            "view": spec.data_view,
            "split_hash": trained.pop("split_hash"),
            "split_protocol": bundle.split_protocol,
            "test_subset_hash": bundle.test_subset_hash,
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


def write_benchmark_manifest(
    path: str | Path,
    specs: list[BenchmarkTrialSpec],
    *,
    provenance: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(specs, key=lambda spec: spec.trial_id)
    if len({spec.trial_id for spec in ordered}) != len(ordered):
        raise ValueError("Manifest contains duplicate trials")
    provenance_hash = provenance_sha256(provenance)
    content = "".join(
        canonical_json(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "provenance": provenance,
                "provenance_sha256": provenance_hash,
                "randomization": seed_streams(spec),
                "dropout_probabilities": benchmark_profile_layers(spec),
                **asdict(spec),
            }
        )
        + "\n"
        for spec in ordered
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def read_benchmark_manifest(path: str | Path) -> list[BenchmarkTrialSpec]:
    specs: list[BenchmarkTrialSpec] = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.pop("schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("Unsupported manifest schema")
        trial_id = row.pop("trial_id")
        config_hash = row.pop("config_hash")
        provenance = row.pop("provenance")
        if row.pop("provenance_sha256") != provenance_sha256(provenance):
            raise ValueError("Manifest provenance hash does not match its contents")
        expected_randomization = row.pop("randomization")
        expected_probabilities = row.pop("dropout_probabilities")
        spec = BenchmarkTrialSpec(**row)
        if trial_id != spec.trial_id:
            raise ValueError("Manifest trial hash does not match its contents")
        if config_hash != spec.config_hash:
            raise ValueError("Manifest config hash does not match its contents")
        if expected_randomization != seed_streams(spec):
            raise ValueError("Manifest randomization does not match its contents")
        if not np.allclose(
            expected_probabilities,
            benchmark_profile_layers(spec),
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("Manifest schedule does not match its contents")
        specs.append(spec)
    return specs


def trial_output_path(run_dir: str | Path, spec: BenchmarkTrialSpec) -> Path:
    return (
        Path(run_dir)
        / "trials"
        / spec.dataset
        / spec.model_kind
        / spec.stage
        / f"{spec.trial_id}.npz"
    )


def trial_checkpoint_path(run_dir: str | Path, spec: BenchmarkTrialSpec) -> Path:
    """Content-addressed best-validation model checkpoint for one trial."""

    return (
        Path(run_dir)
        / "checkpoints"
        / spec.dataset
        / spec.model_kind
        / spec.stage
        / f"{spec.trial_id}.best.pt"
    )


def shard(specs: list[BenchmarkTrialSpec], index: int, count: int) -> list:
    """Deterministic contiguous-stride sharding for Slurm arrays."""

    if count <= 0 or not 0 <= index < count:
        raise ValueError("Invalid shard index or count")
    ordered = sorted(specs, key=lambda spec: spec.trial_id)
    return ordered[index::count]
