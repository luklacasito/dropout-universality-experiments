"""Schema and summary checks for the scale-transfer figure builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dropout_mft.results import save_npz_result

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "scale_transfer"
    / "make_figures.py"
)
SPEC = importlib.util.spec_from_file_location("make_scale_transfer_figures", SCRIPT)
assert SPEC and SPEC.loader
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)
SCHEMA_VERSION = FIGURES.SCHEMA_VERSION


def _trial(
    trial_id: str,
    phase: str,
    *,
    parameterization: str,
    width: int,
    learning_rate: float,
    seed: int,
    profile_id: str = "uniform",
    validation_loss=(1.3, 1.1),
    validation_accuracy=(0.40, 0.50),
    test_loss: float = 1.15,
    test_accuracy: float = 0.52,
    status: str = "complete",
    activation: str = "relu",
    damage: float = 0.25,
    model_kind: str = "mlp",
    dataset: str = "cifar10",
    mean_dropout: float = 0.1,
    budget_space: str = "dropout_probability",
    test_evaluated: bool | None = None,
    optimizer_steps: tuple[int, ...] | None = None,
) -> dict:
    if test_evaluated is None:
        test_evaluated = phase not in {
            "profile_pilot",
            "lr_proxy",
            "lr_proxy_extension",
            "target_oracle",
        }
    validation_loss = np.asarray(validation_loss, dtype=float)
    validation_accuracy = np.asarray(validation_accuracy, dtype=float)
    if optimizer_steps is None:
        optimizer_steps = tuple(range(1, len(validation_accuracy) + 1))
    assert len(optimizer_steps) == len(validation_accuracy)
    return {
        "schema_version": SCHEMA_VERSION,
        "trial": {
            "trial_id": trial_id,
            "config_hash": f"config-{trial_id}",
            "phase": phase,
            "status": status,
            "seed": seed,
        },
        "factors": {
            "phase": phase,
            "model_kind": model_kind,
            "parameterization": parameterization,
            "dataset": dataset,
            "activation": activation,
            "profile_id": profile_id,
            "mean_dropout": mean_dropout,
            "max_dropout": 0.2,
            "budget_space": budget_space,
            "depth": 6,
            "width": width,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "schedule": {"field_damage_mean": damage},
        "curves": {
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "optimizer_steps": np.asarray(optimizer_steps, dtype=int),
        },
        "test": {
            "evaluated": test_evaluated,
            "loss": test_loss if test_evaluated else None,
            "accuracy": test_accuracy if test_evaluated else None,
        },
        "compute": {
            "optimizer_steps": optimizer_steps[-1],
            "estimated_training_flops": 2.0e12,
            "examples_seen": 75 * optimizer_steps[-1],
        },
    }


def test_loader_uses_only_completed_trials(tmp_path):
    complete = _trial(
        "complete",
        "lr_proxy",
        parameterization="sp",
        width=128,
        learning_rate=1e-4,
        seed=0,
    )
    incomplete = _trial(
        "incomplete",
        "lr_proxy",
        parameterization="sp",
        width=128,
        learning_rate=1e-4,
        seed=1,
        status="running",
    )
    save_npz_result(
        tmp_path / "aggregate.npz",
        {"schema_version": SCHEMA_VERSION, "trials": [complete, incomplete]},
    )

    loaded = FIGURES.load_completed_trials(tmp_path, source="aggregate")

    assert [trial["trial"]["trial_id"] for trial in loaded] == ["complete"]


def test_required_phase_failure_is_explicit():
    with pytest.raises(FIGURES.ResultsError, match="profile_confirmation"):
        FIGURES.require_phase([], "profile_confirmation")


def test_validation_endpoints_and_censored_threshold_resources_are_explicit():
    trial = _trial(
        "endpoint",
        "width_transfer",
        parameterization="sp",
        width=256,
        learning_rate=1e-4,
        seed=100,
        validation_loss=np.arange(20, dtype=float),
        validation_accuracy=np.linspace(0.1, 0.9, 20),
        optimizer_steps=tuple(range(1, 21)),
    )
    assert FIGURES._final_validation_loss(trial) == pytest.approx(19.0)
    assert FIGURES._final_validation_accuracy(trial) == pytest.approx(0.9)
    assert FIGURES._validation_loss_final_window_mean(trial) == pytest.approx(18.5)
    assert FIGURES._validation_loss_normalized_auc(trial) == pytest.approx(9.5)

    trial["curves"]["validation_loss"] = np.asarray([1.0, 0.8, 0.6])
    trial["curves"]["optimizer_steps"] = np.asarray([2, 7, 10])
    trial["compute"]["optimizer_steps"] = 10
    trial["compute"]["examples_seen"] = 750
    assert FIGURES._crossing_resources(trial, 0.85) == pytest.approx(
        (7, 525, 1.4e12, True)
    )
    assert FIGURES._crossing_resources(trial, 0.50) == pytest.approx(
        (10, 750, 2.0e12, False)
    )

    trial["curves"]["validation_accuracy"] = np.asarray([0.4, 0.61, 0.62])
    assert FIGURES._crossing_resources(
        trial,
        0.60,
        curve_key="validation_accuracy",
        direction="max",
    ) == pytest.approx((7, 525, 1.4e12, True))
    assert FIGURES._crossing_resources(
        trial,
        0.70,
        curve_key="validation_accuracy",
        direction="max",
    ) == pytest.approx((10, 750, 2.0e12, False))


def test_optional_proxy_extension_is_combined_for_selection_and_lr_figures():
    trials = []
    for phase, seeds in (
        ("lr_proxy", FIGURES.EXPECTED_TUNING_SEEDS),
        ("lr_proxy_extension", FIGURES.EXPECTED_TUNING_EXTENSION_SEEDS),
    ):
        for learning_rate in FIGURES.EXPECTED_LR_GRID:
            for seed in seeds:
                if learning_rate == 1e-4:
                    final_loss = 0.10 if phase == "lr_proxy" else 2.00
                elif learning_rate == 3e-4:
                    final_loss = 1.00 if phase == "lr_proxy" else 0.20
                else:
                    final_loss = 1.50
                trials.append(
                    _trial(
                        f"{phase}-{learning_rate}-{seed}",
                        phase,
                        parameterization="sp",
                        width=128,
                        learning_rate=learning_rate,
                        seed=seed,
                        validation_loss=(2.0, final_loss),
                    )
                )

    records, expected_seeds = FIGURES._proxy_lr_records(trials)
    selected_lr, _ = FIGURES._select_validation_lr(
        records, label="extended proxy", expected_seeds=expected_seeds
    )

    assert expected_seeds == frozenset(range(6))
    assert selected_lr == pytest.approx(3e-4)


def test_zero_shot_rows_cover_profiles_widths_and_censored_compute(tmp_path):
    trials = []
    for profile_id, confirmation_loss in (
        ("quadratic_early", 0.80),
        ("step_early", 0.90),
    ):
        for seed in FIGURES.EXPECTED_TRANSFER_SEEDS:
            trials.append(
                _trial(
                    f"confirmation-relu-{profile_id}-{seed}",
                    "profile_confirmation",
                    parameterization="sp",
                    width=256,
                    learning_rate=1e-4,
                    seed=seed,
                    profile_id=profile_id,
                    activation="relu",
                    validation_loss=(1.10, confirmation_loss),
                )
            )
    for parameterization, proxy_best_lr in (("sp", 1e-4), ("mup", 1e-3)):
        for learning_rate in FIGURES.EXPECTED_LR_GRID:
            proxy_loss = 0.90 if learning_rate == proxy_best_lr else 1.10
            for seed in FIGURES.EXPECTED_TUNING_SEEDS:
                trials.append(
                    _trial(
                        f"proxy-{parameterization}-{learning_rate}-{seed}",
                        "lr_proxy",
                        parameterization=parameterization,
                        width=128,
                        learning_rate=learning_rate,
                        seed=seed,
                        validation_loss=(1.2, proxy_loss),
                    )
                )

        for width in FIGURES.EXPECTED_TRANSFER_WIDTHS:
            for learning_rate in FIGURES.EXPECTED_LR_GRID:
                loss = 0.80 if learning_rate == 1e-3 else 1.05
                for seed in FIGURES.EXPECTED_TUNING_SEEDS:
                    trials.append(
                        _trial(
                            (
                                f"oracle-{parameterization}-{width}-"
                                f"{learning_rate}-{seed}"
                            ),
                            "target_oracle",
                            parameterization=parameterization,
                            width=width,
                            learning_rate=learning_rate,
                            seed=seed,
                            validation_loss=(1.1, loss),
                            validation_accuracy=(0.45, 0.60),
                        )
                    )
        for width in FIGURES.EXPECTED_TRANSFER_WIDTHS:
            for profile_id in FIGURES.TRANSFER_PROFILE_IDS:
                test_loss = 0.90 if profile_id == "quadratic_early" else 0.95
                for seed in FIGURES.EXPECTED_TRANSFER_SEEDS:
                    if profile_id == "quadratic_early" and seed < 105:
                        validation_curve = (1.15, 0.89, 0.85)
                        validation_accuracy = (0.40, 0.66, 0.67)
                    elif profile_id == "quadratic_early":
                        validation_curve = (1.15, 1.00, 0.95)
                        validation_accuracy = (0.40, 0.60, 0.64)
                    else:
                        validation_curve = (1.15, 1.00, 0.90)
                        validation_accuracy = (0.40, 0.60, 0.65)
                    trials.append(
                        _trial(
                            f"transfer-{parameterization}-{width}-{profile_id}-{seed}",
                            "width_transfer",
                            parameterization=parameterization,
                            width=width,
                            learning_rate=proxy_best_lr,
                            seed=seed,
                            profile_id=profile_id,
                            validation_loss=validation_curve,
                            validation_accuracy=validation_accuracy,
                            optimizer_steps=(2, 7, 10),
                            test_loss=test_loss,
                            test_accuracy=0.61,
                        )
                    )

    rows = FIGURES.zero_shot_rows(trials)
    by_key = {
        (row["parameterization"], row["target_width"], row["profile_id"]): row
        for row in rows
    }
    assert len(rows) == 40
    assert {row["target_width"] for row in rows} == {256, 512, 1024, 2048}

    sp = by_key[("sp", 512, "uniform")]
    assert sp["target_oracle_learning_rate"] == pytest.approx(1e-3)
    assert sp["lr_drift_ratio"] == pytest.approx(10.0)
    assert sp["proxy_lr_target_validation_loss"] == pytest.approx(1.05)
    assert sp["validation_regret_absolute"] == pytest.approx(0.25)
    assert sp["test_loss_mean"] == pytest.approx(0.95)
    assert sp["test_accuracy_pct_mean"] == pytest.approx(61.0)
    # The uniform reference is always reported at its full horizon.
    assert sp["validation_loss_terminal_mean"] == pytest.approx(0.90)
    assert sp[
        "restricted_mean_updates_to_paired_uniform_terminal_loss"
    ] == pytest.approx(10.0)
    assert sp[
        "restricted_mean_examples_to_paired_uniform_terminal_loss"
    ] == pytest.approx(750.0)
    assert sp["paired_uniform_terminal_loss_reach_rate"] == pytest.approx(1.0)
    assert sp["paired_uniform_final_validation_accuracy_mean"] == pytest.approx(0.65)
    assert sp[
        "restricted_mean_updates_to_paired_uniform_final_accuracy"
    ] == pytest.approx(10.0)
    assert sp[
        "restricted_mean_examples_to_paired_uniform_final_accuracy"
    ] == pytest.approx(750.0)
    assert sp[
        "restricted_mean_estimated_flops_to_paired_uniform_final_accuracy"
    ] == pytest.approx(2.0e12)
    assert sp["paired_uniform_final_accuracy_reach_rate"] == pytest.approx(1.0)
    assert sp[
        "estimated_flops_savings_fraction_vs_uniform_full_horizon_at_fixed_accuracy"
    ] == pytest.approx(0.0)

    quadratic = by_key[("sp", 512, "quadratic_early")]
    assert quadratic["target_oracle_learning_rate"] is None
    assert quadratic["validation_regret_absolute"] is None
    assert quadratic["validation_loss_terminal_delta_vs_uniform"] == pytest.approx(0.0)
    assert quadratic["validation_loss_final_window_delta_vs_uniform"] == pytest.approx(
        0.0
    )
    assert quadratic[
        "validation_loss_normalized_auc_delta_vs_uniform"
    ] == pytest.approx(-0.0275)
    assert quadratic["test_loss_mean"] == pytest.approx(0.90)
    assert quadratic["paired_test_loss_delta_vs_uniform"] == pytest.approx(-0.05)
    assert quadratic["paired_test_loss_ci95_lower"] == pytest.approx(-0.05)
    assert quadratic["paired_test_loss_ci95_upper"] == pytest.approx(-0.05)
    assert quadratic["paired_permutation_p_value"] == pytest.approx(2 / 1024)
    assert quadratic["holm_p_value_across_widths"] == pytest.approx(8 / 1024)
    assert quadratic["profile_rank_stability_vs_width256"] == pytest.approx(1.0)
    assert quadratic[
        "restricted_mean_updates_to_paired_uniform_terminal_loss"
    ] == pytest.approx(8.5)
    assert quadratic[
        "restricted_mean_examples_to_paired_uniform_terminal_loss"
    ] == pytest.approx(637.5)
    assert quadratic["paired_uniform_terminal_loss_reach_rate"] == pytest.approx(0.5)
    assert quadratic["validation_accuracy_terminal_mean"] == pytest.approx(0.655)
    assert quadratic[
        "restricted_mean_updates_to_paired_uniform_final_accuracy"
    ] == pytest.approx(8.5)
    assert quadratic[
        "restricted_mean_examples_to_paired_uniform_final_accuracy"
    ] == pytest.approx(637.5)
    assert quadratic[
        "restricted_mean_estimated_flops_to_paired_uniform_final_accuracy"
    ] == pytest.approx(1.7e12)
    assert quadratic["paired_uniform_final_accuracy_reach_rate"] == pytest.approx(0.5)
    assert quadratic[
        "estimated_flops_savings_fraction_vs_uniform_full_horizon_at_fixed_accuracy"
    ] == pytest.approx(0.15)
    assert quadratic[
        "restricted_mean_estimated_flops_to_paired_uniform_terminal_loss"
    ] == pytest.approx(1.7e12)

    assert by_key[("sp", 256, "uniform")]["lr_drift_ratio"] == pytest.approx(10.0)
    mup = by_key[("mup", 512, "uniform")]
    assert mup["lr_drift_ratio"] == pytest.approx(1.0)

    csv_path, contrast_csv, tex_path, gates_path = FIGURES.write_zero_shot_summary(
        trials, tmp_path
    )
    assert len(csv_path.read_text().splitlines()) == 41
    latex = tex_path.read_text()
    assert "quadratic early" in latex
    assert "tab:zero_shot_scale_transfer" in latex
    assert "tab:profile_transfer_contrasts" in latex
    assert "tab:profile_validation_compute" in latex
    assert "compute-to-fixed-accuracy" in latex
    assert "tab:profile_validation_loss_diagnostic" in latex
    assert "not statistically independent" in latex
    assert len(contrast_csv.read_text().splitlines()) == 25
    gates = json.loads(gates_path.read_text())
    assert gates["profile_transfer"]["primary"]["candidate_profile"] == (
        "quadratic_early"
    )
    assert (
        gates["profile_transfer"]["primary"]["by_parameterization"]["sp"]["successful"]
        is True
    )
    assert (
        gates["profile_transfer"]["confirmatory"]["by_parameterization"]["sp"][
            "successful"
        ]
        is False
    )
    assert (
        gates["profile_transfer"]["secondary"]["by_parameterization"]["sp"][
            "selected_profile"
        ]
        == "quadratic_early"
    )
    assert gates["mup_learning_rate_transfer"]["evaluable"] is True
    assert gates["mup_learning_rate_transfer"]["successful"] is True


def test_all_figure_builders_accept_complete_phase_grids(tmp_path):
    trials = []
    profile_ids = (
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
    for seed in FIGURES.EXPECTED_TUNING_SEEDS:
        trials.append(
            _trial(
                f"pilot-none-{seed}",
                "profile_pilot",
                parameterization="sp",
                width=256,
                learning_rate=1e-4,
                seed=seed,
                profile_id="none",
                validation_loss=(1.3, 1.12),
                damage=0.0,
                mean_dropout=0.0,
            )
        )
    for budget in FIGURES.EXPECTED_PROFILE_BUDGETS:
        for index, profile_id in enumerate(profile_ids):
            for seed in FIGURES.EXPECTED_TUNING_SEEDS:
                trials.append(
                    _trial(
                        f"pilot-{budget}-{profile_id}-{seed}",
                        "profile_pilot",
                        parameterization="sp",
                        width=256,
                        learning_rate=1e-4,
                        seed=seed,
                        profile_id=profile_id,
                        validation_loss=(1.3, 1.1 - 0.01 * index),
                        damage=0.30 - 0.01 * index,
                        mean_dropout=budget,
                    )
                )

    for activation in ("relu", "gelu"):
        for family in ("quadratic", "step"):
            for orientation, loss in (("early", 0.90), ("late", 1.00)):
                for seed in FIGURES.EXPECTED_TRANSFER_SEEDS:
                    profile_id = f"{family}_{orientation}"
                    trials.append(
                        _trial(
                            f"confirm-{activation}-{profile_id}-{seed}",
                            "profile_confirmation",
                            parameterization="sp",
                            width=256,
                            learning_rate=1e-4,
                            seed=seed,
                            profile_id=profile_id,
                            test_loss=loss + 0.001 * (seed - 100),
                            activation=activation,
                        )
                    )

    for profile_id in (
        "uniform",
        "quadratic_early",
        "quadratic_late",
        "step_early",
        "step_late",
    ):
        for seed in range(200, 206):
            trials.append(
                _trial(
                    f"vit-{profile_id}-{seed}",
                    "vit_confirmation",
                    parameterization="sp",
                    width=128,
                    learning_rate=5e-4,
                    seed=seed,
                    profile_id=profile_id,
                    model_kind="vit",
                    dataset="cifar100",
                    test_loss=1.0 - 0.02 * (profile_id == "quadratic_early"),
                    test_accuracy=0.5 + 0.01 * (profile_id == "quadratic_early"),
                )
            )

    for parameterization in ("sp", "mup"):
        for width, phase in (
            (128, "lr_proxy"),
            (256, "target_oracle"),
            (512, "target_oracle"),
            (1024, "target_oracle"),
            (2048, "target_oracle"),
        ):
            for learning_rate in FIGURES.EXPECTED_LR_GRID:
                loss = 0.8 if learning_rate == 1e-3 else 1.0
                for seed in FIGURES.EXPECTED_TUNING_SEEDS:
                    trials.append(
                        _trial(
                            f"lr-{phase}-{parameterization}-{learning_rate}-{seed}",
                            phase,
                            parameterization=parameterization,
                            width=width,
                            learning_rate=learning_rate,
                            seed=seed,
                            validation_loss=(1.2, loss),
                        )
                    )

    profile_path = FIGURES.make_profile_loss_figure(trials, tmp_path)
    paired_path = FIGURES.make_early_late_figure(trials, tmp_path)
    vit_path = FIGURES.make_vit_confirmation_figure(trials, tmp_path)
    lr_path = FIGURES.make_lr_curve_figure(trials, tmp_path)

    assert profile_path.exists()
    assert paired_path.exists()
    assert vit_path.exists()
    assert lr_path.exists()
