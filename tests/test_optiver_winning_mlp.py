from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dropout_mft.optiver_winning_mlp import (
    DEEP_PROFILE_IDS,
    EPOCH_PROBE_PROTOCOL,
    SHALLOW_PROFILE_IDS,
    ScheduledWinningMLP,
    WinningMLPTrial,
    confirmation_trials,
    epoch_probe_trials,
    matched_deep_width,
    parameter_count,
    profile_layers,
    run_trial,
    save_result,
    select_dropout_budgets,
    selection_trials,
    unique_profiles,
)


def test_confirmation_aggregate_requires_and_summarizes_all_profiles(tmp_path):
    selection = {
        architecture: {
            profile: {"mean_dropout": 0.0 if profile == "none" else 0.05}
            for profile in profiles
        }
        for architecture, profiles in (
            ("exact_shallow", SHALLOW_PROFILE_IDS),
            ("matched_depth12", DEEP_PROFILE_IDS),
        )
    }
    confirmation = tmp_path / "confirmation"
    for trial in confirmation_trials(selection):
        offset = 0.0 if trial.profile_id == "uniform" else 0.1
        save_result(
            confirmation / f"{trial.trial_id}.npz",
            {
                "trial": trial.__dict__,
                "trial_id": trial.trial_id,
                "best_validation_rmspe": 0.2 + offset + 0.001 * trial.seed,
            },
        )

    project = Path(__file__).resolve().parents[1]
    command = runpy.run_path(project / "scripts/run_optiver_winning_mlp_sweep.py")[
        "command_aggregate"
    ]
    command(SimpleNamespace(run_dir=str(tmp_path)))

    summary = json.loads((tmp_path / "aggregate/confirmation-summary.json").read_text())
    assert len(summary["rows"]) == 10
    assert summary["winners"]["exact_shallow"]["profile_id"] == "uniform"
    assert summary["winners"]["matched_depth12"]["profile_id"] == "uniform"


def test_depth_two_nominal_schedules_collapse_to_three_nonzero_shapes():
    nominal = ("uniform", "step_early", "big_step", "linear_early", "linear_late")
    grouped = unique_profiles(nominal, depth=2, mean_dropout=0.10)
    assert len(grouped) == 3
    assert (0.1, 0.1) in grouped
    assert set(grouped[(0.2, 0.0)]) == {"step_early", "big_step", "linear_early"}
    assert grouped[(0.0, 0.2)] == ("linear_late",)


def test_depth_twelve_profiles_are_all_distinct_at_point_one():
    layers = {
        profile_layers(
            profile, depth=12, mean_dropout=0.0 if profile == "none" else 0.10
        )
        for profile in DEEP_PROFILE_IDS
    }
    assert len(layers) == len(DEEP_PROFILE_IDS)


def test_matched_depth12_never_exceeds_winning_shallow_parameter_count():
    kwargs = {"numeric_features": 300, "stock_categories": 128}
    width = matched_deep_width(**kwargs)
    shallow = ScheduledWinningMLP(**kwargs, hidden_width=256, dropout_layers=(0.0, 0.0))
    deep = ScheduledWinningMLP(**kwargs, hidden_width=width, dropout_layers=(0.0,) * 12)
    assert parameter_count(deep) <= parameter_count(shallow)
    next_deep = ScheduledWinningMLP(
        **kwargs, hidden_width=width + 1, dropout_layers=(0.0,) * 12
    )
    assert parameter_count(next_deep) > parameter_count(shallow)


def test_manifest_contains_only_unique_shallow_schedules():
    trials = selection_trials(seeds=(0,))
    shallow = [trial for trial in trials if trial.architecture == "exact_shallow"]
    deep = [trial for trial in trials if trial.architecture == "matched_depth12"]
    assert len(shallow) == 10
    assert len(deep) == 21
    assert len({trial.trial_id for trial in trials}) == 31


def test_validation_selection_precedes_fresh_seed_confirmation():
    results = []
    for trial in selection_trials(seeds=(0, 1, 2)):
        results.append(
            {
                "trial": trial.__dict__,
                "best_validation_rmspe": (
                    0.1 + trial.mean_dropout + 0.001 * trial.seed
                ),
            }
        )
    selected = select_dropout_budgets(results)
    confirmed = confirmation_trials(selected)
    assert len(confirmed) == (len(DEEP_PROFILE_IDS) + 4) * 5
    assert {trial.seed for trial in confirmed} == {100, 101, 102, 103, 104}
    assert all(trial.stage == "confirmation" for trial in confirmed)
    assert all(
        trial.mean_dropout == 0 for trial in confirmed if trial.profile_id == "none"
    )


def test_epoch_probe_is_paired_without_changing_legacy_trial_ids():
    legacy = WinningMLPTrial("exact_shallow", "none", 0.0, 100, stage="confirmation")
    assert legacy.trial_id == "bf857cdc51feabf6d167"
    trials = epoch_probe_trials()
    assert len(trials) == 100
    assert len({trial.trial_id for trial in trials}) == 100
    cells = {}
    for trial in trials:
        cells.setdefault((trial.architecture, trial.profile_id, trial.seed), set()).add(
            trial.epochs
        )
        assert trial.evaluation_protocol == EPOCH_PROBE_PROTOCOL
    assert len(cells) == 50
    assert all(budgets == {30, 100} for budgets in cells.values())


def test_corrected_runner_evaluates_final_fold_once_at_the_end(tmp_path, monkeypatch):
    import dropout_mft.optiver_winning_mlp as winning

    rng = np.random.default_rng(8)
    numeric = rng.normal(size=(64, 6)).astype(np.float32)
    stock = np.arange(64) % 4
    targets = np.asarray(0.4 + 0.02 * numeric[:, 0], dtype=np.float32)
    train, validation, test = np.arange(40), np.arange(40, 56), np.arange(56, 64)
    calls = []

    def fake_evaluate(model, loader, device):
        size = len(loader.dataset)
        calls.append(size)
        value = 0.30 - 0.01 * calls.count(size) if size == 16 else 0.42
        return value, np.full(size, value, dtype=np.float32)

    monkeypatch.setattr(winning, "evaluate", fake_evaluate)
    checkpoint = tmp_path / "best.pt"
    trial = WinningMLPTrial(
        "exact_shallow",
        "early",
        0.05,
        0,
        epochs=2,
        batch_size=20,
        learning_rate=1e-3,
        max_learning_rate=2e-3,
    )
    result = run_trial(
        trial,
        numeric=numeric,
        stock_ids=stock,
        targets=targets,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        checkpoint_path=checkpoint,
        device="cpu",
    )
    assert calls == [16, 16, 8]
    assert result["test_rmspe"] == 0.42
    assert result["evaluation_protocol"] == EPOCH_PROBE_PROTOCOL
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["best_epoch"] == result["best_epoch"]
    assert saved["test_rmspe"] == 0.42


def test_exact_and_deep_cpu_smoke():
    rng = np.random.default_rng(4)
    numeric = rng.normal(size=(96, 8)).astype(np.float32)
    stock = np.arange(96) % 4
    target = np.asarray(0.2 + 0.03 * numeric[:, 0], dtype=np.float32)
    train = np.arange(72)
    validation = np.arange(72, 96)
    for architecture, profile in (
        ("exact_shallow", "early"),
        ("matched_depth12", "linear_late"),
    ):
        trial = WinningMLPTrial(
            architecture=architecture,
            profile_id=profile,
            mean_dropout=0.05,
            seed=0,
            epochs=1,
            batch_size=24,
            learning_rate=1e-3,
            max_learning_rate=2e-3,
        )
        result = run_trial(
            trial,
            numeric=numeric,
            stock_ids=stock,
            targets=target,
            train_indices=train,
            validation_indices=validation,
            device="cpu",
        )
        assert np.isfinite(result["best_validation_rmspe"])
        assert len(result["dropout_layers"]) == trial.depth
        assert np.array_equal(result["validation_targets"], target[validation])
