"""Regression coverage for shared execution and co-located study definitions."""

import importlib
import json
from pathlib import Path

import pytest

from dropout_mft.experiments.benchmark import protocol, workflow
from dropout_mft.experiments.benchmark.__main__ import STUDIES, main
from dropout_mft.results import save_npz_result_atomic
from dropout_mft.schedules import named_profile_layers
from tests.benchmark.cohort_fingerprints import cohort_fingerprints


def test_all_cohort_specs_streams_and_profiles_match_before_refactoring():
    # Captured from the original implementation: 99 ordered stage/cell cases,
    # 3,340 trials. Covers interventions, IDs, every RNG stream and layer value.
    expected = json.loads(
        (Path(__file__).parent / "fixtures/cohort_fingerprints.json").read_text()
    )
    assert cohort_fingerprints() == expected


@pytest.mark.parametrize("study", STUDIES)
def test_every_study_uses_the_common_runner(study, monkeypatch):
    from dropout_mft.experiments.benchmark import cli

    calls = []
    monkeypatch.setattr(
        cli, "command_run", lambda args: calls.append((args.run_dir, args.stage))
    )
    main(
        [
            study,
            "run",
            "--run-dir",
            "runs/test",
            "--stage",
            "confirm",
            "--wandb-mode",
            "disabled",
        ]
    )
    assert calls == [("runs/test", "confirm")]


@pytest.mark.parametrize("study", STUDIES)
def test_study_planning_dispatch(study, monkeypatch):
    module = importlib.import_module(
        f"dropout_mft.experiments.benchmark.{STUDIES[study]}"
    )
    handler = "vision_main" if study == "vision_zero_decay" else "main"
    calls = []
    monkeypatch.setattr(module, handler, lambda args: calls.append(args))
    main([study, "plan", "--help"])
    assert calls == [["plan", "--help"]]


def test_selection_uses_validation_mean_and_deterministic_tie_breaks():
    records = [
        dict(
            cell="data/mlp",
            profile_id="uniform",
            learning_rate=lr,
            mean_dropout=0.1,
            seed=seed,
            validation_loss=loss,
            test_loss=test,
        )
        for lr, losses, test in [(1e-3, (1.0, 3.0), 0.0), (1e-4, (2.0, 2.0), 100.0)]
        for seed, loss in enumerate(losses)
    ]
    selected = workflow.select_candidates(records)["data/mlp"]["uniform"]
    assert selected["learning_rate"] == 1e-4
    assert selected["validation_loss"] == 2.0
    assert selected["seeds"] == 2


def test_stage_planning_requires_every_cell_selection():
    with pytest.raises(SystemExit, match="Missing lr_search selection for fi2010/mlp"):
        workflow.plan_selected_stage(
            "budget_search",
            [("fi2010", "mlp")],
            {},
            lr_search=None,
            budget_search=None,
            confirm=None,
        )


def test_collection_checks_result_identity_and_completeness(tmp_path):
    specs = protocol.lr_search_specs("fi2010", "mlp")[:2]
    workflow.write_immutable_manifest(tmp_path, "lr_search", specs)
    output = protocol.trial_output_path(tmp_path, specs[0])
    save_npz_result_atomic(output, {"trial": {"trial_id": specs[0].trial_id}})
    assert len(workflow.load_completed_trials(tmp_path, "lr_search")) == 1
    with pytest.raises(SystemExit, match="incomplete"):
        workflow.load_completed_trials(tmp_path, "lr_search", require_complete=True)
    save_npz_result_atomic(output, {"trial": {"trial_id": "wrong"}})
    with pytest.raises(ValueError, match="Trial/result mismatch"):
        workflow.load_completed_trials(tmp_path, "lr_search")


def test_paired_metrics_reject_broadcasting_unmatched_seeds():
    with pytest.raises(ValueError, match="aligned"):
        workflow.paired_comparison(
            {"test_losses": [1.0, 2.0], "test_accuracies": [0.3, 0.4]},
            {"test_losses": [1.0], "test_accuracies": [0.3]},
        )


def test_discrete_sampling_conventions_remain_distinct():
    endpoint = named_profile_layers("linear_early", 6, 0.1, 0.2)
    centered = named_profile_layers(
        "linear_early", 6, 0.1, 0.2, sampling="cell_centers"
    )
    assert endpoint[-1] == 0
    assert centered[-1] > 0
    assert sum(endpoint) == pytest.approx(sum(centered))
    with pytest.raises(ValueError, match="Unknown profile"):
        named_profile_layers("quadratic_typo", 6, 0.1, 0.2)
