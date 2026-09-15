"""Protocol guards for the multi-modality benchmark suite."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dropout_mft.experiments.benchmark import cli as benchmark_cli
from dropout_mft.experiments.benchmark.datasets import (
    BENCHMARK_NAMES,
    BENCHMARK_SPECS,
    FI2010_DEFAULT_EMBARGO,
    NESTED_TRAIN_SPLIT_PROTOCOL,
    _anchored_split,
    _balanced_nested_train_split,
    _load_compressed_npz_rows,
    _standardize,
    _standardize_inplace,
    load_benchmark_bundle,
)
from dropout_mft.experiments.benchmark.protocol import (
    BENCHMARK_PROFILE_IDS,
    BUDGET_SEARCH_SEEDS,
    CONFIRM_SEEDS,
    CONTROL_PROFILE_ID,
    LR_SEARCH_SEEDS,
    BenchmarkTrialSpec,
    _schedule_record,
    benchmark_profile_layers,
    budget_search_specs,
    build_benchmark_model,
    bundle_for,
    confirm_specs,
    lr_search_specs,
    read_benchmark_manifest,
    run_benchmark_trial,
    shard,
    trial_checkpoint_path,
    trial_output_path,
    write_benchmark_manifest,
)
from dropout_mft.experiments.scale_transfer.protocol import _provenance, seed_streams
from dropout_mft.wandb_tracking import (
    WandbOptions,
    benchmark_wandb_config,
    benchmark_wandb_history,
    benchmark_wandb_summary,
    finish_benchmark_wandb_run,
    log_benchmark_wandb_result,
    tracking_is_complete,
)
from experiments.benchmark.prepare_data import (
    FI2010_DEFAULT_STOCK,
    FI2010_FOLD_SNAPSHOTS,
    FI2010_STOCK_BOUNDARIES,
    _fi2010_required_window_count,
    _fi2010_stock_block,
    _load_fi2010_rows,
)

MODEL_KINDS = ("mlp", "transformer")


def _spec(**overrides) -> BenchmarkTrialSpec:
    base = {
        "stage": "lr_search",
        "dataset": "openml_jannis",
        "model_kind": "mlp",
        "profile_id": "uniform",
        "mean_dropout": 0.10,
        "learning_rate": 1e-4,
        "seed": 0,
    }
    base.update(overrides)
    return BenchmarkTrialSpec(**base)


def _write_cache(root, name, count=600):
    spec = BENCHMARK_SPECS[name]
    rng = np.random.default_rng(0)
    labels = np.tile(np.arange(spec.classes), count // spec.classes + 1)[:count]
    if spec.image_channels is not None:
        sequence = rng.standard_normal(
            (count, spec.image_channels, spec.image_size, spec.image_size)
        ).astype(np.float32)
        flat = sequence.reshape(count, -1)
    elif spec.vocab_size is not None:
        sequence = rng.integers(0, spec.vocab_size, (count, spec.sequence_length))
        flat = rng.standard_normal((count, spec.mlp_input_dim)).astype(np.float32)
    else:
        sequence = rng.standard_normal(
            (count, spec.sequence_length, spec.input_features)
        ).astype(np.float32)
        flat = sequence.reshape(count, -1)
    digest = hashlib.sha256()
    for array in (flat, sequence, labels):
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    path = root / "benchmarks" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features_mlp=flat,
        features_sequence=sequence,
        labels=labels.astype(np.int64),
        payload_sha256=digest.hexdigest(),
    )
    return path


def _wandb_result(spec: BenchmarkTrialSpec) -> dict:
    return {
        "schema_version": 1,
        "trial": {
            "trial_id": spec.trial_id,
            "config_hash": spec.config_hash,
            "stage": spec.stage,
            "cell": spec.cell,
            "status": "complete",
            "duration_seconds": 12.5,
        },
        "curves": {
            "epoch": np.asarray([0, 1]),
            "train_loss": np.asarray([1.2, 0.8]),
            "train_accuracy": np.asarray([0.4, 0.7]),
            "validation_loss": np.asarray([1.1, 0.9]),
            "validation_accuracy": np.asarray([0.45, 0.65]),
            "global_learning_rate": np.asarray([1e-4, 1e-5]),
            "lr_multiplier": np.asarray([1.0, 0.1]),
            "optimizer_steps": np.asarray([4, 8]),
            "optimizer_group_lrs": np.asarray([[1e-4], [1e-5]]),
        },
        "selection": {
            "criterion": "minimum_validation_loss_v1",
            "selected_epoch": 1,
            "validation_loss": 0.9,
            "validation_accuracy": 0.65,
            "final_validation_loss": 0.9,
            "final_validation_accuracy": 0.65,
        },
        "test": {
            "evaluated": True,
            "epoch": 1,
            "loss": 0.88,
            "accuracy": 0.66,
            "best_validation_checkpoint": {
                "evaluated": True,
                "epoch": 1,
                "loss": 0.88,
                "accuracy": 0.66,
            },
            "fixed_final_epoch": {
                "evaluated": True,
                "epoch": 1,
                "loss": 0.91,
                "accuracy": 0.64,
            },
        },
        "compute": {
            "optimizer_steps": 8,
            "parameter_count": 100,
            "estimated_training_flops": 1_000,
            "examples_seen": 80,
        },
        "schedule": _schedule_record(spec),
        "data": {"split_hash": "split-123"},
        "provenance": {"source_provenance_sha256": "source-123"},
    }


@pytest.mark.parametrize("profile_id", BENCHMARK_PROFILE_IDS)
@pytest.mark.parametrize("mean_dropout", (0.05, 0.10, 0.15, 0.20))
def test_every_profile_preserves_its_budget(profile_id, mean_dropout):
    spec = _spec(
        profile_id=profile_id,
        mean_dropout=mean_dropout,
        max_dropout=0.30 if profile_id == "big_step" else 0.20,
    )
    layers = benchmark_profile_layers(spec)
    assert len(layers) == spec.depth
    assert float(np.mean(layers)) == pytest.approx(mean_dropout, abs=1e-12)
    assert all(0 <= value < 1 for value in layers)


def test_cap_matched_profile_rejects_a_budget_above_its_cap():
    with pytest.raises(ValueError, match="cannot exceed max_dropout"):
        _spec(profile_id="step_early", mean_dropout=0.25, max_dropout=0.20)
    with pytest.raises(ValueError, match="cannot exceed max_dropout"):
        _spec(profile_id="uniform", mean_dropout=0.25, max_dropout=0.20)


def test_saturated_step_partially_fills_its_last_active_layer():
    """At p_bar=0.19 the cap forces five saturated layers plus a remainder."""

    layers = benchmark_profile_layers(
        _spec(profile_id="step_early", mean_dropout=0.19, max_dropout=0.20)
    )
    assert layers[:5] == [0.20] * 5
    assert layers[5] == pytest.approx(0.14)
    assert float(np.mean(layers)) == pytest.approx(0.19, abs=1e-12)


def test_big_step_is_front_loaded_and_uniform_is_flat():
    uniform = benchmark_profile_layers(_spec(profile_id="uniform"))
    step = benchmark_profile_layers(_spec(profile_id="step_early"))
    big = benchmark_profile_layers(_spec(profile_id="big_step", max_dropout=0.30))
    assert len(set(uniform)) == 1
    # Front-loaded means the first half carries strictly more of the budget.
    for profile in (step, big):
        assert sum(profile[:3]) > sum(profile[3:])
        assert profile[-1] == 0.0


def test_control_profile_must_have_zero_budget():
    with pytest.raises(ValueError, match="no-dropout control"):
        _spec(profile_id=CONTROL_PROFILE_ID, mean_dropout=0.1)
    layers = benchmark_profile_layers(
        _spec(profile_id=CONTROL_PROFILE_ID, mean_dropout=0.0)
    )
    assert layers == [0.0] * 6


def test_only_the_confirmation_stage_may_touch_the_test_set():
    with pytest.raises(ValueError, match="confirmation stage"):
        _spec(stage="lr_search", evaluate_test=True)
    with pytest.raises(ValueError, match="confirmation stage"):
        _spec(stage="budget_search", evaluate_test=True)
    assert _spec(stage="confirm", evaluate_test=True).evaluate_test


def test_tuning_and_confirmation_seeds_are_disjoint():
    tuning = set(LR_SEARCH_SEEDS) | set(BUDGET_SEARCH_SEEDS)
    assert not tuning & set(CONFIRM_SEEDS)


def test_profiles_share_initialization_and_minibatch_streams_at_equal_seed():
    """Paired per-seed deltas require the arms to differ only in dropout."""

    streams = [
        seed_streams(_spec(profile_id=profile_id, seed=7))
        for profile_id in BENCHMARK_PROFILE_IDS
    ]
    assert len({stream["initialization_seed"] for stream in streams}) == 1
    assert len({stream["minibatch_seed"] for stream in streams}) == 1


def test_each_profile_is_tuned_independently():
    """A shared learning rate across profiles would strawman the uniform arm."""

    specs = lr_search_specs("openml_jannis", "mlp")
    by_profile: dict[str, set] = {}
    for spec in specs:
        by_profile.setdefault(spec.profile_id, set()).add(spec.learning_rate)
    assert set(by_profile) == set(BENCHMARK_PROFILE_IDS)
    assert all(len(rates) == 5 for rates in by_profile.values())


def test_confirmation_includes_a_no_dropout_control():
    selection = {
        profile: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile in BENCHMARK_PROFILE_IDS
    }
    specs = confirm_specs("openml_jannis", "mlp", selection)
    controls = [spec for spec in specs if spec.profile_id == CONTROL_PROFILE_ID]
    assert len(controls) == len(CONFIRM_SEEDS)
    assert all(spec.evaluate_test for spec in specs)
    assert {spec.seed for spec in specs} == set(CONFIRM_SEEDS)


def test_depth12_pilot_has_140_trials_and_explicit_criticality_labels():
    datasets = ("fi2010", "openml_jannis")
    model_kinds = ("mlp", "transformer")
    lr_specs = [
        spec
        for dataset in datasets
        for model_kind in model_kinds
        for spec in lr_search_specs(dataset, model_kind, depth=12)
    ]
    assert len(lr_specs) == 60
    assert {spec.depth for spec in lr_specs} == {12}
    assert {spec.mean_dropout for spec in lr_specs} == {0.10}

    selection = {
        profile: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile in BENCHMARK_PROFILE_IDS
    }
    confirmation = [
        spec
        for dataset in datasets
        for model_kind in model_kinds
        for spec in confirm_specs(dataset, model_kind, selection, depth=12)
    ]
    assert len(confirmation) == 80
    assert {spec.depth for spec in confirmation} == {12}
    assert len(lr_specs) + len(confirmation) == 140

    mlp_record = _schedule_record(_spec(dataset="fi2010", model_kind="mlp", depth=12))
    transformer_record = _schedule_record(
        _spec(dataset="fi2010", model_kind="transformer", depth=12)
    )
    assert mlp_record["criticality_status"].startswith("near_critical_relu_mlp")
    assert transformer_record["criticality_status"].endswith(
        "not_criticality_calibrated"
    )


def test_depth12_pilot_confirmation_plans_directly_from_lr_selection(
    tmp_path, monkeypatch
):
    selection = {
        f"{dataset}/{model_kind}": {
            profile: {"learning_rate": 1e-4, "mean_dropout": 0.10}
            for profile in BENCHMARK_PROFILE_IDS
        }
        for dataset in ("fi2010", "openml_jannis")
        for model_kind in MODEL_KINDS
    }
    requested_stages: list[str] = []

    def fake_load_selection(_run_dir, stage):
        requested_stages.append(stage)
        return selection

    monkeypatch.setattr(benchmark_cli, "_load_selection", fake_load_selection)
    benchmark_cli.command_plan(
        SimpleNamespace(
            run_dir=tmp_path,
            stage="confirm",
            datasets=["fi2010", "openml_jannis"],
            model_kinds=None,
            depth=12,
            selection_stage="lr_search",
        )
    )
    specs = read_benchmark_manifest(tmp_path / "manifests" / "confirm.jsonl")
    assert requested_stages == ["lr_search"]
    assert len(specs) == 80
    assert {spec.depth for spec in specs} == {12}
    assert {spec.mean_dropout for spec in specs if spec.profile_id != "none"} == {0.10}


def test_budget_search_requires_a_selected_learning_rate():
    with pytest.raises(ValueError, match="No selected learning rate"):
        budget_search_specs("openml_jannis", "mlp", {"uniform": 1e-4})


@pytest.mark.parametrize("dataset", BENCHMARK_NAMES)
@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_every_cell_builds_a_model_with_the_right_output_shape(dataset, model_kind):
    spec = _spec(dataset=dataset, model_kind=model_kind, profile_id="step_early")
    data = BENCHMARK_SPECS[dataset]
    model = build_benchmark_model(spec)
    if model_kind == "mlp":
        inputs = torch.randn(2, data.mlp_input_dim)
    elif data.image_channels is not None:
        inputs = torch.randn(2, data.image_channels, data.image_size, data.image_size)
    elif data.vocab_size is not None:
        inputs = torch.randint(0, data.vocab_size, (2, data.sequence_length))
    else:
        inputs = torch.randn(2, data.sequence_length, data.input_features)
    assert model(inputs).shape == (2, data.classes)


def test_dropout_layer_count_matches_depth_for_both_architectures():
    for model_kind in MODEL_KINDS:
        spec = _spec(model_kind=model_kind)
        model = build_benchmark_model(spec)
        modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        # The MLP has one dropout per hidden block; each transformer block owns
        # a single dropout applied to both residual branches.
        assert len(modules) >= spec.depth


def test_anchored_split_is_forward_in_time_with_an_embargo():
    train, validation, test = _anchored_split(
        1000, train_size=400, validation_size=100, test_size=100, embargo=50
    )
    assert train.max() < validation.min()
    assert validation.max() < test.min()
    assert validation.min() - train.max() > 50
    assert test.min() - validation.max() > 50
    assert not set(train) & set(validation) & set(test)


def test_fi2010_preparation_stops_at_the_end_of_the_locked_split():
    spec = BENCHMARK_SPECS["fi2010"]
    required = _fi2010_required_window_count()
    assert required == 70_200

    _, _, test = _anchored_split(
        required,
        train_size=spec.train_size,
        validation_size=spec.validation_size,
        test_size=spec.test_size,
        embargo=FI2010_DEFAULT_EMBARGO,
    )
    assert test[-1] == required - 1


_FAKE_ROWS = 149
_FAKE_SNAPSHOTS = 400


def _write_fi2010_fold(path: Path, snapshots: int, rows: int = _FAKE_ROWS):
    """A stand-in fold whose value encodes (row, column) for exact assertions.

    Values stay well below 2**24 so float32 represents each one exactly and the
    assertions can pin the precise cell that was read.
    """

    lines = [
        " ".join(f"{row * 1_000 + column}" for column in range(snapshots))
        for row in range(rows)
    ]
    path.write_text("\n".join(lines) + "\n")


def test_fi2010_reads_one_block_from_one_fold(tmp_path):
    path = tmp_path / "fold.txt"
    _write_fi2010_fold(path, _FAKE_SNAPSHOTS)
    start = 300
    block = _load_fi2010_rows(
        path,
        [0, 39, 148],
        start,
        start + 5,
        expected_snapshots=_FAKE_SNAPSHOTS,
        expected_rows=_FAKE_ROWS,
    )

    assert block.shape == (3, 5)
    # Rows arrive in ascending order regardless of the order requested, and the
    # columns come from inside the requested block, not the file prefix.  The
    # old loader always started at column 0 and spilled into the next fold.
    assert block[0].tolist() == [float(start + offset) for offset in range(5)]
    assert block[1].tolist() == [float(39_000 + start + offset) for offset in range(5)]
    assert block[2].tolist() == [float(148_000 + start + offset) for offset in range(5)]


def test_fi2010_stock_blocks_tile_the_whole_fold():
    assert FI2010_STOCK_BOUNDARIES[0] == 0
    assert FI2010_STOCK_BOUNDARIES[-1] == FI2010_FOLD_SNAPSHOTS
    assert list(FI2010_STOCK_BOUNDARIES) == sorted(FI2010_STOCK_BOUNDARIES)
    # The default stock must hold the locked split inside a single block; the
    # whole point of the fix is that preparation never spans two of them.
    start, stop = _fi2010_stock_block(FI2010_DEFAULT_STOCK)
    assert stop - start >= _fi2010_required_window_count() + 99
    # No other block may silently be long enough to mask a wrong default.
    assert stop - start == max(
        FI2010_STOCK_BOUNDARIES[index] - FI2010_STOCK_BOUNDARIES[index - 1]
        for index in range(1, len(FI2010_STOCK_BOUNDARIES))
    )


def test_fi2010_rejects_a_fold_with_unexpected_geometry(tmp_path):
    short = tmp_path / "short.txt"
    _write_fi2010_fold(short, _FAKE_SNAPSHOTS - 1)
    with pytest.raises(SystemExit, match=f"expected {_FAKE_SNAPSHOTS}"):
        _load_fi2010_rows(short, [0], 0, 10, expected_snapshots=_FAKE_SNAPSHOTS)

    missing_rows = tmp_path / "rows.txt"
    _write_fi2010_fold(missing_rows, _FAKE_SNAPSHOTS, rows=_FAKE_ROWS - 1)
    with pytest.raises(SystemExit, match="Expected 149 rows"):
        _load_fi2010_rows(missing_rows, [0], 0, 10, expected_snapshots=_FAKE_SNAPSHOTS)

    # The real fold geometry is what the hardcoded boundaries were verified
    # against, so a differently shaped copy must fail rather than be trusted.
    with pytest.raises(SystemExit, match=f"expected {FI2010_FOLD_SNAPSHOTS}"):
        _load_fi2010_rows(short, [0], 0, 10)


def test_fi2010_rejects_an_unknown_stock():
    with pytest.raises(SystemExit, match="stock must be in 1..5"):
        _fi2010_stock_block(len(FI2010_STOCK_BOUNDARIES))


def test_anchored_split_rejects_a_series_that_is_too_short():
    with pytest.raises(ValueError, match="anchored split needs"):
        _anchored_split(
            100, train_size=40, validation_size=30, test_size=30, embargo=50
        )


def test_both_views_share_split_indices(tmp_path):
    _write_cache(tmp_path, "openml_jannis")
    sizes = {"train_size": 40, "validation_size": 20, "test_size": 20}
    flat = load_benchmark_bundle("openml_jannis", root=tmp_path, view="mlp", **sizes)
    sequence = load_benchmark_bundle(
        "openml_jannis", root=tmp_path, view="sequence", **sizes
    )
    for split in ("train", "validation", "test"):
        left = getattr(flat, split).tensors[1].numpy()
        right = getattr(sequence, split).tensors[1].numpy()
        assert np.array_equal(left, right)


def test_compressed_npz_row_streaming_preserves_requested_order(tmp_path):
    array = np.arange(60 * 7, dtype=np.float32).reshape(60, 7)
    path = tmp_path / "rows.npz"
    np.savez_compressed(path, features=array)
    indices = np.array([59, 0, 17, 18, 3, 41])
    streamed = _load_compressed_npz_rows(
        path, "features", indices, chunk_bytes=7 * 4 * 5
    )
    assert np.array_equal(streamed, array[indices])


@pytest.mark.parametrize("contiguous", (True, False))
def test_inplace_standardization_matches_existing_protocol(contiguous):
    rng = np.random.default_rng(20260812)
    features = rng.standard_normal((40, 3, 4)).astype(np.float32)
    train_indices = np.arange(0, 20)
    if not contiguous:
        rng.shuffle(train_indices)
    expected, _ = _standardize(features.copy(), train_indices)
    source = features.copy()
    pointer = source.__array_interface__["data"][0]
    actual, _ = _standardize_inplace(source, train_indices, chunk_bytes=3 * 4 * 4 * 3)
    assert actual is source
    assert actual.__array_interface__["data"][0] == pointer
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=3e-6, rtol=3e-6)


def test_nested_amazon_train_prefixes_share_validation_and_test():
    labels = np.tile(np.arange(2), 30_000)
    splits = {
        size: _balanced_nested_train_split(
            labels,
            train_size=size,
            validation_size=5_000,
            test_size=10_000,
            max_train_size=20_000,
            seed=20260812,
        )
        for size in (2_000, 5_000, 20_000)
    }
    assert set(splits[2_000][0]) < set(splits[5_000][0])
    assert set(splits[5_000][0]) < set(splits[20_000][0])
    for size in (5_000, 20_000):
        np.testing.assert_array_equal(splits[2_000][1], splits[size][1])
        np.testing.assert_array_equal(splits[2_000][2], splits[size][2])
    for train, validation, test in splits.values():
        assert not set(train) & set(validation)
        assert not set(train) & set(test)
        assert not set(validation) & set(test)
        for indices in (train, validation, test):
            assert len(set(np.bincount(labels[indices]))) == 1


def test_data_regime_amazon_bundles_expose_shared_holdouts(tmp_path):
    _write_cache(tmp_path, "amazon_reviews", count=600)
    hashes = set()
    heldout_labels = []
    for size in (40, 80, 160):
        bundle = load_benchmark_bundle(
            "amazon_reviews",
            root=tmp_path,
            view="mlp",
            train_size=size,
            validation_size=40,
            test_size=80,
            nested_train_max_size=160,
        )
        assert bundle.split_protocol == NESTED_TRAIN_SPLIT_PROTOCOL
        hashes.add(bundle.test_subset_hash)
        heldout_labels.append(
            (
                bundle.validation.tensors[1].numpy(),
                bundle.test.tensors[1].numpy(),
            )
        )
    assert len(hashes) == 1
    for validation, test in heldout_labels[1:]:
        np.testing.assert_array_equal(validation, heldout_labels[0][0])
        np.testing.assert_array_equal(test, heldout_labels[0][1])


def test_splits_are_disjoint_and_class_balanced(tmp_path):
    _write_cache(tmp_path, "openml_jannis")
    bundle = load_benchmark_bundle(
        "openml_jannis",
        root=tmp_path,
        view="mlp",
        train_size=40,
        validation_size=20,
        test_size=20,
    )
    for split in ("train", "validation", "test"):
        labels = getattr(bundle, split).tensors[1].numpy()
        assert len(set(np.bincount(labels))) == 1
    train = {tuple(row) for row in bundle.train.tensors[0].numpy().round(5)}
    test = {tuple(row) for row in bundle.test.tensors[0].numpy().round(5)}
    assert not train & test


def test_jannis_declared_split_fits_its_minority_class():
    spec = BENCHMARK_SPECS["openml_jannis"]
    assert (spec.train_size, spec.validation_size, spec.test_size) == (
        3_840,
        960,
        1_920,
    )
    per_class = (
        spec.train_size + spec.validation_size + spec.test_size
    ) // spec.classes
    assert per_class == 1_680
    assert per_class < 1_687


def test_missing_cache_names_the_preparation_command(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="experiments/benchmark/prepare_data.py"
    ):
        load_benchmark_bundle("fi2010", root=tmp_path)


def test_manifest_round_trip_preserves_every_field(tmp_path):
    specs = lr_search_specs("openml_jannis", "mlp")
    path = tmp_path / "lr_search.jsonl"
    write_benchmark_manifest(path, specs, provenance=_provenance("test", None))
    restored = read_benchmark_manifest(path)
    assert restored == sorted(specs, key=lambda spec: spec.trial_id)


def test_manifest_rejects_tampered_schedules(tmp_path):
    specs = lr_search_specs("openml_jannis", "mlp")[:1]
    path = tmp_path / "lr_search.jsonl"
    write_benchmark_manifest(path, specs, provenance=_provenance("test", None))
    tampered = path.read_text().replace('"mean_dropout":0.1', '"mean_dropout":0.15')
    path.write_text(tampered)
    with pytest.raises(ValueError, match="does not match its contents"):
        read_benchmark_manifest(path)


def test_shards_partition_the_manifest_exactly():
    specs = lr_search_specs("openml_jannis", "mlp")
    count = 7
    collected = [spec for index in range(count) for spec in shard(specs, index, count)]
    assert sorted(collected, key=lambda spec: spec.trial_id) == sorted(
        specs, key=lambda spec: spec.trial_id
    )
    assert len({spec.trial_id for spec in collected}) == len(specs)


def test_trial_runs_and_resumes_without_recomputing(tmp_path):
    _write_cache(tmp_path, "openml_jannis")
    spec = _spec(
        stage="confirm",
        model_kind="mlp",
        profile_id="step_early",
        learning_rate=1e-3,
        epochs=2,
        batch_size=10,
        evaluate_test=True,
        train_size=40,
        validation_size=20,
        test_size=20,
    )
    bundle = bundle_for(spec, root=tmp_path)
    output = trial_output_path(tmp_path / "run", spec)
    first = run_benchmark_trial(spec, bundle, output, device="cpu")
    second = run_benchmark_trial(spec, bundle, output, device="cpu")
    assert first["trial"]["trial_id"] == second["trial"]["trial_id"]
    assert (
        first["selection"]["validation_loss"] == second["selection"]["validation_loss"]
    )
    assert first["test"]["loss"] is not None
    assert (
        first["test"]["best_validation_checkpoint"]["epoch"]
        == first["selection"]["selected_epoch"]
    )
    assert first["test"]["fixed_final_epoch"]["epoch"] == spec.epochs - 1
    assert first["test"]["fixed_final_epoch"]["loss"] is not None
    assert first["selection"]["final_validation_accuracy"] == pytest.approx(
        float(first["curves"]["validation_accuracy"][-1])
    )
    assert first["data"]["split_hash"] == bundle.split_hash
    assert np.allclose(
        first["schedule"]["dropout_probabilities"], benchmark_profile_layers(spec)
    )


def test_trial_atomically_saves_and_verifies_best_checkpoint(tmp_path):
    _write_cache(tmp_path, "openml_jannis")
    spec = _spec(
        stage="confirm",
        model_kind="mlp",
        profile_id="uniform",
        epochs=2,
        batch_size=10,
        evaluate_test=True,
        train_size=40,
        validation_size=20,
        test_size=20,
    )
    bundle = bundle_for(spec, root=tmp_path)
    run_dir = tmp_path / "run"
    output = trial_output_path(run_dir, spec)
    checkpoint = trial_checkpoint_path(run_dir, spec)

    result = run_benchmark_trial(
        spec, bundle, output, device="cpu", checkpoint_path=checkpoint
    )
    assert checkpoint.is_file()
    assert result["checkpoint"]["saved"] is True
    assert (
        result["checkpoint"]["selected_epoch"] == result["selection"]["selected_epoch"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["trial_id"] == spec.trial_id
    assert payload["selected_epoch"] == result["selection"]["selected_epoch"]
    assert payload["model_state_dict"]

    resumed = run_benchmark_trial(
        spec, bundle, output, device="cpu", checkpoint_path=checkpoint
    )
    assert resumed["checkpoint"]["sha256"] == result["checkpoint"]["sha256"]

    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt or mismatched"):
        run_benchmark_trial(
            spec, bundle, output, device="cpu", checkpoint_path=checkpoint
        )


def test_trial_refuses_a_mismatched_dataset_bundle(tmp_path):
    _write_cache(tmp_path, "openml_jannis")
    _write_cache(tmp_path, "fi2010", count=1200)
    spec = _spec(train_size=40, validation_size=20, test_size=20)
    other = load_benchmark_bundle(
        "fi2010",
        root=tmp_path,
        view="mlp",
        train_size=400,
        validation_size=100,
        test_size=100,
        embargo=10,
    )
    with pytest.raises(ValueError, match="does not match trial specification"):
        run_benchmark_trial(
            spec, other, trial_output_path(tmp_path / "run", spec), device="cpu"
        )


def test_wandb_config_records_schedule_criticality_data_and_slurm_ids(monkeypatch):
    spec = _spec(model_kind="transformer", depth=12)
    bundle = SimpleNamespace(split_hash="split-123", split_protocol="balanced_split_v1")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    config = benchmark_wandb_config(spec, bundle)
    assert config["trial_id"] == spec.trial_id
    assert config["dropout_probabilities"] == benchmark_profile_layers(spec)
    assert config["criticality_status"].endswith("not_criticality_calibrated")
    assert config["data_split_hash"] == "split-123"
    assert config["slurm_job_id"] == "12345"
    assert config["slurm_array_task_id"] == "7"


def test_wandb_history_and_summary_include_decision_relevant_metrics():
    spec = _spec()
    result = _wandb_result(spec)
    rows = benchmark_wandb_history(result)
    assert rows == [
        {
            "epoch": 0,
            "train/loss": 1.2,
            "train/accuracy": 0.4,
            "validation/loss": 1.1,
            "validation/accuracy": 0.45,
            "optimizer/global_learning_rate": 1e-4,
            "optimizer/lr_multiplier": 1.0,
            "optimizer/steps": 4.0,
            "optimizer/group_0_lr": 1e-4,
        },
        {
            "epoch": 1,
            "train/loss": 0.8,
            "train/accuracy": 0.7,
            "validation/loss": 0.9,
            "validation/accuracy": 0.65,
            "optimizer/global_learning_rate": 1e-5,
            "optimizer/lr_multiplier": 0.1,
            "optimizer/steps": 8.0,
            "optimizer/group_0_lr": 1e-5,
        },
    ]
    summary = benchmark_wandb_summary(result)
    assert summary["selection/validation_loss"] == 0.9
    assert summary["selection/final_validation_accuracy"] == 0.65
    assert summary["test/best_validation_loss"] == 0.88
    assert summary["test/best_validation_accuracy"] == 0.66
    assert summary["test/final_epoch_loss"] == 0.91
    assert summary["test/final_epoch_accuracy"] == 0.64
    assert summary["compute/parameter_count"] == 100
    assert summary["schedule/field_damage_total"] > 0
    assert summary["data/split_hash"] == "split-123"


def test_wandb_result_logging_adds_curves_summary_and_npz_artifact(tmp_path):
    spec = _spec()
    result = _wandb_result(spec)
    output = tmp_path / "trial.npz"
    output.write_bytes(b"result")

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class FakeWandb:
        Artifact = FakeArtifact

    class FakeRun:
        def __init__(self):
            self.defined = []
            self.logged = []
            self.summary = {}
            self.artifacts = []

        def define_metric(self, *args, **kwargs):
            self.defined.append((args, kwargs))

        def log(self, row):
            self.logged.append(row)

        def log_artifact(self, artifact, aliases):
            self.artifacts.append((artifact, aliases))

    run = FakeRun()
    log_benchmark_wandb_result(run, FakeWandb, result, output)
    assert len(run.logged) == 2
    assert run.logged[-1]["validation/loss"] == 0.9
    assert run.summary["selection/selected_epoch"] == 1
    artifact, aliases = run.artifacts[0]
    assert artifact.kwargs["type"] == "benchmark-trial"
    assert artifact.files == [(str(output), output.name)]
    assert aliases == [spec.stage]


def test_wandb_completion_marker_prevents_duplicate_logging(tmp_path):
    spec = _spec()
    options = WandbOptions(
        project="dropout-benchmark",
        entity="test-team",
        mode="offline",
        run_group="depth12-pilot",
        directory=tmp_path / "wandb",
    )

    class FakeRun:
        id = spec.trial_id
        url = None
        dir = str(tmp_path / "wandb" / "offline-run-test" / "files")

        def __init__(self):
            self.exit_codes = []

        def finish(self, exit_code):
            self.exit_codes.append(exit_code)

    run = FakeRun()
    marker = finish_benchmark_wandb_run(run, options, spec.trial_id)
    assert marker.exists()
    assert run.exit_codes == [0]
    assert tracking_is_complete(options, spec.trial_id)
    different_project = WandbOptions(
        project="another-project",
        entity=options.entity,
        mode=options.mode,
        run_group=options.run_group,
        directory=options.directory,
    )
    assert not tracking_is_complete(different_project, spec.trial_id)
