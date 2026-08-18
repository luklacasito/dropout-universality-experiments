from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.optiver import (
    OPTIVER_FEATURE_NAMES,
    apply_preprocessor,
    fit_train_preprocessor,
    grouped_time_split,
    load_optiver_bundle,
    rmspe,
    run_optiver_canary,
    save_optiver_cache,
)


def _cache(tmp_path, *, groups=20, stocks=4):
    rng = np.random.default_rng(7)
    time_ids = np.repeat(np.arange(groups), stocks)
    stock_ids = np.tile(np.arange(stocks), groups)
    features = rng.normal(size=(len(time_ids), len(OPTIVER_FEATURE_NAMES))).astype(
        np.float32
    )
    targets = np.exp(-5 + 0.12 * features[:, 0] - 0.08 * features[:, 1]).astype(
        np.float32
    )
    path = tmp_path / "optiver.npz"
    save_optiver_cache(
        path,
        features=features,
        targets=targets,
        stock_ids=stock_ids,
        time_ids=time_ids,
    )
    return path, time_ids


def test_group_split_keeps_every_time_bucket_in_exactly_one_partition():
    time_ids = np.repeat(np.arange(30), 5)
    splits = grouped_time_split(time_ids, seed=9)
    covered = np.concatenate(splits)
    assert sorted(covered.tolist()) == list(range(len(time_ids)))
    assert len(np.unique(covered)) == len(time_ids)
    group_sets = [set(time_ids[index].tolist()) for index in splits]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]


def test_preprocessing_uses_training_rows_only_and_produces_finite_values():
    features = np.asarray(
        [[1.0, np.nan], [3.0, 4.0], [1000.0, 900.0]], dtype=np.float32
    )
    median, mean_std = fit_train_preprocessor(features, np.asarray([0, 1]))
    transformed = apply_preprocessor(features, median, mean_std)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:2].mean(axis=0), 0.0, atol=1e-6)
    assert transformed[2, 0] > 100  # held-out outlier did not affect train statistics


def test_cache_views_share_a_group_disjoint_split(tmp_path):
    path, time_ids = _cache(tmp_path)
    first = load_optiver_bundle(path, split_seed=3)
    second = load_optiver_bundle(path, split_seed=3)
    assert first.split_hash == second.split_hash
    assert first.feature_count == len(OPTIVER_FEATURE_NAMES)
    assert len(first.train) + len(first.validation) + len(first.test) == len(time_ids)
    for dataset in (first.train, first.validation, first.test):
        features, target = dataset.tensors
        assert torch_is_finite(features)
        assert torch_is_finite(target)


def torch_is_finite(value):
    import torch

    return bool(torch.isfinite(value).all())


@pytest.mark.parametrize("model_kind", ("mlp", "transformer"))
def test_metrics_and_cpu_canary_smoke(tmp_path, model_kind):
    path, _ = _cache(tmp_path, groups=24, stocks=5)
    bundle = load_optiver_bundle(path, split_seed=4)
    assert rmspe(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])) == 0.0
    result = run_optiver_canary(
        bundle,
        model_kind=model_kind,
        profile="uniform",
        mean_dropout=0.1,
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        seed=0,
        depth=2,
        width=16,
        device="cpu",
    )
    assert len(result["history"]["train_log_mse"]) == 2
    assert np.isfinite(result["history"]["validation_rmspe"]).all()
    assert np.isfinite(list(result["test"].values())).all()
