"""Regression tests for the legacy test subset and new train/validation split."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dropout_mft.training import (
    CIFAR_SPLIT_PROTOCOL,
    LEGACY_TEST_SUBSET_PROTOCOL,
    LEGACY_TEST_SUBSET_SEED,
    legacy_cifar_test_indices,
    load_cifar_bundle,
)


LEGACY_TEST_INDEX_HASH = (
    "6fce1ad32c7b2b132d367364e713742bca4d46be2ea9b80180072113b2afae51"
)


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def test_legacy_test_indices_match_the_notebook_randomstate_sequence():
    mlp_indices = legacy_cifar_test_indices(
        train_population_size=50_000,
        train_subset_size=5_000,
        test_population_size=10_000,
        test_subset_size=5_000,
    )
    vit_indices = legacy_cifar_test_indices(
        train_population_size=50_000,
        train_subset_size=2_000,
        test_population_size=10_000,
        test_subset_size=5_000,
    )

    # Both notebook families use RandomState(0), draw train first, and then
    # retain this exact ordered, unstratified test subset.
    np.testing.assert_array_equal(mlp_indices, vit_indices)
    np.testing.assert_array_equal(
        mlp_indices[:20],
        [
            4223,
            1725,
            4360,
            9043,
            8132,
            5132,
            6570,
            6948,
            9697,
            202,
            1383,
            8750,
            4354,
            8699,
            5087,
            246,
            9339,
            1787,
            3879,
            6053,
        ],
    )
    np.testing.assert_array_equal(
        mlp_indices[-10:],
        [5651, 1377, 5582, 864, 2359, 2026, 705, 4556, 3956, 2684],
    )
    assert _hash_indices(mlp_indices) == LEGACY_TEST_INDEX_HASH
    assert len(np.unique(mlp_indices)) == 5_000


@pytest.mark.parametrize(
    "dataset_name,dataset_class_name,class_count,train_size,validation_size",
    [
        ("cifar10", "CIFAR10", 10, 4_000, 1_000),
        ("cifar100", "CIFAR100", 100, 1_600, 400),
    ],
)
def test_loader_keeps_legacy_test_subset_but_stratifies_train_validation(
    monkeypatch,
    dataset_name,
    dataset_class_name,
    class_count,
    train_size,
    validation_size,
):
    from torchvision import datasets

    class FakeCIFAR:
        def __init__(self, root, train, download):
            del root, download
            size = 50_000 if train else 10_000
            self.data = np.zeros((size, 1, 1, 3), dtype=np.uint8)
            self.targets = (np.arange(size) % class_count).tolist()

    monkeypatch.setattr(datasets, dataset_class_name, FakeCIFAR)
    kwargs = dict(
        train_size=train_size,
        validation_size=validation_size,
        test_size=5_000,
        split_seed=20260812,
    )
    first = load_cifar_bundle(dataset_name, **kwargs)
    second = load_cifar_bundle(dataset_name, **kwargs)
    changed_train_split = load_cifar_bundle(
        dataset_name, **{**kwargs, "split_seed": 20260813}
    )

    assert first.split_protocol == CIFAR_SPLIT_PROTOCOL
    assert first.test_subset_protocol == LEGACY_TEST_SUBSET_PROTOCOL
    assert first.test_subset_seed == LEGACY_TEST_SUBSET_SEED
    assert first.test_subset_hash == LEGACY_TEST_INDEX_HASH
    assert second.test_subset_hash == first.test_subset_hash
    assert changed_train_split.test_subset_hash == first.test_subset_hash
    assert second.split_hash == first.split_hash
    assert changed_train_split.split_hash != first.split_hash

    train_labels = first.train.tensors[1].numpy()
    validation_labels = first.validation.tensors[1].numpy()
    test_labels = first.test.tensors[1].numpy()
    np.testing.assert_array_equal(
        np.bincount(train_labels, minlength=class_count),
        np.full(class_count, train_size // class_count),
    )
    np.testing.assert_array_equal(
        np.bincount(validation_labels, minlength=class_count),
        np.full(class_count, validation_size // class_count),
    )
    # A periodic fake label sequence makes the distinction observable: the
    # legacy test draw is random but deliberately not class-balanced.
    assert not np.all(
        np.bincount(test_labels, minlength=class_count) == 5_000 // class_count
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_subset_size": 0},
        {"train_subset_size": 50_001},
        {"test_subset_size": 0},
    ],
)
def test_legacy_test_indices_reject_invalid_subset_sizes(overrides):
    arguments = {
        "train_population_size": 50_000,
        "train_subset_size": 5_000,
        "test_population_size": 10_000,
        "test_subset_size": 5_000,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        legacy_cifar_test_indices(**arguments)
