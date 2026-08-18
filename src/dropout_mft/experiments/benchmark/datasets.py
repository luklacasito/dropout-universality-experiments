"""Multi-modality benchmark data for the dropout-profile confirmation.

The published results cover CIFAR-10/100 only.  This module adds four further
modalities plus a harder vision task, using the *same* protocol as
:mod:`dropout_mft.training`: a class-balanced train/validation split drawn with
a fixed ``split_seed``, a test set that is never inspected during selection, and
a content-addressed ``split_hash`` so a manifest row pins its exact data.

Every loader returns a :class:`~dropout_mft.training.DatasetBundle`, so
``train_model`` runs unchanged across modalities.

Train sizes are deliberately small relative to model capacity.  The effect under
test is a *regularization* effect; on a task the model cannot overfit, every
dropout profile ties at zero and the comparison is uninformative.  The sizes
below keep each task in the overfitting regime at width 256, depth 6.

Raw-data acquisition is separated from loading: run
``experiments/benchmark/prepare_data.py`` once on a login node, then every compute
job reads prepared ``.npz`` caches with no network access.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from dropout_mft.training import BenchmarkDatasetName, DatasetBundle


BENCHMARK_SPLIT_PROTOCOL = "benchmark_stratified_split_default_rng_v1"
NESTED_TRAIN_SPLIT_PROTOCOL = "benchmark_stratified_nested_train_shared_holdout_v1"
# v2 records that the underlying series comes from a single cumulative fold and
# a single stock block.  v1 caches were built by concatenating consecutive
# FI-2010 folds, which replayed earlier days into the validation and test
# windows; the version is part of every split hash so the two cannot be mixed.
ANCHORED_SPLIT_PROTOCOL = (
    "benchmark_anchored_forward_temporal_split_single_fold_single_stock_v2"
)
FI2010_DEFAULT_EMBARGO = 100

BENCHMARK_NAMES: tuple[BenchmarkDatasetName, ...] = (
    "fi2010",
    "tiny_imagenet",
    "amazon_reviews",
    "speech_commands",
    "openml_jannis",
)


@dataclass(frozen=True)
class BenchmarkDataSpec:
    """Shape and size facts a trial needs before it can build a model."""

    name: BenchmarkDatasetName
    classes: int
    # Flattened dimension seen by the MLP arm.
    mlp_input_dim: int
    # Token-sequence view seen by the transformer arm.
    sequence_length: int
    input_features: int | None
    vocab_size: int | None
    # Image tasks additionally carry a (channels, size) view for the ViT arm.
    image_channels: int | None
    image_size: int | None
    patch_size: int | None
    train_size: int
    validation_size: int
    test_size: int
    split_protocol: str


BENCHMARK_SPECS: dict[BenchmarkDatasetName, BenchmarkDataSpec] = {
    # 100 book snapshots x 40 LOB features (10 levels x {ask,bid} x {price,size}).
    "fi2010": BenchmarkDataSpec(
        name="fi2010",
        classes=3,
        mlp_input_dim=100 * 40,
        sequence_length=100,
        input_features=40,
        vocab_size=None,
        image_channels=None,
        image_size=None,
        patch_size=None,
        train_size=40_000,
        validation_size=10_000,
        test_size=20_000,
        split_protocol=ANCHORED_SPLIT_PROTOCOL,
    ),
    "tiny_imagenet": BenchmarkDataSpec(
        name="tiny_imagenet",
        classes=200,
        mlp_input_dim=3 * 64 * 64,
        sequence_length=64,
        input_features=None,
        vocab_size=None,
        image_channels=3,
        image_size=64,
        patch_size=8,
        train_size=20_000,
        validation_size=5_000,
        test_size=10_000,
        split_protocol=BENCHMARK_SPLIT_PROTOCOL,
    ),
    # MLP arm sees hashed bag-of-words; transformer arm sees token ids.
    "amazon_reviews": BenchmarkDataSpec(
        name="amazon_reviews",
        classes=2,
        mlp_input_dim=4096,
        sequence_length=128,
        input_features=None,
        vocab_size=20_000,
        image_channels=None,
        image_size=None,
        patch_size=None,
        train_size=20_000,
        validation_size=5_000,
        test_size=10_000,
        split_protocol=BENCHMARK_SPLIT_PROTOCOL,
    ),
    # 64 mel bands x 64 frames of log-mel energy, treated as a 1-channel image.
    "speech_commands": BenchmarkDataSpec(
        name="speech_commands",
        classes=35,
        mlp_input_dim=64 * 64,
        sequence_length=64,
        input_features=None,
        vocab_size=None,
        image_channels=1,
        image_size=64,
        patch_size=8,
        # Exact 4:1:2 allocation across 35 classes: 572/143/286 each.
        train_size=20_020,
        validation_size=5_005,
        test_size=10_010,
        split_protocol=BENCHMARK_SPLIT_PROTOCOL,
    ),
    # OpenML jannis: 54 numerical features, 4 classes, no spatial structure.
    "openml_jannis": BenchmarkDataSpec(
        name="openml_jannis",
        classes=4,
        mlp_input_dim=54,
        sequence_length=54,
        input_features=1,
        vocab_size=None,
        image_channels=None,
        image_size=None,
        patch_size=None,
        # The minority class has 1,687 rows.  Use 1,680 per class in an exact
        # 4:1:2 ratio rather than silently changing the balanced-split policy.
        train_size=3_840,
        validation_size=960,
        test_size=1_920,
        split_protocol=BENCHMARK_SPLIT_PROTOCOL,
    ),
}


def cache_path(name: BenchmarkDatasetName, root: str | Path = "data") -> Path:
    return Path(root) / "benchmarks" / f"{name}.npz"


def _indices_hash(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _split_hash(
    name: str,
    protocol: str,
    parts: dict[str, np.ndarray],
    payload_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(protocol.encode())
    digest.update(b"\0")
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(payload_digest.encode())
    for label in sorted(parts):
        values = np.asarray(parts[label], dtype="<i8")
        digest.update(b"\0" + label.encode() + b"\0")
        digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _balanced_split(
    labels: np.ndarray,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw three disjoint class-balanced subsets with a fixed generator.

    This mirrors ``training._balanced_indices`` but draws all three splits from
    one shuffled per-class pool so train, validation, and test cannot overlap.
    """

    classes = np.unique(labels)
    total = train_size + validation_size + test_size
    if total % len(classes):
        raise ValueError(
            f"Total size {total} must divide evenly across {len(classes)} classes"
        )
    for size, label in (
        (train_size, "train"),
        (validation_size, "validation"),
        (test_size, "test"),
    ):
        if size <= 0 or size % len(classes):
            raise ValueError(f"{label}_size must be positive and class-balanced")

    rng = np.random.default_rng(seed)
    per_train = train_size // len(classes)
    per_validation = validation_size // len(classes)
    per_test = test_size // len(classes)
    per_class = per_train + per_validation + per_test
    train, validation, test = [], [], []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {label} has {len(candidates)} examples, needs {per_class}"
            )
        drawn = rng.choice(candidates, per_class, replace=False)
        train.append(drawn[:per_train])
        validation.append(drawn[per_train : per_train + per_validation])
        test.append(drawn[per_train + per_validation :])
    return (
        np.sort(np.concatenate(train)),
        np.sort(np.concatenate(validation)),
        np.sort(np.concatenate(test)),
    )


def _balanced_nested_train_split(
    labels: np.ndarray,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    max_train_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a nested train prefix with fixed validation and test subsets.

    The largest regime is drawn once with the benchmark's original balanced
    splitter.  A domain-separated per-class ordering is then applied only to
    that training reservoir.  Smaller regimes take prefixes of the same
    ordering, so ``train(n)`` is a strict subset of ``train(m)`` for ``n < m``
    while validation and test examples remain byte-for-byte identical.

    Keeping the maximum-size draw identical to :func:`_balanced_split` also
    makes the 20k Amazon regime directly comparable to the existing 20k
    cohort; only the new smaller regimes require the nested prefix rule.
    """

    classes = np.unique(labels)
    for size, label in (
        (train_size, "train"),
        (max_train_size, "max_train"),
        (validation_size, "validation"),
        (test_size, "test"),
    ):
        if size <= 0 or size % len(classes):
            raise ValueError(f"{label}_size must be positive and class-balanced")
    if train_size > max_train_size:
        raise ValueError("train_size cannot exceed max_train_size")

    reservoir, validation, test = _balanced_split(
        labels,
        train_size=max_train_size,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )
    per_train = train_size // len(classes)
    # SeedSequence avoids relying on an arbitrary integer offset while keeping
    # this ordering independent of the RNG stream used by _balanced_split.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E455354]))
    selected: list[np.ndarray] = []
    for label in classes:
        candidates = reservoir[labels[reservoir] == label]
        if len(candidates) != max_train_size // len(classes):
            raise RuntimeError("Nested training reservoir is not class-balanced")
        selected.append(rng.permutation(candidates)[:per_train])
    train = np.sort(np.concatenate(selected))
    return train, validation, test


def _anchored_split(
    count: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    embargo: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous train -> validation -> test in time order, with embargo gaps.

    Order book snapshots are strongly autocorrelated, so a shuffled split leaks
    near-duplicate rows across the boundary and inflates accuracy to a number
    that means nothing.  Splits here are forward in time and separated by an
    embargo of ``embargo`` rows, which must be at least the label horizon so a
    training label cannot be computed from validation-period prices.
    """

    if embargo < 0:
        raise ValueError("embargo must be nonnegative")
    needed = train_size + validation_size + test_size + 2 * embargo
    if count < needed:
        raise ValueError(f"Series has {count} rows, anchored split needs {needed}")
    train_end = train_size
    validation_start = train_end + embargo
    validation_end = validation_start + validation_size
    test_start = validation_end + embargo
    test_end = test_start + test_size
    return (
        np.arange(0, train_end),
        np.arange(validation_start, validation_end),
        np.arange(test_start, test_end),
    )


def _tensor_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    discrete: bool,
) -> TensorDataset:
    selected = _rows_view_or_copy(features, indices)
    if discrete:
        x = torch.from_numpy(np.ascontiguousarray(selected)).long()
    else:
        x = torch.from_numpy(np.ascontiguousarray(selected)).float()
    y = torch.as_tensor(_rows_view_or_copy(labels, indices), dtype=torch.long)
    return TensorDataset(x, y)


def _rows_view_or_copy(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Use a zero-copy slice when indices describe one contiguous interval."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return array[:0]
    start = int(indices[0])
    if np.array_equal(indices, np.arange(start, start + len(indices))):
        return array[start : start + len(indices)]
    return array[indices]


def _read_exact(handle, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError(f"NPY member ended {remaining} bytes early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _load_compressed_npz_rows(
    path: Path,
    key: str,
    indices: np.ndarray,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> np.ndarray:
    """Stream one compressed NPY member and retain only requested rows.

    ``numpy.load`` materializes an entire compressed member.  Tiny ImageNet's
    feature member is about 4.9 GB before selection, which exceeds the 10 GB
    host-memory ceiling once normalization copies are included.  Sequential
    decompression keeps the immutable cache format but retains only the exact
    protocol-selected rows.
    """

    from numpy.lib import format as npy_format

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise ValueError("Requested NPZ rows must be one-dimensional and unique")
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as bundle, bundle.open(member) as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {version} in {member}")
        if fortran or not shape:
            raise ValueError(f"{member} must be a row-major array")
        if requested.size and (requested.min() < 0 or requested.max() >= shape[0]):
            raise IndexError(f"Requested row outside {member} with shape {shape}")

        result = np.empty((len(requested), *shape[1:]), dtype=dtype)
        order = np.argsort(requested)
        sorted_rows = requested[order]
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        rows_per_chunk = max(1, chunk_bytes // row_bytes)
        selected_cursor = 0
        for row_start in range(0, shape[0], rows_per_chunk):
            row_stop = min(row_start + rows_per_chunk, shape[0])
            raw = _read_exact(handle, (row_stop - row_start) * row_bytes)
            selected_stop = int(np.searchsorted(sorted_rows, row_stop, side="left"))
            if selected_stop > selected_cursor:
                block = np.frombuffer(raw, dtype=dtype).reshape(
                    row_stop - row_start, *shape[1:]
                )
                local_rows = sorted_rows[selected_cursor:selected_stop] - row_start
                destinations = order[selected_cursor:selected_stop]
                result[destinations] = block[local_rows]
            selected_cursor = selected_stop
        if selected_cursor != len(requested):
            raise RuntimeError(f"Did not recover every requested row from {member}")
    return result


def _standardize(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Z-score using train statistics only, so no test information leaks."""

    flat = features.reshape(len(features), -1)
    mean = flat[train_indices].mean(axis=0, keepdims=True)
    std = flat[train_indices].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    standardized = ((flat - mean) / std).reshape(features.shape)
    stats = {
        "mean_sha256": hashlib.sha256(
            np.ascontiguousarray(mean, dtype="<f8").tobytes()
        ).hexdigest()[:20],
        "std_sha256": hashlib.sha256(
            np.ascontiguousarray(std, dtype="<f8").tobytes()
        ).hexdigest()[:20],
    }
    return standardized.astype(np.float32), stats


def _standardize_inplace(
    features: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> tuple[np.ndarray, dict]:
    """Train-only z-score with memory bounded independently of dataset size.

    A full Tiny ImageNet view occupies about 4.9 GB.  Calling ``std`` on its
    80k-row training view can allocate another multi-gigabyte temporary and
    exceed Bridges-2's 10 GB cgroup limit.  This implementation merges
    float64 per-chunk moments and normalizes in-place, so its largest workspace
    is bounded by a small multiple of ``chunk_bytes``.
    """

    if features.dtype != np.float32 or not features.flags.writeable:
        features = np.array(features, dtype=np.float32, copy=True, order="C")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    flat = features.reshape(len(features), -1)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1 or len(train_indices) == 0:
        raise ValueError("train_indices must be a nonempty one-dimensional array")
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("train_indices must not contain duplicates")
    if train_indices.min() < 0 or train_indices.max() >= len(flat):
        raise IndexError("train index outside the feature array")

    row_bytes = max(1, flat.shape[1] * flat.dtype.itemsize)
    rows_per_chunk = max(1, chunk_bytes // row_bytes)
    mean64 = np.zeros(flat.shape[1], dtype=np.float64)
    m2 = np.zeros(flat.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), rows_per_chunk):
        index_chunk = train_indices[start : start + rows_per_chunk]
        block = _rows_view_or_copy(flat, index_chunk)
        # The bounded float64 copy permits stable moment accumulation without
        # ever promoting the full training matrix.
        work = np.array(block, dtype=np.float64, copy=True, order="C")
        local_count = len(work)
        local_mean = work.mean(axis=0)
        work -= local_mean
        np.square(work, out=work)
        local_m2 = work.sum(axis=0)
        total = count + local_count
        delta = local_mean - mean64
        m2 += local_m2 + delta * delta * (count * local_count / total)
        mean64 += delta * (local_count / total)
        count = total

    variance = m2 / count
    mean = mean64.astype(np.float32, copy=False).reshape(1, -1)
    std = np.sqrt(variance).astype(np.float32, copy=False).reshape(1, -1)
    std[std < 1e-8] = np.float32(1.0)
    for start in range(0, len(flat), rows_per_chunk):
        block = flat[start : start + rows_per_chunk]
        np.subtract(block, mean, out=block)
        np.divide(block, std, out=block)
    stats = {
        "mean_sha256": hashlib.sha256(
            np.ascontiguousarray(mean, dtype="<f8").tobytes()
        ).hexdigest()[:20],
        "std_sha256": hashlib.sha256(
            np.ascontiguousarray(std, dtype="<f8").tobytes()
        ).hexdigest()[:20],
    }
    return features, stats


def load_benchmark_bundle(
    name: BenchmarkDatasetName,
    *,
    root: str | Path = "data",
    view: str = "mlp",
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
    split_seed: int = 20260812,
    embargo: int | None = None,
    nested_train_max_size: int | None = None,
) -> DatasetBundle:
    """Load one prepared benchmark as a train/validation/test bundle.

    ``view`` selects the feature representation: ``"mlp"`` returns the flat
    vector the MLP consumes, ``"sequence"`` returns the token sequence the
    transformer consumes.  Both views come from the same prepared cache and the
    same split indices, so the two architectures see identical examples.
    """

    if name not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark: {name!r}")
    if view not in {"mlp", "sequence"}:
        raise ValueError(f"Unknown view: {view!r}")
    spec = BENCHMARK_SPECS[name]
    train_size = spec.train_size if train_size is None else train_size
    validation_size = (
        spec.validation_size if validation_size is None else validation_size
    )
    test_size = spec.test_size if test_size is None else test_size

    path = cache_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared cache {path}. Run:\n"
            f"  python experiments/benchmark/prepare_data.py {name} --root {root}"
        )
    with np.load(path, allow_pickle=False) as payload:
        key = "features_sequence" if view == "sequence" else "features_mlp"
        if key not in payload:
            available = sorted(payload.keys())
            raise KeyError(f"Cache {path} has no {key!r}; found {available}")
        labels = payload["labels"].astype(np.int64)
        payload_digest = str(payload["payload_sha256"])

    discrete = view == "sequence" and spec.vocab_size is not None

    split_protocol = spec.split_protocol
    if spec.split_protocol == ANCHORED_SPLIT_PROTOCOL:
        if nested_train_max_size is not None:
            raise ValueError("nested training splits are not valid for temporal data")
        horizon_embargo = FI2010_DEFAULT_EMBARGO if embargo is None else embargo
        train_indices, validation_indices, test_indices = _anchored_split(
            len(labels),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            embargo=horizon_embargo,
        )
    elif nested_train_max_size is not None:
        train_indices, validation_indices, test_indices = _balanced_nested_train_split(
            labels,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            max_train_size=nested_train_max_size,
            seed=split_seed,
        )
        split_protocol = NESTED_TRAIN_SPLIT_PROTOCOL
    else:
        train_indices, validation_indices, test_indices = _balanced_split(
            labels,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            seed=split_seed,
        )

    split_hash = _split_hash(
        name,
        split_protocol,
        {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        },
        payload_digest,
    )

    if name == "tiny_imagenet":
        selected_global = np.concatenate(
            (train_indices, validation_indices, test_indices)
        )
        features = _load_compressed_npz_rows(path, key, selected_global)
        labels = labels[selected_global]
        train_stop = len(train_indices)
        validation_stop = train_stop + len(validation_indices)
        train_local = np.arange(0, train_stop)
        validation_local = np.arange(train_stop, validation_stop)
        test_local = np.arange(validation_stop, len(selected_global))
        if not discrete:
            features, _ = _standardize_inplace(features, train_local)
        tensor_indices = (train_local, validation_local, test_local)
    else:
        with np.load(path, allow_pickle=False) as payload:
            features = payload[key]
        if not discrete:
            features, _ = _standardize(features, train_indices)
        tensor_indices = (train_indices, validation_indices, test_indices)

    tensor_train, tensor_validation, tensor_test = tensor_indices
    return DatasetBundle(
        train=_tensor_dataset(features, labels, tensor_train, discrete=discrete),
        validation=_tensor_dataset(
            features, labels, tensor_validation, discrete=discrete
        ),
        test=_tensor_dataset(features, labels, tensor_test, discrete=discrete),
        split_hash=split_hash,
        dataset=name,
        split_protocol=split_protocol,
        test_subset_hash=_indices_hash(test_indices),
        test_subset_protocol=split_protocol,
        test_subset_seed=None
        if split_protocol == ANCHORED_SPLIT_PROTOCOL
        else split_seed,
    )
