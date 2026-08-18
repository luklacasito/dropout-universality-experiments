"""Leakage-audited Optiver realized-volatility benchmark utilities.

The Kaggle training table contains one target per ``(stock_id, time_id)`` and
the book/trade parquet files contain observations from the preceding ten-minute
feature window.  ``time_id`` is an anonymized bucket identifier, not a reliable
chronological timestamp.  We therefore keep every stock sharing a ``time_id``
in the same split and describe the local protocol honestly as *group-disjoint*,
not chronological.  Kaggle's held-out competition test remains the genuine
future-period evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPConfig, SequenceTransformer, build_mlp, make_optimizer
from .schedules import schedule_layers
from .training import make_multiplicative_cosine_scheduler, seed_everything


OPTIVER_COMPETITION = "optiver-realized-volatility-prediction"
OPTIVER_SPLIT_PROTOCOL = "optiver_time_id_group_disjoint_hash_v1"
OPTIVER_CACHE_SCHEMA = 1
OPTIVER_FEATURE_NAMES = (
    "book_wap1_rv",
    "book_wap2_rv",
    "book_wap1_rv_last300",
    "book_wap2_rv_last300",
    "book_spread_mean",
    "book_spread_std",
    "book_spread_max",
    "book_spread_last300_mean",
    "book_imbalance_mean",
    "book_imbalance_std",
    "book_bid_size_sum",
    "book_ask_size_sum",
    "book_updates",
    "book_updates_last300",
    "trade_price_rv",
    "trade_price_rv_last300",
    "trade_price_std",
    "trade_size_sum",
    "trade_size_mean",
    "trade_order_count_sum",
    "trade_order_count_mean",
    "trade_count",
    "trade_count_last300",
)


def rmspe(target: np.ndarray, prediction: np.ndarray) -> float:
    """Competition metric: root mean squared percentage error."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("target and prediction must be nonempty with equal shapes")
    if np.any(~np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("Optiver targets must be finite and strictly positive")
    if np.any(~np.isfinite(prediction)):
        raise ValueError("predictions must be finite")
    return float(np.sqrt(np.mean(np.square((target - prediction) / target))))


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = float(np.square(target - prediction).sum())
    total = float(np.square(target - target.mean()).sum())
    return float("nan") if total == 0 else 1.0 - residual / total


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def grouped_time_split(
    time_ids: np.ndarray,
    *,
    seed: int = 20260813,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole anonymized time buckets, preventing cross-stock leakage."""

    time_ids = np.asarray(time_ids, dtype=np.int64)
    if time_ids.ndim != 1 or not len(time_ids):
        raise ValueError("time_ids must be a nonempty one-dimensional array")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("split fractions must lie strictly between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    groups = np.unique(time_ids)
    if len(groups) < 3:
        raise ValueError("at least three distinct time_id groups are required")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    validation_count = max(1, round(len(groups) * validation_fraction))
    test_count = max(1, round(len(groups) * test_fraction))
    if validation_count + test_count >= len(groups):
        raise ValueError("split fractions leave no training groups")
    validation_groups = groups[:validation_count]
    test_groups = groups[validation_count : validation_count + test_count]
    validation = np.flatnonzero(np.isin(time_ids, validation_groups))
    test = np.flatnonzero(np.isin(time_ids, test_groups))
    train = np.flatnonzero(
        ~np.isin(time_ids, np.concatenate((validation_groups, test_groups)))
    )
    return train, validation, test


def fit_train_preprocessor(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit finite-value imputation and z-score statistics on training rows only."""

    values = np.asarray(features, dtype=np.float32)
    train = values[np.asarray(train_indices, dtype=np.int64)].astype(np.float64)
    train[~np.isfinite(train)] = np.nan
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(train), train, median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return np.asarray(median, dtype=np.float32), np.stack((mean, std)).astype(
        np.float32
    )


def apply_preprocessor(
    features: np.ndarray, median: np.ndarray, mean_std: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    values = np.where(np.isfinite(values), values, np.asarray(median, dtype=np.float32))
    mean, std = np.asarray(mean_std, dtype=np.float32)
    return np.asarray((values - mean) / std, dtype=np.float32)


def save_optiver_cache(
    path: str | Path,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    stock_ids: np.ndarray,
    time_ids: np.ndarray,
) -> Path:
    path = Path(path)
    features = np.ascontiguousarray(features, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)
    stock_ids = np.ascontiguousarray(stock_ids, dtype=np.int32)
    time_ids = np.ascontiguousarray(time_ids, dtype=np.int32)
    count = len(features)
    if features.shape != (count, len(OPTIVER_FEATURE_NAMES)):
        raise ValueError("Optiver feature matrix has the wrong shape")
    if any(len(value) != count for value in (targets, stock_ids, time_ids)):
        raise ValueError("Optiver cache columns disagree on row count")
    if np.any(~np.isfinite(targets)) or np.any(targets <= 0):
        raise ValueError("Optiver targets must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    digest = _digest_arrays(features, targets, stock_ids, time_ids)
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(OPTIVER_CACHE_SCHEMA),
        features=features,
        targets=targets,
        stock_ids=stock_ids,
        time_ids=time_ids,
        feature_names=np.asarray(OPTIVER_FEATURE_NAMES),
        payload_sha256=np.asarray(digest),
    )
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class OptiverBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    split_hash: str
    feature_count: int


def load_optiver_bundle(
    cache: str | Path,
    *,
    split_seed: int = 20260813,
    max_train_rows: int | None = None,
) -> OptiverBundle:
    with np.load(cache, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != OPTIVER_CACHE_SCHEMA:
            raise ValueError("Unsupported Optiver cache schema")
        features = payload["features"].astype(np.float32)
        targets = payload["targets"].astype(np.float32)
        time_ids = payload["time_ids"].astype(np.int64)
        payload_digest = str(payload["payload_sha256"])
    train, validation, test = grouped_time_split(time_ids, seed=split_seed)
    if max_train_rows is not None:
        if max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")
        train = train[:max_train_rows]
    median, mean_std = fit_train_preprocessor(features, train)
    features = apply_preprocessor(features, median, mean_std)
    log_targets = np.log(targets).astype(np.float32)
    digest = hashlib.sha256()
    digest.update(OPTIVER_SPLIT_PROTOCOL.encode())
    digest.update(payload_digest.encode())
    for indices in (train, validation, test):
        digest.update(np.asarray(indices, dtype="<i8").tobytes())

    def dataset(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(features[indices])),
            torch.from_numpy(np.ascontiguousarray(log_targets[indices, None])),
        )

    return OptiverBundle(
        train=dataset(train),
        validation=dataset(validation),
        test=dataset(test),
        split_hash=digest.hexdigest(),
        feature_count=features.shape[1],
    )


def build_optiver_model(
    model_kind: str,
    dropout_layers: list[float],
    *,
    feature_count: int,
    width: int,
) -> nn.Module:
    if model_kind == "mlp":
        return build_mlp(
            MLPConfig(
                input_dim=feature_count,
                width=width,
                output_dim=1,
                depth=len(dropout_layers),
                activation="relu",
            ),
            dropout_layers,
        )
    if model_kind == "transformer":
        heads = 4 if width % 4 == 0 else 1
        return SequenceTransformer(
            dropout_layers,
            sequence_length=feature_count,
            input_features=1,
            dimension=width,
            heads=heads,
            output_dim=1,
        )
    raise ValueError(f"Unknown model kind: {model_kind!r}")


@torch.no_grad()
def evaluate_regression(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    logs, targets = [], []
    for features, log_target in loader:
        logs.append(model(features.to(device)).float().cpu().numpy().reshape(-1))
        targets.append(log_target.numpy().reshape(-1))
    predicted_log = np.concatenate(logs)
    target_log = np.concatenate(targets)
    prediction = np.exp(np.clip(predicted_log, -20, 5))
    target = np.exp(target_log)
    return {
        "log_mse": float(np.mean(np.square(predicted_log - target_log))),
        "rmspe": rmspe(target, prediction),
        "r2": r2_score(target, prediction),
    }


def run_optiver_canary(
    bundle: OptiverBundle,
    *,
    model_kind: str,
    profile: str,
    mean_dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    depth: int = 3,
    width: int = 64,
    device: str = "auto",
) -> dict:
    """Run a bounded real-data smoke test; selection/test sweeps come later."""

    profile_map = {
        "none": "none",
        "uniform": "constant",
        "step_early": "reverse_step",
        "big_step": "big_step",
        "linear_early": "reverse_linear",
        "linear_late": "linear",
    }
    if profile not in profile_map:
        raise ValueError(f"Unknown Optiver profile: {profile!r}")
    seed_everything(seed)
    dropout = schedule_layers(profile_map[profile], depth, mean_dropout, h_max=0.20)
    model = build_optiver_model(
        model_kind, dropout, feature_count=bundle.feature_count, width=width
    )
    resolved = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    model.to(resolved)
    optimizer = (
        make_optimizer(model, learning_rate=learning_rate, weight_decay=0.0)
        if model_kind == "mlp"
        else torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    )
    scheduler = make_multiplicative_cosine_scheduler(
        optimizer, epochs=epochs, floor_ratio=1e-3
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        bundle.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        bundle.validation, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(bundle.test, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    history = {
        "train_log_mse": [],
        "validation_log_mse": [],
        "validation_rmspe": [],
        "validation_r2": [],
    }
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, target in train_loader:
            features, target = features.to(resolved), target.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        validation = evaluate_regression(model, validation_loader, resolved)
        history["train_log_mse"].append(loss_sum / count)
        history["validation_log_mse"].append(validation["log_mse"])
        history["validation_rmspe"].append(validation["rmspe"])
        history["validation_r2"].append(validation["r2"])
        scheduler.step()
    test = evaluate_regression(model, test_loader, resolved)
    return {
        "model_kind": model_kind,
        "profile": profile,
        "dropout_layers": dropout,
        "split_hash": bundle.split_hash,
        "history": {key: np.asarray(value) for key, value in history.items()},
        "test": test,
    }
