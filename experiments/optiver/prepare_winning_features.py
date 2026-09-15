#!/usr/bin/env python
"""Create an immutable schedule cache from the winning notebook's exports.

After the public notebook builds ``X``, ``y`` and ``folds``, export them with:

    X.reset_index(drop=True).to_feather("X.f")
    y.reset_index(drop=True).to_frame("target").to_feather("y.f")
    np.savez("winning_folds.npz",
             selection_train_indices=folds[-2][0],
             selection_validation_indices=folds[-2][1],
             confirmation_train_indices=folds[-1][0],
             test_indices=folds[-1][1])

The stock identifier remains categorical; every other column is numerical.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

NULL_CHECK_COLUMNS = (
    "book.log_return1.realized_volatility",
    "book_150.log_return1.realized_volatility",
    "book_300.log_return1.realized_volatility",
    "book_450.log_return1.realized_volatility",
    "trade.log_return.realized_volatility",
    "trade_150.log_return.realized_volatility",
    "trade_300.log_return.realized_volatility",
    "trade_450.log_return.realized_volatility",
)


def prepare(x_path: Path, y_path: Path, fold_path: Path, output: Path) -> Path:
    import pandas as pd

    frame = pd.read_feather(x_path)
    target_frame = pd.read_feather(y_path)
    if "stock_id" not in frame:
        raise ValueError("Winning X export must contain stock_id")
    if len(target_frame.columns) != 1:
        raise ValueError("Winning y export must contain exactly one target column")
    stock_ids = frame.pop("stock_id").to_numpy(np.int64)
    # The notebook adds these indicators inside preprocess_nn immediately
    # before StandardScaler.  Materialize the same columns in the immutable
    # cache so the standalone runner receives the exact winning MLP input.
    for name in NULL_CHECK_COLUMNS:
        if name in frame:
            frame[f"{name}_isnull"] = frame[name].isnull().astype(np.float32)
    feature_names = frame.columns.to_numpy(dtype=str)
    numeric = frame.to_numpy(np.float32)
    targets = target_frame.iloc[:, 0].to_numpy(np.float32)
    split_names = (
        "selection_train_indices",
        "selection_validation_indices",
        "confirmation_train_indices",
        "test_indices",
    )
    with np.load(fold_path, allow_pickle=False) as split:
        missing = set(split_names) - set(split.files)
        if missing:
            raise ValueError(f"Winning fold export is missing {sorted(missing)}")
        indices = {name: split[name].astype(np.int64) for name in split_names}
    if any(
        value.ndim != 1
        or len(value) == 0
        or value.min() < 0
        or value.max() >= len(numeric)
        for value in indices.values()
    ):
        raise ValueError("Fold indices must be nonempty, one-dimensional, and in range")
    if len(numeric) != len(targets):
        raise ValueError("X and y exports disagree on row count")
    if len(
        np.intersect1d(
            indices["selection_train_indices"],
            indices["selection_validation_indices"],
        )
    ):
        raise ValueError("Selection train and validation indices overlap")
    if len(
        np.intersect1d(indices["confirmation_train_indices"], indices["test_indices"])
    ):
        raise ValueError("Confirmation train and test indices overlap")
    if not set(indices["selection_train_indices"]).issubset(
        set(indices["confirmation_train_indices"])
    ):
        raise ValueError("Forward folds are not nested as expected")
    if not set(indices["selection_validation_indices"]).issubset(
        set(indices["confirmation_train_indices"])
    ):
        raise ValueError("Selection validation is not absorbed into confirmation train")
    if np.any(targets <= 0) or np.any(~np.isfinite(targets)):
        raise ValueError("Optiver targets must be finite and positive")
    digest = hashlib.sha256()
    for array in (numeric, stock_ids, targets, *indices.values()):
        digest.update(np.ascontiguousarray(array).tobytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        numeric=numeric,
        stock_ids=stock_ids,
        targets=targets,
        feature_names=feature_names,
        **indices,
        payload_sha256=np.asarray(digest.hexdigest()),
        source_protocol=np.asarray(
            "nyanpn_forward_fold_selection_then_final_fold_test_v2"
        ),
    )
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--y", type=Path, required=True)
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"wrote {prepare(args.x, args.y, args.fold, args.output)}")


if __name__ == "__main__":
    main()
