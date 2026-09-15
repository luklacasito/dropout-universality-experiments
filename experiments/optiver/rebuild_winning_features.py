#!/usr/bin/env python
"""Rebuild the public Optiver winner's feature table on modern pandas.

The Kaggle notebook is kept as the authoritative recipe.  This runner executes
only its public feature/fold cells, with three compatibility-only changes:

* local raw-data paths replace Kaggle's ``../input`` layout;
* ``groupby.apply`` calls use ``group_keys=False`` (the pre-pandas-2 behavior);
* removed positional ``DataFrame.pivot`` calls are rewritten with keywords.
* Mahalanobis covariance matrices are explicitly inverted to the ``VI`` form
  required by current scikit-learn/SciPy.

No feature, neighbor, ordering, or fold definition is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

FEATURE_CELLS = (0, 2, 4, 5, 7, 8, 10, 18, 19, 20, 22, 23, 24, 25, 27, 28, 30)


def _canonical_build_paths(
    notebook_path: Path,
    raw: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Resolve paths before the notebook runner changes its working directory."""

    return tuple(
        path.expanduser().resolve() for path in (notebook_path, raw, output_dir)
    )


def _cell_source(notebook: dict, index: int) -> str:
    source = "".join(notebook["cells"][index].get("source", []))
    source = source.replace("%matplotlib inline", "")
    source = source.replace("import lightgbm as lgb\n", "")
    source = source.replace("from IPython.display import display\n", "")
    source = source.replace("tqdm_notebook as tqdm", "tqdm as tqdm")
    source = source.replace("n_iter=2000", "max_iter=2000")
    # Threaded joblib preserves the notebook's stock-level parallelism without
    # macOS semaphore/process-pool failures and avoids duplicating large frames.
    source = source.replace(
        "Parallel(n_jobs=-1)", "Parallel(n_jobs=-1, prefer='threads')"
    )
    source = source.replace(
        "Parallel(n_jobs=4, verbose=51)",
        "Parallel(n_jobs=4, verbose=51, prefer='threads')",
    )
    source = source.replace(
        ".groupby(['time_id'])", ".groupby(['time_id'], group_keys=False)"
    )
    source = source.replace(
        ".groupby('time_id')", ".groupby('time_id', group_keys=False)"
    )
    source = source.replace(
        ".pivot('time_id', 'stock_id', ",
        ".pivot(index='time_id', columns='stock_id', values=",
    )
    # The public notebook passes a covariance matrix as ``V``.  Current
    # scikit-learn delegates pairwise neighbor queries to SciPy with X and Y,
    # where the Mahalanobis implementation requires the inverse covariance
    # ``VI`` explicitly.  Supplying inv(V) preserves the same distance metric.
    source = source.replace(
        "metric_params={'V':np.cov(pivot.values.T)}",
        "metric_params={'VI':np.linalg.inv(np.cov(pivot.values.T))}",
    )
    # Every replaced pivot originally ended at the next closing parenthesis.
    # Its keyword form has the same arity, so no additional syntax is needed.
    if index == 0:
        source = source.replace("DATA_DIR = '../input'", "DATA_DIR = RAW_PARENT")
        source = source.replace(
            "USE_PRECOMPUTE_FEATURES = True", "USE_PRECOMPUTE_FEATURES = False"
        )
        source = source.replace("PREDICT_CNN = True", "PREDICT_CNN = False")
        source = source.replace("PREDICT_MLP = True", "PREDICT_MLP = False")
        source = source.replace("PREDICT_GBDT = True", "PREDICT_GBDT = False")
    if index == 5:
        marker = "test = pd.read_csv"
        source = source[: source.index(marker)]
    if index == 27:
        source = source.replace(
            "'/kaggle/input/optiver-realized-volatility-prediction/",
            "os.path.join(DATA_DIR, 'optiver-realized-volatility-prediction', '",
        )
        # The glob replacement above opens os.path.join; close it before the
        # recursive glob's quote terminator.
        source = source.replace("/**/*.parquet')})", "/**/*.parquet'))})")
    return source


def _write_feather_atomic(frame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.f")
    frame.reset_index(drop=True).to_feather(temporary)
    temporary.replace(path)


def _install_cached_base_build(namespace: dict, output_dir: Path, jobs: int) -> None:
    """Wrap the notebook's exact per-stock functions in resumable caches."""

    import pandas as pd
    from joblib import Parallel, delayed

    cache_root = output_dir / "base-stock-features"
    cache_root.mkdir(parents=True, exist_ok=True)

    def cached(
        stock_id: int,
        block,
        kind: str,
        builder: Callable,
    ):
        path = cache_root / f"{kind}-stock_id={stock_id}.f"
        if path.is_file():
            return pd.read_feather(path)
        frame = builder(stock_id, block)
        _write_feather_atomic(frame, path)
        print(f"cached {kind} stock_id={stock_id}", flush=True)
        return frame

    def make_features(base, block):
        stock_ids = sorted(set(base["stock_id"]))
        books = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(cached)(
                stock_id,
                block,
                "book",
                namespace["make_book_feature"],
            )
            for stock_id in stock_ids
        )
        trades = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(cached)(
                stock_id,
                block,
                "trade",
                namespace["make_trade_feature"],
            )
            for stock_id in stock_ids
        )
        book = pd.concat(books)
        trade = pd.concat(trades)
        frame = pd.merge(base, book, on=["stock_id", "time_id"], how="left")
        return pd.merge(frame, trade, on=["stock_id", "time_id"], how="left")

    def make_features_v2(base, block):
        stock_ids = sorted(set(base["stock_id"]))
        books = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(cached)(
                stock_id,
                block,
                "book-v2",
                namespace["make_book_feature_v2"],
            )
            for stock_id in stock_ids
        )
        return pd.merge(
            base,
            pd.concat(books),
            on=["stock_id", "time_id"],
            how="left",
        )

    namespace["make_features"] = make_features
    namespace["make_features_v2"] = make_features_v2


def rebuild(
    notebook_path: Path,
    raw: Path,
    output_dir: Path,
    *,
    jobs: int = 4,
) -> tuple[Path, Path, Path]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    notebook_path, raw, output_dir = _canonical_build_paths(
        notebook_path,
        raw,
        output_dir,
    )
    if not (raw / "train.csv").is_file() or not (raw / "book_train.parquet").is_dir():
        raise FileNotFoundError("raw must contain train.csv and book_train.parquet")
    notebook = json.loads(notebook_path.read_text())
    notebook_sha256 = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the notebook's expected competition-directory name without
    # copying 1.6 GB: create a temporary parent symlink inside the output dir.
    link = output_dir / "optiver-realized-volatility-prediction"
    if link.exists() or link.is_symlink():
        if link.resolve() != raw.resolve():
            raise RuntimeError(f"Refusing to replace unrelated path {link}")
    else:
        link.symlink_to(raw.resolve(), target_is_directory=True)

    namespace = {
        "__name__": "__optiver_winner_feature_build__",
        "RAW_PARENT": str(output_dir),
        "display": lambda *_args, **_kwargs: None,
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        for cell in FEATURE_CELLS:
            print(f"==> winning notebook feature cell {cell}", flush=True)
            source = _cell_source(notebook, cell)
            exec(compile(source, f"winning-notebook-cell-{cell}", "exec"), namespace)
            if cell == 4:
                _install_cached_base_build(namespace, output_dir, jobs)
            if cell == 5:
                namespace["df"].reset_index(drop=True).to_feather(
                    output_dir / "features_v2.f"
                )
    finally:
        os.chdir(old_cwd)

    frame = namespace["df_train"].reset_index(drop=True)
    get_x = namespace["get_X"]
    x = get_x(frame)
    y = frame[["target"]]
    folds = namespace["folds"]
    if len(folds) < 2:
        raise RuntimeError("Winning notebook did not produce at least two folds")
    x_path = output_dir / "X.f"
    y_path = output_dir / "y.f"
    fold_path = output_dir / "winning_folds.npz"
    x.to_feather(x_path)
    y.to_feather(y_path)
    np.savez(
        fold_path,
        selection_train_indices=np.asarray(folds[-2][0], dtype=np.int64),
        selection_validation_indices=np.asarray(folds[-2][1], dtype=np.int64),
        confirmation_train_indices=np.asarray(folds[-1][0], dtype=np.int64),
        test_indices=np.asarray(folds[-1][1], dtype=np.int64),
    )
    metadata = {
        "source_notebook_sha256": notebook_sha256,
        "source_notebook": str(notebook_path.resolve()),
        "raw_data": str(raw.resolve()),
        "feature_rows": len(x),
        "feature_columns": len(x.columns),
        "generated_unix_time": time.time(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    (output_dir / "feature-provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"exported X={x.shape} y={y.shape} folds={fold_path}")
    return x_path, y_path, fold_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rebuild(args.notebook, args.raw, args.output_dir, jobs=args.jobs)


if __name__ == "__main__":
    main()
