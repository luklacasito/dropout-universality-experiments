#!/usr/bin/env python
"""Aggregate the Optiver Kaggle parquet files into the shared compact cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.optiver.data import (  # noqa: E402
    OPTIVER_FEATURE_NAMES,
    save_optiver_cache,
)


def _rv(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        return 0.0
    returns = np.diff(np.log(values))
    return float(np.sqrt(np.square(returns).sum()))


def _stock_id(path: Path) -> int:
    try:
        return int(path.name.split("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Expected a stock_id=<integer> directory, got {path}"
        ) from exc


def _aggregate_stock(book_path: Path, trade_path: Path | None):
    import pandas as pd

    book = pd.read_parquet(book_path)
    required = {
        "time_id",
        "seconds_in_bucket",
        "bid_price1",
        "ask_price1",
        "bid_price2",
        "ask_price2",
        "bid_size1",
        "ask_size1",
        "bid_size2",
        "ask_size2",
    }
    missing = required - set(book.columns)
    if missing:
        raise ValueError(f"{book_path} is missing columns {sorted(missing)}")
    denominator1 = (book.bid_size1 + book.ask_size1).replace(0, np.nan)
    denominator2 = (book.bid_size2 + book.ask_size2).replace(0, np.nan)
    book["wap1"] = (
        book.bid_price1 * book.ask_size1 + book.ask_price1 * book.bid_size1
    ) / denominator1
    book["wap2"] = (
        book.bid_price2 * book.ask_size2 + book.ask_price2 * book.bid_size2
    ) / denominator2
    book["spread"] = (book.ask_price1 - book.bid_price1) / (
        (book.ask_price1 + book.bid_price1) / 2
    )
    total_size = book.bid_size1 + book.ask_size1
    book["imbalance"] = (book.bid_size1 - book.ask_size1) / total_size.replace(
        0, np.nan
    )

    rows = []
    for time_id, group in book.groupby("time_id", sort=False):
        last = group[group.seconds_in_bucket >= 300]
        rows.append(
            {
                "time_id": int(time_id),
                "book_wap1_rv": _rv(group.wap1),
                "book_wap2_rv": _rv(group.wap2),
                "book_wap1_rv_last300": _rv(last.wap1),
                "book_wap2_rv_last300": _rv(last.wap2),
                "book_spread_mean": group.spread.mean(),
                "book_spread_std": group.spread.std(),
                "book_spread_max": group.spread.max(),
                "book_spread_last300_mean": last.spread.mean(),
                "book_imbalance_mean": group.imbalance.mean(),
                "book_imbalance_std": group.imbalance.std(),
                "book_bid_size_sum": group.bid_size1.sum(),
                "book_ask_size_sum": group.ask_size1.sum(),
                "book_updates": len(group),
                "book_updates_last300": len(last),
            }
        )
    output = pd.DataFrame(rows).set_index("time_id")

    if trade_path is not None and trade_path.exists():
        trade = pd.read_parquet(trade_path)
        required_trade = {
            "time_id",
            "seconds_in_bucket",
            "price",
            "size",
            "order_count",
        }
        missing = required_trade - set(trade.columns)
        if missing:
            raise ValueError(f"{trade_path} is missing columns {sorted(missing)}")
        trade_rows = []
        for time_id, group in trade.groupby("time_id", sort=False):
            last = group[group.seconds_in_bucket >= 300]
            trade_rows.append(
                {
                    "time_id": int(time_id),
                    "trade_price_rv": _rv(group.price),
                    "trade_price_rv_last300": _rv(last.price),
                    "trade_price_std": group.price.std(),
                    "trade_size_sum": group["size"].sum(),
                    "trade_size_mean": group["size"].mean(),
                    "trade_order_count_sum": group.order_count.sum(),
                    "trade_order_count_mean": group.order_count.mean(),
                    "trade_count": len(group),
                    "trade_count_last300": len(last),
                }
            )
        if trade_rows:
            output = output.join(
                pd.DataFrame(trade_rows).set_index("time_id"), how="left"
            )
    return output.reset_index()


def prepare(raw: Path, output: Path, *, max_stocks: int | None = None) -> Path:
    import pandas as pd

    train_csv = raw / "train.csv"
    book_root = raw / "book_train.parquet"
    trade_root = raw / "trade_train.parquet"
    if not train_csv.is_file() or not book_root.is_dir():
        raise FileNotFoundError(
            "Expected train.csv and book_train.parquet under the raw directory. "
            "Download and unzip the Kaggle competition first."
        )
    target = pd.read_csv(train_csv)
    if not {"stock_id", "time_id", "target"} <= set(target.columns):
        raise ValueError("train.csv must contain stock_id, time_id, and target")
    book_paths = sorted(book_root.glob("stock_id=*"), key=_stock_id)
    if max_stocks is not None:
        book_paths = book_paths[:max_stocks]
    if not book_paths:
        raise FileNotFoundError(f"No stock_id=* partitions found under {book_root}")
    frames = []
    for number, book_path in enumerate(book_paths, 1):
        stock = _stock_id(book_path)
        trade_path = trade_root / book_path.name
        frame = _aggregate_stock(book_path, trade_path if trade_root.is_dir() else None)
        frame["stock_id"] = stock
        frames.append(frame)
        print(
            f"[{number}/{len(book_paths)}] stock_id={stock} buckets={len(frame)}",
            flush=True,
        )
    features = pd.concat(frames, ignore_index=True).merge(
        target, on=["stock_id", "time_id"], how="inner", validate="one_to_one"
    )
    for name in OPTIVER_FEATURE_NAMES:
        if name not in features:
            features[name] = np.nan
    return save_optiver_cache(
        output,
        features=features.loc[:, OPTIVER_FEATURE_NAMES].to_numpy(np.float32),
        targets=features.target.to_numpy(np.float32),
        stock_ids=features.stock_id.to_numpy(np.int32),
        time_ids=features.time_id.to_numpy(np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-stocks", type=int)
    args = parser.parse_args()
    path = prepare(args.raw, args.output, max_stocks=args.max_stocks)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
