#!/usr/bin/env python
"""Verify an Optiver cache or run a bounded MLP/Transformer smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.optiver.data import (  # noqa: E402
    load_optiver_bundle,
    run_optiver_canary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--model", choices=("mlp", "transformer"), default="mlp")
    parser.add_argument("--profile", choices=("none", "uniform"), default="uniform")
    parser.add_argument("--mean-dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_optiver_bundle(args.cache, max_train_rows=args.max_train_rows)
    print(
        f"PASS cache train={len(bundle.train)} validation={len(bundle.validation)} "
        f"test={len(bundle.test)} features={bundle.feature_count} "
        f"split_hash={bundle.split_hash[:16]}"
    )
    if args.verify_only:
        return
    result = run_optiver_canary(
        bundle,
        model_kind=args.model,
        profile=args.profile,
        mean_dropout=0.0 if args.profile == "none" else args.mean_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=0,
        device=args.device,
    )
    serializable = {
        **result,
        "history": {key: value.tolist() for key, value in result["history"].items()},
    }
    print(json.dumps(serializable, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, payload=np.asarray(serializable, dtype=object))


if __name__ == "__main__":
    main()
