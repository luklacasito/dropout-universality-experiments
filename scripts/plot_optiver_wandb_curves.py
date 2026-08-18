#!/usr/bin/env python3
"""Export and plot the completed Optiver confirmation learning curves from W&B."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


EXPECTED_RUNS = 50
EXPECTED_EPOCHS = 30
METRIC_KEYS = ("epoch", "train/rmspe", "validation/rmspe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=10,
        help="Maximum histories to fetch in this invocation (0 means all).",
    )
    return parser.parse_args()


def export_histories(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]
    path = f"{entity}/{project}"
    filters = {
        "state": "finished",
        "jobType": "confirmation",
        "tags": {"$in": ["optiver"]},
    }
    api = wandb.Api(timeout=120)
    runs = api.runs(
        path,
        filters=filters,
        order="+created_at",
        per_page=100,
        include_sweeps=False,
    )
    if len(runs) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, found {len(runs)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "optiver-winning-mlp-confirmation-curves.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        frame = pd.DataFrame()
    complete_ids = set()
    if not frame.empty:
        counts = frame.groupby("run_id").size()
        complete_ids = set(counts[counts == EXPECTED_EPOCHS].index.astype(str))

    pending = [run for run in runs if run.id not in complete_ids]
    if args.max_new_runs:
        pending = pending[: args.max_new_runs]
    print(
        f"project={path} exact_runs={len(runs)} cached={len(complete_ids)} "
        f"fetching={len(pending)}",
        flush=True,
    )
    for index, run in enumerate(pending, 1):
        config = run.config
        history = run.history(samples=100, keys=list(METRIC_KEYS), pandas=True)
        history = history.dropna(subset=list(METRIC_KEYS)).copy()
        if len(history) != EXPECTED_EPOCHS:
            raise RuntimeError(f"{run.id}: expected 30 history rows, found {len(history)}")
        addition = pd.DataFrame(
            {
                "architecture": config["architecture"],
                "profile": config["profile_id"],
                "p": float(config["mean_dropout"]),
                "seed": int(config["seed"]),
                "epoch": history["epoch"].astype(int).to_numpy(),
                "train_rmspe": history["train/rmspe"].astype(float).to_numpy(),
                "validation_rmspe": history["validation/rmspe"]
                .astype(float)
                .to_numpy(),
                "run_id": run.id,
            }
        )
        frame = pd.concat([frame, addition], ignore_index=True)
        frame = frame.drop_duplicates(["run_id", "epoch"], keep="last")
        frame.to_csv(csv_path, index=False)
        print(f"  fetched {index}/{len(pending)}: {run.id}", flush=True)

    counts = frame.groupby("run_id").size() if not frame.empty else pd.Series()
    complete = int((counts == EXPECTED_EPOCHS).sum()) == EXPECTED_RUNS
    print(
        f"cached_complete_runs={int((counts == EXPECTED_EPOCHS).sum())}/"
        f"{EXPECTED_RUNS} rows={len(frame)}",
        flush=True,
    )
    return frame, complete


def plot(frame: pd.DataFrame, output_dir: Path) -> Path:
    colors = {
        "none": "#222222",
        "uniform": "#4C78A8",
        "early": "#E45756",
        "late": "#72B7B2",
        "step_early": "#E45756",
        "big_step": "#F2CF5B",
        "linear_early": "#B279A2",
        "linear_late": "#54A24B",
    }
    labels = {
        "none": "No dropout",
        "uniform": "Uniform",
        "early": "Early dropout",
        "late": "Late dropout",
        "step_early": "Step-early",
        "big_step": "Big-step",
        "linear_early": "Linear-early",
        "linear_late": "Linear-late",
    }
    orders = {
        "exact_shallow": ["none", "uniform", "early", "late"],
        "matched_depth12": [
            "none",
            "uniform",
            "step_early",
            "big_step",
            "linear_early",
            "linear_late",
        ],
    }
    titles = {
        "exact_shallow": "Exact winning MLP (2 hidden layers)",
        "matched_depth12": "Parameter-matched depth-12 MLP",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10.5), sharex=True, constrained_layout=True
    )
    for row, (architecture, profiles) in enumerate(orders.items()):
        for column, (metric, metric_title) in enumerate(
            [
                ("train_rmspe", "Training RMSPE"),
                ("validation_rmspe", "Validation RMSPE"),
            ]
        ):
            axis = axes[row, column]
            zoom = inset_axes(
                axis, width="48%", height="43%", loc="upper right", borderpad=1.15
            )
            for profile in profiles:
                subset = frame[
                    (frame.architecture == architecture) & (frame.profile == profile)
                ]
                if subset.seed.nunique() != 5:
                    raise RuntimeError(
                        f"{architecture}/{profile}: expected five seeds, found "
                        f"{subset.seed.nunique()}"
                    )
                p = float(subset.p.iloc[0])
                grouped = subset.groupby("epoch")[metric]
                mean = grouped.mean()
                lower_quartile = grouped.quantile(0.25)
                upper_quartile = grouped.quantile(0.75)
                x = mean.index.to_numpy()
                y = mean.to_numpy()
                legend = labels[profile] + ("" if p == 0 else f" (p̄={p:g})")
                axis.plot(x, y, label=legend, color=colors[profile], linewidth=2.1)
                axis.fill_between(
                    x,
                    lower_quartile.to_numpy(),
                    upper_quartile.to_numpy(),
                    color=colors[profile],
                    alpha=0.10,
                    linewidth=0,
                )
                late = x >= 8
                zoom.plot(x[late], y[late], color=colors[profile], linewidth=1.7)
            axis.set_yscale("log")
            axis.set_xlim(0, 29)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("RMSPE (log scale; lower is better)")
            axis.set_title(
                f"{titles[architecture]} — {metric_title}",
                fontsize=13.2,
                fontweight="bold",
            )
            axis.grid(alpha=0.25, which="both")
            axis.legend(
                frameon=True,
                fontsize=8.5,
                loc="lower left",
                ncol=2 if len(profiles) > 4 else 1,
            )
            zoom.set_xlim(8, 29)
            zoom.set_title("Epochs 8–29 (linear zoom)", fontsize=8.5)
            late_values = frame[
                (frame.architecture == architecture) & (frame.epoch >= 8)
            ][metric]
            low = float(late_values.quantile(0.01))
            high = float(late_values.quantile(0.99))
            padding = 0.08 * (high - low)
            zoom.set_ylim(max(0, low - padding), high + padding)
            zoom.grid(alpha=0.20)
            zoom.tick_params(labelsize=7)

    figure.suptitle(
        "Optiver winning-feature MLP: confirmation learning curves",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.005,
        "Mean across five fresh seeds (100–104); shading is the interquartile range. Insets reveal "
        "late-epoch separation. Test RMSPE is a single best-validation-checkpoint "
        "evaluation, so no per-epoch test curve exists.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    output = output_dir / "optiver-winning-mlp-confirmation-training-curves.png"
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    frame, complete = export_histories(args)
    if complete:
        output = plot(frame, args.output_dir)
        print(f"figure={output}", flush=True)
    else:
        print("figure=pending-more-history-chunks", flush=True)


if __name__ == "__main__":
    main()
