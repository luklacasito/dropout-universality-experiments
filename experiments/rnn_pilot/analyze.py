#!/usr/bin/env python3
"""Summarize complete paired seeds; standard library only, no GPU needed."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ARMS = ("none", "uniform", "early_3_3", "early_2_4", "linear_decreasing", "late_3_3")
COMPARISONS = (
    ("early_3_3", "uniform"), ("early_2_4", "uniform"), ("linear_decreasing", "uniform"),
    ("early_3_3", "late_3_3"), ("early_2_4", "early_3_3"),
    ("linear_decreasing", "early_3_3"),
    ("early_3_3", "none"), ("early_2_4", "none"), ("linear_decreasing", "none"),
)


def analyze(root: Path, seeds: list[int]):
    rows, results, reference = [], {}, None
    for seed in seeds:
        for arm in ARMS:
            path = root / f"seed-{seed}" / arm / "result.json"
            if not path.exists():
                raise ValueError(f"Incomplete pilot: missing {path}")
            result = json.loads(path.read_text())
            if result["seed"] != seed or result["arm"] != arm:
                raise ValueError(f"Result identity mismatch: {path}")
            config = {k: v for k, v in result["config"].items() if k != "seed"}
            if reference is None:
                reference = config
            if config != reference:
                raise ValueError(f"Mixed configurations/source/environments: {path}")
            results[seed, arm] = result
            for length, metrics in result["test"].items():
                rows.append({"seed": seed, "arm": arm, "length": int(length),
                             "test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"],
                             "best_epoch": result["best_epoch"],
                             "train_accuracy": result["train_clean"]["accuracy"],
                             "blank_cue_accuracy": result["blank_cue_test"]["accuracy"],
                             "seconds": result["seconds"]})
    lengths = sorted({r["length"] for r in rows})
    comparisons = []
    for length in lengths:
        for arm, baseline in COMPARISONS:
            for seed in seeds:
                a = results[seed, arm]["test"][str(length)]
                b = results[seed, baseline]["test"][str(length)]
                comparisons.append({"length": length, "arm": arm, "baseline": baseline,
                                    "seed": seed, "accuracy_delta_pp": 100 * (a["accuracy"] - b["accuracy"]),
                                    "loss_delta": a["loss"] - b["loss"]})
    for name, data in (("results.csv", rows), ("paired.csv", comparisons)):
        with (root / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = ["# RNN depth-dropout pilot", "",
             f"Complete paired seeds: {seeds}. Chance accuracy: 12.5%. Training length: {reference['length']}.",
             "", "Validation selects the checkpoint; test sets are evaluated only afterward.",
             "Positive arms match raw dropout probability, not a derived RNN effective field.",
             "Means ± sample SD across training seeds; the data splits are fixed.", "",
             "| Length | Schedule | Test accuracy (%) | Test cross-entropy |",
             "|---:|---|---:|---:|"]
    def fmt(values, scale=1):
        values = [scale * x for x in values]
        return f"{mean(values):.3f} ± {stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f} (one seed)"
    for length in lengths:
        for arm in ARMS:
            subset = [r for r in rows if r["length"] == length and r["arm"] == arm]
            lines.append(f"| {length} | {arm} | {fmt([r['test_accuracy'] for r in subset], 100)} | {fmt([r['test_loss'] for r in subset])} |")
    lines += ["", "## Paired differences", "", "Differences are first schedule minus second; positive accuracy and negative loss favor the first.", "",
              "| Length | Comparison | Accuracy delta (pp) | Loss delta |",
              "|---:|---|---:|---:|"]
    for length in lengths:
        for arm, baseline in COMPARISONS:
            subset = [r for r in comparisons if r["length"] == length and r["arm"] == arm and r["baseline"] == baseline]
            lines.append(f"| {length} | {arm} − {baseline} | {fmt([r['accuracy_delta_pp'] for r in subset])} | {fmt([r['loss_delta'] for r in subset])} |")
    blank = [r["blank_cue_test"]["accuracy"] for r in results.values()]
    lines += ["", "## Interpretation", "",
              f"Blank-cue accuracy range: {100 * min(blank):.2f}–{100 * max(blank):.2f}% (chance 12.5%).",
              "Primary comparisons: early_3_3, early_2_4, and linear_decreasing versus uniform at the training length.",
              "early_3_3 versus late_3_3 tests ordering; comparisons among early profiles test concentration and smoothness.",
              "Budget is the sum of six probabilities, not the expected count of dropped units; the first site has input dimension nine.",
              "Longer lengths test delay generalization, not fitted correlation-length exponents.",
              "If all arms remain near chance, the training setup failed to learn; if all saturate, compare loss and treat accuracy as inconclusive.",
              "If no-dropout wins, this pilot does not establish a benefit from regularization.",
              "Three seeds are exploratory evidence, not a significance claim or a universality test."]
    report = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be unique")
    analyze(args.out, args.seeds)
