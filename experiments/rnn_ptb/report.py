#!/usr/bin/env python3
"""Validation-only rate selection and paired PTB results (standard library only)."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ARMS = ("none", "uniform", "early_3_3", "early_2_4", "linear_decreasing", "late_3_3")
CALIBRATION_RATES = (0., .01, .025, .05, .1, .2)


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def select_calibration(root):
    results = []
    for p in CALIBRATION_RATES:
        arm = "none" if p == 0 else "uniform"
        r = json.loads((root / "calibrate" / "seed-0" / f"{arm}-p{p:g}" / "result.json").read_text())
        if r["stage"] != "calibrate" or r["seed"] != 0 or r["arm"] != arm or r["mean_p"] != p or r["test"] is not None:
            raise ValueError("Invalid calibration identity or test leakage")
        if results and r["protocol"] != results[0]["protocol"]:
            raise ValueError("Mixed calibration protocols")
        results.append(r)
    best = min(results[1:], key=lambda r: (r["best_valid_loss"], r["mean_p"]))
    selection = {"protocol": best["protocol"], "selected_mean_p": best["mean_p"],
                 "dropout_helped_calibration_seed": best["best_valid_loss"] < results[0]["best_valid_loss"],
                 "none_valid_perplexity": results[0]["best_valid_perplexity"],
                 "selected_valid_perplexity": best["best_valid_perplexity"],
                 "calibration": [{"mean_p": r["mean_p"], "valid_perplexity": r["best_valid_perplexity"],
                                  "fingerprint": r["fingerprint"]} for r in results]}
    atomic_json(root / "selection.json", selection)
    print(json.dumps(selection, indent=2))
    return selection


def analyze(root):
    selection = json.loads((root / "selection.json").read_text())
    p = selection["selected_mean_p"]
    records, rows = {}, []
    for seed in (1, 2, 3):
        for arm in ARMS:
            r = json.loads((root / "confirm" / f"seed-{seed}" / f"{arm}-p{p:g}" / "result.json").read_text())
            if (r["protocol"] != selection["protocol"] or r["arm"] != arm or
                    r["seed"] != seed or r["mean_p"] != p or r["stage"] != "confirm" or r["test"] is None):
                raise ValueError("Mismatched confirmation result")
            records[seed, arm] = r
            rows.append({"seed": seed, "arm": arm, "mean_p": p, "best_epoch": r["best_epoch"],
                         "train_perplexity": r["train_clean"]["perplexity"],
                         "valid_perplexity": r["best_valid_perplexity"],
                         "test_perplexity": r["test"]["perplexity"], "test_loss": r["test"]["loss"]})
    paired = []
    for seed in (1, 2, 3):
        for arm, baseline in [(a, "uniform") for a in ARMS if a != "uniform"] + [("early_3_3", "late_3_3")]:
            a, b = records[seed, arm]["test"], records[seed, baseline]["test"]
            paired.append({"seed": seed, "arm": arm, "baseline": baseline,
                           "test_ppl_delta": a["perplexity"] - b["perplexity"],
                           "test_loss_delta": a["loss"] - b["loss"]})
    for name, data in (("results.csv", rows), ("paired.csv", paired)):
        with (root / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = ["# Penn Treebank six-layer LSTM pilot", "",
             f"Uniform rate selected using validation only, seed 0: {p:g}.",
             f"Dropout beat no dropout on that calibration seed: {selection['dropout_helped_calibration_seed']}.",
             "Confirmation uses fresh paired seeds 1, 2, 3; full standard PTB splits.",
             "Mean ± sample SD; lower perplexity is better.", "",
             "| Schedule | Clean train PPL | Validation PPL | Test PPL |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        parts = []
        for key in ("train_perplexity", "valid_perplexity", "test_perplexity"):
            vals = [r[key] for r in rows if r["arm"] == arm]
            parts.append(f"{mean(vals):.2f} ± {stdev(vals):.2f}")
        lines.append(f"| {arm} | " + " | ".join(parts) + " |")
    lines += ["", "This is a small six-layer model study, not a reproduction of published state-of-the-art scores.",
              "Nonzero schedules match mean probability; an RNN effective-field budget is not derived.",
              "No-dropout remains a baseline even if calibration favors zero. Three seeds are exploratory."]
    report = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--select", action="store_true")
    args = p.parse_args()
    select_calibration(args.out) if args.select else analyze(args.out)
