#!/usr/bin/env python3
"""Compare the added increasing schedule with the completed parent controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


def analyze(base_out, out, if_ready=False):
    selection = json.loads((base_out / "selection.json").read_text())
    p = selection["selected_mean_p"]
    arms = ("none", "uniform", "linear_decreasing", "linear_increasing")
    extension_hash = hashlib.sha256((Path(__file__).parent / "extend.py").read_bytes()).hexdigest()
    rows = []
    for seed in (1, 2, 3):
        for arm in arms:
            root = out if arm == "linear_increasing" else base_out
            path = root / "confirm" / f"seed-{seed}" / f"{arm}-p{p:g}" / "result.json"
            if if_ready and not path.exists():
                print("Comparison pending remaining results")
                return
            r = json.loads(path.read_text())
            expected = dict(selection["protocol"])
            if arm == "linear_increasing":
                expected["linear_extension_sha256"] = extension_hash
            if (r["protocol"] != expected or r["arm"] != arm or r["seed"] != seed
                    or r["mean_p"] != p or r["stage"] != "confirm" or r["test"] is None):
                raise ValueError(f"Mismatched result: {path}")
            rows.append({"seed": seed, "arm": arm, "train_ppl": r["train_clean"]["perplexity"],
                         "valid_ppl": r["best_valid_perplexity"], "test_ppl": r["test"]["perplexity"]})
    with (out / "linear_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# PTB linear schedule follow-up", "",
             f"Mean dropout {p:g}; paired seeds 1, 2, 3; mean ± sample SD; lower PPL is better.", "",
             "| Schedule | Train PPL | Validation PPL | Test PPL |", "|---|---:|---:|---:|"]
    for arm in arms:
        values = [[r[key] for r in rows if r["arm"] == arm] for key in ("train_ppl", "valid_ppl", "test_ppl")]
        lines.append(f"| {arm} | " + " | ".join(f"{mean(v):.2f} ± {stdev(v):.2f}" for v in values) + " |")
    differences = []
    for seed in (1, 2, 3):
        pair = {r["arm"]: r["test_ppl"] for r in rows if r["seed"] == seed}
        differences.append(pair["linear_increasing"] - pair["linear_decreasing"])
    lines += ["", f"Paired test PPL difference (increasing minus decreasing): {mean(differences):.2f} ± {stdev(differences):.2f}.",
              "Exploratory extension requested after partial parent results; no new rate tuning.",
              "Parent controls are reused unchanged. Equal mean probability is not a derived RNN field budget."]
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    analyze(args.base_out, args.out, args.if_ready)
