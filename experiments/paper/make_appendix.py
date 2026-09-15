#!/usr/bin/env python
"""Build v2 figures and manuscript tables from checked-in, run-level results.

No network access is needed. Test-loss winners are descriptive: they are chosen
from the observed confirmation results, not an independent validation selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dropout_mft.paths import project_root
from dropout_mft.results import load_npz_result
from dropout_mft.style import COLORS, apply_paper_style

ROOT = project_root()
PROFILE_LABELS = {
    "none_tuned": "No dropout (tuned)",
    "none": "No dropout",
    "uniform": "Uniform",
    "step_early": "Step early",
    "big_step": "Big step",
    "linear_early": "Linear decreasing",
    "linear_late": "Linear increasing",
    "step_late": "Step late",
    "quadratic_early": "Quadratic early",
    "quadratic_late": "Quadratic late",
    "quartic_early": "Quartic early",
    "quartic_late": "Quartic late",
}
PROFILE_COLORS = dict(
    zip(
        PROFILE_LABELS,
        [
            COLORS["baseline"],
            COLORS["neutral"],
            COLORS["kink"],
            COLORS["smooth"],
            COLORS["teal"],
            COLORS["dark_gold"],
            COLORS["rose"],
            COLORS["muted_blue"],
            "#668866",
            "#887799",
            "#88AA66",
            "#AA6688",
        ],
    )
)
DATASET_LABELS = {
    "fi2010": "FI-2010",
    "openml_jannis": "Jannis",
    "amazon_reviews": "Amazon Reviews",
    "speech_commands": "Speech Commands",
    "tiny_imagenet": "Tiny ImageNet",
}


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Metrics must be nonempty and finite")
    return float(values.mean()), float(
        values.std(ddof=1) / np.sqrt(len(values))
    ) if len(values) > 1 else 0.0


def describe_group(rows):
    config = rows[0]["config"]
    cohort = config["cohort_id"]
    dataset = DATASET_LABELS[config["dataset"]]
    model = "MLP" if config["model_kind"] == "mlp" else "Transformer"
    if "data-regime" in cohort:
        setting = f"{config['train_size']:,} train"
        section = "Data-regime comparisons"
    elif "sidecar" in cohort:
        setting, section = "Linear follow-up", "Architecture and duration follow-ups"
    elif "100epoch" in cohort:
        setting, section = "100 epochs", "Architecture and duration follow-ups"
    else:
        setting = "Zero weight decay"
        section = "Multi-dataset benchmark"
    label = f"{dataset} / {model} / {setting}"
    return dataset, model, setting, section, label


def summarize_group(rows):
    """Keep every arm visible; only full paired seed sets can win the table."""
    by_profile = defaultdict(list)
    common_keys = (
        "dataset",
        "model_kind",
        "depth",
        "width",
        "epochs",
        "train_size",
        "data_split_hash",
        "weight_decay",
    )
    first = rows[0]["config"]
    for row in rows:
        config, summary = row["config"], row["summary"]
        if (
            row["state"] != "finished"
            or summary.get("trial/status") != "complete"
            or summary.get("test/evaluated") is not True
        ):
            raise ValueError(f"Run is not a completed test evaluation: {row['id']}")
        if any(config.get(k) != first.get(k) for k in common_keys):
            raise ValueError(
                "Do not pool different datasets, splits, or training protocols"
            )
        history = row["history"]
        if [int(p["epoch"]) for p in history] != list(range(config["epochs"])):
            raise ValueError(f"Incomplete epoch history: {row['id']}")
        by_profile[config["profile_id"]].append(row)
    arms = {}
    expected = set(range(100, 110 if "100epoch" in first["cohort_id"] else 105))
    for profile, runs in by_profile.items():
        runs.sort(key=lambda r: r["config"]["seed"])
        seeds = [r["config"]["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate paired seed in {profile}")
        for key in ("mean_dropout", "learning_rate", "dropout_probabilities"):
            if any(r["config"][key] != runs[0]["config"][key] for r in runs):
                raise ValueError(f"Mixed profile settings: {profile}/{key}")
        losses = [r["summary"]["test/loss"] for r in runs]
        accuracies = [100 * r["summary"]["test/accuracy"] for r in runs]
        loss, loss_sem = mean_sem(losses)
        accuracy, accuracy_sem = mean_sem(accuracies)
        arms[profile] = {
            "runs": runs,
            "seeds": seeds,
            "n": len(runs),
            "complete": set(seeds) == expected,
            "loss": loss,
            "loss_sem": loss_sem,
            "accuracy": accuracy,
            "accuracy_sem": accuracy_sem,
            "p": runs[0]["config"]["mean_dropout"],
            "lr": runs[0]["config"]["learning_rate"],
        }
    eligible = [
        p
        for p, a in arms.items()
        if p not in ("none", "none_tuned", "uniform") and a["complete"]
    ]
    winner = min(eligible, key=lambda p: (arms[p]["loss"], p)) if eligible else None
    baseline = arms.get("uniform")
    paired = None
    if baseline and baseline["complete"] and winner:
        chosen = arms[winner]
        if baseline["seeds"] != chosen["seeds"]:
            raise ValueError("Winner and uniform baseline must use identical seeds")
        differences = np.array(
            [r["summary"]["test/loss"] for r in chosen["runs"]]
        ) - np.array([r["summary"]["test/loss"] for r in baseline["runs"]])
        delta, delta_sem = mean_sem(differences)
        paired = {
            "delta": delta,
            "delta_sem": delta_sem,
            "reduction_percent": 100
            * (baseline["loss"] - chosen["loss"])
            / baseline["loss"],
        }
    return {
        "arms": arms,
        "winner": winner,
        "paired": paired,
        "expected_seeds": sorted(expected),
        "complete": all(a["complete"] for a in arms.values()),
        "config": first,
    }


def plot_group(summary, output):
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    metrics = [
        ("train/loss", "Training loss", False),
        ("validation/loss", "Validation loss", False),
        ("train/accuracy", "Training accuracy (%)", True),
        ("validation/accuracy", "Validation accuracy (%)", True),
    ]
    for ax, (metric, title, percent) in zip(axes.flat, metrics):
        for profile in PROFILE_LABELS:
            if profile not in summary["arms"]:
                continue
            arm = summary["arms"][profile]
            values = np.array(
                [[p[metric] for p in r["history"]] for r in arm["runs"]], dtype=float
            )
            if percent:
                values *= 100
            mean = values.mean(axis=0)
            sem = (
                values.std(axis=0, ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else np.zeros_like(mean)
            )
            x = np.arange(len(mean))
            color = PROFILE_COLORS[profile]
            ax.plot(x, mean, color=color, label=PROFILE_LABELS[profile])
            lower = (
                np.maximum(mean - sem, 1e-10) if metric == "train/loss" else mean - sem
            )
            ax.fill_between(x, lower, mean + sem, color=color, alpha=0.14, linewidth=0)
        ax.set_title(title)
        ax.set_ylabel(title)
        if metric == "train/loss":
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5 if len(labels) > 6 else 3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(
        output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def tex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def estimate(mean, sem, places=3):
    return rf"\({mean:.{places}f}\pm{sem:.{places}f}\)"


def original_rows():
    """Original paper endpoints stay separate from validation-selected benchmarks."""
    rows = []
    cases = [
        (
            "MLP schedules",
            "CIFAR-10",
            "mlp/dropout_experiment_results.npz",
            "results",
            None,
            "constant",
            ["step", "reverse_step", "linear", "reverse_linear"],
        ),
        (
            "MLP budget controls",
            "CIFAR-10",
            "mlp/dropout_MLP_triple.npz",
            "results",
            None,
            "constant",
            ["big_step", "reverse_step"],
        ),
        (
            "ReLU p=0.1",
            "CIFAR-10",
            "sweeps/h_bar_sweep_results.npz",
            "all_results",
            0.1,
            "constant",
            ["big_step", "reverse_step"],
        ),
        (
            "GELU p=0.1",
            "CIFAR-10",
            "sweeps/gelu_h_bar_sweep_results.npz",
            "all_results",
            0.1,
            "constant",
            ["big_step", "reverse_step"],
        ),
    ]
    labels = {
        "step": "Step late",
        "reverse_step": "Step early",
        "linear": "Linear increasing",
        "reverse_linear": "Linear decreasing",
        "big_step": "Big step",
    }
    for label, dataset, path, key, budget, base, candidates in cases:
        results = load_npz_result(ROOT / "results" / path)[key]

        def entry(profile):
            return results[profile if budget is None else str((budget, profile))]

        candidates = [
            p
            for p in candidates
            if (p if budget is None else str((budget, p))) in results
        ]
        winner = min(
            candidates, key=lambda p: np.array(entry(p)["test_loss"])[:, -1].mean()
        )
        a, b = entry(winner), entry(base)
        rows.append((label, dataset, labels[winner], a, b))
    data = json.loads(
        (ROOT / "results/transformer/vit_dropout_results.json").read_text()
    )
    winner = min(
        ("reverse_step", "reverse_linear"),
        key=lambda p: np.array(data[p]["test_loss"])[:, -1].mean(),
    )
    rows.append(
        (
            "ViT",
            "CIFAR-100",
            {"reverse_step": "Step early", "reverse_linear": "Linear decreasing"}[
                winner
            ],
            data[winner],
            data["constant"],
        )
    )
    data = json.loads((ROOT / "results/transformer/ablation_results.json").read_text())
    rows.append(
        (
            "ViT both-block ablation",
            "CIFAR-10",
            "Step early",
            data["both_reverse_step"],
            data["both_constant"],
        )
    )
    output = []
    for label, dataset, winner, a, b in rows:
        loss, sem = mean_sem(np.asarray(a["test_loss"])[:, -1])
        base_loss, base_sem = mean_sem(np.asarray(b["test_loss"])[:, -1])
        acc, acc_sem = mean_sem(np.asarray(a["test_acc"])[:, -1])
        base_acc, base_acc_sem = mean_sem(np.asarray(b["test_acc"])[:, -1])
        output.append(
            dict(
                dataset=dataset,
                model=label,
                winner=winner,
                n=len(a["test_loss"]),
                uniform_loss=base_loss,
                uniform_loss_sem=base_sem,
                winner_loss=loss,
                winner_loss_sem=sem,
                uniform_accuracy=base_acc,
                winner_accuracy=acc,
                reduction_percent=100 * (base_loss - loss) / base_loss,
            )
        )
    return output


def build(output_dir, paper_dir=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = ROOT / "figures/paper/experiments/benchmarks"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / "results/benchmarks/confirmation.json.gz", "rt") as stream:
        payload = json.load(stream)
    grouped = defaultdict(list)
    for row in payload["runs"]:
        grouped[row["group"]].append(row)
    entries = []
    for i, (group, rows) in enumerate(sorted(grouped.items()), 1):
        summary = summarize_group(rows)
        dataset, model, setting, section, label = describe_group(rows)
        slug = f"benchmark-{i:02d}-{summary['config']['dataset']}-{summary['config']['model_kind']}"
        plot_group(summary, figure_dir / slug)
        entries.append(
            dict(
                group=group,
                summary=summary,
                dataset=dataset,
                model=model,
                setting=setting,
                section=section,
                label=label,
                slug=slug,
            )
        )
    table_rows = []
    for entry in entries:
        s = entry["summary"]
        winner = s["winner"]
        arm = s["arms"][winner] if winner else None
        base = s["arms"].get("uniform")
        table_rows.append(
            dict(
                dataset=entry["dataset"],
                model=entry["model"],
                setting=entry["setting"],
                section=entry["section"],
                group=entry["group"],
                winner=PROFILE_LABELS[winner] if winner else "Unavailable",
                n=arm["n"] if arm else 0,
                complete=s["complete"],
                uniform_loss=base["loss"] if base else None,
                uniform_loss_sem=base["loss_sem"] if base else None,
                winner_loss=arm["loss"] if arm else None,
                winner_loss_sem=arm["loss_sem"] if arm else None,
                uniform_accuracy=base["accuracy"] if base else None,
                winner_accuracy=arm["accuracy"] if arm else None,
                reduction_percent=s["paired"]["reduction_percent"]
                if s["paired"]
                else None,
            )
        )
    with (output_dir / "benchmark_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    (output_dir / "original_results.json").write_text(
        json.dumps(original_rows(), indent=2) + "\n"
    )
    write_tables(output_dir, table_rows, original_rows())
    write_appendix(output_dir, entries)
    add_profile_pilot(output_dir, figure_dir)
    write_index(output_dir, entries, table_rows)
    if paper_dir:
        (paper_dir / "sections").mkdir(exist_ok=True)
        shutil.copy2(
            ROOT / "paper/benchmark_discussion.tex",
            paper_dir / "sections/benchmark_discussion.tex",
        )
        for filename in (
            "benchmark_table.tex",
            "original_results_table.tex",
            "experimental_appendix.tex",
        ):
            destination = paper_dir / "sections" / filename
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(output_dir / filename, destination)
        destination = paper_dir / "figures/experiments/benchmarks"
        destination.mkdir(parents=True, exist_ok=True)
        for source in figure_dir.glob("*.pdf"):
            shutil.copy2(source, destination / source.name)
    print(
        f"Built {len(entries)} figures and result rows from {len(payload['runs'])} recorded runs"
    )


def write_tables(output, rows, originals):
    def table_start(caption, label):
        return [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{" + caption + "}",
            rf"\label{{{label}}}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}llrlrrrr@{}}",
            r"\toprule",
            r"Dataset / setting & Model & Seeds & Best nonuniform & Uniform CE & Best CE & Reduction & Acc. U / best \\",
            r"\midrule",
        ]

    lines = table_start(
        "Cross-dataset confirmation results. CE is test cross-entropy at each run's minimum-validation-loss checkpoint; uncertainties are SEM across seeds. The best nonuniform profile minimizes observed mean test CE among profiles with the full paired seed set, so these are descriptive best-observed comparisons, not independent tests of a prespecified winner. Reduction is relative to uniform; accuracies are percentages. A dagger marks incomplete candidate coverage. Missing baselines are shown explicitly.",
        "tab:benchmark_results",
    )
    for section in [
        "Multi-dataset benchmark",
        "Data-regime comparisons",
        "Architecture and duration follow-ups",
    ]:
        lines += [rf"\multicolumn{{8}}{{l}}{{\textit{{{section}}}}} \\"]
        for r in rows:
            if r["section"] != section:
                continue
            suffix = r"$^{\dagger}$" if not r["complete"] else ""
            dataset = r["dataset"] + (
                " " + r["setting"] if r["section"] == "Data-regime comparisons" else ""
            )
            if r["setting"] == "100 epochs":
                dataset += " (100 ep.)"
            if r["setting"] == "Linear follow-up":
                dataset += " (linear)"
            baseline = (
                estimate(r["uniform_loss"], r["uniform_loss_sem"])
                if r["uniform_loss"] is not None
                else "--"
            )
            winning = estimate(r["winner_loss"], r["winner_loss_sem"])
            reduction = (
                rf"{r['reduction_percent']:+.2f}\%"
                if r["reduction_percent"] is not None
                else "--"
            )
            acc = (
                f"{r['uniform_accuracy']:.2f} / {r['winner_accuracy']:.2f}"
                if r["uniform_accuracy"] is not None
                else f"-- / {r['winner_accuracy']:.2f}"
            )
            lines.append(
                f"{tex(dataset)}{suffix} & {r['model']} & {r['n']} & {tex(r['winner'])} & {baseline} & {winning} & {reduction} & {acc}"
                + r" \\"
            )
        lines += [r"\addlinespace[3pt]"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "benchmark_table.tex").write_text("\n".join(lines) + "\n")
    lines = table_start(
        "Original paper experiments, recomputed from the saved arrays at the final epoch. These endpoints differ from the validation-selected checkpoints in Table~\\ref{tab:benchmark_results}. Best denotes the lowest observed mean final test loss among the compared nonuniform schedules. Uncertainties are SEM; accuracies are percentages.",
        "tab:loss_improvements",
    )
    for r in originals:
        lines.append(
            f"{r['dataset']} & {r['model']} & {r['n']} & {r['winner']} & {estimate(r['uniform_loss'], r['uniform_loss_sem'])} & {estimate(r['winner_loss'], r['winner_loss_sem'])} & {r['reduction_percent']:+.2f}\\% & {r['uniform_accuracy']:.2f} / {r['winner_accuracy']:.2f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "original_results_table.tex").write_text("\n".join(lines) + "\n")


def write_appendix(output, entries):
    lines = [
        r"\clearpage",
        r"\section{Extended Experiments Across Datasets}",
        r"\label{app:extended_experiments}",
        r"The extended experiments ask how spatial allocation behaves across tasks, architectures, and training-data budgets. We preserve each cohort's split, optimization settings, and random seeds; cohorts are not pooled across weight decay or training duration. Table~\ref{tab:benchmark_results} reports every recovered confirmation comparison, including missing baseline information and incomplete candidate coverage.",
        r"\paragraph{Evaluation and selection.} Each profile's learning rate and mean dropout were selected by validation screening before confirmation. The recorded test loss is evaluated at the minimum-validation-loss checkpoint. Some cohorts also record a fixed-final-epoch test endpoint; that endpoint is reported separately below when available. The best nonuniform profile in the main table is selected descriptively from confirmation test means. Its improvement is therefore subject to winner-selection optimism and is not a prespecified significance claim. Profiles with missing paired seeds remain visible below but are ineligible to win the main table.",
        r"\paragraph{Figure convention.} We use the original paper's serif typography, schedule palette, four-panel layout, and mean $\pm$ SEM bands. The newer runs record training and validation at every epoch, so the right-hand panels show validation rather than invented test trajectories. The tables report the observed test endpoints. Training loss uses a logarithmic axis.",
        r"\paragraph{Budgets and architecture.} The extended benchmark compares tuned mean dropout probabilities; the tuned probability may differ across profiles, so these comparisons do not isolate placement at an equal budget. The big-step profile is cap-exempt, and its local probability can exceed the primary $0.20$ cap. Transformer profiles are empirical architecture extensions; the MLP reference-field diagnostic is not a derived Transformer correlation recursion.",
        r"\paragraph{Coverage.} The Amazon 20,000-example MLP cohort has four of five runs for each linear direction; other recorded candidates have their full seed sets. Two linear follow-up cohorts contain their selected linear profile and tuned no-dropout control, but their matching uniform-baseline source has not been recovered in this export. Their uniform comparisons remain unreported rather than borrowing a zero-weight-decay baseline from another protocol.",
    ]
    for entry in entries:
        s = entry["summary"]
        c = s["config"]
        lines += [
            r"\clearpage",
            r"\subsection{" + tex(entry["label"]) + "}",
            f"Depth {c['depth']}, width {c['width']}; {c['train_size']:,}/{c['validation_size']:,}/{c['test_size']:,} training/validation/test examples; {c['epochs']} epochs, batch size {c['batch_size']}, weight decay \\({c['weight_decay']:g}\\). The recorded split is held fixed across profiles.",
            r"\begin{figure}[ht]",
            r"\centering",
            rf"\includegraphics[width=0.92\textwidth]{{figures/experiments/benchmarks/{entry['slug']}.pdf}}",
            r"\caption{"
            + tex(entry["label"])
            + r". Full recorded learning curves; bands show SEM across the available seeds for each profile. Sample counts and tuned probabilities are given below.}",
            rf"\label{{fig:{entry['slug']}}}",
            r"\end{figure}",
            r"\begin{center}\small",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Profile & $n$ & $\bar p$ & Learning rate & Test CE & Test acc. (\%) & Final test CE \\",
            r"\midrule",
        ]
        for profile in PROFILE_LABELS:
            if profile not in s["arms"]:
                continue
            a = s["arms"][profile]
            final = [r["summary"].get("test/final_epoch_loss") for r in a["runs"]]
            final_text = (
                estimate(*mean_sem(final))
                if all(v is not None for v in final)
                else "--"
            )
            lines.append(
                f"{tex(PROFILE_LABELS[profile])} & {a['n']} & {a['p']:.2f} & {a['lr']:.0e} & {estimate(a['loss'], a['loss_sem'])} & {estimate(a['accuracy'], a['accuracy_sem'], 2)} & {final_text}"
                + r" \\"
            )
        lines += [r"\bottomrule\end{tabular}\end{center}"]
        if not s["complete"]:
            lines.append(
                r"The linear profiles are incomplete ($n=4$); their curves and endpoint estimates use only the recorded runs. The main table selects among profiles with all five paired seeds."
            )
        if s["paired"]:
            d = s["paired"]
            lines.append(
                r"For the best observed nonuniform profile, the paired test-CE difference (profile minus uniform) is "
                + estimate(d["delta"], d["delta_sem"])
                + " (SEM)."
            )
    (output / "experimental_appendix.tex").write_text("\n".join(lines) + "\n")


def add_profile_pilot(output, figure_dir):
    trials = load_npz_result(ROOT / "results/profiles/geometry_pilot.npz")["trials"]
    grouped = defaultdict(list)
    for trial in trials:
        if trial["test"].get("evaluated"):
            raise ValueError("Geometry pilot must keep the test set sealed")
        grouped[trial["factors"]["profile_id"]].append(trial)
    arms = {}
    for profile, records in grouped.items():
        runs = []
        for record in sorted(records, key=lambda r: r["factors"]["seed"]):
            curves = record["curves"]
            history = [
                {
                    "epoch": int(epoch),
                    **{
                        f"{split}/{metric}": float(curves[f"{split}_{metric}"][index])
                        for split in ("train", "validation")
                        for metric in ("loss", "accuracy")
                    },
                }
                for index, epoch in enumerate(curves["epoch"])
            ]
            runs.append({"history": history})
        arms[profile] = {"runs": runs}
    plot_group({"arms": arms}, figure_dir / "profile-geometry-pilot")
    lines = [
        r"\clearpage",
        r"\subsection{CIFAR-10 profile-geometry pilot}",
        r"The validation-only geometry pilot contains 30 runs: ten profiles at mean probability $\bar p=0.10$, with three seeds per profile, six-layer width-256 ReLU MLPs, 4,000 training examples, 1,000 validation examples, and 75 epochs. It compares profile shape and exact early/late reversals. Test data remain sealed, so this pilot contributes no test-loss winner to the main tables.",
        r"\begin{figure}[ht]\centering",
        r"\includegraphics[width=0.94\textwidth]{figures/experiments/benchmarks/profile-geometry-pilot.pdf}",
        r"\caption{Profile geometry at fixed mean dropout. Training and validation curves use the original paper style and show mean $\pm$ SEM over three seeds.}",
        r"\label{fig:profile-geometry-pilot}\end{figure}",
        r"\begin{center}\small\begin{tabular}{@{}lrr@{}}\toprule",
        r"Profile & Final validation CE & Final validation accuracy (\%) \\",
        r"\midrule",
    ]
    for profile in PROFILE_LABELS:
        if profile not in grouped:
            continue
        loss = mean_sem([r["curves"]["validation_loss"][-1] for r in grouped[profile]])
        accuracy = mean_sem(
            [100 * r["curves"]["validation_accuracy"][-1] for r in grouped[profile]]
        )
        lines.append(
            f"{PROFILE_LABELS[profile]} & {estimate(*loss)} & {estimate(*accuracy, places=2)}"
            + r" \\"
        )
    lines += [
        r"\bottomrule\end{tabular}\end{center}",
        r"A complete width-transfer result set was not available in the recovered records. The retained protocol defines those future comparisons, but the geometry pilot alone does not establish successful $\mu$P transfer.",
    ]
    with (output / "experimental_appendix.tex").open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def write_index(output, entries, rows):
    lines = [
        "# Results and manuscript",
        "",
        "[Main manuscript PDF](dropout-universality-v2.pdf) · [Full results CSV](benchmark_results.csv) · [Appendix source](experimental_appendix.tex)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python experiments/paper/make_appendix.py",
        "# Also copy the generated sections and figures into a manuscript checkout:",
        "python experiments/paper/make_appendix.py --paper-dir /path/to/icml26",
        "```",
        "",
        "Figures and tables rebuild offline from the versioned run export. The winning profile minimizes observed confirmation test loss among complete nonuniform arms; it is a descriptive result, not an independently tested selection. All other profiles remain in the appendix. Uncertainties are SEM.",
        "",
        "The full manuscript is maintained in the separate `icml26` checkout. This directory contains its compiled v2 PDF, [main-body discussion](benchmark_discussion.tex), and generated LaTeX tables and appendix; rebuilding the full PDF requires that manuscript checkout and LaTeX. The discussion is edited by hand; the builder copies it without rewriting it.",
        "",
        "## Benchmark results",
        "",
        "| Dataset / setting | Model | Best nonuniform | Uniform test CE | Best test CE | Reduction | Seeds |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for entry, r in zip(entries, rows):
        base = (
            f"{r['uniform_loss']:.4f}"
            if r["uniform_loss"] is not None
            else "Unavailable"
        )
        delta = (
            f"{r['reduction_percent']:+.2f}%"
            if r["reduction_percent"] is not None
            else "—"
        )
        lines.append(
            f"| [{r['dataset']} / {r['setting']}](../figures/paper/experiments/benchmarks/{entry['slug']}.pdf) | {r['model']} | {r['winner']} | {base} | {r['winner_loss']:.4f} | {delta} | {r['n']} |"
        )
    lines += [
        "",
        "Amazon 20k MLP has two missing linear-profile runs. Those arms are shown with n=4 in the appendix and cannot win the main comparison. Linear follow-up uniform baselines have not been recovered; they are not substituted with another cohort's baselines.",
        "",
        "## Original paper",
        "",
        "Original final-epoch comparisons are recomputed separately in [original_results.json](original_results.json) and [the LaTeX table](original_results_table.tex). Rebuild the original curves with `python experiments/paper/make_figures.py --all`.",
        "",
        "## Profile and width-transfer studies",
        "",
        "[Geometry-pilot figure](../figures/paper/experiments/benchmarks/profile-geometry-pilot.pdf)",
        "",
        "The 30-run validation-only geometry pilot is preserved in `results/profiles/geometry_pilot.npz`. Full width-transfer outcomes are not present in the recovered result set, so no transfer success is claimed. The protocol and commands remain under `experiments/scale_transfer/`.",
        "",
        "## Provenance",
        "",
        "The export contains each run's ID, source URL, scientific configuration, split hash, endpoint metrics, and every recorded epoch. See [the data README](../results/benchmarks/README.md).",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    build(args.output_dir, args.paper_dir)
