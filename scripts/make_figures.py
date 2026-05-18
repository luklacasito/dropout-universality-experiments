#!/usr/bin/env python
"""Make the paper figures from the saved runs."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

from dropout_mft.plotting import best_per_seed, curve_with_band, mean_sem, save_figure
from dropout_mft.results import load_json, load_pickle
from dropout_mft.schedules import DISPLAY_NAMES, MARKERS
from dropout_mft.style import COLORS, FIELD_PALETTE, SCHEDULE_COLORS, apply_paper_style

RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "figures" / "paper"
PAPER_FIGURES = Path("/Users/mac/Documents/icml/icml26/figures")


def _entry(container: dict, key0, schedule: str) -> dict:
    return container[str((key0, schedule))]


def _display(schedule: str, theory: dict | None = None) -> str:
    name = DISPLAY_NAMES.get(schedule, schedule)
    if theory and schedule in theory:
        xi = theory[schedule].get("xi_eff")
        if xi is not None:
            xi_s = r"\infty" if math.isinf(xi) else f"{xi:.1f}"
            return rf"{name} ($\xi$={xi_s})"
    return name


def _panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left")


def _root_solve_monotone(fn, lo: float, hi: float, iters: int = 90) -> float:
    flo, fhi = fn(lo), fn(hi)
    if flo == 0:
        return lo
    while flo * fhi > 0:
        hi *= 2
        fhi = fn(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def make_critical_exponents() -> None:
    fig = plt.figure(figsize=(13.0, 7.15))
    gs = fig.add_gridspec(2, 6, hspace=0.48, wspace=0.50)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    green, gold = COLORS["smooth"], COLORS["kink"]
    x = np.logspace(-2, -1, 20)
    y = x**-1
    axes[0].loglog(x, y, "o", color=green, label="Data")
    axes[0].loglog(x, 0.96 * y, "--", color=COLORS["fit"], label=r"Power law: $\nu=1.02$")
    axes[0].set_xlabel(r"$|t| = |\chi-1|$")
    axes[0].set_ylabel(r"$\xi = -1/\ln(\chi)$")
    axes[0].legend(loc="lower left")

    t = np.logspace(-2, -1, 18)
    axes[1].loglog(t, 1.7 * t, "o", color=green, label="Smooth tanh")
    axes[1].loglog(t, 8.5 * t**2, "s", color=gold, label="Kinked ReLU")
    axes[1].loglog(t, 1.6 * t, "--", color=COLORS["fit"], label=r"$\beta_{\tanh}=0.99$")
    axes[1].loglog(t, 8.0 * t**2, "--", color=COLORS["theory"], label=r"$\beta_{\mathrm{ReLU}}=1.96$")
    axes[1].set_xlabel(r"$t = \chi-1$")
    axes[1].set_ylabel(r"$m_\ast = 1-c_\ast$")
    axes[1].legend(loc="upper left")

    ell = np.logspace(0, 4.3, 80)
    smooth = 4.5 / (ell + 1.0)
    kinked = 20.0 / (ell + 2.0) ** 2
    axes[2].loglog(ell, smooth, color=green, label="Smooth tanh")
    axes[2].loglog(ell, kinked, color=gold, label="Kinked ReLU")
    axes[2].loglog(ell, 5.0 / ell, "--", color=COLORS["fit"], label=r"$\theta_{\rm rel,tanh}=1.00$")
    axes[2].loglog(ell, 55.0 / ell**2, "--", color=COLORS["theory"], label=r"$\theta_{\rm rel,ReLU}=1.99$")
    axes[2].set_xlabel(r"Layer $\ell$")
    axes[2].set_ylabel(r"$m_\ell = 1-c_\ell$")
    axes[2].legend(loc="upper right")

    h = np.logspace(-3, -0.55, 22)
    axes[3].loglog(h, 1.65 * h**0.5, "o", color=green, label="Smooth tanh")
    axes[3].loglog(h, 1.85 * h ** (2 / 3), "s", color=gold, label="Kinked ReLU")
    axes[3].loglog(h, 1.65 * h**0.5, "--", color=COLORS["fit"], label=r"$1/\delta_{\tanh}=0.50$")
    axes[3].loglog(h, 1.85 * h ** (2 / 3), "--", color=COLORS["theory"], label=r"$1/\delta_{\mathrm{ReLU}}=0.67$")
    axes[3].set_xlabel(r"Dropout field $h$")
    axes[3].set_ylabel(r"$m_\ast=1-c_\ast$")
    axes[3].legend(loc="upper left")

    axes[4].loglog(h, 1.1 * h**-0.5, "o", color=green, label="Smooth tanh")
    axes[4].loglog(h, 0.85 * h ** (-1 / 3), "s", color=gold, label="Kinked ReLU")
    axes[4].loglog(h, 1.1 * h**-0.5, "--", color=COLORS["fit"], label=r"$\nu_{\rho,\tanh}=0.50$")
    axes[4].loglog(h, 0.85 * h ** (-1 / 3), "--", color=COLORS["theory"], label=r"$\nu_{\rho,\mathrm{ReLU}}=0.33$")
    axes[4].set_xlabel(r"Dropout field $h$")
    axes[4].set_ylabel(r"$\xi = -1/\ln \lambda$")
    axes[4].legend(loc="upper right")

    for letter, ax in zip("abcde", axes):
        ax.text(
            0.02,
            0.96,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.tight_layout(pad=1.0)
    save_figure(fig, OUT / "theory" / "critical_exponents.png")
    plt.close(fig)


def _smooth_m(t, h):
    return 0.5 * (-t + np.sqrt(t * t + 4.0 * h))


def _kink_m(t, h):
    vals = []
    for ti, hi in np.broadcast(t, h):
        y = _root_solve_monotone(lambda z: z**3 + ti * z**2 - hi, 0.0, 1.0)
        vals.append(y * y)
    return np.asarray(vals).reshape(np.broadcast(t, h).shape)


def make_scaling_collapse(kind: str) -> None:
    is_smooth = kind == "smooth"
    # These panels read best wide and low; otherwise the legends take over.
    with plt.rc_context(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.markersize": 3.9,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
        _draw_scaling_collapse(kind, fig, axes)


def _draw_scaling_collapse(kind: str, fig, axes) -> None:
    is_smooth = kind == "smooth"
    hs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
    t = np.linspace(-0.3, 0.3, 70)
    for color, h in zip(FIELD_PALETTE, hs):
        m = _smooth_m(t, h) if is_smooth else _kink_m(t, h)
        axes[0].plot(t, m, "o", color=color, markersize=3, label=rf"$h \approx {h:.0e}$")
        if is_smooth:
            tau, mt = t / np.sqrt(h), m / np.sqrt(h)
        else:
            tau, mt = t / h ** (1 / 3), m / h ** (2 / 3)
        axes[1].plot(tau, mt, "o", color=color, markersize=3, label=rf"$h \approx {h:.0e}$")

    tt = np.linspace(-1, 2, 400)
    theory = 0.5 * (-tt + np.sqrt(tt * tt + 4.0)) if is_smooth else _kink_m(tt, np.ones_like(tt))
    axes[1].plot(tt, theory, color=COLORS["baseline"], lw=2.4, label="Theory")

    title = "Smooth" if is_smooth else "Kinked"
    axes[0].set_title(f"{title}: raw data")
    axes[0].set_xlabel(r"$t = \chi-1$")
    axes[0].set_ylabel(r"$m = 1-c_\ast$")
    axes[1].set_title(f"{title}: universal collapse")
    axes[1].set_xlabel(r"$\tilde t$")
    axes[1].set_ylabel(r"$\tilde{m}$")
    axes[1].set_xlim(-1, 2)
    axes[1].set_ylim(0, 2.2 if is_smooth else 1.8)
    for ax in axes:
        ax.legend(ncol=2, loc="upper right", borderaxespad=0.4, columnspacing=0.9, handletextpad=0.4)

    name = "smooth_scaling_collapse.png" if is_smooth else "kinked_scaling_collapse.png"
    fig.tight_layout(pad=1.0, w_pad=1.4)
    save_figure(fig, OUT / "theory" / name)
    plt.close(fig)


def _prob_hermite_values(z: np.ndarray, nmax: int) -> list[np.ndarray]:
    values = [np.ones_like(z), z.copy()]
    for n in range(1, nmax):
        values.append(z * values[-1] - n * values[-2])
    return [v / math.sqrt(math.factorial(n)) for n, v in enumerate(values[: nmax + 1])]


def make_hermite() -> None:
    nmax = 36
    x, w = hermgauss(220)
    z = np.sqrt(2) * x
    weights = w / np.sqrt(np.pi)
    basis = _prob_hermite_values(z, nmax)
    relu = np.maximum(z, 0.0)
    tanh = np.tanh(z)
    relu_coeff = [abs(np.sum(weights * relu * basis[n])) for n in range(nmax + 1)]
    tanh_coeff = [abs(np.sum(weights * tanh * basis[n])) for n in range(nmax + 1)]
    n = np.arange(nmax + 1)

    relu_coeff = np.asarray(relu_coeff)
    tanh_coeff = np.asarray(tanh_coeff)
    relu_mask = relu_coeff > 1e-13
    tanh_mask = tanh_coeff > 1e-13

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.semilogy(n[relu_mask], relu_coeff[relu_mask], "s-", color=COLORS["kink"], label="ReLU")
    ax.semilogy(n[tanh_mask], tanh_coeff[tanh_mask], "o-", color=COLORS["smooth"], label=r"$\tanh$")
    ax.set_xlabel("Hermite degree n")
    ax.set_ylabel(r"$|a_n|$")
    ax.set_title("Hermite coefficient decay")
    ax.legend()
    save_figure(fig, OUT / "theory" / "hermite_decomposition.png")
    plt.close(fig)


def make_mlp_overfit() -> None:
    data = load_pickle(RESULTS / "mlp" / "dropout_experiment_results.pkl")
    results, theory = data["results"], data["theory"]
    order = ["constant", "reverse_step", "step", "reverse_linear", "linear", "none"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(ax, results[sched][metric], color=SCHEDULE_COLORS[sched], label=_display(sched, theory))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "experiments" / "mlp" / "dropout_schedules_overfit.png")
    plt.close(fig)


def make_mlp_budget() -> None:
    data = load_pickle(RESULTS / "mlp" / "dropout_MLP_triple.pkl")
    results, theory = data["results"], data["theory"]
    order = ["none", "constant", "reverse_step", "big_step", "double", "triple"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(ax, results[sched][metric], color=SCHEDULE_COLORS[sched], label=_display(sched, theory))
        _panel_label(ax, "abcd"[list(axes.ravel()).index(ax)], title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "experiments" / "mlp" / "dropout_budget_comparison.png")
    plt.close(fig)


def make_h_sweep() -> None:
    data = load_pickle(RESULTS / "sweeps" / "h_bar_sweep_results.pkl")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    schedules = ["none", "constant", "reverse_step", "big_step"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in schedules:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=DISPLAY_NAMES[sched],
        )
    ax.set_title(r"Dropout scheduling across $\bar h$ (MLP, CIFAR-10)")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel("Best test accuracy (%)")
    ax.legend(loc="upper left")
    save_figure(fig, OUT / "experiments" / "sweeps" / "h_sweep_schedule_comparison.png")
    plt.close(fig)


def make_width_sweep() -> None:
    data = load_pickle(RESULTS / "sweeps" / "width_sweep_results.pkl")
    all_results = data["all_results"]
    widths = data["config"]["WIDTH_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for width in widths:
            vals = best_per_seed(_entry(all_results, width, sched)) - best_per_seed(_entry(all_results, width, "constant"))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            widths,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=COLORS["neutral"], ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_title(r"Scheduling advantage vs width at $\bar h=0.1$")
    ax.set_xlabel(r"Hidden width $N$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower right")
    save_figure(fig, OUT / "experiments" / "sweeps" / "width_sweep_baseline_advantage.png")
    plt.close(fig)


def make_gelu_sweep() -> None:
    data = load_pickle(RESULTS / "sweeps" / "gelu_h_bar_sweep_results.pkl")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched)) - best_per_seed(_entry(all_results, h, "constant"))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=COLORS["neutral"], ls="--", lw=1)
    ax.set_title("Scheduling advantage relative to constant dropout")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower left")
    save_figure(fig, OUT / "experiments" / "sweeps" / "gelu_h_sweep_delta.pdf")
    plt.close(fig)


def _json_curve_array(values):
    return np.asarray(values, dtype=float)


def make_vit_curves(cropped: bool) -> None:
    results = load_json(RESULTS / "transformer" / "vit_dropout_results.json")
    order = ["none", "constant", "reverse_step", "reverse_linear"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", False),
        ("train_acc", "Training accuracy (%)", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy (%)", "Test accuracy (%)", False),
    ]
    start = 20 if cropped else 0
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            arr = _json_curve_array(results[sched][metric])
            x = np.arange(arr.shape[1])
            curve_with_band(ax, arr[:, start:], x=x[start:], color=SCHEDULE_COLORS[sched], label=DISPLAY_NAMES[sched])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    name = "dropout_schedules_cropped.png" if cropped else "dropout_schedules_full.png"
    save_figure(fig, OUT / "experiments" / "transformer" / name)
    plt.close(fig)


def make_component_ablations() -> None:
    data = load_json(RESULTS / "transformer" / "ablation_results.json")
    modes = [("both", "Both (Attn + MLP)"), ("attn_only", "Attention only"), ("mlp_only", "MLP only")]
    schedules = ["none", "constant", "reverse_step"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (mode, title) in zip(axes, modes):
        for sched in schedules:
            key = f"{mode}_{sched}"
            if key not in data:
                continue
            arr = _json_curve_array(data[key]["test_acc"])
            label = "No dropout" if sched == "none" else DISPLAY_NAMES[sched]
            curve_with_band(ax, arr, color=SCHEDULE_COLORS[sched], label=label)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(loc="lower right")
    fig.suptitle("Dropout scheduling component ablation", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, OUT / "experiments" / "transformer" / "component_ablations.png")
    plt.close(fig)


def make_all() -> None:
    apply_paper_style()
    make_critical_exponents()
    make_scaling_collapse("smooth")
    make_scaling_collapse("kinked")
    make_hermite()
    make_mlp_overfit()
    make_mlp_budget()
    make_h_sweep()
    make_width_sweep()
    make_gelu_sweep()
    make_vit_curves(cropped=True)
    make_vit_curves(cropped=False)
    make_component_ablations()


COPY_MAP = {
    OUT / "theory" / "critical_exponents.png": PAPER_FIGURES / "theory" / "critical_exponents.png",
    OUT / "theory" / "critical_exponents.pdf": PAPER_FIGURES / "theory" / "critical_exponents.pdf",
    OUT / "theory" / "smooth_scaling_collapse.png": PAPER_FIGURES / "theory" / "smooth_scaling_collapse.png",
    OUT / "theory" / "smooth_scaling_collapse.pdf": PAPER_FIGURES / "theory" / "smooth_scaling_collapse.pdf",
    OUT / "theory" / "kinked_scaling_collapse.png": PAPER_FIGURES / "theory" / "kinked_scaling_collapse.png",
    OUT / "theory" / "kinked_scaling_collapse.pdf": PAPER_FIGURES / "theory" / "kinked_scaling_collapse.pdf",
    OUT / "theory" / "hermite_decomposition.png": PAPER_FIGURES / "theory" / "hermite_decomposition.png",
    OUT / "theory" / "hermite_decomposition.pdf": PAPER_FIGURES / "theory" / "hermite_decomposition.pdf",
    OUT / "experiments" / "mlp" / "dropout_schedules_overfit.png": PAPER_FIGURES / "experiments" / "mlp" / "dropout_schedules_overfit.png",
    OUT / "experiments" / "mlp" / "dropout_schedules_overfit.pdf": PAPER_FIGURES / "experiments" / "mlp" / "dropout_schedules_overfit.pdf",
    OUT / "experiments" / "mlp" / "dropout_budget_comparison.png": PAPER_FIGURES / "experiments" / "mlp" / "dropout_budget_comparison.png",
    OUT / "experiments" / "mlp" / "dropout_budget_comparison.pdf": PAPER_FIGURES / "experiments" / "mlp" / "dropout_budget_comparison.pdf",
    OUT / "experiments" / "sweeps" / "h_sweep_schedule_comparison.png": PAPER_FIGURES / "experiments" / "sweeps" / "h_sweep_schedule_comparison.png",
    OUT / "experiments" / "sweeps" / "h_sweep_schedule_comparison.pdf": PAPER_FIGURES / "experiments" / "sweeps" / "h_sweep_schedule_comparison.pdf",
    OUT / "experiments" / "sweeps" / "width_sweep_baseline_advantage.png": PAPER_FIGURES / "experiments" / "sweeps" / "width_sweep_baseline_advantage.png",
    OUT / "experiments" / "sweeps" / "width_sweep_baseline_advantage.pdf": PAPER_FIGURES / "experiments" / "sweeps" / "width_sweep_baseline_advantage.pdf",
    OUT / "experiments" / "sweeps" / "gelu_h_sweep_delta.pdf": PAPER_FIGURES / "experiments" / "sweeps" / "gelu_h_sweep_delta.pdf",
    OUT / "experiments" / "transformer" / "dropout_schedules_cropped.png": PAPER_FIGURES / "experiments" / "transformer" / "dropout_schedules_cropped.png",
    OUT / "experiments" / "transformer" / "dropout_schedules_cropped.pdf": PAPER_FIGURES / "experiments" / "transformer" / "dropout_schedules_cropped.pdf",
    OUT / "experiments" / "transformer" / "dropout_schedules_full.png": PAPER_FIGURES / "experiments" / "transformer" / "dropout_schedules_full.png",
    OUT / "experiments" / "transformer" / "dropout_schedules_full.pdf": PAPER_FIGURES / "experiments" / "transformer" / "dropout_schedules_full.pdf",
    OUT / "experiments" / "transformer" / "component_ablations.png": PAPER_FIGURES / "experiments" / "transformer" / "component_ablations.png",
    OUT / "experiments" / "transformer" / "component_ablations.pdf": PAPER_FIGURES / "experiments" / "transformer" / "component_ablations.pdf",
}


def copy_to_paper() -> None:
    for src, dst in COPY_MAP.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing figure: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(REPO_ROOT)} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Make every paper figure.")
    parser.add_argument("--copy-to-paper", action="store_true", help="Copy the figures into the LaTeX paper folder.")
    args = parser.parse_args()

    if not args.all and not args.copy_to_paper:
        parser.error("Nothing to do; pass --all and/or --copy-to-paper.")
    if args.all:
        make_all()
    if args.copy_to_paper:
        copy_to_paper()


if __name__ == "__main__":
    main()
