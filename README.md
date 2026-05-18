# Dropout Mean-Field Theory Experiments

This is the accompanying code for the ICML 2026 paper:

**Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos**  
Lucas Fernandez Sarmiento. Accepted at ICML 2026.

The paper studies how dropout changes mean-field signal propagation in wide neural networks. The main point is that dropout does not simply destroy the edge-of-chaos fixed-point picture; it displaces it, giving a small field in the correlation dynamics. That field leads to scaling laws, different smooth/kinked activation classes, and a simple prescription for depth-dependent dropout schedules.

This repository keeps the experimental side of that story in one place. It contains the saved runs used for the paper figures, the notebooks used to produce or check those runs, and a small shared plotting/utility package so the figures can be remade without rerunning the training jobs.

## Layout

```text
src/dropout_mft/     Schedules, result loading, and plotting style
scripts/             Scripts for remaking the figures
results/             Saved runs used in the paper
figures/paper/       Figures used in the paper
notebooks/           Notebooks for reruns and small changes
```

The exploratory ResNet trial is intentionally not included. It was useful while thinking through the project, but it is not part of the reported experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or, with conda/mamba:

```bash
mamba env create -f environment.yml
conda activate dropout-mft-experiments
```

The pickle files were saved with NumPy 2.x. There is a small compatibility loader so they can also be read with older NumPy 1.x installs.

## Make Figures

```bash
python scripts/make_figures.py --all
```

To copy the figures into the local LaTeX paper folder as well:

```bash
python scripts/make_figures.py --all --copy-to-paper
```

The default target is `/Users/mac/Documents/icml/icml26/figures`.

## Reproducibility Notes

The plotting script reads the saved runs, so remaking the paper figures should not require GPU time. The notebooks are there for rerunning a sweep, changing a schedule, or checking the details behind a saved result.

Figures are exported as vector PDFs for the paper, with PNG copies kept for quick browsing. The plotting palette follows the green/gold/teal/brown palette used in the paper figures; schedules also differ by markers or line styles, so the plots do not rely on hue alone.

The training notebooks use deterministic seeds of the form `42 + sim` for repeated runs. When a fixed data subset is used, the subset seed is `0`. The saved result files are the source for the camera-ready figures.

What is included:

- CIFAR-10 MLP scheduling runs in the overfitting regime.
- Matched-budget controls that separate schedule shape from total dropout budget.
- ReLU sweeps over the mean dropout field and hidden width.
- A GELU sweep checking the same scheduling logic for a smooth activation.
- CIFAR-100 Vision Transformer scheduling runs and component ablations.
- Mean-field theory figures for critical exponents, collapse diagnostics, and Hermite structure.
