# Results and manuscript

[Main manuscript PDF](dropout-universality-v2.pdf) · [Full results CSV](benchmark_results.csv) · [Appendix source](experimental_appendix.tex)

## Reproduce

```bash
python experiments/paper/make_appendix.py
# Also copy the generated sections and figures into a manuscript checkout:
python experiments/paper/make_appendix.py --paper-dir /path/to/icml26
```

Figures and tables rebuild offline from the versioned run export. The winning profile minimizes observed confirmation test loss among complete nonuniform arms; it is a descriptive result, not an independently tested selection. All other profiles remain in the appendix. Uncertainties are SEM.

The full manuscript is maintained in the separate `icml26` checkout. This directory contains its compiled v2 PDF, [main-body discussion](benchmark_discussion.tex), and generated LaTeX tables and appendix; rebuilding the full PDF requires that manuscript checkout and LaTeX. The discussion is edited by hand; the builder copies it without rewriting it.

## Benchmark results

| Dataset / setting | Model | Best nonuniform | Uniform test CE | Best test CE | Reduction | Seeds |
|---|---|---|---:|---:|---:|---:|
| [Amazon Reviews / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-01-amazon_reviews-mlp.pdf) | MLP | Step early | 0.4280 | 0.4061 | +5.12% | 5 |
| [Amazon Reviews / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-02-amazon_reviews-transformer.pdf) | Transformer | Linear increasing | 0.2931 | 0.2832 | +3.40% | 5 |
| [Speech Commands / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-03-speech_commands-mlp.pdf) | MLP | Step early | 1.0731 | 0.9367 | +12.71% | 5 |
| [Speech Commands / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-04-speech_commands-transformer.pdf) | Transformer | Big step | 0.7432 | 0.7067 | +4.91% | 5 |
| [Amazon Reviews / 2,000 train](../figures/paper/experiments/benchmarks/benchmark-05-amazon_reviews-mlp.pdf) | MLP | Big step | 0.6717 | 0.6431 | +4.26% | 5 |
| [Amazon Reviews / 2,000 train](../figures/paper/experiments/benchmarks/benchmark-06-amazon_reviews-transformer.pdf) | Transformer | Step early | 0.4938 | 0.5201 | -5.34% | 5 |
| [Amazon Reviews / 20,000 train](../figures/paper/experiments/benchmarks/benchmark-07-amazon_reviews-mlp.pdf) | MLP | Step early | 0.4225 | 0.4073 | +3.58% | 5 |
| [Amazon Reviews / 20,000 train](../figures/paper/experiments/benchmarks/benchmark-08-amazon_reviews-transformer.pdf) | Transformer | Linear increasing | 0.2836 | 0.2864 | -0.97% | 5 |
| [Amazon Reviews / 5,000 train](../figures/paper/experiments/benchmarks/benchmark-09-amazon_reviews-mlp.pdf) | MLP | Big step | 0.5816 | 0.5378 | +7.54% | 5 |
| [Amazon Reviews / 5,000 train](../figures/paper/experiments/benchmarks/benchmark-10-amazon_reviews-transformer.pdf) | Transformer | Linear increasing | 0.4047 | 0.4041 | +0.15% | 5 |
| [Tiny ImageNet / 80,000 train](../figures/paper/experiments/benchmarks/benchmark-11-tiny_imagenet-mlp.pdf) | MLP | Step early | 4.5755 | 4.5179 | +1.26% | 5 |
| [Tiny ImageNet / 80,000 train](../figures/paper/experiments/benchmarks/benchmark-12-tiny_imagenet-transformer.pdf) | Transformer | Linear decreasing | 3.0347 | 3.0040 | +1.01% | 5 |
| [FI-2010 / Linear follow-up](../figures/paper/experiments/benchmarks/benchmark-13-fi2010-transformer.pdf) | Transformer | Linear decreasing | Unavailable | 0.9299 | — | 5 |
| [Jannis / Linear follow-up](../figures/paper/experiments/benchmarks/benchmark-14-openml_jannis-transformer.pdf) | Transformer | Linear decreasing | Unavailable | 0.9923 | — | 5 |
| [Jannis / 100 epochs](../figures/paper/experiments/benchmarks/benchmark-15-openml_jannis-transformer.pdf) | Transformer | Linear decreasing | 0.9878 | 0.9883 | -0.05% | 10 |
| [Tiny ImageNet / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-16-tiny_imagenet-mlp.pdf) | MLP | Step early | 4.8682 | 4.7737 | +1.94% | 5 |
| [Tiny ImageNet / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-17-tiny_imagenet-transformer.pdf) | Transformer | Step early | 3.9761 | 3.8897 | +2.17% | 5 |
| [FI-2010 / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-18-fi2010-mlp.pdf) | MLP | Big step | 1.0934 | 1.0544 | +3.57% | 5 |
| [FI-2010 / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-19-fi2010-transformer.pdf) | Transformer | Linear increasing | 0.9416 | 0.9093 | +3.43% | 5 |
| [Jannis / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-20-openml_jannis-mlp.pdf) | MLP | Big step | 1.1235 | 1.0870 | +3.25% | 5 |
| [Jannis / Zero weight decay](../figures/paper/experiments/benchmarks/benchmark-21-openml_jannis-transformer.pdf) | Transformer | Step early | 0.9933 | 0.9883 | +0.50% | 5 |

Amazon 20k MLP has two missing linear-profile runs. Those arms are shown with n=4 in the appendix and cannot win the main comparison. Linear follow-up uniform baselines have not been recovered; they are not substituted with another cohort's baselines.

## Original paper

Original final-epoch comparisons are recomputed separately in [original_results.json](original_results.json) and [the LaTeX table](original_results_table.tex). Rebuild the original curves with `python experiments/paper/make_figures.py --all`.

## Profile and width-transfer studies

[Geometry-pilot figure](../figures/paper/experiments/benchmarks/profile-geometry-pilot.pdf)

The 30-run validation-only geometry pilot is preserved in `results/profiles/geometry_pilot.npz`. Full width-transfer outcomes are not present in the recovered result set, so no transfer success is claimed. The protocol and commands remain under `experiments/scale_transfer/`.

## Provenance

The export contains each run's ID, source URL, scientific configuration, split hash, endpoint metrics, and every recorded epoch. See [the data README](../results/benchmarks/README.md).
