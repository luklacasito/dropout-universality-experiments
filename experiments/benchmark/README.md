# Multi-Modality Benchmark Suite

Use one command surface for all studies:

```bash
python -m dropout_mft.experiments.benchmark STUDY COMMAND --help
```

`STUDY` is `benchmark`, `zero_decay`, `vision_zero_decay`, `data_regimes`,
`vision`, `sidecar`, or `jannis_100`. Each study keeps its definitions, planning,
and custom analysis together under `src/dropout_mft/experiments/benchmark/`.
Tiny ImageNet zero-decay is a preset of `zero_decay.py`. Execution and common
selection/aggregation use the shared runner; sidecar and Jannis retain their
specific analysis requirements. All commands and Slurm launchers below use the
unified entry point; the obsolete Python launcher wrappers have been removed.

The published results establish uniform-versus-front-loaded dropout at fixed
budget on CIFAR-10/100 with an MLP and a ViT. This suite runs the same
comparison on five further tasks spanning finance, vision, text, audio, and
tabular data, for both an MLP and a transformer, at width 256 and depth 6.

This is an **additive** cohort. It does not modify `scale_transfer.TrialSpec`,
any published manifest, or any file under `results/`. The saved inputs to the
camera-ready figures remain unchanged.

## Tasks

| Task | Modality | Classes | MLP view | Transformer view |
|---|---|---|---|---|
| `fi2010` | Limit order book | 3 | 100x40 flattened | 100 snapshot tokens, 40 features |
| `tiny_imagenet` | Vision, 64x64 | 200 | 12288-d flat | ViT, 8x8 patches |
| `amazon_reviews` | Text | 2 | 4096-d signed hashing | 128 token ids, 20k vocab |
| `speech_commands` | Audio | 35 | 64x64 log-mel flat | ViT on spectrogram, 8x8 patches |
| `openml_jannis` | Tabular | 4 | 54 features | 54 feature tokens (FT-style) |

Jannis is highly imbalanced: its minority class has 1,687 examples.  Its
locked class-balanced split therefore uses 3,840 train, 960 validation, and
1,920 test examples (1,680 examples per class in an exact 4:1:2 ratio).  The
larger generic 20k/5k/10k split is infeasible for this dataset and is rejected
by the split guard rather than silently sampling with replacement.

Train sizes are deliberately small relative to capacity. The effect under test
is a *regularization* effect; on a task the model cannot overfit, every profile
ties at zero and the comparison says nothing.

## Profiles

All arms spend the **same mean dropout budget** at depth 6:

| Profile | At `p̄ = 0.10` | Role |
|---|---|---|
| `uniform` | `(.1,.1,.1,.1,.1,.1)` | Control |
| `step_early` | `(.2,.2,.2,0,0,0)` | Paper's cap-matched front-loaded profile |
| `big_step` | `(.3,.3,0,0,0,0)` | Historical h-sweep profile, cap-exempt |
| `none` | all zero | Does dropout help on this task at all |

`big_step` intentionally breaks the 0.20 cap, exactly as it does in
the original CIFAR-10 budget-control experiment, so it is reported as an extension rather than a
cap-matched primary arm. At the top of the budget grid (`p̄ = 0.20`) it reaches
`p = 0.60` in its first two layers, which is a qualitatively different regime;
expect the response curve to turn over there and report it rather than trimming
the grid.

## Protocol

Three staged passes per (task, architecture) cell:

1. **`lr_search`** Five log-spaced learning rates at `p̄ = 0.10`, one seed,
   **per profile**.
2. **`budget_search`** `p̄ ∈ {0.05, 0.10, 0.15, 0.20}` at each profile's selected
   learning rate, three seeds. Doubles as the dropout-response curve.
3. **`confirm`** Five *fresh* seeds (100-104, disjoint from the tuning seeds
   0-2), at the selected `(learning_rate, p̄)`, plus the no-dropout control.
   Only this stage evaluates the test set.

710 trials total: 150 + 360 + 200.

Three properties are enforced in code, not by convention, because each is a way
this experiment could quietly produce a wrong answer:

- **Each profile is tuned independently** (`test_each_profile_is_tuned_independently`).
  The profiles differ in effective regularization strength, so a shared learning
  rate silently favours whichever arm the grid was centred on. Reusing the
  front-loaded arm's hyperparameters for uniform would confound allocation
  with hyperparameter choice.
- **Only `confirm` may touch the test set**
  (`test_only_the_confirmation_stage_may_touch_the_test_set`). `BenchmarkTrialSpec`
  raises if any other stage sets `evaluate_test`. Selection reads validation loss
  only.
- **Arms are paired at equal seed**
  (`test_profiles_share_initialization_and_minibatch_streams_at_equal_seed`).
  Initialization and minibatch order derive from the base seed alone, so at a
  given seed the profiles differ only in their dropout masks. This is what makes
  the paired per-seed delta a valid variance-reduced estimator.

`aggregate` reports **paired per-seed deltas with a 95% bootstrap CI**, seed
standard deviation alongside every mean, and a per-seed win rate. It does not
report two independent means with error bars, which would understate the
pairing and overstate the uncertainty.

## FI-2010 specifics

The financial task is the one most easily done wrong, so its protocol is
stricter than the others.

- **Anchored forward splits, never shuffled.** Windows overlap by 99 of 100
  snapshots, so a random split puts near-duplicates on both sides of the
  boundary and yields accuracy that means nothing. `_anchored_split` cuts
  train, validation, and test forward in time with an embargo of 100 rows
  between them, at least the label horizon, so a training label cannot be
  computed from validation-period prices.
- **Raw book only.** Only the first 40 of the 144 published columns are used
  (10 levels x {ask, bid} x {price, volume}), so the model sees the book rather
  than someone else's feature engineering.
- **Report the horizon.** `--horizon-index` selects among the published
  horizons k ∈ {1, 2, 3, 5, 10}; the default is k = 10. Short horizons are close
  to microstructure noise. If the effect appears at one horizon and vanishes at
  another, that is a result worth stating, not a grid to prune.
- **The reference point is DeepLOB**, not this MLP. A depth-6 MLP at width 256
  is well below the published state of the art, deliberately: this measures a
  regularization effect, it does not chase a leaderboard.
- **No P&L claim.** The metric stops at classification. Do not translate it into
  a backtested return.

## Running on Bridges-2

Caches are built once on CPU, then compute jobs read prepared `.npz` files.
Tiny-ImageNet and FI-2010 need their archives staged manually; the preparation
script prints the exact command and expected layout when they are missing.

```bash
export PROJECT_DIR=$PROJECT/dropout-universality-experiments
export RUN_DIR=$PROJECT/runs/benchmark-suite-20260811
export VENV_DIR=$PROJECT/venvs/dropout311
export DATA_ROOT=$PROJECT/data

# Local sanity check, no data and no GPU needed.
python -m dropout_mft.experiments.benchmark benchmark smoke
python -m dropout_mft.experiments.benchmark benchmark cost

# One CPU job to build all five caches.
sbatch -A YOUR_ALLOCATION --export=ALL,PROJECT_DIR=$PROJECT_DIR,\
DATA_ROOT=$DATA_ROOT,VENV_DIR=$VENV_DIR \
  experiments/benchmark/slurm/prepare_data.sbatch

# Check every cache loads before spending any GPU time on it.
python -m dropout_mft.experiments.benchmark benchmark verify-data --data-root $DATA_ROOT

# Then the whole three-stage chain, with Slurm dependencies between stages.
./experiments/benchmark/slurm/submit_suite.sh -A YOUR_ALLOCATION
```

The chain submits: `lr_search` array, then a CPU job that selects and plans
`budget_search`, then that array, then select and plan `confirm`, then the
confirmation array, then `aggregate`. Each link waits on `afterok` of the
previous one.

Before committing the full array, time one cell:

```bash
sbatch -A YOUR_ALLOCATION --array=0-0 \
  --export=ALL,STAGE=lr_search,NUM_SHARDS=150,FILTER_DATASET=openml_jannis,\
PROJECT_DIR=$PROJECT_DIR,RUN_DIR=$RUN_DIR,VENV_DIR=$VENV_DIR,DATA_ROOT=$DATA_ROOT \
  experiments/benchmark/slurm/run_h100.sbatch
```

Progress and results:

```bash
python -m dropout_mft.experiments.benchmark benchmark status --run-dir $RUN_DIR
python -m dropout_mft.experiments.benchmark benchmark aggregate --run-dir $RUN_DIR
```

Every trial is content-addressed by its spec hash and written atomically, so a
requeued, resubmitted, or preempted array task re-verifies finished trials and
recomputes nothing. Resubmitting the chain after a partial failure is safe.

## Depth-12 separation pilot

Before expanding the full suite to a second depth, a separate fixed-budget
pilot tests whether the longer propagation distance makes schedule placement
easier to detect.  It uses FI-2010 and OpenML Jannis, both model kinds, and
depth 12 at mean dropout 0.10.  Each profile receives an independent five-point
learning-rate search (60 trials total), followed by the three profiles and the
no-dropout control at five fresh paired seeds (80 trials), for 140 trials total.

The MLP remains at the same fixed near-critical ReLU initialization
(`sigma_w_sq=1.98`, `sigma_b_sq=0.02`) across profiles.  This is not exact
per-profile criticality: retuning initialization after applying dropout would
change the treatment.  The pre-LayerNorm transformer is explicitly recorded as
not criticality-calibrated; its MLP field diagnostics remain a proxy.

Use a dedicated run directory so these manifests cannot mix with depth 6:

```bash
export RUN_DIR=$PROJECT/runs/benchmark-depth12-pilot-20260811
export MAX_CONCURRENT=4
export CONTROL_GPU_TYPE=v100-16
export WANDB_PROJECT=dropout-universality-benchmarks
export WANDB_MODE=offline
# Optional when your default W&B account is not the intended destination:
# export WANDB_ENTITY=your-user-or-team
./experiments/benchmark/slurm/submit_depth12_pilot.sh -A YOUR_ALLOCATION
```

## Weights & Biases tracking

Every GPU trial is a separate deterministic W&B run, grouped as
`RUN_DIR/stage/dataset/model`.  The run records the complete immutable config,
exact layerwise dropout profile, criticality label, Slurm identifiers, epoch
loss/accuracy/LR curves, selection and test metrics, compute totals, and W&B's
automatic GPU/CPU system metrics.  The authoritative compressed result is also
logged as a `benchmark-trial` artifact.

Bridges-2 defaults to `WANDB_MODE=offline`: training writes W&B data below
`$RUN_DIR/wandb` and never depends on compute-node internet access.  After the
array has finished, authenticate once on a login node and upload all completed
runs:

```bash
source "$VENV_DIR/bin/activate"
wandb login
export WANDB_PROJECT=dropout-universality-benchmarks
RUN_DIR="$RUN_DIR" VENV_DIR="$VENV_DIR" \
  ./experiments/benchmark/slurm/sync_wandb.sh
```

Each successfully finished W&B run gets a marker under
`$RUN_DIR/wandb/tracked`.  Resubmitted Slurm shards therefore skip duplicate
logging.  A result created before tracking was enabled has no marker and is
backfilled once, without retraining.

## Dependencies beyond the base environment

- `torchaudio` for Speech Commands
- `datasets` for Amazon reviews
- `Pillow` for Tiny-ImageNet
- `scikit-learn` for the OpenML tabular task (already required)

## Known limitations

- The text vocabulary and the hashing feature map are fit on the unlabeled pool
  rather than on the training split alone. This is a label-free transductive
  feature extractor, the same status as a pretrained tokenizer, and the MLP view
  is additionally z-scored using train rows only. It is not label leakage, but
  it is worth stating rather than leaving implicit.
- The transformer arms carry MLP-derived field diagnostics as a *proxy*, flagged
  as `mlp_inspired_profile_proxy_not_a_transformer_recursion` in every result
  file. The mean-field recursion is derived for MLPs; the profile is transferred
  to transformers as an empirical claim, matching how the paper already treats
  its ViT results.
- Effect sizes on FI-2010 are expected to be small relative to seed variance.
  Report the seed standard deviation next to any claimed gain.
