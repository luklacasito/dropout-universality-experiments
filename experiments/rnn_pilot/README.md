# Small Bridges-2 RNN dropout pilot: six layers

Question: does concentrating dropout at the bottom of a stacked LSTM improve
generalization or retention relative to uniform and late dropout?

This is a depth-scheduling experiment, not a schedule over sequence positions or
training epochs. It does not measure critical exponents. Only PyTorch and Python
are required; there are no downloads, W&B connections, or editable-package imports.

## Frozen pilot design

- Six unidirectional LSTM layers, width 64, 186,120 parameters. Six dropout
  sites, one on the input to each layer: the raw nine-channel input and the five
  interlayer connections. No dropout on cell states, recurrent edges, or readout.
  The previous five-layer package used only interlayer sites; this is a new cohort.
- Six arms (rates in order from input to final layer):

  | Schedule | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 |
  |---|---:|---:|---:|---:|---:|---:|
  | none | 0 | 0 | 0 | 0 | 0 | 0 |
  | uniform | .10 | .10 | .10 | .10 | .10 | .10 |
  | early_3_3 | .20 | .20 | .20 | 0 | 0 | 0 |
  | early_2_4 | .30 | .30 | 0 | 0 | 0 | 0 |
  | linear_decreasing | .20 | .16 | .12 | .08 | .04 | 0 |
  | late_3_3 | 0 | 0 | 0 | .20 | .20 | .20 |

- Inverted masks sampled independently per sequence/channel/layer/minibatch,
  then held fixed across time. Dropout is disabled for all evaluation.
- Positive arms match sum of raw probabilities (0.6), mean 0.1. The 2/4 profile
  uses a higher active rate to preserve that budget. `--mean-p` rescales all
  profiles and must be below 1/3. Because the first site has nine channels and
  later sites have 64, this does not match expected numbers of dropped units.
  The paper's effective RNN field has not been derived; do not call this an
  exact field-budget match.
- Synthetic delayed 8-class recall: a noisy one-hot cue occupies the first four
  positions, marked by an extra input channel. Remaining positions contain
  label-independent Gaussian distractors. Predict only at the final position.
- Train length 32; held-out tests at 32, 64, and 128. Longer test examples extend
  the same prefixes, so their labels/cues/distractors before the extension match.
- Fixed independent train/validation/test splits: 1,024 / 1,024 / 2,048 examples.
  Three seeds (0,1,2) vary initialization, training order, and masks, not the splits.
- 40 epochs, batch 64, Adam LR 0.001, no weight decay, gradient clipping at 1,
  FP32. Forget-gate bias sums to +1. Same initialization and minibatch order
  within each paired seed; each dropout site has a separate RNG stream.
- Best validation cross-entropy selects a checkpoint, then test is evaluated.
  Also evaluate with cue values removed; accuracy should approach 12.5% chance.

Primary comparisons: paired test cross-entropy of each early profile (3/3, 2/4,
linear) minus uniform, at length 32. Early 3/3 versus late 3/3 tests ordering;
2/4 versus 3/3 and linear versus 3/3 compare concentration and smoothness.
No-dropout establishes whether regularization helps. Longer-delay accuracy is
secondary. Three seeds are a directional pilot, not a significance test across
multiple profile comparisons.
Chance-level performance means the model did not learn; ceiling accuracy means
accuracy cannot resolve the comparison. A win does not establish universality.

## Local verification

From the repository or extracted source root:

```bash
python -m unittest discover -s experiments/rnn_pilot -v
python experiments/rnn_pilot/run.py --out /tmp/rnn-six-layer-smoke \
  --hidden 8 --length 8 --eval-lengths 8 12 --epochs 1 \
  --train-size 16 --valid-size 16 --test-size 16 --batch-size 8
python experiments/rnn_pilot/analyze.py --out /tmp/rnn-six-layer-smoke --seeds 0
```

Use a fresh output path for a changed configuration or source. CPU smoke tests
exercise the pipeline, not the experiment's statistical hypothesis.

## Transfer and run on Bridges-2

The standalone archive contains `experiments/rnn_pilot/` and can be extracted
without the rest of the repository. Example transfer from your Mac:

```bash
scp /Users/mac/Documents/icml/rnn-pilot-bridges2.tar.gz \
  lucasf@data.bridges2.psc.edu:/ocean/projects/phy260030p/lucasf/
```

After logging in to Bridges-2, reuse the existing PyTorch environment:

```bash
export PROJECT_DIR=/ocean/projects/phy260030p/$USER/rnn-pilot-6layer-source-20260914
export VENV_DIR=/ocean/projects/phy260030p/$USER/venvs/dropout-py311-cu121
export RUN_DIR=/ocean/projects/phy260030p/$USER/rnn-pilot-6layer-20260914
mkdir -p "$PROJECT_DIR" "$RUN_DIR/logs"
tar -xzf /ocean/projects/phy260030p/$USER/rnn-pilot-bridges2.tar.gz -C "$PROJECT_DIR"
cd "$PROJECT_DIR"
sha256sum -c experiments/rnn_pilot/SHA256SUMS
source "$VENV_DIR/bin/activate"
python -m unittest discover -s experiments/rnn_pilot -v
```

First run a two-epoch, full-size no-dropout GPU canary in a **separate** output
directory; use it to check CUDA compatibility and estimate epoch time:

```bash
mkdir -p "${RUN_DIR}-canary/logs"
sbatch -A phy260030p --array=0 --time=00:10:00 \
  --output="${RUN_DIR}-canary/logs/%A_%a.out" \
  --export=ALL,RUN_DIR="${RUN_DIR}-canary",CANARY=1 \
  experiments/rnn_pilot/slurm/run_bridges2.sbatch
```

When the canary succeeds, submit all 18 trials, at most one GPU at a time:

```bash
sbatch -A phy260030p --array=0-2%1 \
  --output="$RUN_DIR/logs/%A_%a.out" \
  --export=ALL,CANARY=0 \
  experiments/rnn_pilot/slurm/run_bridges2.sbatch
```

Each array element runs six schedules for one seed on one H100 with a one-hour
wall limit: at most three allocated GPU-hours across the array, not a measured
runtime estimate. Use the canary's epoch timings to check that 240 training
epochs plus evaluation fit per element; increase the limit if needed. `%3`
runs all three seeds concurrently. `--gpus=v100-32:1` can override the GPU type
for a new run directory. The job requests `GPU-shared` using the
[PSC batch-job convention](https://www.psc.edu/resources/bridges-2/user-guide/).

No jobs are submitted by preparing or extracting this package.

## Results and recovery

After all jobs finish:

```bash
python experiments/rnn_pilot/analyze.py --out "$RUN_DIR"
```

The analyzer requires complete matched seeds and rejects mixed settings. It
writes `summary.md`, `results.csv`, and per-seed differences in `paired.csv`.
Every trial saves configuration/source hash/environment, epoch history,
validation-selected `best.pt`, clean training metrics, tests, and elapsed time.
The report gives means and sample SDs across seeds, not significance claims.

Resubmission skips verified completed trials. Interrupted trials restart from
epoch one. A hard-killed process can leave `running.lock`: confirm the job is no
longer active before removing that one stale lock. Never remove a live lock.
Changing source, configuration, or recorded environment requires a new output
directory. The checkpoint is for inspection/inference, not mid-epoch resume.

Relevant foundations: [the paper's scheduling argument](https://arxiv.org/abs/2605.21648),
[Gal and Ghahramani on temporal mask reuse](https://arxiv.org/abs/1512.05287), and
[Chen et al. on RNN signal propagation](https://proceedings.mlr.press/v80/chen18i.html).
