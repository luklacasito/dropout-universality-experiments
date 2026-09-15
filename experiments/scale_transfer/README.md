# Profile Geometry and Width Transfer

This study tests early/late profile reversals and the transfer of tuned settings
across network widths under standard and μP parameterization.

- [Repository setup](../../README.md#get-started)
- [Available results](../../paper/README.md)
- [Protocol implementation](../../src/dropout_mft/experiments/scale_transfer/protocol.py)

## Continuous Profiles and μP Width Transfer

The scale-transfer extension keeps every MLP at depth `L=6`. Here `128`
means the proxy **width**, not depth. Standard μP is used only for
widthwise transfer; no depth-transfer claim is made.

The canonical profile builder supports uniform, linear, quadratic, quartic,
and saturated-step profiles with exact early/late reversals. Every primary
profile preserves the discrete mean dropout probability and obeys
`p_max=0.20`. The one-step field mapping is retained as an analytic diagnostic,
not as a separate training phase: it is exactly the raw dropout probability in
the zero-bias case and differs negligibly for the selected near-critical ReLU
initialization. All new results are written under a user-selected run
directory; the published result files are never modified.

Run the deterministic CPU smoke test locally, then freeze a portable source
snapshot before transferring anything to Bridges-2:

```bash
python experiments/scale_transfer/run.py smoke
python experiments/scale_transfer/run.py snapshot-source \
  --output /path/to/dropout-source-snapshot \
  --slurm-script experiments/scale_transfer/slurm/coord_check_h100.sbatch \
  --slurm-script experiments/scale_transfer/slurm/run_phase_h100.sbatch
```

This snapshot contains a deterministic archive including dirty and untracked
files, the source-tree hash and Git status, and copied, hashed Slurm scripts.
Transfer the snapshot and restore or copy that exact checkout to Bridges-2.
After activating the finalized Bridges virtual environment, verify the checkout
and capture the cluster package lock before writing manifests:

```bash
python experiments/scale_transfer/run.py plan \
  --run-dir "$RUN_DIR" --phase all \
  --source-snapshot "$SOURCE_SNAPSHOT"
```

The plan copies the portable source artifacts under `RUN_DIR/provenance/` and
adds an installed-package lock from the active Bridges environment. A changed
checkout, snapshot, environment provenance, or Slurm script fails rather than
silently relabeling the experiment. Then run the non-vacuous coordinate check:

```bash
python experiments/scale_transfer/run.py coord-check \
  --run-dir "$RUN_DIR" \
  --widths 64 128 256 512 1024 2048 \
  --seeds 0 1 2 --steps 2 --learning-rate 3e-3 --strict \
  --output "$RUN_DIR/coord_check.npz"
```

Every real `run` command refuses to start unless `RUN_DIR/coord_check.npz`
records a passing strict check over exactly these widths and seeds with at
least two nontrivial updates. Both hidden- and output-coordinate slopes must
remain within `+/-0.1`, and the artifact must match the frozen source
provenance. For an artifact stored elsewhere, pass
`--coord-check-artifact PATH` to `run`. The CPU `smoke` command remains exempt.

This plans 30 `p=0.10` geometry-pilot trials, 100 MLP confirmation trials, 36
proxy-LR trials, 400 width-transfer trials after proxy selection, 144
all-width target-oracle diagnostics, and a 30-trial residual/ViT bridge: 740
trials in the base protocol. If the proxy-LR leave-one-seed-out stability gate
fails, a validation-only 36-run extension is triggered before width transfer;
it is not counted in the base 740. The bridge uses five profiles (uniform,
quadratic early/late, and step early/late), six paired seeds, 10 residual
blocks, the original unit residual-branch scale, and a global gradient-norm
clipping threshold of 1.0.
Run a phase locally or shard its deterministic manifest:

```bash
python experiments/scale_transfer/run.py run \
  --run-dir results/scale_transfer \
  --phase lr_proxy --device cuda --download

python experiments/scale_transfer/run.py run \
  --run-dir results/scale_transfer \
  --phase profile_confirmation \
  --shard-index 0 --num-shards 4 --device cuda
```

On first execution, the immutable pre-data plan is paired with the realized
dataset hash in `manifests/PHASE.locked.jsonl`; the same hash is recorded in
`manifests/data_splits.json` and in every trial result. Resuming or aggregating
fails if any of those bindings disagree.

Initialization, minibatch order, and dropout use separately recorded random
streams. Exact early/late reversal pairs share dropout common random numbers;
unrelated profile families do not.

Only after all 36 proxy trials complete, lock the SP and μP learning
rates from final validation cross-entropy. This writes the 400-trial
width-transfer manifest:

```bash
python experiments/scale_transfer/run.py select \
  --run-dir results/scale_transfer

python experiments/scale_transfer/run.py run \
  --run-dir results/scale_transfer \
  --phase width_transfer --device cuda
```

Run `target_oracle` only after zero-shot transfer, then aggregate and analyze.
This order is enforced: the oracle command requires the complete 400-trial
locked width-transfer cohort, including its final test evaluations.

```bash
python experiments/scale_transfer/run.py run \
  --run-dir results/scale_transfer \
  --phase target_oracle --device cuda

python experiments/scale_transfer/run.py aggregate \
  --run-dir results/scale_transfer

python experiments/scale_transfer/make_figures.py \
  --run-dir results/scale_transfer --only all
```

The analysis fails rather than fabricating panels when a required phase or
paired seed is missing. Test-set metrics are sealed for pilot, proxy-LR, and
oracle-grid trials; locked confirmation and transfer trials evaluate the test
split exactly once at the final epoch.

The primary endpoint is final benchmark cross-entropy. Secondary validation
summaries include terminal loss, the final-10%-epoch mean, and normalized loss
AUC. The prespecified compute endpoint uses each seed's paired, same-width and
same-parameterization uniform final validation accuracy as a fixed target. It
reports censored restricted-mean optimizer updates, examples, and estimated
dense FLOPs; non-reachers pay the full horizon. Time to the paired uniform
terminal validation loss remains a separately labeled diagnostic. Any reported
FLOP reduction is an early-stop counterfactual under the stated dense-training
estimate, not a measurement of hardware-level dropout sparsity.

The new 4,000/1,000 CIFAR-10 and 1,600/400 CIFAR-100 train/validation splits
are class-balanced and controlled by `split_seed`. The 5,000-example test set
is the locked historical benchmark test set from the original notebooks, not
a fresh holdout: it is the same unstratified `np.random.RandomState(0)` draw,
made after the legacy training draw from the same RNG, that informed the
published work. Its ordered index hash is recorded in every split manifest and
trial result so it cannot silently drift, and extension claims are interpreted
conditionally on that historical benchmark.

The residual/ViT bridge is an architectural boundary test rather than a direct
test of the plain-MLP mean-field recursion. Its residual branches alter global
correlation dynamics, so opposite early-versus-late signs in the MLP and bridge
cohorts are predeclared evidence of architecture-dependent effect modification,
not a failed or selectively discarded experiment. The bridge remains standard
parameterization only; no Transformer-μP claim is made. It retains the
published small-ViT recipe and does not reproduce the separate ResNet-56
protocol (27 scheduled blocks, SGD, `p=0.05`, and no gradient clipping).

### Bridges-2 H100 launch

Use one H100 per independent shard; the study does not use DDP or NCCL. Keep
the source checkout, data, and run directory on Ocean, and submit from an Ocean
log directory so scheduler logs do not modify the frozen source tree:

```bash
export PROJECT_DIR=/ocean/projects/PROJECT/USER/dropout-universality-experiments
export RUN_DIR=/ocean/projects/PROJECT/USER/dropout-scale-transfer
export SOURCE_SNAPSHOT=/ocean/projects/PROJECT/USER/dropout-source-snapshot
export VENV_DIR=/ocean/projects/PROJECT/USER/venvs/dropout-py311
export DATA_ROOT=/ocean/projects/PROJECT/USER/data
mkdir -p "$RUN_DIR/logs"
cd "$RUN_DIR/logs"
```

After transferring the portable source snapshot and exact checkout, create and
activate the Python 3.11 environment from `pyproject.toml`, run the cluster-side
`plan --source-snapshot` command above, stage CIFAR once, and submit the strict
H100 coordinate check:

```bash
sbatch -A PROJECT --export=ALL \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/coord_check_h100.sbatch"
```

Do not submit several `%4` arrays simultaneously if the allocation is capped
at four GPUs. Run these independent phases sequentially (or add Slurm
dependencies), each with at most four active array tasks:

```bash
# 30-run geometry pilot
sbatch -A PROJECT --array=0-3%4 \
  --export=ALL,PHASE=profile_pilot,NUM_SHARDS=4 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"

# 100-run MLP confirmation
sbatch -A PROJECT --array=0-9%4 \
  --export=ALL,PHASE=profile_confirmation,NUM_SHARDS=10 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"

# 36-run proxy LR grid
sbatch -A PROJECT --array=0-3%4 \
  --export=ALL,PHASE=lr_proxy,NUM_SHARDS=4 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"

# 30-run residual/ViT bridge
sbatch -A PROJECT --array=0-3%4 \
  --export=ALL,PHASE=vit_confirmation,NUM_SHARDS=4 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"
```

Run `select` only after the proxy grid completes. If its leave-one-seed-out
choice is unstable, the command writes the prespecified 36-run
`lr_proxy_extension` manifest (fresh seeds 3--5), blocks transfer, and tells
you to run that phase before repeating `select`.

Before releasing the 400-run transfer array, time one genuine width-2048
uniform `μP` trial. It is part of the locked cohort and is automatically
reused when the full array resumes:

```bash
sbatch -A PROJECT --array=0-0%1 \
  --export=ALL,PHASE=width_transfer,NUM_SHARDS=1,FILTER_WIDTH=2048,FILTER_PARAMETERIZATION=mup,FILTER_PROFILE=uniform,FILTER_SEED=100 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"

sbatch -A PROJECT --array=0-39%4 \
  --export=ALL,PHASE=width_transfer,NUM_SHARDS=40 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"
```

Only after all 400 zero-shot trials finish may the all-width oracle run start:

```bash
sbatch -A PROJECT --array=0-15%4 \
  --export=ALL,PHASE=target_oracle,NUM_SHARDS=16 \
  "$PROJECT_DIR/experiments/scale_transfer/slurm/run_phase_h100.sbatch"
```

The scripts request `GPU-shared`, one `h100-80`, eight CPU cores, BF16 when
supported, and shared Ocean results. The width-2048 timing—not the much smaller
width-128 proxy—should determine the final wall-time request and cost estimate.
