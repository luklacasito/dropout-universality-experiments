# Standard Penn Treebank RNN dropout pilot

This replaces the synthetic delayed-recall experiment with **word-level Penn
Treebank language modeling**, using the original preprocessed PTB files from
[Zaremba's reference repository](https://github.com/wojzaremba/lstm/tree/master/data).
The motivating regularization benchmark is
[Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329).
This is a small six-layer architecture study, not a reproduction of published
state-of-the-art perplexities.

## Dataset and task

Use the full standard splits, without train-set subsampling:

| Split | Tokens including end-of-sentence markers |
|---|---:|
| Train | 929,589 |
| Validation | 73,760 |
| Test | 82,430 |

Vocabulary: 10,000 words, constructed from training text only. Newlines become
`<eos>`; unknown words map to the training `<unk>` token. The runner checks each
split's SHA-256 and token count. Data hashes are embedded in `run.py` and saved
in every result. Dataset downloads happen before submitting compute jobs.

Predict the next word at every time step. Training uses contiguous streams,
batch 64, truncated BPTT length 35, and carried hidden/cell states detached at
chunk boundaries. States reset between epochs and evaluation passes. Validation
uses batch 10, test uses batch 1. As in standard stream batching, training and
validation omit short terminal remainders and one initial target per stream;
every pass reports the exact number of scored tokens.

## Model and schedules

Six LSTM layers, hidden size 128, learned 128-dimensional word embeddings,
untied output decoder, FP32, Adam LR .001, gradient clipping at 1, total forget
bias +1. The plan uses 20 epochs. Each scheduled task runs one complete trial,
keeping individual GPU reservations small; the canary checks full-epoch timing.

Dropout acts on the **embedding representation** and five interlayer inputs:
six equal-width sites. It does not directly erase one-hot word IDs. Masks are
independent across channels, streams, and layers, locked within each BPTT chunk,
and resampled at the next chunk. Cell states, recurrent edges, and decoder have
no direct dropout. Evaluation disables dropout.

At selected average probability `p`, profiles are:

| Profile | Six probabilities |
|---|---|
| none | 0, 0, 0, 0, 0, 0 |
| uniform | p, p, p, p, p, p |
| early_3_3 | 2p, 2p, 2p, 0, 0, 0 |
| early_2_4 | 3p, 3p, 0, 0, 0, 0 |
| linear_decreasing | 2p, 1.6p, 1.2p, .8p, .4p, 0 |
| late_3_3 | 0, 0, 0, 2p, 2p, 2p |

Nonzero profiles match mean probability; no exact RNN effective-field budget
is claimed. This changes both dataset and input representation relative to the
synthetic pilot; it is not a controlled attribution of the previous failure.

## Validation-first protocol

1. Canary: one full epoch of uniform .1 dropout, to verify CUDA, data, checkpoint
   writes, validation, and timing. No test evaluation.
2. Calibration seed 0: no dropout plus uniform `.01, .025, .05, .1, .2`, all at
   the same optimization budget. Best validation cross-entropy selects the
   checkpoint for each trial and the best **positive** probability for comparison.
   No test token encoding or evaluation occurs in calibration.
3. Confirmation: all six profiles at that frozen positive budget, using fresh
   paired seeds 1, 2, 3. No-dropout remains a control. Checkpoints are selected by
   validation loss and then evaluated on test. This adds 18 runs.

Selection explicitly records whether the best positive rate beat no dropout.
If zero wins calibration, confirmation still measures all profiles and the
no-dropout control; the report must not claim that a useful regularization regime
has been established. Neither positive-only selection nor three seeds establishes
statistical significance. Model initialization, data order, optimizer, and
dropout streams are paired across profiles within each confirmation seed.

## Verify locally

```bash
python -m unittest discover -s experiments/rnn_ptb -v
python experiments/rnn_ptb/run.py --stage canary \
  --data-root /path/to/ptb-data --out /tmp/ptb-smoke \
  --hidden 8 --batch-size 2 --bptt 5 --epochs 1 --max-batches 2 --threads 1
```

Batch limits are only allowed in canary stage. Scientific calibration and
confirmation always use the full splits.

## Bridges-2

Use a new source/output directory; leave the completed synthetic experiment
intact. Transfer the source and the three `ptb.*.txt` files before submitting.
Activate the existing `dropout-py311-cu121` environment. Export `PROJECT_DIR`,
`DATA_ROOT`, `RUN_DIR`, and `VENV_DIR`; the launcher needs no package installation
or network connection. It defaults to one V100-32, two CPUs, 4 GB RAM, 20 minutes.

```bash
sbatch -A phy260030p --time=00:05:00 --export=ALL,STAGE=canary \
  experiments/rnn_ptb/slurm/run.sbatch
# After canary succeeds and timing is checked:
sbatch -A phy260030p --array=0-5%1 --export=ALL,STAGE=calibrate,EPOCHS=20 \
  experiments/rnn_ptb/slurm/run.sbatch
# Use the returned calibration job ID as CALIBRATION_ID:
sbatch -A phy260030p --array=0-17%1 --dependency=afterok:$CALIBRATION_ID \
  --kill-on-invalid-dep=yes --export=ALL,STAGE=confirm,EPOCHS=20 \
  experiments/rnn_ptb/slurm/run.sbatch
```

After all calibration tasks pass, the first confirmation task writes
`selection.json` from validation results; confirmation verifies the
same source, data, optimization settings, and recorded environment. Slurm CLI
wall-time overrides, if needed after timing, are recorded in submission metadata.
After all confirmation jobs finish:

```bash
python experiments/rnn_ptb/report.py --out "$RUN_DIR"
```

Outputs: validation-selected `best.pt`, resumable `latest.pt`, epoch histories,
configuration/data/source hashes, `selection.json`, per-trial results, final
`results.csv`, `paired.csv`, and `summary.md`. A completed trial is reused only
with an identical fingerprint; interrupted trials resume at epoch boundaries.
If a hard kill leaves `running.lock`, verify the old job has ended before removing
that lock. No checkpoint is loaded from an external source.
