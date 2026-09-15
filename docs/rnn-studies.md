# RNN studies

These three small studies run independently of the `dropout_mft` package. They
need Python and PyTorch. Their original sources and launch records are preserved
in `experiments/`; this guide provides the repository-level entry points.

| Study | Task | Main files |
|---|---|---|
| [Synthetic pilot](../experiments/rnn_pilot/) | Delayed recall with six LSTM layers | `run.py`, `analyze.py` |
| [Penn Treebank](../experiments/rnn_ptb/) | Next-word prediction with validation-only dropout calibration | `run.py`, `report.py` |
| [Linear follow-up](../experiments/rnn_ptb_linear/) | Compare increasing and decreasing layer probabilities | `extend.py`, `compare.py` |

The synthetic pilot generates its data. PTB uses the three original preprocessed
`ptb.train.txt`, `ptb.valid.txt`, and `ptb.test.txt` files in a supplied directory;
the runner verifies hashes and token counts. See the PTB README for the data
source and complete protocol.

## Run the local checks

From the repository root:

```bash
make test-rnn
python experiments/rnn_pilot/run.py --help
python experiments/rnn_ptb/run.py --help
python experiments/rnn_ptb_linear/extend.py --help
```

Each test suite runs in a separate process because these standalone folders each
use local module names such as `run` and `report`. Their unit tests use small CPU
examples and do not need the full PTB data.

## Frozen sources and cluster runs

`run.py` and related analysis sources contribute to each run's source fingerprint.
The linear follow-up also checks the exact parent fingerprint. Formatting those
files would change run identity, so normal repository formatting excludes them.
The `SHA256SUMS` files verify each original source bundle from the repository root.

The experiment READMEs retain historical paths, account names, and job IDs as
launch records. For a new submission:

1. Transfer a copy of the required experiment folders. The linear follow-up needs
   both `rnn_ptb/` and `rnn_ptb_linear/`.
2. Set `PROJECT_DIR`, `RUN_DIR`, and `VENV_DIR` to your own cluster directories.
   PTB also needs `DATA_ROOT`; its linear follow-up needs `BASE_RUN_DIR` containing
   the parent selection and results.
3. Use your own allocation and the appropriate GPU type. Replace historical
   dependency IDs with the jobs returned by your new submissions.
4. Run a canary and inspect its timing before submitting the full array.

The linear follow-up is fixed to the original selected mean probability, training
settings, and source/environment identity. It reuses the parent controls and is
an exploratory comparison, not an independent confirmation. A new protocol
needs a new study identity.
