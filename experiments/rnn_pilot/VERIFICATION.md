# Six-layer pilot verification, 2026-09-14

Environment: Python 3.11.4, PyTorch 2.1.0, macOS CPU. No Bridges-2 jobs were submitted; CUDA execution and GPU runtime remain unverified.

Passed:

- Six protocol tests: matched six-site budgets and linear increments; actual placement at all six layer inputs; masks locked across time and independent across examples; paired data prefixes; equal initialization/evaluation predictions across arms; gradients reach the first layer.
- All six schedules completed training/checkpoint/test smoke runs.
- Analysis produced every schedule and all nine paired comparisons at both smoke-test lengths.
- Completed-trial reuse, changed-configuration rejection, and incomplete-pilot rejection checks passed.
- Bash syntax and Python execution checks passed.

Learnability check: packaged six-layer model, width 64, 1,024 training examples, 40 epochs, seed 0, no dropout. The source hash matches the packaged runner.

Best validation checkpoint: epoch 39.

| Measurement | Value |
|---|---:|
| Parameters | 186,120 |
| Best validation cross-entropy | 0.2604 |
| Clean training accuracy | 97.17% |
| Test accuracy, 32 steps | 94.34% |
| Test accuracy, 64 steps | 82.08% |
| Test accuracy, 128 steps | 39.11% |
| Test accuracy after blanking cue | 13.18% |
| Chance | 12.5% |

These measurements establish task learnability and memory sensitivity, not a dropout benefit. They replace the previous five-layer sanity check and are not part of the Bridges-2 comparison.
