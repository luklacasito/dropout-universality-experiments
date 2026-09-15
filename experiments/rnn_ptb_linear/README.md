# Linear depth schedules on PTB

Follow-up requested after partial results from the original six-schedule sweep.
That sweep already contains linear decreasing. Add only linear increasing,
using the same selected mean probability (0.05), seeds 1–3, full PTB splits,
six layers, width 128, and 20-epoch training budget.

- Decreasing, already queued: 0.10, 0.08, 0.06, 0.04, 0.02, 0.
- Increasing, added here: 0, 0.02, 0.04, 0.06, 0.08, 0.10.
- Uniform and no-dropout controls are reused from the original sweep.

The adapter imports the frozen original runner and adds one probability profile.
It validates the complete original protocol and source hash before training, and
records its own source hash in every checkpoint and result. It uses a separate
output directory. Original source, live jobs, and outputs are not modified.
Increasing is exactly the reversed decreasing probability vector, matching both
the mean and the distribution of layer probabilities. This does not establish
an exact effective-field budget for RNNs.

Submit array `1-3%1`, with `--dependency=afterok:45975344` and
`--kill-on-invalid-dep=yes`. Each task requests one V100, 2 CPUs, 4 GB RAM,
20 minutes (60 GPU minutes maximum across the three tasks).
The last completed task automatically writes `summary.md` and
`linear_results.csv`, comparing both directions and controls across paired seeds.
This is an exploratory follow-up, not a new independent confirmation study.
