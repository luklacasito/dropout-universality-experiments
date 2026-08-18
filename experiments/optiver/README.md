# Optiver Experiments

Two isolated studies live here:

- `run_canary.py` compares the shared MLP and Transformer implementations on a
  group-disjoint Optiver cache.
- `run_winning_mlp.py` reproduces the public winning MLP and a
  parameter-matched depth-12 variant before comparing dropout schedules.

Data preparation is deliberately separate from training:

```bash
python experiments/optiver/prepare_data.py --help
python experiments/optiver/prepare_winning_features.py --help
python experiments/optiver/rebuild_winning_features.py --help
```

Inspect the local smoke commands before submitting the full immutable cohort:

```bash
python experiments/optiver/run_canary.py --help
python experiments/optiver/run_winning_mlp.py --help
```

Bridges-2 launchers are under `slurm/`. Every trial and manifest remains
content-addressed, so moving these drivers does not alter model, split, seed,
or optimization behavior.
