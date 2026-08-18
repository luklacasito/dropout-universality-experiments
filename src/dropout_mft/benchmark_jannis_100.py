"""Pre-registered 100-epoch, ten-seed Jannis Transformer confirmation.

The original depth-12 cohort trains for 50 epochs.  Extending a cosine schedule
to 100 epochs changes every learning-rate trajectory, so this cohort reruns all
ten seeds under the longer horizon instead of pooling incompatible protocols.
The six arms use the validation-selected learning rate from the preceding
screens and differ only in their fixed layerwise dropout profile.
"""

from __future__ import annotations

from .benchmark_suite import BenchmarkTrialSpec, _spec_defaults


JANNIS_100_COHORT_ID = "jannis-transformer-depth12-100epoch-10seed-v1"
JANNIS_100_DEPTH = 12
JANNIS_100_EPOCHS = 100
JANNIS_100_SEEDS = tuple(range(100, 110))
JANNIS_100_PROFILES = (
    "uniform",
    "step_early",
    "big_step",
    "linear_early",
    "linear_late",
    "none_tuned",
)

# All six previous validation-only screens selected the same initial LR.
JANNIS_100_SELECTED_LR = {profile: 1e-4 for profile in JANNIS_100_PROFILES}


def jannis_100epoch_specs() -> list[BenchmarkTrialSpec]:
    """Return 60 paired confirmation trials with retained best checkpoints."""

    defaults = _spec_defaults("openml_jannis", "transformer")
    defaults["epochs"] = JANNIS_100_EPOCHS
    specs: list[BenchmarkTrialSpec] = []
    for profile_id in JANNIS_100_PROFILES:
        for seed in JANNIS_100_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=0.0 if profile_id == "none_tuned" else 0.10,
                    max_dropout=0.30 if profile_id == "big_step" else 0.20,
                    learning_rate=JANNIS_100_SELECTED_LR[profile_id],
                    seed=seed,
                    depth=JANNIS_100_DEPTH,
                    evaluate_test=True,
                    cohort_id=JANNIS_100_COHORT_ID,
                    **defaults,
                )
            )
    return specs


def jannis_100epoch_trial_count() -> int:
    return len(jannis_100epoch_specs())
