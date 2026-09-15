"""Characterization vectors captured before workflow consolidation."""

import hashlib
import json
from dataclasses import asdict

from dropout_mft.experiments.benchmark import data_regimes as d
from dropout_mft.experiments.benchmark import jannis_100 as j
from dropout_mft.experiments.benchmark import protocol as b
from dropout_mft.experiments.benchmark import sidecar as s
from dropout_mft.experiments.benchmark import vision as v
from dropout_mft.experiments.benchmark import zero_decay as vz
from dropout_mft.experiments.benchmark import zero_decay as z
from dropout_mft.experiments.scale_transfer.protocol import seed_streams


def cohort_fingerprints():
    out = {}

    def add(name, specs):
        payload = [
            {
                "spec": asdict(x),
                "trial_id": x.trial_id,
                "streams": seed_streams(x),
                "profile": b.benchmark_profile_layers(x),
            }
            for x in specs
        ]
        raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        out[name] = {"trials": len(specs), "sha256": hashlib.sha256(raw).hexdigest()}

    def selections(profiles):
        return {
            p: {
                "profile_id": p,
                "learning_rate": 1e-4,
                "mean_dropout": 0.0 if p == "none_tuned" else 0.1,
            }
            for p in profiles
        }

    for dataset in z.ZERO_DECAY_SUPPORTED_DATASETS:
        for model in ["mlp", "transformer"]:
            key = f"{dataset}/{model}"
            base = selections(b.BENCHMARK_PROFILE_IDS)
            rates = {p: c["learning_rate"] for p, c in base.items()}
            add("benchmark/lr/" + key, b.lr_search_specs(dataset, model))
            add("benchmark/budget/" + key, b.budget_search_specs(dataset, model, rates))
            add("benchmark/confirm/" + key, b.confirm_specs(dataset, model, base))
            sel = selections(z.ZERO_DECAY_PROFILE_IDS)
            rates = {p: c["learning_rate"] for p, c in sel.items()}
            add("zero/lr/" + key, z.zero_decay_lr_search_specs(dataset, model))
            add(
                "zero/budget/" + key,
                z.zero_decay_budget_search_specs(dataset, model, rates),
            )
            add(
                "zero/confirm/" + key,
                z.zero_decay_confirm_specs(dataset, model, sel, sel),
            )
    for model in ["mlp", "transformer"]:
        sel = selections(z.ZERO_DECAY_PROFILE_IDS)
        rates = {p: c["learning_rate"] for p, c in sel.items()}
        add("vision_zero/lr/" + model, vz.vision_zero_decay_lr_search_specs(model))
        add(
            "vision_zero/budget/" + model,
            vz.vision_zero_decay_budget_search_specs(model, rates),
        )
        add(
            "vision_zero/confirm/" + model,
            vz.vision_zero_decay_confirm_specs(model, sel, sel),
        )
        add("vision/lr/" + model, v.vision_lr_search_specs(model))
        add(
            "vision/confirm/" + model,
            v.vision_confirm_specs(model, selections(v.VISION_PROFILE_IDS)),
        )
        for regime in d.DATA_REGIME_IDS:
            add(
                "regime/lr/" + regime + "/" + model,
                d.data_regime_lr_search_specs(regime, model),
            )
            add(
                "regime/budget/" + regime + "/" + model,
                d.data_regime_budget_search_specs(regime, model, rates),
            )
            add(
                "regime/confirm/" + regime + "/" + model,
                d.data_regime_confirm_specs(regime, model, sel, sel),
            )
    for dataset in s.SIDECAR_DATASETS:
        add("sidecar/lr/" + dataset, s.sidecar_lr_search_specs(dataset))
        selection = {
            "selected_linear": {
                "profile_id": "linear_early",
                "learning_rate": 1e-4,
                "mean_dropout": 0.1,
            },
            "none_tuned": {
                "profile_id": "none_tuned",
                "learning_rate": 1e-4,
                "mean_dropout": 0.0,
            },
        }
        add("sidecar/confirm/" + dataset, s.sidecar_confirm_specs(dataset, selection))
    add("jannis/confirm", j.jannis_100epoch_specs())
    return out
