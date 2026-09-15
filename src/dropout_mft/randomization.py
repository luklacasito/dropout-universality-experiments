"""Deterministic named random streams shared by modern experiment protocols."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _derived_seed(stream: str, payload: object) -> int:
    digest = hashlib.sha256(f"{stream}\0{canonical_json(payload)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _profile_pair_family(profile_id: str) -> str:
    for suffix in ("_early", "_late"):
        if profile_id.endswith(suffix):
            return profile_id[: -len(suffix)]
    return profile_id


def seed_streams(spec) -> dict:
    """Return independent seeds, with dropout CRNs shared by reversal pairs."""

    base = {"base_seed": spec.seed}
    pair_payload = asdict(spec)
    pair_payload["profile_id"] = _profile_pair_family(spec.profile_id)
    pair_key = hashlib.sha256(canonical_json(pair_payload).encode()).hexdigest()[:20]
    return {
        "scheme": "sha256_named_streams_v1",
        "base_seed": spec.seed,
        "initialization_seed": _derived_seed("initialization", base),
        "minibatch_seed": _derived_seed("minibatch", base),
        "dropout_seed": _derived_seed(
            "dropout_common_random_numbers", {"pair_key": pair_key}
        ),
        "dropout_crn_group": pair_key,
    }
