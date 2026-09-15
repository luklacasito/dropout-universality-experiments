#!/usr/bin/env python
"""Refresh the paper's auditable W&B export; figure generation itself is offline.

Only recorded confirmation runs are read. Epoch coverage is checked before the
snapshot is replaced. Requires access to the existing W&B project.
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import wandb
from wandb_graphql.language import parser

from dropout_mft.paths import project_root

PROJECT = "dropout-universality-benchmarks"
ENTITY = "llama2cmu"
METRICS = [
    "epoch",
    "train/loss",
    "train/accuracy",
    "validation/loss",
    "validation/accuracy",
]
RUNS_QUERY = parser.parse("""
query($cursor:String,$filters:JSONString) {
  project(entityName:"llama2cmu",name:"dropout-universality-benchmarks") {
    runs(first:100,after:$cursor,filters:$filters) {
      pageInfo { endCursor hasNextPage }
      edges { node { name displayName group state config summaryMetrics } }
    }
  }
}
""")
HISTORY_QUERY = parser.parse("""
query($name:String!,$specs:[JSONString!]!) {
  project(entityName:"llama2cmu",name:"dropout-universality-benchmarks") {
    run(name:$name) { sampledHistory(specs:$specs) }
  }
}
""")


def confirmation_runs(api):
    rows = []
    cursor = None
    while True:
        response = api.client.execute(
            RUNS_QUERY,
            variable_values={
                "cursor": cursor,
                "filters": json.dumps({"config.stage": "confirm"}),
            },
        )["project"]["runs"]
        for edge in response["edges"]:
            row = edge["node"]
            config = {
                key: value["value"]
                for key, value in json.loads(row["config"]).items()
                if not key.startswith(("_", "slurm_"))
            }
            summary = {
                key: value
                for key, value in json.loads(row["summaryMetrics"]).items()
                if not key.startswith("_")
            }
            if row["state"] != "finished" or summary.get("trial/status") != "complete":
                continue
            if summary.get("test/evaluated") is not True:
                continue
            rows.append(
                {
                    "id": row["name"],
                    "name": row["displayName"],
                    "group": row["group"],
                    "state": row["state"],
                    "config": config,
                    "summary": summary,
                    "url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{row['name']}",
                }
            )
        if not response["pageInfo"]["hasNextPage"]:
            return rows
        cursor = response["pageInfo"]["endCursor"]


def export(output):
    rows = confirmation_runs(wandb.Api(timeout=120))
    clients = threading.local()

    def attach_history(row):
        if not hasattr(clients, "api"):
            clients.api = wandb.Api(timeout=120)
        history = clients.api.client.execute(
            HISTORY_QUERY,
            variable_values={
                "name": row["id"],
                "specs": [json.dumps({"keys": ["_step"] + METRICS, "samples": 1000})],
            },
        )["project"]["run"]["sampledHistory"][0]
        epochs = {
            int(point["epoch"]): {key: point[key] for key in METRICS}
            for point in history
            if all(point.get(key) is not None for key in METRICS)
        }
        if set(epochs) != set(range(row["config"]["epochs"])):
            raise ValueError(f"Incomplete epoch history: {row['id']}")
        return {**row, "history": [epochs[e] for e in sorted(epochs)]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(attach_history, rows))
    payload = {
        "schema_version": 1,
        "project": f"{ENTITY}/{PROJECT}",
        "retrieved_on": date.today().isoformat(),
        "selection_note": "Best observed nonuniform confirmation test loss among complete paired seed sets; descriptive, not a prespecified independent test.",
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with gzip.GzipFile(temporary, "wb", mtime=0) as stream:
            stream.write(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(rows)} confirmation runs to {output}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results/benchmarks/confirmation.json.gz",
    )
    export(cli.parse_args().output)
