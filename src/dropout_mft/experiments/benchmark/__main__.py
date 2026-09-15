"""One command surface for related benchmark studies.

Example: python -m dropout_mft.experiments.benchmark zero_decay plan --help
"""

from __future__ import annotations

import argparse
import importlib
import sys

STUDIES = {
    "benchmark": "cli",
    "zero_decay": "zero_decay",
    "data_regimes": "data_regimes",
    "vision": "vision",
    "vision_zero_decay": "zero_decay",
    "sidecar": "sidecar",
    "jannis_100": "jannis_100",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.epilog = "Commands: plan, run, select, aggregate, status, cost, smoke, verify-data. Pass STUDY COMMAND --help for its options."
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    study = parser.parse_args(argv[:1]).study
    module = importlib.import_module(
        f"dropout_mft.experiments.benchmark.{STUDIES[study]}"
    )
    remaining = argv[1:]
    # The vision preset shares the zero-decay implementation, with its original
    # dataset/cohort defaults and narrower planning/cost arguments.
    if study == "vision_zero_decay" and (
        not remaining or remaining[0] in ("plan", "cost", "-h", "--help")
    ):
        module.vision_main(remaining)
        return
    # Running manifests is the same operation for every dataset/cohort. Studies
    # override only planning, selection, or analysis where their protocol differs.
    if remaining and remaining[0] in (
        "run",
        "select",
        "aggregate",
        "status",
        "smoke",
        "verify-data",
    ):
        handler = "command_" + remaining[0].replace("-", "_")
        if not hasattr(module, handler):
            module = importlib.import_module("dropout_mft.experiments.benchmark.cli")
    module.main(remaining)


if __name__ == "__main__":
    main()
