"""Keep experiment notebooks on the canonical schedule implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPOSITORY_ROOT / "notebooks"

SCHEDULE_NOTEBOOKS = {
    "mlp/mlp_dropout_budget_controls.ipynb",
    "mlp/mlp_dropout_scheduling_overfit.ipynb",
    "sweeps/gelu_h_sweep.ipynb",
    "sweeps/relu_h_sweep.ipynb",
    "sweeps/relu_width_sweep.ipynb",
    "transformer/vit_cifar100_dropout_scheduling.ipynb",
    "transformer/vit_component_ablation.ipynb",
}

XI_NOTEBOOKS = {
    "mlp/mlp_dropout_budget_controls.ipynb",
    "mlp/mlp_dropout_scheduling_overfit.ipynb",
    "sweeps/gelu_h_sweep.ipynb",
    "sweeps/relu_h_sweep.ipynb",
    "sweeps/relu_width_sweep.ipynb",
}


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        _cell_source(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_all_notebooks_are_valid_json_and_have_no_local_schedule_implementations():
    forbidden = ("def get_dropout_schedule", "def compute_effective_xi")
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        source = _code_source(path)
        for definition in forbidden:
            assert definition not in source, (
                f"{path.relative_to(NOTEBOOK_ROOT)} redefines {definition}"
            )


@pytest.mark.parametrize("relative_path", sorted(SCHEDULE_NOTEBOOKS))
def test_schedule_notebooks_import_and_use_canonical_builder(relative_path):
    source = _code_source(NOTEBOOK_ROOT / relative_path)
    assert "from dropout_mft.schedules import" in source
    assert "schedule_layers" in source
    assert "schedule_layers(" in source


@pytest.mark.parametrize("relative_path", sorted(XI_NOTEBOOKS))
def test_xi_notebooks_import_and_use_canonical_effective_xi(relative_path):
    source = _code_source(NOTEBOOK_ROOT / relative_path)
    assert "from dropout_mft.schedules import" in source
    assert "effective_xi" in source
    assert "effective_xi(" in source
