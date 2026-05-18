"""Read the saved runs used by the paper figures.

Some saved pickle files were produced with NumPy 2.x and refer to the
``numpy._core`` namespace. The compatibility unpickler maps those names back
to ``numpy.core`` so the files also open in NumPy 1.x environments.
"""

from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path
from typing import Any


class CompatUnpickler(pickle.Unpickler):
    """Accept NumPy 2.x pickle module names on NumPy 1.x."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return CompatUnpickler(handle).load()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def parse_tuple_key(key: str) -> tuple[Any, ...]:
    """Turn a stringified sweep key back into a tuple."""

    value = ast.literal_eval(key)
    if not isinstance(value, tuple):
        raise ValueError(f"Expected tuple-like key, got {key!r}")
    return value
