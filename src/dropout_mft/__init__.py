"""Code used across the dropout mean-field experiments."""

from .numerics import find_positive_root, solve_bracketed, solve_with_expanding_upper
from .paths import project_root
from .results import load_json, load_npz_result, save_npz_result
from .schedules import (
    DISPLAY_NAMES,
    FIELD_EXPONENTS,
    effective_xi,
    field_damage,
    power_profile_layers,
    schedule_layers,
)
from .style import COLORS, SCHEDULE_COLORS, apply_paper_style, schedule_color

__all__ = [
    "COLORS",
    "DISPLAY_NAMES",
    "FIELD_EXPONENTS",
    "SCHEDULE_COLORS",
    "apply_paper_style",
    "effective_xi",
    "field_damage",
    "find_positive_root",
    "load_json",
    "load_npz_result",
    "project_root",
    "power_profile_layers",
    "save_npz_result",
    "schedule_color",
    "schedule_layers",
    "solve_bracketed",
    "solve_with_expanding_upper",
]
