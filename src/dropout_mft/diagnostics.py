"""Coordinate checks for validating the muP implementation."""

from __future__ import annotations

import platform
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .models import MLPConfig, build_mlp, make_optimizer


def mup_coordinate_check(
    *,
    widths: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
    depth: int = 6,
    input_dim: int = 32,
    output_dim: int = 10,
    batch_size: int = 32,
    steps: int = 2,
    learning_rate: float = 3e-3,
    seed: int = 0,
    seeds: Sequence[int] | None = None,
    device: str = "cpu",
) -> dict:
    """Measure hidden and output coordinates over width and real updates.

    A strict check should pass several ``seeds``.  The relatively large default
    learning rate and the explicit update-size diagnostics prevent a vacuous
    pass in which every model is effectively unchanged.  ``seed`` remains as a
    backwards-compatible shorthand for a one-seed diagnostic.
    """

    widths = tuple(int(width) for width in widths)
    if (
        len(widths) < 2
        or any(width <= 0 for width in widths)
        or any(right <= left for left, right in zip(widths, widths[1:]))
    ):
        raise ValueError(
            "At least two strictly increasing positive widths are required"
        )
    if steps < 1:
        raise ValueError("steps must be at least one")
    seeds = (int(seed),) if seeds is None else tuple(int(value) for value in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of unique integers")
    criterion = nn.CrossEntropyLoss()
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA coordinate check requested but CUDA is unavailable")
    seed_values: list[np.ndarray] = []
    for current_seed in seeds:
        torch.manual_seed(current_seed)
        generator = torch.Generator().manual_seed(current_seed)
        inputs = torch.randn(batch_size, input_dim, generator=generator)
        targets = torch.randint(output_dim, (batch_size,), generator=generator)
        records: dict[int, np.ndarray] = {}

        for width in widths:
            config = MLPConfig(
                input_dim=input_dim,
                width=width,
                output_dim=output_dim,
                depth=depth,
                activation="relu",
                zero_readout=True,
            )
            model = build_mlp(
                config,
                [0.1] * depth,
                parameterization="mup",
                base_width=widths[0],
                delta_width=widths[1],
            ).to(resolved_device)
            optimizer = make_optimizer(
                model, learning_rate=learning_rate, weight_decay=0.0
            )
            x = inputs.to(resolved_device)
            y = targets.to(resolved_device)
            per_step: list[list[float]] = []
            for step_index in range(steps + 1):
                model.train()
                features: list[torch.Tensor] = []
                hooks = [
                    layer.register_forward_hook(
                        lambda _module, _inputs, output, sink=features: sink.append(
                            output.detach()
                        )
                    )
                    for layer in model.hidden
                ]
                logits = model(x)
                for hook in hooks:
                    hook.remove()
                per_step.append(
                    [float(feature.abs().mean()) for feature in features]
                    + [float(logits.detach().abs().mean())]
                )
                if step_index < steps:
                    optimizer.zero_grad(set_to_none=True)
                    criterion(logits, y).backward()
                    optimizer.step()
            records[width] = np.asarray(per_step)
            del model, optimizer
        seed_values.append(np.stack([records[width] for width in widths], axis=0))

    # Shape: seed, width, update (including update zero), hidden/output coordinate.
    values = np.stack(seed_values, axis=0)
    log_widths = np.log(np.asarray(widths, dtype=float))
    slopes = np.empty((len(seeds), steps + 1, depth + 1))
    for seed_index in range(len(seeds)):
        for step_index in range(steps + 1):
            for layer in range(depth + 1):
                slopes[seed_index, step_index, layer] = np.polyfit(
                    log_widths,
                    np.log(np.maximum(values[seed_index, :, step_index, layer], 1e-30)),
                    deg=1,
                )[0]
    hidden_slopes_after_update = slopes[:, 1:, :-1]
    output_slopes_after_update = slopes[:, 1:, -1]
    hidden_change = np.abs(values[:, :, -1, :-1] - values[:, :, 0, :-1]) / np.maximum(
        values[:, :, 0, :-1], 1e-30
    )
    max_hidden_change_by_seed = np.max(hidden_change, axis=(1, 2))
    min_output_after_update_by_seed = np.min(values[:, :, 1:, -1], axis=(1, 2))
    max_hidden_slope = float(np.max(np.abs(hidden_slopes_after_update)))
    max_output_slope = float(np.max(np.abs(output_slopes_after_update)))
    nontrivial_update = bool(
        np.min(max_hidden_change_by_seed) >= 1e-3
        and np.min(min_output_after_update_by_seed) >= 1e-3
    )
    return {
        "widths": np.asarray(widths),
        "seeds": np.asarray(seeds),
        "learning_rate": float(learning_rate),
        "device": str(resolved_device),
        "device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "coordinate_l1": values,
        "log_width_slopes": slopes,
        "max_abs_hidden_slope_after_update": max_hidden_slope,
        "max_abs_output_slope_after_update": max_output_slope,
        "max_relative_hidden_change_by_seed": max_hidden_change_by_seed,
        "min_output_l1_after_update_by_seed": min_output_after_update_by_seed,
        "passes_nontrivial_update": nontrivial_update,
        "passes_threshold_0p1": bool(
            max_hidden_slope <= 0.1 and max_output_slope <= 0.1
        ),
    }
