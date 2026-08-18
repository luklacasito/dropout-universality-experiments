"""Deterministic CPU smoke tests for models, optimizers, and training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dropout_mft.models import (
    MLPConfig,
    TinyViT,
    build_mlp,
    make_optimizer,
    parameter_group_lr_ratios,
)
from dropout_mft.training import (
    TrainingConfig,
    make_multiplicative_cosine_scheduler,
    multiplicative_cosine_factor,
    seed_everything,
    synthetic_bundle,
    train_model,
)


def _small_sp_run():
    seed_everything(314)
    model = build_mlp(
        MLPConfig(
            input_dim=8,
            width=12,
            output_dim=3,
            depth=2,
            zero_readout=False,
        ),
        [0.0, 0.0],
        parameterization="sp",
    )
    optimizer = make_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
    bundle = synthetic_bundle(
        input_dim=8,
        classes=3,
        train_size=24,
        validation_size=12,
        test_size=12,
        seed=71,
    )
    result = train_model(
        model,
        optimizer,
        bundle,
        TrainingConfig(
            epochs=2,
            batch_size=6,
            learning_rate=3e-3,
            lr_floor_ratio=0.1,
            weight_decay=0.0,
            seed=19,
            device="cpu",
        ),
    )
    return model, result


def test_sp_cpu_training_is_deterministic_and_test_is_final_only():
    first_model, first = _small_sp_run()
    second_model, second = _small_sp_run()

    for first_parameter, second_parameter in zip(
        first_model.parameters(), second_model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
    for name, values in first["history"].items():
        np.testing.assert_array_equal(values, second["history"][name])
    assert first["final_test_loss"] == pytest.approx(
        second["final_test_loss"], rel=0, abs=0
    )
    assert first["final_test_accuracy"] == pytest.approx(
        second["final_test_accuracy"], rel=0, abs=0
    )
    assert "test_loss" not in first["history"]
    assert "test_accuracy" not in first["history"]
    np.testing.assert_allclose(first["history"]["lr_multiplier"], [1.0, 0.1])
    np.testing.assert_allclose(first["history"]["global_learning_rate"], [3e-3, 3e-4])
    assert first["optimizer_steps"] == 8


def test_training_applies_configured_global_gradient_clipping(monkeypatch):
    seed_everything(2718)
    model = build_mlp(
        MLPConfig(input_dim=8, width=12, output_dim=3, depth=2),
        [0.0, 0.0],
        parameterization="sp",
    )
    optimizer = make_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
    bundle = synthetic_bundle(
        input_dim=8,
        classes=3,
        train_size=24,
        validation_size=12,
        test_size=12,
        seed=72,
    )
    original_clip = torch.nn.utils.clip_grad_norm_
    observed_max_norms = []

    def recording_clip(parameters, max_norm, *args, **kwargs):
        observed_max_norms.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    result = train_model(
        model,
        optimizer,
        bundle,
        TrainingConfig(
            epochs=1,
            batch_size=6,
            learning_rate=3e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            seed=20,
            device="cpu",
        ),
    )

    assert observed_max_norms == [1.0] * 4
    assert result["training_config"]["gradient_clip_norm"] == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_training_rejects_invalid_gradient_clip_norm(value):
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        TrainingConfig(gradient_clip_norm=value)


def _overfitting_run(*, restore_best_validation: bool):
    """A run small enough that validation loss turns up well before the end."""

    seed_everything(31415)
    model = build_mlp(
        MLPConfig(input_dim=8, width=64, output_dim=3, depth=3),
        [0.0, 0.0, 0.0],
        parameterization="sp",
    )
    optimizer = make_optimizer(model, learning_rate=3e-2, weight_decay=0.0)
    bundle = synthetic_bundle(
        input_dim=8,
        classes=3,
        train_size=24,
        validation_size=48,
        test_size=48,
        seed=73,
    )
    return train_model(
        model,
        optimizer,
        bundle,
        TrainingConfig(
            epochs=40,
            batch_size=6,
            learning_rate=3e-2,
            lr_floor_ratio=0.5,
            weight_decay=0.0,
            seed=21,
            device="cpu",
            restore_best_validation=restore_best_validation,
        ),
    )


def test_restoring_the_best_epoch_matches_the_selection_criterion():
    result = _overfitting_run(restore_best_validation=True)
    validation_loss = np.asarray(result["history"]["validation_loss"], dtype=float)

    # The epoch the test set is read at must equal the epoch selection scores,
    # otherwise the reported test metric belongs to a different model.
    assert result["test_epoch"] == int(np.argmin(validation_loss))
    assert result["test_protocol"] == "best_validation_epoch_single_evaluation_v1"
    assert result["final_epoch_test_epoch"] == 39
    assert result["final_epoch_test_protocol"] == (
        "fixed_final_epoch_single_evaluation_v1"
    )
    assert result["final_epoch_test_loss"] is not None
    assert result["final_epoch_test_accuracy"] is not None
    # Guard the fixture itself: with no early turn there is nothing to restore.
    assert result["test_epoch"] < 39


def test_final_epoch_protocol_remains_the_default():
    result = _overfitting_run(restore_best_validation=False)
    assert result["test_epoch"] == 39
    assert result["test_protocol"] == "final_epoch_single_evaluation_v1"


def test_sp_mlp_and_tiny_vit_forward_shapes_on_cpu():
    seed_everything(1)
    mlp = build_mlp(
        MLPConfig(input_dim=8, width=6, output_dim=4, depth=2),
        [0.0, 0.0],
        parameterization="sp",
    )
    assert mlp(torch.randn(5, 8)).shape == (5, 4)

    vit = TinyViT(
        [0.0, 0.0],
        dimension=8,
        heads=2,
        mlp_ratio=2,
        output_dim=7,
        patch_size=8,
    )
    assert vit(torch.randn(2, 3, 32, 32)).shape == (2, 7)


@pytest.mark.parametrize(
    "epoch,expected",
    [
        (-3, 1.0),
        (0, 1.0),
        (2, 0.55),
        (4, 0.1),
        (10, 0.1),
    ],
)
def test_multiplicative_cosine_factor(epoch, expected):
    assert multiplicative_cosine_factor(
        epoch, epochs=4, floor_ratio=0.1
    ) == pytest.approx(expected)


def test_multiplicative_scheduler_preserves_optimizer_group_lr_ratios():
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "lr": 0.1},
            {"params": [second], "lr": 0.02},
        ]
    )
    scheduler = make_multiplicative_cosine_scheduler(
        optimizer, epochs=4, floor_ratio=0.1
    )

    assert parameter_group_lr_ratios(optimizer) == pytest.approx((1.0, 0.2))
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        assert parameter_group_lr_ratios(optimizer) == pytest.approx((1.0, 0.2))
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.01, 0.002]
    )


def test_mup_build_forward_and_optimizer_when_dependency_is_available():
    mup = pytest.importorskip("mup")
    seed_everything(9)
    model = build_mlp(
        MLPConfig(input_dim=8, width=16, output_dim=3, depth=2),
        [0.0, 0.0],
        parameterization="mup",
        base_width=4,
        delta_width=8,
    )
    optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    assert model(torch.randn(4, 8)).shape == (4, 3)
    assert isinstance(model.readout, mup.MuReadout)
    assert all(hasattr(parameter, "infshape") for parameter in model.parameters())
    assert len(optimizer.param_groups) >= 1
    assert all(
        np.isfinite(group["lr"]) and group["lr"] > 0 for group in optimizer.param_groups
    )


def test_direct_mup_gelu_build_is_rejected_until_initializer_is_defined():
    with pytest.raises(ValueError, match="ReLU-specific"):
        build_mlp(
            MLPConfig(
                input_dim=8,
                width=16,
                output_dim=3,
                depth=2,
                activation="gelu",
            ),
            [0.0, 0.0],
            parameterization="mup",
            base_width=4,
            delta_width=8,
        )


def test_sp_and_mup_critical_variance_and_controlled_zero_readout():
    pytest.importorskip("mup")
    for parameterization in ("sp", "mup"):
        seed_everything(12)
        model = build_mlp(
            MLPConfig(
                input_dim=128,
                width=512,
                output_dim=3,
                depth=2,
                sigma_w_sq=1.98,
                sigma_b_sq=0.02,
                zero_readout=True,
            ),
            [0.0, 0.0],
            parameterization=parameterization,
            base_width=64,
            delta_width=128,
        )
        for layer in model.hidden:
            realized = float(layer.weight.var()) * layer.weight.shape[1]
            assert realized == pytest.approx(1.98, rel=0.06)
        assert torch.count_nonzero(model.readout.weight) == 0


def test_miniature_mup_proxy_to_target_width_transfer_smoke():
    pytest.importorskip("mup")
    bundle = synthetic_bundle(
        input_dim=8,
        classes=3,
        train_size=24,
        validation_size=12,
        test_size=12,
        seed=90,
    )
    final_losses = []
    for width in (8, 16):
        seed_everything(44)
        model = build_mlp(
            MLPConfig(
                input_dim=8,
                width=width,
                output_dim=3,
                depth=2,
                zero_readout=True,
            ),
            [0.1, 0.0],
            parameterization="mup",
            base_width=4,
            delta_width=8,
        )
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        result = train_model(
            model,
            optimizer,
            bundle,
            TrainingConfig(
                epochs=1,
                batch_size=6,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=44,
                device="cpu",
            ),
        )
        final_losses.append(result["final_test_loss"])
    assert np.all(np.isfinite(final_losses))
