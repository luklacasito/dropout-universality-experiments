"""Deterministic data splits and training loops for transfer experiments."""

from __future__ import annotations

import contextlib
import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CifarDatasetName = Literal["cifar10", "cifar100"]
BenchmarkDatasetName = Literal[
    "fi2010",
    "tiny_imagenet",
    "amazon_reviews",
    "speech_commands",
    "openml_jannis",
]
# ``DatasetBundle`` is shared by the frozen CIFAR protocol and the additive
# multi-modality benchmark suite.  Widening the alias does not touch any
# published manifest hash, which is computed from ``TrialSpec`` fields only.
DatasetName = Literal[
    "cifar10",
    "cifar100",
    "fi2010",
    "tiny_imagenet",
    "amazon_reviews",
    "speech_commands",
    "openml_jannis",
]

LEGACY_TEST_SUBSET_SEED = 0
LEGACY_TEST_SUBSET_PROTOCOL = (
    "numpy_randomstate0_unstratified_choice_after_train_choice_v1"
)
CIFAR_SPLIT_PROTOCOL = (
    "stratified_train_validation_default_rng_plus_legacy_test_subset_v1"
)


@dataclass(frozen=True)
class DatasetBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    split_hash: str
    dataset: DatasetName
    split_protocol: str
    test_subset_hash: str
    test_subset_protocol: str
    test_subset_seed: int | None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 75
    batch_size: int = 75
    learning_rate: float = 1e-4
    lr_floor_ratio: float = 1e-3
    weight_decay: float = 1e-7
    gradient_clip_norm: float | None = None
    seed: int = 0
    stochastic_seed: int | None = None
    evaluate_test: bool = True
    # When true, the primary test endpoint is the minimum-validation-loss
    # checkpoint.  Benchmark confirmation additionally records the fixed final
    # epoch as a preregistered secondary endpoint; neither test metric is used
    # for model or hyperparameter selection.  Defaults to false to leave the
    # frozen CIFAR and scale-transfer protocols exactly as published.
    restore_best_validation: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.lr_floor_ratio <= 1:
            raise ValueError("lr_floor_ratio must be in [0, 1]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")


def training_config_from_trial(
    spec, randomization: dict, *, device="auto", restore_best_validation=False
) -> TrainingConfig:
    """Translate common trial fields without changing the study's test policy."""
    return TrainingConfig(
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        lr_floor_ratio=spec.lr_floor_ratio,
        weight_decay=spec.weight_decay,
        gradient_clip_norm=spec.gradient_clip_norm,
        seed=randomization["minibatch_seed"],
        stochastic_seed=randomization["dropout_seed"],
        evaluate_test=spec.evaluate_test,
        restore_best_validation=restore_best_validation,
        device=device,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if size <= 0 or size % len(classes):
        raise ValueError(f"Requested size {size} must be positive and class-balanced")
    per_class = size // len(classes)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} examples")
        selected.append(rng.choice(candidates, per_class, replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def _split_train_validation(
    labels: np.ndarray,
    *,
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pool = _balanced_indices(labels, train_size + validation_size, seed)
    pool_labels = labels[pool]
    validation_local = _balanced_indices(pool_labels, validation_size, seed + 1)
    validation_mask = np.zeros(len(pool), dtype=bool)
    validation_mask[validation_local] = True
    return pool[~validation_mask], pool[validation_mask]


def _indices_hash(indices: np.ndarray) -> str:
    """Hash an ordered index vector using a platform-independent representation."""

    values = np.asarray(indices, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def legacy_cifar_test_indices(
    *,
    train_population_size: int,
    train_subset_size: int,
    test_population_size: int,
    test_subset_size: int,
    seed: int = LEGACY_TEST_SUBSET_SEED,
) -> np.ndarray:
    """Reproduce the locked historical CIFAR benchmark test set.

    The legacy MLP and ViT notebooks instantiate ``np.random.RandomState(0)``,
    draw the (unstratified) training subset first, and then draw the test subset
    from the same RNG with ``choice(..., replace=False)``.  The returned order is
    significant: the notebooks index the test arrays in exactly this order.

    ``train_subset_size`` is the size of the legacy training draw.  In the new
    protocol it equals ``train_size + validation_size``; the indices from this
    compatibility draw are deliberately discarded because the new train and
    validation sets remain fixed and stratified.
    """

    sizes = {
        "train_population_size": train_population_size,
        "train_subset_size": train_subset_size,
        "test_population_size": test_population_size,
        "test_subset_size": test_subset_size,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in sizes.values()
    ):
        raise ValueError("population and subset sizes must be integers")
    if train_population_size <= 0 or test_population_size <= 0:
        raise ValueError("population sizes must be positive")
    if not 0 < train_subset_size <= train_population_size:
        raise ValueError("train_subset_size must lie in [1, train_population_size]")
    if test_subset_size <= 0:
        raise ValueError("test_subset_size must be positive")

    rng = np.random.RandomState(seed)
    if train_subset_size < train_population_size:
        # Preserve the legacy RNG advance even though the new stratified split
        # uses independently generated train/validation indices.
        rng.choice(train_population_size, train_subset_size, replace=False)
    if test_subset_size >= test_population_size:
        return np.arange(test_population_size, dtype=np.int64)
    return np.asarray(
        rng.choice(test_population_size, test_subset_size, replace=False),
        dtype=np.int64,
    )


def _to_tensor_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> TensorDataset:
    x = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).float().div_(255)
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    x = (x - mean_tensor) / std_tensor
    y = torch.as_tensor(labels[indices], dtype=torch.long)
    return TensorDataset(x, y)


def load_cifar_bundle(
    dataset: CifarDatasetName,
    *,
    root: str | Path = "data",
    train_size: int = 4000,
    validation_size: int = 1000,
    test_size: int = 5000,
    split_seed: int = 20260812,
    download: bool = False,
) -> DatasetBundle:
    """Load stratified train/validation data and the historical benchmark test set.

    ``split_seed`` controls only the new class-balanced train/validation split.
    Test selection intentionally reproduces the paper notebooks' unstratified
    ``RandomState(0)`` draw and never inspects test labels.
    """

    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("CIFAR loading requires torchvision") from exc

    if dataset == "cifar10":
        cls = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        cls = datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:  # pragma: no cover - Literal protects normal callers
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    train_source = cls(root=str(root), train=True, download=download)
    test_source = cls(root=str(root), train=False, download=download)
    train_labels = np.asarray(train_source.targets)
    test_labels = np.asarray(test_source.targets)
    train_indices, validation_indices = _split_train_validation(
        train_labels,
        train_size=train_size,
        validation_size=validation_size,
        seed=split_seed,
    )
    test_indices = legacy_cifar_test_indices(
        train_population_size=len(train_source.data),
        train_subset_size=train_size + validation_size,
        test_population_size=len(test_source.data),
        test_subset_size=test_size,
    )
    test_subset_hash = _indices_hash(test_indices)

    digest = hashlib.sha256()
    digest.update(CIFAR_SPLIT_PROTOCOL.encode())
    digest.update(b"\0")
    digest.update(dataset.encode())
    for label, values in (
        (b"train", train_indices),
        (b"validation", validation_indices),
        (b"test", test_indices),
    ):
        digest.update(b"\0" + label + b"\0")
        digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
        digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))

    return DatasetBundle(
        train=_to_tensor_dataset(
            train_source.data,
            train_labels,
            train_indices,
            mean=mean,
            std=std,
        ),
        validation=_to_tensor_dataset(
            train_source.data,
            train_labels,
            validation_indices,
            mean=mean,
            std=std,
        ),
        test=_to_tensor_dataset(
            test_source.data,
            test_labels,
            test_indices,
            mean=mean,
            std=std,
        ),
        split_hash=digest.hexdigest(),
        dataset=dataset,
        split_protocol=CIFAR_SPLIT_PROTOCOL,
        test_subset_hash=test_subset_hash,
        test_subset_protocol=LEGACY_TEST_SUBSET_PROTOCOL,
        test_subset_seed=LEGACY_TEST_SUBSET_SEED,
    )


def multiplicative_cosine_factor(
    epoch: int,
    *,
    epochs: int,
    floor_ratio: float,
) -> float:
    progress = min(max(epoch, 0), epochs) / epochs
    return floor_ratio + (1.0 - floor_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def make_multiplicative_cosine_scheduler(
    optimizer,
    *,
    epochs: int,
    floor_ratio: float = 1e-3,
):
    """Cosine decay that preserves every optimizer group's LR ratio."""

    # LambdaLR is stepped after each epoch.  Using epochs - 1 as the decay
    # horizon makes the multiplier used by the final training epoch equal to
    # floor_ratio, while the first epoch still uses the original group LRs.
    decay_steps = max(epochs - 1, 1)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: multiplicative_cosine_factor(
            epoch, epochs=decay_steps, floor_ratio=floor_ratio
        ),
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(dataset: TensorDataset, batch_size: int, *, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss_sum += float(criterion(logits, targets))
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


def train_model(
    model: nn.Module,
    optimizer,
    bundle: DatasetBundle,
    config: TrainingConfig,
    *,
    return_best_state: bool = False,
) -> dict:
    """Train using validation only, then evaluate preregistered test endpoints."""

    # Model initialization happens before this function.  The optional
    # stochastic seed gives dropout an independent, deterministic stream while
    # the explicit loader generator continues to pair minibatch order by seed.
    seed_everything(
        config.seed if config.stochastic_seed is None else config.stochastic_seed
    )
    device = _resolve_device(config.device)
    model.to(device)
    train_loader = _loader(
        bundle.train, config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = _loader(
        bundle.validation, config.batch_size, shuffle=False, seed=config.seed
    )
    test_loader = (
        _loader(bundle.test, config.batch_size, shuffle=False, seed=config.seed)
        if config.evaluate_test
        else None
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = make_multiplicative_cosine_scheduler(
        optimizer,
        epochs=config.epochs,
        floor_ratio=config.lr_floor_ratio,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "first_optimizer_group_lr": [],
        "global_learning_rate": [],
        "lr_multiplier": [],
        "optimizer_group_lrs": [],
        "optimizer_steps": [],
    }
    optimizer_steps = 0
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if use_amp and torch.cuda.is_bf16_supported()
                else torch.cuda.amp.autocast(enabled=use_amp)
            )
            with autocast if use_amp else contextlib.nullcontext():
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if config.gradient_clip_norm is not None:
                # Unscale before clipping so the configured norm has the same
                # meaning under CUDA AMP as in the original ViT recipe.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * targets.numel()
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
            count += targets.numel()

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        if config.restore_best_validation and validation_loss < best_validation_loss:
            # Strict `<` keeps the first minimum, matching `np.argmin` on the
            # recorded validation curve, so `selected_epoch` and the epoch the
            # test set is evaluated at cannot disagree.
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in model.state_dict().items()
            }
        history["train_loss"].append(loss_sum / count)
        history["train_accuracy"].append(correct / count)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        multiplier = multiplicative_cosine_factor(
            epoch,
            epochs=max(config.epochs - 1, 1),
            floor_ratio=config.lr_floor_ratio,
        )
        history["first_optimizer_group_lr"].append(
            float(optimizer.param_groups[0]["lr"])
        )
        history["global_learning_rate"].append(config.learning_rate * multiplier)
        history["lr_multiplier"].append(multiplier)
        history["optimizer_group_lrs"].append(
            [float(group["lr"]) for group in optimizer.param_groups]
        )
        history["optimizer_steps"].append(optimizer_steps)
        scheduler.step()

    final_epoch_test_loss, final_epoch_test_accuracy = None, None
    if test_loader is not None and config.restore_best_validation:
        # The model still holds the fixed-final-epoch weights.  Reading the test
        # set here is reporting only: training and checkpoint selection have
        # already finished, and neither endpoint feeds a decision.
        final_epoch_test_loss, final_epoch_test_accuracy = evaluate(
            model, test_loader, device
        )

    if config.restore_best_validation:
        test_epoch = best_epoch
        test_protocol = "best_validation_epoch_single_evaluation_v1"
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        test_epoch = config.epochs - 1
        test_protocol = "final_epoch_single_evaluation_v1"
    if test_loader is None:
        test_loss, test_accuracy = None, None
    elif config.restore_best_validation and test_epoch == config.epochs - 1:
        # Both preregistered endpoints coincide, so the fixed-final pass above
        # is also the selected-checkpoint evaluation.
        test_loss, test_accuracy = final_epoch_test_loss, final_epoch_test_accuracy
    elif config.evaluate_test:
        test_loss, test_accuracy = evaluate(model, test_loader, device)
    else:
        test_loss, test_accuracy = None, None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    examples_seen = len(bundle.train) * config.epochs
    # Standard dense-training estimate: forward + backward ~= 6 FLOPs/parameter/example.
    estimated_flops = 6 * parameter_count * examples_seen
    result = {
        "training_config": asdict(config),
        "dataset": bundle.dataset,
        "split_hash": bundle.split_hash,
        "history": {key: np.asarray(value) for key, value in history.items()},
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_epoch_test_loss": final_epoch_test_loss,
        "final_epoch_test_accuracy": final_epoch_test_accuracy,
        "final_epoch_test_epoch": config.epochs - 1,
        "final_epoch_test_protocol": "fixed_final_epoch_single_evaluation_v1",
        "test_epoch": test_epoch,
        "test_protocol": test_protocol,
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_count,
        "estimated_training_flops": estimated_flops,
        "examples_seen": examples_seen,
        "device": str(device),
    }
    if return_best_state:
        if not config.restore_best_validation:
            raise ValueError("return_best_state requires restore_best_validation=True")
        if best_state is None:  # pragma: no cover - positive epochs guarantee this
            raise RuntimeError("No best-validation checkpoint was captured")
        result["best_state_dict"] = best_state
    return result


def synthetic_bundle(
    *,
    input_dim: int = 16,
    classes: int = 3,
    train_size: int = 48,
    validation_size: int = 24,
    test_size: int = 24,
    seed: int = 0,
) -> DatasetBundle:
    """Small deterministic classification problem for CPU smoke tests."""

    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(input_dim, classes, generator=generator)

    def make(count: int) -> TensorDataset:
        x = torch.randn(count, input_dim, generator=generator)
        y = (x @ teacher).argmax(dim=1)
        return TensorDataset(x, y)

    return DatasetBundle(
        train=make(train_size),
        validation=make(validation_size),
        test=make(test_size),
        split_hash=hashlib.sha256(f"synthetic:{seed}".encode()).hexdigest(),
        dataset="cifar10",
        split_protocol="synthetic_generated_v1",
        test_subset_hash=hashlib.sha256(
            f"synthetic-test:{seed}:{test_size}".encode()
        ).hexdigest(),
        test_subset_protocol="synthetic_generated_v1",
        test_subset_seed=seed,
    )
