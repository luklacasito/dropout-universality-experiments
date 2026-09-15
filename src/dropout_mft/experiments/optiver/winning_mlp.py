"""Schedule-aware replicas of the first-place Optiver solution's MLP.

The public notebook's MLP is

``numeric + stock embedding -> [Linear, Dropout, BN, ReLU] x 2 -> Linear(1)``.

Its submitted configuration sets both hidden dropouts and embedding dropout to
zero.  With only two hidden layers, the paper's nominal schedule catalog
collapses to three nonzero shapes.  This module makes that equivalence explicit
and also provides a parameter-matched depth-12 version where the schedules are
genuinely distinct.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dropout_mft.schedules import comparison_profile_cap, named_profile_layers

SHALLOW_PROFILE_IDS = ("none", "uniform", "early", "late")
DEEP_PROFILE_IDS = (
    "none",
    "uniform",
    "step_early",
    "big_step",
    "linear_early",
    "linear_late",
)
PROFILE_TO_SCHEDULE = {
    "none": "none",
    "uniform": "constant",
    "early": "reverse_step",
    "late": "step",
    "step_early": "reverse_step",
    "big_step": "big_step",
    "linear_early": "reverse_linear",
    "linear_late": "linear",
}
SELECTION_SEEDS = (0, 1, 2)
CONFIRMATION_SEEDS = (100, 101, 102, 103, 104)
EPOCH_PROBE_BUDGETS = (30, 100)
EPOCH_PROBE_COHORT_ID = "optiver-winning-mlp-epoch-budget-corrected-v1"
EPOCH_PROBE_PROTOCOL = "penultimate_validation_checkpoint_then_final_fold_once_v1"
EPOCH_PROBE_DROPOUT = {
    "exact_shallow": {
        "none": 0.0,
        "uniform": 0.025,
        "early": 0.025,
        "late": 0.025,
    },
    "matched_depth12": {
        "none": 0.0,
        "uniform": 0.025,
        "step_early": 0.025,
        "big_step": 0.025,
        "linear_early": 0.025,
        "linear_late": 0.05,
    },
}
SHALLOW_DROPOUT_GRID = (0.025, 0.05, 0.10)
# Stop below the fixed 0.20 step cap: at mean p=.20, capped step-early is
# exactly uniform [0.20] x depth and would be a redundant trial.
DEEP_DROPOUT_GRID = (0.025, 0.05, 0.10, 0.15)


def profile_layers(
    profile_id: str, *, depth: int, mean_dropout: float
) -> tuple[float, ...]:
    if profile_id not in PROFILE_TO_SCHEDULE:
        raise ValueError(f"Unknown profile: {profile_id!r}")
    if profile_id == "none":
        if mean_dropout != 0:
            raise ValueError("No-dropout profile requires mean_dropout=0")
        return (0.0,) * depth
    if not 0 < mean_dropout < 0.5:
        raise ValueError("mean_dropout must lie strictly between 0 and 0.5")
    # Keep the paper's step control capped at 0.20.  Big-step is deliberately
    # cap-exempt and spends the same mean budget in the first third; linear
    # profiles only need a 2p cap.  Using one permissive cap for every profile
    # would make depth-12 step-early collapse onto big-step at p=.10.
    profile_id = {"early": "step_early", "late": "step_late"}.get(
        profile_id, profile_id
    )
    cap = comparison_profile_cap(profile_id, mean_dropout)
    return tuple(
        float(value)
        for value in named_profile_layers(profile_id, depth, mean_dropout, cap)
    )


def canonical_profile_key(layers: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(value, 12) for value in layers)


def unique_profiles(
    profile_ids: tuple[str, ...], *, depth: int, mean_dropout: float
) -> dict[tuple[float, ...], tuple[str, ...]]:
    """Return exact layer vectors and every nominal name that maps to them."""

    grouped: dict[tuple[float, ...], list[str]] = {}
    for profile_id in profile_ids:
        budget = 0.0 if profile_id == "none" else mean_dropout
        key = canonical_profile_key(
            profile_layers(profile_id, depth=depth, mean_dropout=budget)
        )
        grouped.setdefault(key, []).append(profile_id)
    return {key: tuple(names) for key, names in grouped.items()}


class WinningFeatureDataset(Dataset):
    def __init__(
        self, numeric: np.ndarray, stock_ids: np.ndarray, targets: np.ndarray
    ) -> None:
        self.numeric = torch.as_tensor(numeric, dtype=torch.float32)
        stock_ids = np.asarray(stock_ids, dtype=np.int64)
        if stock_ids.ndim == 1:
            stock_ids = stock_ids[:, None]
        if stock_ids.ndim != 2:
            raise ValueError("stock_ids must have shape [examples] or [examples, 1]")
        self.stock_ids = torch.as_tensor(stock_ids, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, index: int):
        return self.numeric[index], self.stock_ids[index], self.targets[index]


class ScheduledWinningMLP(nn.Module):
    """Winning PyTorch MLP with one explicit dropout probability per block."""

    def __init__(
        self,
        *,
        numeric_features: int,
        stock_categories: int,
        hidden_width: int,
        dropout_layers: tuple[float, ...],
        embedding_dim: int = 30,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if numeric_features <= 0 or stock_categories <= 0 or hidden_width <= 0:
            raise ValueError("model dimensions must be positive")
        if not dropout_layers:
            raise ValueError("at least one hidden layer is required")
        self.embedding = nn.Embedding(stock_categories, embedding_dim)
        blocks = []
        input_width = numeric_features + embedding_dim
        for dropout in dropout_layers:
            if not 0 <= dropout < 1:
                raise ValueError("dropout probabilities must lie in [0, 1)")
            components: list[nn.Module] = [
                nn.Linear(input_width, hidden_width),
                nn.Dropout(dropout),
            ]
            if batch_norm:
                components.append(nn.BatchNorm1d(hidden_width))
            components.append(nn.ReLU())
            blocks.append(nn.Sequential(*components))
            input_width = hidden_width
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(hidden_width, 1)

    def forward(self, numeric: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(stock_ids[:, 0])
        value = torch.cat((numeric, embedded), dim=1)
        for block in self.blocks:
            value = block(value)
        return self.readout(value).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def matched_deep_width(
    *,
    numeric_features: int,
    stock_categories: int,
    shallow_width: int = 256,
    shallow_depth: int = 2,
    deep_depth: int = 12,
    embedding_dim: int = 30,
) -> int:
    """Largest deep width no more parameterized than the exact shallow MLP."""

    zeros_shallow = (0.0,) * shallow_depth
    target = parameter_count(
        ScheduledWinningMLP(
            numeric_features=numeric_features,
            stock_categories=stock_categories,
            hidden_width=shallow_width,
            dropout_layers=zeros_shallow,
            embedding_dim=embedding_dim,
        )
    )
    best = 1
    for width in range(1, shallow_width + 1):
        count = parameter_count(
            ScheduledWinningMLP(
                numeric_features=numeric_features,
                stock_categories=stock_categories,
                hidden_width=width,
                dropout_layers=(0.0,) * deep_depth,
                embedding_dim=embedding_dim,
            )
        )
        if count <= target:
            best = width
        else:
            break
    return best


@dataclass(frozen=True)
class WinningMLPTrial:
    architecture: Literal["exact_shallow", "matched_depth12"]
    profile_id: str
    mean_dropout: float
    seed: int
    stage: Literal["selection", "confirmation", "epoch_probe"] = "selection"
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 0.002
    max_learning_rate: float = 0.0055
    weight_decay: float = 1e-7
    cohort_id: str | None = None
    evaluation_protocol: str | None = None

    @property
    def depth(self) -> int:
        return 2 if self.architecture == "exact_shallow" else 12

    @property
    def trial_id(self) -> str:
        values = asdict(self)
        # Omit absent follow-up metadata so all legacy trial IDs stay stable.
        if self.cohort_id is None and self.evaluation_protocol is None:
            values.pop("cohort_id")
            values.pop("evaluation_protocol")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def __post_init__(self) -> None:
        allowed = (
            SHALLOW_PROFILE_IDS
            if self.architecture == "exact_shallow"
            else DEEP_PROFILE_IDS
        )
        if self.profile_id not in allowed:
            raise ValueError(
                f"{self.profile_id!r} is not valid for {self.architecture}"
            )
        if self.profile_id == "none" and self.mean_dropout != 0:
            raise ValueError("no dropout requires zero mean")
        if self.profile_id != "none" and self.mean_dropout <= 0:
            raise ValueError("nonzero profiles require positive mean dropout")
        if self.stage not in {"selection", "confirmation", "epoch_probe"}:
            raise ValueError(f"Unknown trial stage: {self.stage!r}")
        if self.stage == "epoch_probe":
            expected = EPOCH_PROBE_DROPOUT[self.architecture][self.profile_id]
            if self.mean_dropout != expected:
                raise ValueError(
                    f"Corrected probe requires frozen mean dropout {expected}"
                )
            if self.epochs not in EPOCH_PROBE_BUDGETS:
                raise ValueError(f"Unknown epoch-probe budget: {self.epochs}")
            if self.cohort_id != EPOCH_PROBE_COHORT_ID:
                raise ValueError("Corrected probe requires its immutable cohort ID")
            if self.evaluation_protocol != EPOCH_PROBE_PROTOCOL:
                raise ValueError("Corrected probe requires its immutable test protocol")


def selection_trials(
    *, seeds: tuple[int, ...] = SELECTION_SEEDS
) -> list[WinningMLPTrial]:
    """Validation-only dropout sweep on the penultimate forward fold."""

    trials = []
    for seed in seeds:
        trials.append(WinningMLPTrial("exact_shallow", "none", 0.0, seed))
        for profile in SHALLOW_PROFILE_IDS[1:]:
            for mean_dropout in SHALLOW_DROPOUT_GRID:
                trials.append(
                    WinningMLPTrial("exact_shallow", profile, mean_dropout, seed)
                )
        trials.append(WinningMLPTrial("matched_depth12", "none", 0.0, seed))
        for profile in DEEP_PROFILE_IDS[1:]:
            for mean_dropout in DEEP_DROPOUT_GRID:
                trials.append(
                    WinningMLPTrial("matched_depth12", profile, mean_dropout, seed)
                )
    return trials


def select_dropout_budgets(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Choose one budget per profile by mean validation RMSPE across seeds."""

    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for result in results:
        trial = WinningMLPTrial(**result["trial"])
        if trial.stage != "selection":
            continue
        grouped[(trial.architecture, trial.profile_id, trial.mean_dropout)].append(
            float(result["best_validation_rmspe"])
        )
    selected: dict[str, dict[str, dict]] = {}
    for architecture, profile_ids in (
        ("exact_shallow", SHALLOW_PROFILE_IDS),
        ("matched_depth12", DEEP_PROFILE_IDS),
    ):
        selected[architecture] = {}
        for profile_id in profile_ids:
            candidates = []
            for (arch, profile, mean_dropout), values in grouped.items():
                if arch == architecture and profile == profile_id:
                    candidates.append(
                        (float(np.mean(values)), mean_dropout, len(values))
                    )
            if not candidates:
                raise ValueError(
                    f"No selection results for {architecture}/{profile_id}"
                )
            mean_rmspe, mean_dropout, seed_count = min(candidates)
            selected[architecture][profile_id] = {
                "mean_dropout": mean_dropout,
                "selection_validation_rmspe": mean_rmspe,
                "selection_seed_count": seed_count,
            }
    return selected


def confirmation_trials(
    selection: dict[str, dict[str, dict]],
    *,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
) -> list[WinningMLPTrial]:
    """Fresh-seed trials on the untouched final forward-time fold."""

    trials = []
    for architecture, profile_ids in (
        ("exact_shallow", SHALLOW_PROFILE_IDS),
        ("matched_depth12", DEEP_PROFILE_IDS),
    ):
        for profile_id in profile_ids:
            mean_dropout = float(selection[architecture][profile_id]["mean_dropout"])
            for seed in seeds:
                trials.append(
                    WinningMLPTrial(
                        architecture,
                        profile_id,
                        mean_dropout,
                        seed,
                        stage="confirmation",
                    )
                )
    return trials


def epoch_probe_trials() -> list[WinningMLPTrial]:
    """Paired 30/100-epoch trials using the already-selected schedules."""

    return [
        WinningMLPTrial(
            architecture=architecture,
            profile_id=profile_id,
            mean_dropout=mean_dropout,
            seed=seed,
            epochs=epochs,
            stage="epoch_probe",
            cohort_id=EPOCH_PROBE_COHORT_ID,
            evaluation_protocol=EPOCH_PROBE_PROTOCOL,
        )
        for epochs in EPOCH_PROBE_BUDGETS
        for architecture, profiles in EPOCH_PROBE_DROPOUT.items()
        for profile_id, mean_dropout in profiles.items()
        for seed in CONFIRMATION_SEEDS
    ]


def rmspe_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.square((target - prediction) / target)))


def rmspe(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square((target - prediction) / target))))


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for numeric, stock, target in loader:
        predictions.append(
            model(numeric.to(device), stock.to(device)).float().cpu().numpy()
        )
        targets.append(target.numpy())
    predictions_array = np.concatenate(predictions)
    targets_array = np.concatenate(targets)
    return rmspe(targets_array, predictions_array), predictions_array


def run_trial(
    trial: WinningMLPTrial,
    *,
    numeric: np.ndarray,
    stock_ids: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """Run one paired trial using the winning optimizer and OneCycle recipe.

    When ``test_indices`` is supplied, validation alone selects the checkpoint.
    The final test loader is not constructed or evaluated until training has
    finished and the best-validation state has been restored.  This is the
    corrected protocol used by the 30-vs-100 epoch-budget probe; legacy callers
    that omit ``test_indices`` retain their original validation-only behavior.
    """

    random.seed(trial.seed)
    np.random.seed(trial.seed)
    torch.manual_seed(trial.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial.seed)
    numeric = np.asarray(numeric, dtype=np.float32)
    stock_ids = np.asarray(stock_ids, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if test_indices is not None:
        test_indices = np.asarray(test_indices, dtype=np.int64)
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError("train and test indices must be disjoint")
        if np.intersect1d(validation_indices, test_indices).size:
            raise ValueError("validation and test indices must be disjoint")
    # Match the winning notebook: fit StandardScaler on the full feature table
    # supplied to train_nn before slicing its one CV fold.  This behavior is
    # preserved for exact reproduction and called out in result metadata.
    mean = np.nanmean(numeric, axis=0)
    std = np.nanstd(numeric, axis=0)
    std[std < 1e-8] = 1.0
    numeric = np.nan_to_num((numeric - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(stock_ids < 0):
        raise ValueError("stock identifiers must be non-negative")
    # The public winning notebook hard-codes n_categories=[128].  Preserve that
    # embedding table exactly while remaining safe for any larger export.
    categories = max(128, int(stock_ids.max()) + 1)
    layers = profile_layers(
        trial.profile_id, depth=trial.depth, mean_dropout=trial.mean_dropout
    )
    hidden_width = (
        256
        if trial.architecture == "exact_shallow"
        else matched_deep_width(
            numeric_features=numeric.shape[1], stock_categories=categories
        )
    )
    model = ScheduledWinningMLP(
        numeric_features=numeric.shape[1],
        stock_categories=categories,
        hidden_width=hidden_width,
        dropout_layers=layers,
    )
    resolved = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    model.to(resolved)
    train_dataset = WinningFeatureDataset(
        numeric[train_indices], stock_ids[train_indices], targets[train_indices]
    )
    validation_dataset = WinningFeatureDataset(
        numeric[validation_indices],
        stock_ids[validation_indices],
        targets[validation_indices],
    )
    generator = torch.Generator().manual_seed(trial.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=trial.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=trial.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.1,
        div_factor=1e3,
        max_lr=trial.max_learning_rate,
        epochs=trial.epochs,
        steps_per_epoch=len(train_loader),
    )
    history = {"train_rmspe": [], "validation_rmspe": []}
    best_rmspe = math.inf
    best_epoch = -1
    best_predictions = None
    best_state = None
    for epoch in range(trial.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for feature_batch, stock_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(resolved)
            stock_batch = stock_batch.to(resolved)
            target_batch = target_batch.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature_batch, stock_batch)
            loss = rmspe_loss(target_batch, prediction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(target_batch)
            count += len(target_batch)
        validation_rmspe, predictions = evaluate(model, validation_loader, resolved)
        history["train_rmspe"].append(loss_sum / count)
        history["validation_rmspe"].append(validation_rmspe)
        if validation_rmspe < best_rmspe:
            best_rmspe = validation_rmspe
            best_epoch = epoch
            best_predictions = predictions.copy()
            if test_indices is not None:
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    result = {
        "trial": asdict(trial),
        "trial_id": trial.trial_id,
        "dropout_layers": np.asarray(layers),
        "hidden_width": hidden_width,
        "parameter_count": parameter_count(model),
        "history": {key: np.asarray(value) for key, value in history.items()},
        "best_validation_rmspe": best_rmspe,
        "best_epoch": best_epoch,
        "best_validation_predictions": best_predictions,
        "validation_targets": targets[validation_indices].copy(),
        "preprocessing_protocol": "winning_notebook_full_X_standard_scaler_v1",
    }
    if test_indices is not None:
        if best_state is None:
            raise RuntimeError("No best-validation checkpoint was captured")
        model.load_state_dict(best_state)
        test_dataset = WinningFeatureDataset(
            numeric[test_indices], stock_ids[test_indices], targets[test_indices]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=trial.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_rmspe, test_predictions = evaluate(model, test_loader, resolved)
        result.update(
            {
                "test_rmspe": test_rmspe,
                "test_predictions": test_predictions,
                "test_targets": targets[test_indices].copy(),
                "evaluation_protocol": EPOCH_PROBE_PROTOCOL,
            }
        )
        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(f".tmp{checkpoint.suffix}")
            torch.save(
                {
                    "trial": asdict(trial),
                    "trial_id": trial.trial_id,
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_rmspe": best_rmspe,
                    "test_rmspe": test_rmspe,
                    "evaluation_protocol": result["evaluation_protocol"],
                },
                temporary,
            )
            temporary.replace(checkpoint)
            result["checkpoint_path"] = str(checkpoint)
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires test_indices")
    return result


def save_result(path: str | Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, payload=np.asarray(result, dtype=object))
    temporary.replace(path)


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "numeric",
            "stock_ids",
            "targets",
            "selection_train_indices",
            "selection_validation_indices",
            "confirmation_train_indices",
            "test_indices",
        }
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"Winning feature cache is missing {sorted(missing)}")
        return {key: payload[key] for key in required}
