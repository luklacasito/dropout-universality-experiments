"""Shared neural-network models for scale-transfer experiments.

The original paper notebooks define their models inline.  This module keeps the
same plain, fully connected MLP while making the parameterization explicit so
standard-parameterization and muP runs share every architectural choice other
than the pieces that muP requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ActivationName = Literal["relu", "gelu"]
Parameterization = Literal["sp", "mup"]


def _require_mup():
    try:
        import mup
        from mup import MuReadout, set_base_shapes
        from mup.optim import MuAdamW
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "muP support requires the pinned optional dependency `mup==1.0.0`."
        ) from exc
    return mup, MuReadout, set_base_shapes, MuAdamW


@dataclass(frozen=True)
class MLPConfig:
    """Architecture and initialization shared by SP and muP MLPs."""

    input_dim: int = 3 * 32 * 32
    width: int = 256
    output_dim: int = 10
    depth: int = 6
    activation: ActivationName = "relu"
    sigma_w_sq: float = 1.98
    sigma_b_sq: float = 0.02
    # Profile studies retain the paper's Gaussian readout.  Controlled SP/muP
    # transfer studies opt into the same zero-readout condition for both arms.
    zero_readout: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.width <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim, width, and output_dim must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {self.activation!r}")
        if self.sigma_w_sq <= 0 or self.sigma_b_sq < 0:
            raise ValueError("sigma_w_sq must be positive and sigma_b_sq nonnegative")


class CriticalMLP(nn.Module):
    """Plain depth-dependent-dropout MLP used by the paper experiments.

    ``depth`` is the number of hidden affine/nonlinear/dropout blocks.  The
    classifier is a separate readout, matching the existing notebooks.
    """

    def __init__(
        self,
        config: MLPConfig,
        dropout_layers: list[float] | tuple[float, ...],
        *,
        parameterization: Parameterization = "sp",
    ) -> None:
        super().__init__()
        if len(dropout_layers) != config.depth:
            raise ValueError(
                f"Expected {config.depth} dropout probabilities, got {len(dropout_layers)}"
            )
        if any(not 0 <= value < 1 for value in dropout_layers):
            raise ValueError("Dropout probabilities must lie in [0, 1)")
        if parameterization not in {"sp", "mup"}:
            raise ValueError(f"Unknown parameterization: {parameterization!r}")

        self.config = config
        self.parameterization = parameterization
        hidden: list[nn.Linear] = [nn.Linear(config.input_dim, config.width)]
        hidden.extend(
            nn.Linear(config.width, config.width) for _ in range(config.depth - 1)
        )
        self.hidden = nn.ModuleList(hidden)
        self.dropouts = nn.ModuleList(nn.Dropout(float(p)) for p in dropout_layers)
        self.activation = nn.ReLU() if config.activation == "relu" else nn.GELU()

        if parameterization == "mup":
            _, MuReadout, _, _ = _require_mup()
            self.readout = MuReadout(
                config.width,
                config.output_dim,
                readout_zero_init=config.zero_readout,
            )
        else:
            self.readout = nn.Linear(config.width, config.output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        for layer, dropout in zip(self.hidden, self.dropouts, strict=True):
            x = dropout(self.activation(layer(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.forward_features(x))


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 4,
        *,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("ViT dimension must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dimension // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, tokens, dimension)
        return self.projection(attended)


class _TransformerMLP(nn.Module):
    def __init__(self, dimension: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dimension * ratio)
        self.first = nn.Linear(dimension, hidden)
        self.second = nn.Linear(hidden, dimension)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.activation(self.first(x)))


class _TransformerBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.first_norm = nn.LayerNorm(dimension)
        self.second_norm = nn.LayerNorm(dimension)
        self.attention = _Attention(dimension, heads)
        self.mlp = _TransformerMLP(dimension, ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.first_norm(x)))
        return x + self.dropout(self.mlp(self.second_norm(x)))


class TinyViT(nn.Module):
    """The existing CIFAR ViT, centralized for the profile confirmation."""

    def __init__(
        self,
        dropout_layers: list[float] | tuple[float, ...],
        *,
        dimension: int = 128,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int = 100,
        patch_size: int = 4,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch = _PatchEmbed(
            dimension, patch_size, in_channels=in_channels, image_size=image_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(
            torch.zeros(1, self.patch.num_patches + 1, dimension)
        )
        self.blocks = nn.ModuleList(
            _TransformerBlock(dimension, heads, mlp_ratio, float(dropout))
            for dropout in dropout_layers
        )
        self.norm = nn.LayerNorm(dimension)
        self.readout = nn.Linear(dimension, output_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.readout(self.norm(x)[:, 0])


class SequenceTransformer(nn.Module):
    """TinyViT's block stack over a generic token sequence.

    The benchmark suite needs a transformer arm for three non-image tasks, and
    all three reduce to the same shape once embedded:

    * limit order book -- ``(batch, snapshots, book_features)`` continuous,
    * text -- ``(batch, tokens)`` integer ids into a vocabulary,
    * tabular -- ``(batch, features, 1)`` continuous, one token per feature,
      which is the numerical-embedding half of an FT-Transformer.

    Passing ``vocab_size`` selects the embedding path; otherwise a linear
    projection maps ``input_features`` to ``dimension``.  Blocks, dropout
    placement, and readout are the same objects TinyViT uses, so the profile
    comparison means the same thing across architectures.
    """

    def __init__(
        self,
        dropout_layers: list[float] | tuple[float, ...],
        *,
        sequence_length: int,
        dimension: int = 128,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int = 2,
        input_features: int | None = None,
        vocab_size: int | None = None,
        padding_index: int | None = None,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if (vocab_size is None) == (input_features is None):
            raise ValueError("Pass exactly one of vocab_size or input_features")

        if vocab_size is not None:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be positive")
            self.embed = nn.Embedding(vocab_size, dimension, padding_idx=padding_index)
            self.token_kind = "discrete"
        else:
            if input_features <= 0:
                raise ValueError("input_features must be positive")
            self.embed = nn.Linear(input_features, dimension)
            self.token_kind = "continuous"

        self.sequence_length = sequence_length
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, dimension))
        self.blocks = nn.ModuleList(
            _TransformerBlock(dimension, heads, mlp_ratio, float(dropout))
            for dropout in dropout_layers
        )
        self.norm = nn.LayerNorm(dimension)
        self.readout = nn.Linear(dimension, output_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(TinyViT._initialize)
        if isinstance(self.embed, nn.Embedding):
            nn.init.trunc_normal_(self.embed.weight, std=0.02)
            if padding_index is not None:
                with torch.no_grad():
                    self.embed.weight[padding_index].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_kind == "discrete":
            x = self.embed(x.long())
        else:
            # Tabular inputs arrive flat; one token per feature.
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = self.embed(x)
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.shape[1]}"
            )
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.readout(self.norm(x)[:, 0])


def _initialize_sp(model: CriticalMLP) -> None:
    config = model.config
    for layer in model.hidden:
        fan_in = layer.weight.shape[1]
        nn.init.normal_(layer.weight, mean=0.0, std=(config.sigma_w_sq / fan_in) ** 0.5)
        if config.sigma_b_sq:
            nn.init.normal_(layer.bias, mean=0.0, std=config.sigma_b_sq**0.5)
        else:
            nn.init.zeros_(layer.bias)
    if config.zero_readout:
        nn.init.zeros_(model.readout.weight)
        nn.init.zeros_(model.readout.bias)
    else:
        fan_in = model.readout.weight.shape[1]
        nn.init.normal_(
            model.readout.weight,
            mean=0.0,
            std=(config.sigma_w_sq / fan_in) ** 0.5,
        )
        if config.sigma_b_sq:
            nn.init.normal_(model.readout.bias, mean=0.0, std=config.sigma_b_sq**0.5)
        else:
            nn.init.zeros_(model.readout.bias)


def _initialize_mup(model: CriticalMLP) -> None:
    mup, _, _, _ = _require_mup()
    config = model.config
    critical_rescale = (config.sigma_w_sq / 2.0) ** 0.5
    for layer in model.hidden:
        # muP's Kaiming initializer applies the appropriate width multiplier.
        mup.init.kaiming_normal_(
            layer.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="relu",
        )
        with torch.no_grad():
            layer.weight.mul_(critical_rescale)
        if config.sigma_b_sq:
            mup.init.normal_(layer.bias, mean=0.0, std=config.sigma_b_sq**0.5)
        else:
            nn.init.zeros_(layer.bias)
    if config.zero_readout:
        nn.init.zeros_(model.readout.weight)
        if model.readout.bias is not None:
            nn.init.zeros_(model.readout.bias)


def build_mlp(
    config: MLPConfig,
    dropout_layers: list[float] | tuple[float, ...],
    *,
    parameterization: Parameterization = "sp",
    base_width: int = 64,
    delta_width: int = 128,
) -> CriticalMLP:
    """Construct and initialize an SP or muP model.

    Base and delta models always have the same depth as the target; only width
    changes.  This is essential to keep the experiment a width-transfer test.
    """

    if parameterization == "mup" and config.activation != "relu":
        raise ValueError(
            "The current muP critical initializer is ReLU-specific; "
            "use standard parameterization for GELU."
        )
    model = CriticalMLP(config, dropout_layers, parameterization=parameterization)
    if parameterization == "sp":
        _initialize_sp(model)
        return model
    if not 0 < base_width < delta_width:
        raise ValueError("muP requires 0 < base_width < delta_width")
    _, _, set_base_shapes, _ = _require_mup()
    base_config = MLPConfig(**{**config.__dict__, "width": base_width})
    delta_config = MLPConfig(**{**config.__dict__, "width": delta_width})
    base = CriticalMLP(base_config, dropout_layers, parameterization="mup")
    delta = CriticalMLP(delta_config, dropout_layers, parameterization="mup")
    set_base_shapes(model, base, delta=delta)
    _initialize_mup(model)
    return model


def make_optimizer(
    model: CriticalMLP,
    *,
    learning_rate: float,
    weight_decay: float = 1e-7,
):
    """Return AdamW with the correct parameterization-specific groups."""

    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if model.parameterization == "mup":
        _, _, _, MuAdamW = _require_mup()
        return MuAdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def parameter_group_lr_ratios(optimizer) -> tuple[float, ...]:
    """Normalize optimizer-group learning rates by the first group."""

    rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not rates or rates[0] == 0:
        raise ValueError("Optimizer must have a nonzero first learning rate")
    return tuple(rate / rates[0] for rate in rates)
