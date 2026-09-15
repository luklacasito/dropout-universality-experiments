#!/usr/bin/env python3
"""Small, standalone depth-dropout LSTM pilot; requires only PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("none", "uniform", "early_3_3", "early_2_4", "linear_decreasing", "late_3_3")
LAYERS = 6
CLASSES = 8
CUE_STEPS = 4


def probabilities(arm: str, mean_p: float) -> list[float]:
    """Six layer-input sites; positive arms have identical sum(p)."""
    if not 0 <= mean_p < 1 / 3:
        raise ValueError("mean_p must lie in [0, 1/3) so the 2/4 rate stays below one")
    return {"none": [0.] * LAYERS, "uniform": [mean_p] * LAYERS,
            "early_3_3": [2 * mean_p] * 3 + [0.] * 3,
            "early_2_4": [3 * mean_p] * 2 + [0.] * 4,
            "linear_decreasing": [2 * mean_p * (1 - i / (LAYERS - 1)) for i in range(LAYERS)],
            "late_3_3": [0.] * 3 + [2 * mean_p] * 3}[arm]


def locked_dropout(x: torch.Tensor, p: float, generator: torch.Generator,
                   training: bool) -> torch.Tensor:
    if not training or p == 0:
        return x
    # One mask per example/channel, reused at every sequence position.
    u = torch.rand((x.shape[0], 1, x.shape[2]), device=x.device, generator=generator)
    return x * (u >= p).to(x.dtype) / (1 - p)


class StackedLSTM(nn.Module):
    def __init__(self, hidden: int, rates: list[float]):
        super().__init__()
        if len(rates) != LAYERS or any(not 0 <= p < 1 for p in rates):
            raise ValueError("Provide six layer-input probabilities in [0, 1)")
        self.rates = rates
        self.layers = nn.ModuleList([
            nn.LSTM(CLASSES + 1 if i == 0 else hidden, hidden, batch_first=True)
            for i in range(LAYERS)
        ])
        self.readout = nn.Linear(hidden, CLASSES)
        # Total forget bias +1; do not mask the cell state or recurrent edges.
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
            with torch.no_grad():
                layer.bias_ih_l0[hidden:2 * hidden].fill_(1.)

    def forward(self, x: torch.Tensor, generators: list[torch.Generator]) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            # Six sites: raw sequence input, then five interlayer connections.
            x = locked_dropout(x, self.rates[i], generators[i], self.training)
            x, _ = layer(x)
        return self.readout(x[:, -1])


def dataset(n: int, length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy 8-way cue in first four steps; label-independent distractors after.

    Time-major random draws make shorter examples exact prefixes of longer ones
    with the same seed/n. Labels are balanced to within one example.
    """
    if length <= CUE_STEPS:
        raise ValueError("sequence length must exceed four cue steps")
    gen = torch.Generator().manual_seed(seed)
    y = (torch.arange(n) % CLASSES)[torch.randperm(n, generator=gen)]
    noise = torch.randn((length, n, CLASSES), generator=gen).transpose(0, 1)
    x = torch.zeros(n, length, CLASSES + 1)
    x[:, :, :CLASSES] = noise * 0.5
    x[:, :CUE_STEPS, :CLASSES] += F.one_hot(y, CLASSES).float()[:, None, :]
    x[:, :CUE_STEPS, -1] = 1.  # Marker identifies the cue, never the class.
    return x, y


def mask_generators(seed: int, device: torch.device) -> list[torch.Generator]:
    # Separate streams prevent inactive dropout sites shifting other sites' RNG.
    return [torch.Generator(device=device).manual_seed(20_000 + seed * 10 + i)
            for i in range(LAYERS)]


@torch.no_grad()
def evaluate(model, data, batch_size, device, generators):
    model.eval()
    x, y = data
    loss, correct = 0., 0
    for start in range(0, len(y), batch_size):
        target = y[start:start + batch_size].to(device)
        logits = model(x[start:start + batch_size].to(device), generators)
        loss += F.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(-1) == target).sum().item()
    return {"loss": loss / len(y), "accuracy": correct / len(y)}


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def train(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = {k: v for k, v in vars(args).items()
              if k not in ("out", "arms", "command")}
    config["out"] = None  # Output location is not an experimental treatment.
    config["layers"] = LAYERS
    config["dropout_sites"] = "input_to_each_layer"
    config["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["torch"] = str(torch.__version__)
    config["python"] = platform.python_version()
    config["cuda"] = torch.version.cuda
    config["cudnn"] = torch.backends.cudnn.version()
    config["hardware"] = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    # Fixed data splits across seeds: paired seeds measure training variability.
    train_data = dataset(args.train_size, args.length, 101)
    valid_data = dataset(args.valid_size, args.length, 202)
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        trial = args.out / f"seed-{args.seed}" / arm
        trial.mkdir(parents=True, exist_ok=True)
        if (trial / "config.json").exists():
            previous = json.loads((trial / "config.json").read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError(f"Configuration/source/environment changed: use a new output directory ({trial})")
        if (trial / "result.json").exists():
            result = json.loads((trial / "result.json").read_text())
            if result["fingerprint"] != fingerprint:
                raise ValueError(f"Configuration/source/environment changed: use a new output directory ({trial})")
            print(f"Verified complete: {trial}", flush=True)
            continue
        # Exclusive lock prevents overlapping submissions corrupting a trial.
        with (trial / "running.lock").open("x") as lock:
            lock.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', 'local')}\n")
        try:
            atomic_json(trial / "config.json", {**config, "arm": arm, "fingerprint": fingerprint})
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = StackedLSTM(args.hidden, probabilities(arm, args.mean_p)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            order_gen = torch.Generator().manual_seed(10_000 + args.seed)
            generators = mask_generators(args.seed, device)
            best_loss, best_epoch, best_state = math.inf, 0, None
            history = []
            start_time = time.monotonic()
            x, y = train_data
            for epoch in range(1, args.epochs + 1):
                model.train()
                permutation = torch.randperm(len(y), generator=order_gen)
                total_loss, total_correct, grad_sum = 0., 0, 0.
                for index in permutation.split(args.batch_size):
                    inputs, targets = x[index].to(device), y[index].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs, generators)
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += loss.item() * len(index)
                    total_correct += (logits.argmax(-1) == targets).sum().item()
                    grad_sum += float(grad)
                valid = evaluate(model, valid_data, args.batch_size, device, generators)
                row = {"epoch": epoch, "train_loss_dropout_on": total_loss / len(y),
                       "train_accuracy_dropout_on": total_correct / len(y),
                       "valid_loss": valid["loss"], "valid_accuracy": valid["accuracy"],
                       "mean_preclip_grad_norm": grad_sum / math.ceil(len(y) / args.batch_size),
                       "seconds": time.monotonic() - start_time}
                history.append(row)
                atomic_json(trial / "history.json", history)
                if valid["loss"] < best_loss:
                    best_loss, best_epoch = valid["loss"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                    print(json.dumps({"seed": args.seed, "arm": arm, **row}), flush=True)
            model.load_state_dict(best_state)
            torch.save({"state_dict": best_state, "config": config, "arm": arm,
                        "best_epoch": best_epoch}, trial / "best.pt")
            # Test splits touched only after validation-selected checkpoint is fixed.
            test = {str(length): evaluate(model, dataset(args.test_size, length, 303),
                                         args.batch_size, device, generators)
                    for length in sorted(set([args.length, *args.eval_lengths]))}
            # Label-independent blanking of the cue should remove predictive signal.
            blank_x, blank_y = dataset(args.test_size, args.length, 303)
            blank_x[:, :CUE_STEPS, :CLASSES] = 0.
            blank = evaluate(model, (blank_x, blank_y), args.batch_size, device, generators)
            clean_train = evaluate(model, train_data, args.batch_size, device, generators)
            result = {"fingerprint": fingerprint, "config": config, "arm": arm,
                      "rates": probabilities(arm, args.mean_p), "seed": args.seed,
                      "best_epoch": best_epoch, "best_valid_loss": best_loss,
                      "train_clean": clean_train, "test": test, "blank_cue_test": blank,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "seconds": time.monotonic() - start_time}
            atomic_json(trial / "result.json", result)
            print(json.dumps({"completed": str(trial), "test": test}), flush=True)
        finally:
            (trial / "running.lock").unlink()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--valid-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--mean-p", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=4)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    for key in ("hidden", "train_size", "valid_size", "test_size", "epochs", "batch_size", "threads"):
        if getattr(args, key) <= 0:
            raise SystemExit(f"{key} must be positive")
    if args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise SystemExit("seed must be nonnegative and lr must be positive and finite")
    probabilities("uniform", args.mean_p)
    if min([args.length, *args.eval_lengths]) <= CUE_STEPS:
        raise SystemExit("All lengths must exceed four cue steps")
    train(args)
