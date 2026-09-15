#!/usr/bin/env python3
"""Six-layer Penn Treebank language-model pilot, with validation-only calibration."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from report import ARMS, CALIBRATION_RATES, atomic_json, select_calibration

EXPECTED = {
    "train": ("fcea919f6cf83f35d4d00c6cbf08040d13d4155226340912e2fef9c9c4102cbf", 929589),
    "valid": ("c9fe6985fe0d4ccb578183407d7668fc6066c20700cb4cf87d8ff1cc34df1bf2", 73760),
    "test": ("dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0", 82430),
}


def probabilities(arm, p):
    if not 0 <= p < 1 / 3:
        raise ValueError("Mean dropout must be in [0, 1/3)")
    return {"none": [0.] * 6, "uniform": [p] * 6,
            "early_3_3": [2 * p] * 3 + [0.] * 3,
            "early_2_4": [3 * p] * 2 + [0.] * 4,
            "linear_decreasing": [2 * p * (1 - i / 5) for i in range(6)],
            "late_3_3": [0.] * 3 + [2 * p] * 3}[arm]


def read_words(root, split):
    raw = (root / f"ptb.{split}.txt").read_bytes()
    digest, expected_count = EXPECTED[split]
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Unexpected PTB {split} file hash")
    words = raw.decode("utf-8").replace("\n", " <eos> ").split()
    if len(words) != expected_count:
        raise ValueError(f"Unexpected PTB {split} token count")
    return words


def vocabulary(words):
    counts = Counter(words)
    return {w: i for i, w in enumerate(sorted(counts, key=lambda w: (-counts[w], w)))}


def encode(words, vocab):
    return torch.tensor([vocab.get(w, vocab["<unk>"]) for w in words], dtype=torch.long)


def batchify(tokens, batch_size, device):
    # Contiguous streams, as in standard truncated-BPTT word-language-model code.
    count = (len(tokens) // batch_size) * batch_size
    if count // batch_size < 2:
        raise ValueError("Not enough tokens for a next-word batch")
    return tokens[:count].reshape(batch_size, -1).to(device)


def batches(streams, bptt):
    for start in range(0, streams.shape[1] - 1, bptt):
        stop = min(start + bptt, streams.shape[1] - 1)
        yield streams[:, start:stop], streams[:, start + 1:stop + 1]


def detach_state(state):
    return None if state is None else [(h.detach(), c.detach()) for h, c in state]


def locked_dropout(x, p, gen, training):
    if not training or p == 0:
        return x
    u = torch.rand((x.shape[0], 1, x.shape[-1]), device=x.device, generator=gen)
    return x * (u >= p).to(x.dtype) / (1 - p)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden, rates):
        super().__init__()
        if len(rates) != 6 or any(not 0 <= p < 1 for p in rates):
            raise ValueError("Six valid layer-input probabilities required")
        self.rates = rates
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.LSTM(hidden, hidden, batch_first=True) for _ in range(6)])
        self.decoder = nn.Linear(hidden, vocab_size)
        nn.init.uniform_(self.embedding.weight, -.1, .1)
        nn.init.uniform_(self.decoder.weight, -.1, .1)
        nn.init.zeros_(self.decoder.bias)
        for layer in self.layers:
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
            with torch.no_grad():
                layer.bias_ih_l0[hidden:2 * hidden].fill_(1.)

    def forward(self, tokens, state, generators):
        x = self.embedding(tokens)
        new_state = []
        for i, layer in enumerate(self.layers):
            x = locked_dropout(x, self.rates[i], generators[i], self.training)
            x, carry = layer(x, None if state is None else state[i])
            new_state.append(carry)
        return self.decoder(x), new_state


def pass_epoch(model, streams, bptt, generators, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    state, total_loss, count, grad_sum, steps = None, 0., 0, 0., 0
    with torch.set_grad_enabled(training):
        for index, (inputs, targets) in enumerate(batches(streams, bptt)):
            if max_batches and index >= max_batches:
                break
            state = detach_state(state)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, state = model(inputs, state, generators)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite language-model loss")
            if training:
                loss.backward()
                grad_sum += float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step()
            total_loss += loss.item() * targets.numel()
            count += targets.numel()
            steps += 1
    loss = total_loss / count
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": count,
            "steps": steps, "mean_preclip_grad_norm": grad_sum / steps if training else None}


def atomic_checkpoint(path, contents):
    tmp = path.with_suffix(".tmp")
    torch.save(contents, tmp)
    tmp.replace(path)


def protocol(args):
    code = b"".join((Path(__file__).parent / name).read_bytes() for name in ("run.py", "report.py"))
    return {"benchmark": "PTB word-level, original preprocessed 10k vocabulary",
            "data_sha256": {s: h for s, (h, _) in EXPECTED.items()},
            "source_sha256": hashlib.sha256(code).hexdigest(), "layers": 6,
            "hidden": args.hidden, "batch_size": args.batch_size, "eval_batch_size": 10,
            "bptt": args.bptt, "epochs": args.epochs, "lr": args.lr,
            "max_batches": args.max_batches, "device": args.device, "threads": args.threads,
            "torch": str(torch.__version__), "python": platform.python_version(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "hardware": torch.cuda.get_device_name() if args.device == "cuda" else platform.machine(),
            "dropout_sites": "embedding plus five interlayer connections",
            "mask_reuse": "locked within BPTT chunk; resampled at next chunk"}


def trial(args, arm, p, seed, corpus, proto):
    folder = args.out / args.stage / f"seed-{seed}" / f"{arm}-p{p:g}"
    folder.mkdir(parents=True, exist_ok=True)
    config = {"protocol": proto, "stage": args.stage, "arm": arm, "mean_p": p, "seed": seed}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (folder / "config.json").exists():
        if json.loads((folder / "config.json").read_text())["fingerprint"] != fingerprint:
            raise ValueError(f"Changed source/config/environment: use a new run directory ({folder})")
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("Completed-result fingerprint mismatch")
        print(f"Verified complete: {folder}", flush=True)
        return result
    with (folder / "running.lock").open("x") as lock:
        lock.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', 'local')}\n")
    try:
        atomic_json(folder / "config.json", {**config, "fingerprint": fingerprint})
        torch.manual_seed(seed)
        model = LanguageModel(len(corpus["vocab"]), args.hidden, probabilities(arm, p)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        generators = [torch.Generator(device=args.device).manual_seed(20_000 + seed * 10 + i) for i in range(6)]
        best_loss, best_epoch, first_epoch, history, prior_seconds = math.inf, 0, 1, [], 0.
        if (folder / "latest.pt").exists():
            saved = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint fingerprint mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            for gen, state in zip(generators, saved["mask_rng"]):
                gen.set_state(state)
            best_loss, best_epoch = saved["best_loss"], saved["best_epoch"]
            history, first_epoch = saved["history"], saved["epoch"] + 1
            prior_seconds = history[-1]["seconds"]
        start = time.monotonic()
        for epoch in range(first_epoch, args.epochs + 1):
            tick = time.monotonic()
            train = pass_epoch(model, corpus["train"], args.bptt, generators, optimizer, args.max_batches)
            valid = pass_epoch(model, corpus["valid"], args.bptt, generators, max_batches=args.max_batches)
            row = {"epoch": epoch, "train_dropout_on": train, "valid": valid,
                   "epoch_seconds": time.monotonic() - tick, "seconds": prior_seconds + time.monotonic() - start}
            history.append(row)
            if valid["loss"] < best_loss:
                best_loss, best_epoch = valid["loss"], epoch
                atomic_checkpoint(folder / "best.pt", {"model": model.state_dict(), "config": config,
                                                       "fingerprint": fingerprint, "epoch": epoch})
            atomic_checkpoint(folder / "latest.pt", {
                "fingerprint": fingerprint, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "mask_rng": [g.get_state() for g in generators], "epoch": epoch,
                "best_loss": best_loss, "best_epoch": best_epoch, "history": history})
            atomic_json(folder / "history.json", history)
            print(json.dumps({"stage": args.stage, "seed": seed, "arm": arm, "mean_p": p, **row}), flush=True)
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        train_clean = pass_epoch(model, corpus["train"], args.bptt, generators, max_batches=args.max_batches)
        test = None
        if args.stage == "confirm":
            # No test evaluation or test-token encoding during calibration.
            test_words = read_words(args.data_root, "test")
            test_stream = batchify(encode(test_words, corpus["vocab"]), 1, args.device)
            test = pass_epoch(model, test_stream, args.bptt, generators, max_batches=args.max_batches)
        result = {**config, "fingerprint": fingerprint, "rates": probabilities(arm, p),
                  "best_epoch": best_epoch, "best_valid_loss": best_loss,
                  "best_valid_perplexity": math.exp(best_loss), "train_clean": train_clean,
                  "test": test, "parameters": sum(v.numel() for v in model.parameters()),
                  "seconds": prior_seconds + time.monotonic() - start}
        atomic_json(folder / "result.json", result)
        return result
    finally:
        (folder / "running.lock").unlink()


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage != "canary" and args.max_batches:
        raise ValueError("Batch-limited smoke checks are allowed only in canary stage")
    train_words = read_words(args.data_root, "train")
    vocab = vocabulary(train_words)
    if len(vocab) != 10000:
        raise ValueError("Expected the standard 10,000-word PTB training vocabulary")
    corpus = {"vocab": vocab,
              "train": batchify(encode(train_words, vocab), args.batch_size, args.device),
              "valid": batchify(encode(read_words(args.data_root, "valid"), vocab), 10, args.device)}
    proto = protocol(args)
    if args.stage == "canary":
        trial(args, "uniform", .1, 0, corpus, proto)
    elif args.stage == "calibrate":
        rates = CALIBRATION_RATES if args.trial_index is None else [CALIBRATION_RATES[args.trial_index]]
        for p in rates:
            trial(args, "none" if p == 0 else "uniform", p, 0, corpus, proto)
        if args.trial_index is None:
            select_calibration(args.out)
    else:
        if args.seed not in (1, 2, 3):
            raise ValueError("Confirmation uses fresh seeds 1, 2, 3; seed 0 is for calibration")
        # Confirmation is released only after all calibration array tasks pass.
        # A concurrency limit of one ensures there is only one selection writer.
        if not (args.out / "selection.json").exists():
            select_calibration(args.out)
        selection = json.loads((args.out / "selection.json").read_text())
        if selection["protocol"] != proto:
            raise ValueError("Calibration and confirmation protocols differ")
        arms = ARMS if args.trial_index is None else [ARMS[args.trial_index]]
        for arm in arms:
            trial(args, arm, selection["selected_mean_p"], args.seed, corpus, proto)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("canary", "calibrate", "confirm"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial-index", type=int, choices=range(6), default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bptt", type=int, default=35)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0)
    args = p.parse_args()
    for field in ("hidden", "batch_size", "bptt", "epochs", "threads"):
        if getattr(args, field) <= 0:
            p.error(f"{field} must be positive")
    if args.max_batches < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("max-batches must be nonnegative; lr must be finite and positive")
    main(args)
