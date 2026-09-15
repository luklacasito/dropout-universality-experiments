#!/usr/bin/env python3
"""Add a reversed linear schedule to the frozen PTB experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Keep the original training kernel and its source fingerprint intact.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rnn_ptb"))
import run as base

BASE_SOURCE_SHA256 = "e540d9c914f8443da54000775f20ad00e5620fb7ccdec1d971ca8cb881384596"
original_probabilities = base.probabilities


def probabilities(arm, p):
    if arm == "linear_increasing":
        return list(reversed(original_probabilities("linear_decreasing", p)))
    return original_probabilities(arm, p)


def validate_protocol(current, selected):
    if current != selected or current["source_sha256"] != BASE_SOURCE_SHA256:
        raise ValueError("Follow-up must match the frozen parent source, data, settings and environment")


def main(cli):
    selection = json.loads((cli.base_out / "selection.json").read_text())
    args = SimpleNamespace(stage="confirm", out=cli.out, data_root=cli.data_root,
                           device="cuda", hidden=128, batch_size=64, bptt=35,
                           epochs=20, lr=.001, threads=2, max_batches=0)
    if not base.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base.torch.set_num_threads(args.threads)
    base.torch.use_deterministic_algorithms(True)
    base.torch.backends.cudnn.benchmark = False
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False
    proto = base.protocol(args)
    validate_protocol(proto, selection["protocol"])
    # Record the adapter explicitly; never pretend this is an unchanged source.
    proto["linear_extension_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    train_words = base.read_words(args.data_root, "train")
    vocab = base.vocabulary(train_words)
    if len(vocab) != 10000:
        raise ValueError("Expected standard 10,000-word PTB vocabulary")
    corpus = {"vocab": vocab,
              "train": base.batchify(base.encode(train_words, vocab), args.batch_size, args.device),
              "valid": base.batchify(base.encode(base.read_words(args.data_root, "valid"), vocab), 10, args.device)}
    base.probabilities = probabilities
    base.trial(args, "linear_increasing", selection["selected_mean_p"], cli.seed, corpus, proto)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    main(parser.parse_args())
