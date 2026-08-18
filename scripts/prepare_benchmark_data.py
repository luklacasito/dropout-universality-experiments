#!/usr/bin/env python
"""Build the benchmark caches once, on a login node, then never touch the network.

Each task is written to ``<root>/benchmarks/<name>.npz`` holding

    features_mlp       flat float32 view consumed by the MLP arm
    features_sequence  token-sequence view consumed by the transformer arm
    labels             int64 class ids
    payload_sha256     digest of the arrays, folded into every split hash

Compute nodes on Bridges-2 have no outbound network access from the compute
partitions, so this must be run separately before submitting any array job:

    python scripts/prepare_benchmark_data.py all --root "$PROJECT_DATA"

Two tasks need a manual download because their hosts do not allow scripted
fetches; the script prints the exact command and expected layout when the raw
files are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout_mft.benchmarks import (  # noqa: E402
    BENCHMARK_NAMES,
    BENCHMARK_SPECS,
    FI2010_DEFAULT_EMBARGO,
)


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
FI2010_INFO = (
    "FI-2010 must be downloaded manually from the Fairdata repository:\n"
    "  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649\n"
    "Unpack it so the z-scored no-auction training files are at:\n"
    "  <raw>/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/"
    "NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_*.txt"
)

# The published FI-2010 training files are *anchored cumulative folds*, not
# disjoint chunks: Train_CF_{i+1} contains every snapshot of Train_CF_i plus the
# next day.  Verified against the distributed archive by exact column count and
# label multiset:
#
#     Train_CF_2 (77,909) == Train_CF_1 (39,512) + Test_CF_1 (38,397)
#     Train_CF_3 (106,444) == Train_CF_2 (77,909) + Test_CF_2 (28,535)
#
# Concatenating consecutive folds therefore replays earlier days under a second,
# different z-scoring, which puts the same market events on both sides of a
# forward split.  Read exactly one fold, and stay inside one stock's contiguous
# block within it, so the series is a single instrument moving forward in time
# under a single normalization.
FI2010_FOLD_FILE = "Train_Dst_NoAuction_ZScore_CF_9.txt"
FI2010_FOLD_ROWS = 149
FI2010_FOLD_SNAPSHOTS = 362_400
# Train_CF_9 is stock-major: five contiguous nine-day blocks.  Boundaries were
# recovered by locating each stock's day-one signature (from Train_CF_1) inside
# Train_CF_9, and are asserted against the file's shape at preparation time.
FI2010_STOCK_BOUNDARIES = (0, 27_868, 90_673, 147_849, 228_633, 362_400)
FI2010_DEFAULT_STOCK = 5  # longest block, 133,767 snapshots
FI2010_BOOK_FEATURES = 40
FI2010_LABEL_ROW_BASE = 144


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
    return digest.hexdigest()


def _save(root: Path, name: str, features_mlp, features_sequence, labels) -> None:
    features_mlp = np.ascontiguousarray(features_mlp)
    features_sequence = np.ascontiguousarray(features_sequence)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    spec = BENCHMARK_SPECS[name]

    if features_mlp.shape[0] != labels.shape[0]:
        raise ValueError("features_mlp and labels disagree on example count")
    if features_sequence.shape[0] != labels.shape[0]:
        raise ValueError("features_sequence and labels disagree on example count")
    flat = int(np.prod(features_mlp.shape[1:]))
    if flat != spec.mlp_input_dim:
        raise ValueError(
            f"{name}: MLP view has {flat} features, spec expects {spec.mlp_input_dim}"
        )
    if features_sequence.shape[1] != spec.sequence_length and spec.image_size is None:
        raise ValueError(
            f"{name}: sequence view has length {features_sequence.shape[1]}, "
            f"spec expects {spec.sequence_length}"
        )
    observed = int(labels.max()) + 1
    if observed != spec.classes:
        raise ValueError(f"{name}: found {observed} classes, spec expects {spec.classes}")
    needed = spec.train_size + spec.validation_size + spec.test_size
    if len(labels) < needed:
        raise ValueError(f"{name}: {len(labels)} examples, protocol needs {needed}")

    path = root / "benchmarks" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features_mlp=features_mlp,
        features_sequence=features_sequence,
        labels=labels,
        payload_sha256=_digest(features_mlp, features_sequence, labels),
    )
    temporary.replace(path)
    size_gb = path.stat().st_size / 1e9
    print(
        f"wrote {path} examples={len(labels)} classes={spec.classes} "
        f"mlp={features_mlp.shape[1:]} seq={features_sequence.shape[1:]} "
        f"({size_gb:.2f} GB)"
    )


def _fi2010_required_window_count() -> int:
    spec = BENCHMARK_SPECS["fi2010"]
    return (
        spec.train_size
        + spec.validation_size
        + spec.test_size
        + 2 * FI2010_DEFAULT_EMBARGO
    )


def _fi2010_stock_block(stock: int) -> tuple[int, int]:
    """Return the [start, stop) column range of one stock's nine-day block."""

    stocks = len(FI2010_STOCK_BOUNDARIES) - 1
    if not 1 <= stock <= stocks:
        raise SystemExit(f"FI-2010 stock must be in 1..{stocks}, got {stock}")
    return FI2010_STOCK_BOUNDARIES[stock - 1], FI2010_STOCK_BOUNDARIES[stock]


def _load_fi2010_rows(
    path: Path,
    rows: list[int],
    start: int,
    stop: int,
    *,
    expected_snapshots: int = FI2010_FOLD_SNAPSHOTS,
    expected_rows: int = FI2010_FOLD_ROWS,
) -> np.ndarray:
    """Parse selected rows of one fold over one column slice.

    ``Train_CF_9`` is 149 x 362,400 and roughly 864 MB of text, while the locked
    split needs only the 40 raw book rows plus one label row over a single
    stock's block.  Parsing exactly those keeps preparation to seconds and tens
    of megabytes rather than loading the whole table.  Reading a single file is
    also the point of the fix: the loader can no longer spill into the next
    cumulative fold and replay days it has already seen.
    """

    wanted = {row: position for position, row in enumerate(sorted(set(rows)))}
    block = np.empty((len(wanted), stop - start), dtype=np.float32)
    width: int | None = None
    seen = 0
    with path.open() as handle:
        for index, line in enumerate(handle):
            seen = index + 1
            if index not in wanted:
                continue
            tokens = line.split()
            if width is None:
                width = len(tokens)
                if width != expected_snapshots:
                    raise SystemExit(
                        f"{path.name} has {width} snapshots, expected "
                        f"{expected_snapshots}.  The stock-block boundaries in "
                        "FI2010_STOCK_BOUNDARIES were verified against that exact "
                        "file and cannot be trusted for a different copy."
                    )
                if stop > width:
                    raise SystemExit(
                        f"Requested columns [{start}, {stop}) exceed {path.name}"
                    )
            elif len(tokens) != width:
                raise SystemExit(
                    f"{path.name} row {index} has {len(tokens)} columns, "
                    f"expected {width}"
                )
            block[wanted[index]] = np.asarray(tokens[start:stop], dtype=np.float32)
    if seen != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows in {path.name}, found {seen}")
    return block


def prepare_fi2010(
    root: Path,
    raw: Path,
    *,
    window: int = 100,
    horizon_index: int = 4,
    stock: int = FI2010_DEFAULT_STOCK,
):
    """Sliding windows over the z-scored no-auction limit order book.

    The published files are ``(149, T)``: rows 0-143 are engineered features
    whose first 40 entries are the raw book (10 levels x {ask,bid} x
    {price,volume}), and rows 144-148 are mid-price movement labels at horizons
    k in {1, 2, 3, 5, 10}.  Only the raw 40 are used, so the model sees the book
    rather than someone else's feature engineering.

    The series is taken from a **single** cumulative fold and a **single**
    stock's contiguous nine-day block inside it.  Both restrictions matter.
    Reading more than one fold replays days the earlier fold already contained
    under a second z-scoring, and crossing a stock boundary splices two
    instruments into one window.  Either one silently breaks the forward split.

    Examples stay in time order.  ``benchmarks._anchored_split`` then cuts them
    forward in time with an embargo, which is the only defensible protocol here:
    consecutive windows overlap by 99 snapshots, so a shuffled split would put
    near-duplicates on both sides of the boundary.
    """

    pattern = "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training"
    directory = raw / pattern
    path = directory / FI2010_FOLD_FILE
    if not path.exists():
        raise SystemExit(f"Missing FI-2010 fold {path}\n\n{FI2010_INFO}")

    required_windows = _fi2010_required_window_count()
    required_snapshots = required_windows + window - 1
    start, block_stop = _fi2010_stock_block(stock)
    available = block_stop - start
    if available < required_snapshots:
        longest = max(
            range(1, len(FI2010_STOCK_BOUNDARIES)),
            key=lambda index: FI2010_STOCK_BOUNDARIES[index]
            - FI2010_STOCK_BOUNDARIES[index - 1],
        )
        raise SystemExit(
            f"FI-2010 stock {stock} has a {available}-snapshot block, the locked "
            f"split needs {required_snapshots}.  Use --fi2010-stock {longest}."
        )
    stop = start + required_snapshots
    label_row = FI2010_LABEL_ROW_BASE + horizon_index
    series = _load_fi2010_rows(
        path, [*range(FI2010_BOOK_FEATURES), label_row], start, stop
    )
    print(
        f"FI-2010 fold={FI2010_FOLD_FILE} stock={stock} "
        f"columns=[{start}, {stop}) of {FI2010_FOLD_SNAPSHOTS} "
        f"-> {required_snapshots} snapshots for {required_windows} windows"
    )

    book = series[:FI2010_BOOK_FEATURES].T.astype(np.float32)
    # Labels are coded 1/2/3 (up / stationary / down); shift to 0-based.
    labels_all = series[-1].astype(np.int64) - 1
    if labels_all.min() < 0 or labels_all.max() > 2:
        raise SystemExit("FI-2010 labels are not in the expected 1..3 coding")

    count = len(book) - window + 1
    if count != required_windows:
        raise RuntimeError(
            f"FI-2010 preparation produced {count} windows, expected {required_windows}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(book, window, axis=0)
    # sliding_window_view yields (count, features, window); the model wants
    # (count, window, features) so time is the token axis.
    sequence = np.ascontiguousarray(windows.transpose(0, 2, 1)[:count])
    labels = labels_all[window - 1 : window - 1 + count]
    print(
        f"FI-2010 windows={count} horizon_index={horizon_index} "
        f"class balance={np.bincount(labels) / len(labels)}"
    )
    _save(root, "fi2010", sequence.reshape(count, -1), sequence, labels)


def prepare_tiny_imagenet(root: Path, raw: Path):
    """64x64 RGB from the official training split, label ids sorted by wnid."""

    archive = raw / "tiny-imagenet-200.zip"
    directory = raw / "tiny-imagenet-200"
    if not directory.exists():
        if not archive.exists():
            raise SystemExit(
                f"Missing {archive}. On a login node run:\n"
                f"  curl -L -o {archive} {TINY_IMAGENET_URL}"
            )
        print(f"unpacking {archive}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(raw)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("Tiny-ImageNet preparation requires Pillow") from exc

    train_root = directory / "train"
    wnids = sorted(entry.name for entry in train_root.iterdir() if entry.is_dir())
    if len(wnids) != 200:
        raise SystemExit(f"Expected 200 wnids, found {len(wnids)}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for index, wnid in enumerate(wnids):
        for path in sorted((train_root / wnid / "images").glob("*.JPEG")):
            with Image.open(path) as handle:
                images.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
            labels.append(index)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/200 classes, {len(images)} images")

    # Channel-first float in [0, 1]; the loader z-scores with train statistics.
    stacked = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    labels_array = np.asarray(labels, dtype=np.int64)
    _save(root, "tiny_imagenet", stacked.reshape(len(stacked), -1), stacked, labels_array)


_TOKEN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def prepare_amazon_reviews(root: Path, *, pool: int = 60_000, seed: int = 20260812):
    """Amazon review polarity: hashed bag of words for the MLP, ids for the transformer.

    The vocabulary and hashing are label-free feature extractors fit on the
    unlabeled pool, which is standard practice for a fixed tokenizer.  No label
    from any split is consulted, and the loader still z-scores the MLP view
    using train rows only.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("Amazon preparation requires the `datasets` package") from exc

    spec = BENCHMARK_SPECS["amazon_reviews"]
    # Draw only from the HF train split so the held-out HF test split stays
    # untouched and available as an external check later.
    # Use the canonical namespace explicitly.  Bare legacy dataset names are
    # no longer accepted by newer huggingface_hub URI parsing.
    stream = load_dataset(
        "fancyzhx/amazon_polarity", split="train", streaming=True
    )
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    for row in stream:
        texts.append(f"{row['title']} {row['content']}")
        labels.append(int(row["label"]))
        if len(texts) >= pool:
            break
    print(f"amazon: pooled {len(texts)} reviews")

    tokenized = [_tokenize(text) for text in texts]
    counts = Counter(token for tokens in tokenized for token in tokens)
    # Index 0 is the padding token, so the vocabulary holds vocab_size - 1 words.
    vocabulary = {
        token: index + 1
        for index, (token, _) in enumerate(counts.most_common(spec.vocab_size - 1))
    }

    length = spec.sequence_length
    sequences = np.zeros((len(tokenized), length), dtype=np.int64)
    hashed = np.zeros((len(tokenized), spec.mlp_input_dim), dtype=np.float32)
    for row, tokens in enumerate(tokenized):
        ids = [vocabulary[token] for token in tokens if token in vocabulary]
        truncated = ids[:length]
        sequences[row, : len(truncated)] = truncated
        for token in tokens:
            # Signed hashing keeps the bag-of-words view unbiased.
            code = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(code, "big")
            hashed[row, value % spec.mlp_input_dim] += 1.0 if value >> 63 else -1.0
    # Length normalization so review length does not dominate the MLP view.
    norms = np.linalg.norm(hashed, axis=1, keepdims=True)
    hashed /= np.where(norms < 1e-8, 1.0, norms)

    order = rng.permutation(len(texts))
    _save(
        root,
        "amazon_reviews",
        hashed[order],
        sequences[order],
        np.asarray(labels, dtype=np.int64)[order],
    )


def prepare_speech_commands(root: Path, raw: Path, *, mels: int = 64, frames: int = 64):
    """Log-mel spectrograms of the 35-keyword Speech Commands V2 set."""

    try:
        import torch
        import torchaudio
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit(
            "Speech Commands preparation requires torchaudio:\n"
            "  pip install torchaudio"
        ) from exc

    raw.mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=str(raw), download=True, subset="training"
    )
    sample_rate = 16_000
    # hop_length puts a 1 s clip at just over `frames` columns, then centre crop.
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=sample_rate // frames,
        n_mels=mels,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    use_torchaudio_loader = bool(torchaudio.list_audio_backends())
    if not use_torchaudio_loader:
        from scipy.io import wavfile

    # Metadata lookup does not decode audio, so it also works on systems where
    # TorchAudio was installed without an optional WAV backend (notably macOS).
    labels_seen = sorted(
        {dataset.get_metadata(index)[2] for index in range(len(dataset))}
    )
    label_ids = {label: index for index, label in enumerate(labels_seen)}
    print(f"speech: {len(dataset)} clips across {len(labels_seen)} keywords")

    spectrograms = np.zeros((len(dataset), 1, mels, frames), dtype=np.float32)
    labels = np.zeros(len(dataset), dtype=np.int64)
    batch_size = 256
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        clips = torch.zeros(stop - start, sample_rate)
        for offset, index in enumerate(range(start, stop)):
            relative_path, rate, label, *_ = dataset.get_metadata(index)
            path = Path(dataset._archive) / relative_path
            if use_torchaudio_loader:
                waveform, loaded_rate = torchaudio.load(path)
                rate = loaded_rate
            else:
                loaded_rate, samples = wavfile.read(path)
                rate = int(loaded_rate)
                scale = float(
                    max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
                )
                waveform = torch.from_numpy(samples.astype(np.float32) / scale)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.transpose(0, 1)
            if rate != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, rate, sample_rate
                )
            usable = min(waveform.shape[-1], sample_rate)
            clips[offset, :usable] = waveform[0, :usable]
            labels[index] = label_ids[label]
        batch_spectrograms = to_db(transform(clips))[..., :frames]
        spectrograms[start:stop, 0, :, : batch_spectrograms.shape[-1]] = (
            batch_spectrograms.numpy()
        )
        if stop % 20_000 < batch_size or stop == len(dataset):
            print(f"  {stop}/{len(dataset)} clips")

    _save(
        root,
        "speech_commands",
        spectrograms.reshape(len(spectrograms), -1),
        spectrograms,
        labels,
    )


def prepare_openml_jannis(root: Path, raw: Path):
    """OpenML jannis: 54 numerical features, 4 classes, no spatial structure."""

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("Tabular preparation requires scikit-learn") from exc

    raw.mkdir(parents=True, exist_ok=True)
    bundle = fetch_openml(
        "jannis", version=1, as_frame=True, data_home=str(raw), parser="auto"
    )
    features = bundle.data.to_numpy(dtype=np.float32)
    categories = sorted(bundle.target.unique())
    mapping = {value: index for index, value in enumerate(categories)}
    labels = bundle.target.map(mapping).to_numpy(dtype=np.int64)
    print(f"jannis: {features.shape} classes={np.bincount(labels)}")
    if not np.isfinite(features).all():
        raise SystemExit("jannis contains non-finite values")
    # One token per feature for the transformer arm, the numerical-embedding
    # half of an FT-Transformer.
    _save(root, "openml_jannis", features, features[:, :, None], labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*BENCHMARK_NAMES, "all"))
    parser.add_argument("--root", default="data", help="cache destination")
    parser.add_argument("--raw", default=None, help="raw download area")
    parser.add_argument("--pool", type=int, default=60_000)
    parser.add_argument("--horizon-index", type=int, default=4, choices=range(5))
    parser.add_argument(
        "--fi2010-stock",
        type=int,
        default=FI2010_DEFAULT_STOCK,
        choices=range(1, len(FI2010_STOCK_BOUNDARIES)),
        help="which stock's contiguous nine-day block to use within one fold",
    )
    args = parser.parse_args()

    root = Path(args.root)
    raw = Path(args.raw) if args.raw else root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks = BENCHMARK_NAMES if args.task == "all" else (args.task,)

    failures: list[str] = []
    for task in tasks:
        print(f"\n=== {task} ===")
        try:
            if task == "fi2010":
                prepare_fi2010(
                    root,
                    raw,
                    horizon_index=args.horizon_index,
                    stock=args.fi2010_stock,
                )
            elif task == "tiny_imagenet":
                prepare_tiny_imagenet(root, raw)
            elif task == "amazon_reviews":
                prepare_amazon_reviews(root, pool=args.pool)
            elif task == "speech_commands":
                prepare_speech_commands(root, raw)
            else:
                prepare_openml_jannis(root, raw)
        except SystemExit as exc:
            print(f"SKIPPED {task}: {exc}")
            failures.append(task)
    if failures:
        raise SystemExit(f"\nIncomplete: {', '.join(failures)}")


if __name__ == "__main__":
    main()
