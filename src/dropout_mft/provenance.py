"""Frozen, content-addressed provenance for manifest-driven experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable

PROVENANCE_SCHEMA_VERSION = 1
_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "results",
    "venv",
}
_EXCLUDED_FILE_NAMES = {".DS_Store"}
_EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
_SCHEDULER_LOG_PATTERNS = ("slurm-*.out", "slurm-*.err")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a provenance value deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _source_entries(
    source_root: Path,
    *,
    excluded_paths: Iterable[Path] = (),
) -> list[Path]:
    source_root = source_root.resolve()
    excluded = tuple(path.resolve() for path in excluded_paths)
    entries: list[Path] = []
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if any(_is_within(path.resolve(), item) for item in excluded):
            continue
        if path.is_dir():
            continue
        if path.name in _EXCLUDED_FILE_NAMES or path.suffix in _EXCLUDED_FILE_SUFFIXES:
            continue
        if any(path.match(pattern) for pattern in _SCHEDULER_LOG_PATTERNS):
            continue
        entries.append(path)
    return sorted(entries, key=lambda item: item.relative_to(source_root).as_posix())


def source_tree_sha256(
    source_root: str | Path,
    *,
    excluded_paths: Iterable[str | Path] = (),
) -> str:
    """Hash paths, executable bits, symlink targets, and bytes in the source tree."""

    root = Path(source_root).resolve()
    excluded = tuple(Path(path) for path in excluded_paths)
    digest = hashlib.sha256()
    for path in _source_entries(root, excluded_paths=excluded):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.readlink(path).encode() + b"\0")
            continue
        executable = bool(path.stat().st_mode & 0o111)
        digest.update(b"file\0" + (b"x\0" if executable else b"-\0"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_source_archive(
    source_root: Path,
    output: Path,
    *,
    excluded_paths: Iterable[Path] = (),
) -> None:
    """Write a deterministic tar snapshot that includes dirty and untracked files."""

    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in _source_entries(source_root, excluded_paths=excluded_paths):
            relative = path.relative_to(source_root).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if info.isfile():
                info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    os.replace(temporary, output)


def _git_identity(source_root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unknown",
            "dirty": None,
            "status_porcelain_sha256": None,
            "status_entries": [],
        }
    return {
        "commit": commit.decode(errors="replace"),
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "status_entries": [
            entry.decode(errors="replace") for entry in status.split(b"\0") if entry
        ],
    }


def _environment_lock_payload() -> dict:
    distributions = sorted(
        {
            (
                (distribution.metadata.get("Name") or "unknown")
                .lower()
                .replace("_", "-"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
        }
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": [
            {"name": name, "version": version} for name, version in distributions
        ],
    }


def provenance_sha256(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def freeze_source_snapshot(
    output: str | Path,
    *,
    source_root: str | Path | None = None,
    slurm_scripts: Iterable[str | Path] = (),
) -> dict:
    """Create a portable pre-transfer snapshot of source and Slurm inputs."""

    output = Path(output).resolve()
    root = Path(source_root).resolve() if source_root else repository_root()
    if _is_within(output, root):
        raise ValueError("The source snapshot must be stored outside the source tree")
    resolved_slurm = tuple(Path(path).resolve() for path in slurm_scripts)
    for source in resolved_slurm:
        if not source.is_file():
            raise FileNotFoundError(f"Slurm script does not exist: {source}")
    record_path = output / "source-snapshot.json"
    if record_path.exists():
        record = load_source_snapshot(output, source_root=root, validate_source=True)
        requested_hashes = {sha256_file(path) for path in resolved_slurm}
        frozen_hashes = {item["sha256"] for item in record["slurm_scripts"]}
        if requested_hashes != frozen_hashes:
            raise ValueError(
                "Requested Slurm scripts differ from the portable source snapshot"
            )
        return record
    if output.exists():
        raise ValueError(
            f"Source snapshot directory is incomplete or occupied: {output}"
        )

    source_hash_before = source_tree_sha256(root)
    git = _git_identity(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_dir.mkdir(exist_ok=False)
    try:
        archive_path = temporary_dir / "source-tree.tar"
        _write_source_archive(root, archive_path)
        if source_tree_sha256(root) != source_hash_before:
            raise RuntimeError(
                "Source tree changed while its portable snapshot was being frozen"
            )

        frozen_slurm: list[dict] = []
        for index, source in enumerate(resolved_slurm):
            slurm_dir = temporary_dir / "slurm"
            slurm_dir.mkdir(parents=True, exist_ok=True)
            destination = slurm_dir / f"{index:02d}-{source.name}"
            shutil.copyfile(source, destination)
            frozen_slurm.append(
                {
                    "source_name": source.name,
                    "frozen_path": f"slurm/{destination.name}",
                    "sha256": sha256_file(destination),
                }
            )

        record = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "artifact_type": "portable_source_snapshot",
            "git": git,
            "source_tree_sha256": source_hash_before,
            "source_archive": {
                "format": "deterministic_tar_v1",
                "path": "source-tree.tar",
                "sha256": sha256_file(archive_path),
            },
            "slurm_scripts": frozen_slurm,
        }
        _write_json_atomic(temporary_dir / "source-snapshot.json", record)
        os.replace(temporary_dir, output)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return record


def load_source_snapshot(
    snapshot: str | Path,
    *,
    source_root: str | Path | None = None,
    validate_source: bool = True,
) -> dict:
    """Validate a portable source snapshot and optionally the current checkout."""

    snapshot = Path(snapshot).resolve()
    root = Path(source_root).resolve() if source_root else repository_root()
    if _is_within(snapshot, root):
        raise ValueError("The source snapshot must be stored outside the source tree")
    record_path = snapshot / "source-snapshot.json"
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Portable source snapshot is absent or invalid: {record_path}"
        ) from exc
    if (
        record.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or record.get("artifact_type") != "portable_source_snapshot"
    ):
        raise ValueError(f"Unsupported portable source snapshot: {record_path}")
    _verify_frozen_file(
        snapshot, record.get("source_archive", {}), label="source snapshot archive"
    )
    for item in record.get("slurm_scripts", []):
        _verify_frozen_file(
            snapshot,
            {"path": item.get("frozen_path"), "sha256": item.get("sha256")},
            label="source snapshot Slurm script",
        )
    if validate_source and source_tree_sha256(root) != record.get("source_tree_sha256"):
        raise ValueError(
            "Current source tree differs from the portable source snapshot; "
            "restore its archive before planning experiments."
        )
    return record


def freeze_provenance(
    run_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    slurm_scripts: Iterable[str | Path] = (),
    source_snapshot: str | Path | None = None,
) -> dict:
    """Freeze source, environment, and optional Slurm scripts before manifests.

    An existing record is immutable. Reusing a run directory after changing the
    source tree fails with an actionable error instead of silently relabeling it.
    """

    run_dir = Path(run_dir).resolve()
    root = Path(source_root).resolve() if source_root else repository_root()
    provenance_dir = run_dir / "provenance"
    record_path = provenance_dir / "provenance.json"
    resolved_slurm = tuple(Path(path).resolve() for path in slurm_scripts)
    snapshot_path = Path(source_snapshot).resolve() if source_snapshot else None
    if snapshot_path is not None and resolved_slurm:
        raise ValueError(
            "Slurm scripts are already frozen in --source-snapshot; do not pass both"
        )
    for source in resolved_slurm:
        if not source.is_file():
            raise FileNotFoundError(f"Slurm script does not exist: {source}")
    snapshot_record = (
        load_source_snapshot(snapshot_path, source_root=root, validate_source=True)
        if snapshot_path is not None
        else None
    )
    excluded_paths = (run_dir,) if _is_within(run_dir, root) else ()
    if record_path.exists():
        record = load_frozen_provenance(
            run_dir,
            source_root=root,
            validate_source=True,
        )
        requested_hashes = {sha256_file(path) for path in resolved_slurm}
        frozen_hashes = {item["sha256"] for item in record["slurm_scripts"]}
        if requested_hashes and requested_hashes != frozen_hashes:
            raise ValueError(
                "Requested Slurm scripts differ from the frozen provenance; "
                "use a new run directory."
            )
        if snapshot_record is not None and record.get("source_snapshot", {}).get(
            "sha256"
        ) != provenance_sha256(snapshot_record):
            raise ValueError(
                "Requested source snapshot differs from the frozen run provenance"
            )
        return record

    if snapshot_record is not None:
        environment = _environment_lock_payload()
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = run_dir / f".provenance.{os.getpid()}.tmp"
        temporary_dir.mkdir(exist_ok=False)
        try:
            archive_source = snapshot_path / snapshot_record["source_archive"]["path"]
            archive_path = temporary_dir / "source-tree.tar"
            shutil.copyfile(archive_source, archive_path)
            environment_path = temporary_dir / "environment.lock.json"
            _write_json_atomic(environment_path, environment)

            frozen_slurm: list[dict] = []
            for item in snapshot_record["slurm_scripts"]:
                source = snapshot_path / item["frozen_path"]
                slurm_dir = temporary_dir / "slurm"
                slurm_dir.mkdir(parents=True, exist_ok=True)
                destination = slurm_dir / Path(source).name
                shutil.copyfile(source, destination)
                frozen_slurm.append(
                    {
                        "source_name": item["source_name"],
                        "frozen_path": f"provenance/slurm/{destination.name}",
                        "sha256": sha256_file(destination),
                    }
                )

            record = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "git": snapshot_record["git"],
                "source_tree_sha256": snapshot_record["source_tree_sha256"],
                "source_snapshot": {
                    "artifact_type": "portable_source_snapshot",
                    "sha256": provenance_sha256(snapshot_record),
                },
                "source_archive": {
                    "format": "deterministic_tar_v1",
                    "path": "provenance/source-tree.tar",
                    "sha256": sha256_file(archive_path),
                },
                "environment_lock": {
                    "path": "provenance/environment.lock.json",
                    "sha256": sha256_file(environment_path),
                },
                "slurm_scripts": frozen_slurm,
            }
            _write_json_atomic(temporary_dir / "provenance.json", record)
            os.replace(temporary_dir, provenance_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return record

    source_hash_before = source_tree_sha256(root, excluded_paths=excluded_paths)
    git = _git_identity(root)
    environment = _environment_lock_payload()
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = run_dir / f".provenance.{os.getpid()}.tmp"
    temporary_dir.mkdir(exist_ok=False)

    archive_path = temporary_dir / "source-tree.tar"
    _write_source_archive(root, archive_path, excluded_paths=excluded_paths)
    source_hash_after = source_tree_sha256(root, excluded_paths=excluded_paths)
    if source_hash_after != source_hash_before:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise RuntimeError("Source tree changed while its provenance was being frozen")

    environment_path = temporary_dir / "environment.lock.json"
    _write_json_atomic(environment_path, environment)

    frozen_slurm: list[dict] = []
    slurm_dir = temporary_dir / "slurm"
    for index, source in enumerate(resolved_slurm):
        slurm_dir.mkdir(parents=True, exist_ok=True)
        destination = slurm_dir / f"{index:02d}-{source.name}"
        shutil.copyfile(source, destination)
        frozen_slurm.append(
            {
                "source_name": source.name,
                "frozen_path": f"provenance/slurm/{destination.name}",
                "sha256": sha256_file(destination),
            }
        )

    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "git": git,
        "source_tree_sha256": source_hash_before,
        "source_archive": {
            "format": "deterministic_tar_v1",
            "path": "provenance/source-tree.tar",
            "sha256": sha256_file(archive_path),
        },
        "environment_lock": {
            "path": "provenance/environment.lock.json",
            "sha256": sha256_file(environment_path),
        },
        "slurm_scripts": frozen_slurm,
    }
    _write_json_atomic(temporary_dir / "provenance.json", record)
    try:
        os.replace(temporary_dir, provenance_dir)
    except OSError:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if record_path.exists():
            return load_frozen_provenance(
                run_dir,
                source_root=root,
                validate_source=True,
            )
        raise
    return record


def _verify_frozen_file(run_dir: Path, item: dict, *, label: str) -> None:
    try:
        path = (run_dir / item["path"]).resolve()
        expected_hash = item["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid {label} provenance entry") from exc
    if (
        not _is_within(path, run_dir)
        or not path.is_file()
        or sha256_file(path) != expected_hash
    ):
        raise ValueError(f"Frozen {label} is absent or has been modified: {path}")


def load_frozen_provenance(
    run_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    validate_source: bool = True,
    validate_environment: bool = True,
) -> dict:
    run_dir = Path(run_dir).resolve()
    root = Path(source_root).resolve() if source_root else repository_root()
    record_path = run_dir / "provenance" / "provenance.json"
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Frozen provenance is absent or invalid: {record_path}"
        ) from exc
    if record.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported frozen provenance: {record_path}")
    _verify_frozen_file(
        run_dir, record.get("source_archive", {}), label="source archive"
    )
    _verify_frozen_file(
        run_dir, record.get("environment_lock", {}), label="environment lock"
    )
    if validate_environment:
        environment_path = run_dir / record["environment_lock"]["path"]
        try:
            frozen_environment = json.loads(environment_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Frozen environment lock is invalid: {environment_path}"
            ) from exc
        if frozen_environment != _environment_lock_payload():
            raise ValueError(
                "Current Python environment differs from the frozen environment lock; "
                "restore the planned Bridges environment or use a new run directory."
            )
    for item in record.get("slurm_scripts", []):
        _verify_frozen_file(
            run_dir,
            {"path": item.get("frozen_path"), "sha256": item.get("sha256")},
            label="Slurm script",
        )
    if validate_source:
        excluded_paths = (run_dir,) if _is_within(run_dir, root) else ()
        current_hash = source_tree_sha256(root, excluded_paths=excluded_paths)
        if current_hash != record.get("source_tree_sha256"):
            raise ValueError(
                "Current source tree differs from the frozen provenance; "
                "restore the frozen source archive or use a new run directory."
            )
    return record


def git_commit() -> str:
    """Read HEAD without forking after PyTorch has started worker threads."""

    git_entry = repository_root() / ".git"
    try:
        if git_entry.is_file():
            pointer = git_entry.read_text().strip()
            if not pointer.startswith("gitdir:"):
                return "unknown"
            git_dir = (git_entry.parent / pointer.split(":", 1)[1].strip()).resolve()
        else:
            git_dir = git_entry
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        reference = head.split(":", 1)[1].strip()
        loose_ref = git_dir / reference
        if loose_ref.exists():
            return loose_ref.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    except (OSError, ValueError):
        pass
    return "unknown"


def runtime_provenance(device: str, source: dict | None) -> dict:
    import numpy as np
    import torch

    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CLUSTER_NAME",
        "SLURMD_NODENAME",
    )
    runtime = {
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": device,
        "slurm": {
            key.lower(): os.environ[key] for key in slurm_keys if key in os.environ
        },
    }
    return {
        "source": source,
        "source_provenance_sha256": (
            provenance_sha256(source) if source is not None else None
        ),
        "runtime": runtime,
    }
