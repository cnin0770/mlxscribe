"""Resumable job checkpoints and structured progress events."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import IO

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
WORK_DIR_NAME = ".mlxscribe-work"
HASH_SAMPLE_BYTES = 1024 * 1024


def work_paths(output_dir: Path, stem: str, job: str, work_dir: Path | None) -> tuple[Path, tuple[Path, ...]]:
    """Where a job keeps its checkpoint, and which empty parents to remove afterwards.

    The default is ``<output>/.mlxscribe-work/<stem>/<job>``; ``work_dir`` replaces
    ``<output>/.mlxscribe-work/<stem>`` and is never removed.
    """
    if work_dir is not None:
        return work_dir / job, ()
    root = output_dir / WORK_DIR_NAME / stem
    return root / job, (root, root.parent)


def file_fingerprint(path: Path) -> dict:
    """Identify a media file cheaply: size, mtime, and a hash of its first and last MiB."""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_SAMPLE_BYTES))
        if stat.st_size > HASH_SAMPLE_BYTES:
            handle.seek(max(HASH_SAMPLE_BYTES, stat.st_size - HASH_SAMPLE_BYTES))
            digest.update(handle.read(HASH_SAMPLE_BYTES))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def content_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Checkpoint:
    """A work directory that holds finished units of one job, keyed by a settings fingerprint.

    ``manifest.json`` records the job kind, its fingerprint, how many units it has, and
    which ones are done, so a later run (or an app watching the directory) can tell what
    is left. Each finished unit is its own JSON file, written atomically.
    """

    def __init__(self, path: Path, *, kind: str, fingerprint: dict, total: int) -> None:
        self.path = path
        self.kind = kind
        # Round-trip through JSON so tuples and lists compare equal to a saved manifest.
        self.fingerprint = json.loads(json.dumps(fingerprint))
        self.total = total
        self.completed: set[str] = set()

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        kind: str,
        fingerprint: dict,
        total: int,
        resume: bool,
    ) -> tuple["Checkpoint", str | None]:
        """Open ``path`` for this job, discarding anything that does not match.

        Returns the checkpoint and a message to show the user, or ``None`` when
        there was nothing to resume or discard.
        """
        checkpoint = cls(path, kind=kind, fingerprint=fingerprint, total=total)
        message = None
        previous = read_json(path / MANIFEST_NAME) if path.exists() else None
        if path.exists():
            if not resume:
                message = f"Discarding previous work in {path} (--no-resume)."
            elif previous is None:
                message = f"Discarding unreadable work in {path}; starting fresh."
            else:
                changed = changed_keys(previous, kind, checkpoint.fingerprint)
                if changed:
                    message = (
                        f"Previous work in {path} was made with different settings "
                        f"({', '.join(changed)}); starting fresh."
                    )
                else:
                    checkpoint.completed = {
                        name
                        for name in previous.get("completed", [])
                        if read_json(path / f"{name}.json") is not None
                    }
                    if checkpoint.completed:
                        message = (
                            f"Resuming from {path}: {len(checkpoint.completed)}/{total} "
                            "already done."
                        )
            if message and not checkpoint.completed:
                shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        checkpoint.write_manifest(status="running")
        return checkpoint, message

    def load(self, name: str) -> dict | None:
        if name not in self.completed:
            return None
        return read_json(self.path / f"{name}.json")

    def save(self, name: str, data: dict) -> None:
        write_json_atomic(self.path / f"{name}.json", data)
        self.completed.add(name)
        self.write_manifest(status="running")

    def write_manifest(self, *, status: str) -> None:
        write_json_atomic(
            self.path / MANIFEST_NAME,
            {
                "version": MANIFEST_VERSION,
                "kind": self.kind,
                "status": status,
                "fingerprint": self.fingerprint,
                "total": self.total,
                "completed": sorted(self.completed),
                "updated_at": time.time(),
            },
        )

    def finish(self, *, keep: bool, prune: tuple[Path, ...] = ()) -> None:
        """Mark the job complete, or remove its work directory unless ``keep`` is set.

        Directories in ``prune`` (innermost first) are removed too once they are empty.
        """
        if keep:
            self.write_manifest(status="complete")
            return
        shutil.rmtree(self.path, ignore_errors=True)
        for parent in prune:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()


def changed_keys(manifest: dict, kind: str, fingerprint: dict) -> list[str]:
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("kind") != kind:
        return ["job format"]
    previous = manifest.get("fingerprint") or {}
    keys = sorted(set(previous) | set(fingerprint))
    return [key for key in keys if previous.get(key) != fingerprint.get(key)]


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_json_atomic(path: Path, data: object) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


class ProgressWriter:
    """Append one JSON object per line describing job phases, chunks and batches.

    Every event has ``event`` and ``time`` (Unix seconds); the rest depends on the event.
    A writer without a path does nothing, so callers never need to check.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._handle: IO[str] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: object) -> None:
        if self._handle is None:
            return
        record = {"event": event, "time": round(time.time(), 3), **fields}
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
