from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Dict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: str | Path | None = None) -> Dict[str, object]:
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def file_identity(path: str | Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def assert_file_identity(
    path: str | Path, expected: object, *, label: str
) -> Dict[str, object]:
    if not isinstance(expected, dict) or "sha256" not in expected:
        raise ValueError(f"Missing recorded identity for {label}")
    actual = file_identity(path)
    if actual["sha256"] != str(expected["sha256"]):
        raise ValueError(f"Recorded identity mismatch for {label}: {path}")
    return actual
