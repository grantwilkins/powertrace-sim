from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_dir_for_file(path: str | Path) -> None:
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fields: Optional[Sequence[str]] = None,
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    resolved_fields = fieldnames if fieldnames is not None else fields
    if resolved_fields is None:
        raise TypeError("write_csv() missing required argument: 'fields'")

    ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resolved_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _ensure_dir(path: str | Path) -> None:
    ensure_dir(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _safe_slug(text: str) -> str:
    return safe_slug(text)


def resolve_existing_path(path_str: str, base_dir: str | Path) -> Optional[str]:
    path_text = str(path_str).strip()
    if path_text == "":
        return None

    repo_root = Path(__file__).resolve().parents[2]
    repo_name = repo_root.name
    raw = Path(path_text)

    if raw.is_absolute():
        if raw.exists():
            return str(raw)

        # Handle manifests copied across machines with stale absolute roots.
        parts = raw.parts
        if repo_name in parts:
            i = parts.index(repo_name)
            suffix = Path(*parts[i + 1 :]) if (i + 1) < len(parts) else Path()
            remapped = repo_root / suffix
            if remapped.exists():
                return str(remapped)
        return None

    local = Path(path_text)
    if local.exists():
        return str(local)

    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)

    # Pair manifests often store paths relative to repo root (e.g. "data/...").
    from_repo_root = repo_root / raw
    if from_repo_root.exists():
        return str(from_repo_root)

    return None


def power_timestamp_to_epoch(ts_text: str) -> Optional[float]:
    text = ts_text.strip()
    formats = (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            dt_obj = datetime.strptime(text, fmt)
            # Power CSV timestamps are timezone-naive wall times.
            # Treat as UTC to avoid host-local timezone drift.
            return dt_obj.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        dt_obj = datetime.fromisoformat(text.replace("/", "-"))
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.timestamp()
    except ValueError:
        return None
