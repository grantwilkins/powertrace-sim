#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

DEFAULT_ALLOWED_JSON_PREFIX = "data/sharegpt-benchmark"
DEFAULT_REQUEST_TIMESTAMP_POLICY = "recorded_only"
REQUEST_TIMESTAMP_POLICIES = ("recorded_only", "allow_synthesized")


def normalize_request_timestamp_policy(policy: str) -> str:
    value = str(policy).strip().lower()
    if value not in REQUEST_TIMESTAMP_POLICIES:
        raise ValueError(
            f"Invalid request timestamp policy '{policy}'. "
            f"Expected one of {REQUEST_TIMESTAMP_POLICIES}."
        )
    return value


def request_timestamp_policy_requires_recorded(policy: str) -> bool:
    return normalize_request_timestamp_policy(policy) == "recorded_only"


def request_json_has_recorded_timestamps(payload: Mapping[str, object]) -> bool:
    raw = payload.get("request_timestamps")
    return bool(isinstance(raw, list) and len(raw) > 0)


def resolve_existing_path_default(raw_path: str, base_dir: str) -> Optional[str]:
    text = str(raw_path).strip()
    if text == "":
        return None
    p = Path(text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(Path(base_dir) / p)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _normalize_prefix(prefix: str) -> str:
    text = str(prefix).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    while text.startswith("/"):
        text = text[1:]
    return text.rstrip("/")


def _normalize_manifest_path(text: str) -> str:
    out = str(text).replace("\\", "/").strip()
    while out.startswith("./"):
        out = out[2:]
    return out


def _path_is_under_prefix(path_text: str, prefix_text: str) -> bool:
    if prefix_text == "":
        return True
    norm_path = _normalize_manifest_path(path_text)
    norm_prefix = _normalize_prefix(prefix_text)
    # Allow wildcard-like prefix behavior (e.g., data/sharegpt-benchmark*).
    if norm_path == norm_prefix or norm_path.startswith(norm_prefix):
        return True
    return False


def is_allowed_sharegpt_pair_json(
    *,
    dataset_dir: str,
    json_path_raw: str,
    resolved_json_path: Optional[str],
    allowed_json_prefix: str,
) -> Tuple[bool, str]:
    prefix = _normalize_prefix(allowed_json_prefix)
    if prefix == "":
        return True, "allowed_prefix_empty"

    if _path_is_under_prefix(dataset_dir, prefix):
        return True, "allowed_by_dataset_dir"
    if _path_is_under_prefix(json_path_raw, prefix):
        return True, "allowed_by_json_path"

    if resolved_json_path is not None:
        try:
            rel = Path(resolved_json_path).resolve().relative_to(Path.cwd().resolve())
            rel_text = rel.as_posix()
            if _path_is_under_prefix(rel_text, prefix):
                return True, "allowed_by_resolved_json_path"
        except Exception:
            pass

    return False, "outside_allowed_json_prefix"


@dataclass
class PairManifestPolicyOutput:
    pair_map: Dict[str, str]
    summary: Dict[str, object]
    rejected_rows: List[Dict[str, str]]


def load_pair_manifest_map_with_policy(
    pair_manifest_csv: str,
    *,
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
    resolve_existing_path_fn: Optional[Callable[[str, str], Optional[str]]] = None,
    include_rejected_rows: bool = False,
    max_rejected_rows: int = 20000,
) -> PairManifestPolicyOutput:
    policy = normalize_request_timestamp_policy(request_timestamp_policy)
    prefix = str(allowed_json_prefix).strip()
    resolver = resolve_existing_path_fn or resolve_existing_path_default
    base_dir = str(Path(pair_manifest_csv).resolve().parent)

    pair_map: Dict[str, str] = {}
    rejected_rows: List[Dict[str, str]] = []
    reason_counts: Dict[str, int] = {}

    num_rows_total = 0
    num_rows_matched = 0
    num_rows_kept = 0
    num_duplicate_pair_keys_overwritten = 0

    def _reject(reason: str, row: Mapping[str, object]) -> None:
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        if include_rejected_rows and len(rejected_rows) < int(max_rejected_rows):
            rejected_rows.append(
                {
                    "pair_key": str(row.get("pair_key", "")).strip(),
                    "reason": str(reason),
                    "dataset_dir": str(row.get("dataset_dir", "")).strip(),
                    "json_path": str(row.get("json_path", "")).strip(),
                    "status": str(row.get("status", "")).strip(),
                }
            )

    with open(pair_manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_rows_total += 1
            if str(row.get("status", "")).strip() != "matched":
                _reject("status_not_matched", row)
                continue
            num_rows_matched += 1

            pair_key = str(row.get("pair_key", "")).strip()
            dataset_dir = str(row.get("dataset_dir", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if pair_key == "":
                _reject("missing_pair_key", row)
                continue
            if json_path_raw == "":
                _reject("missing_json_path", row)
                continue

            json_path = resolver(json_path_raw, base_dir)
            if json_path is None:
                _reject("json_path_not_found", row)
                continue

            allowed, reason = is_allowed_sharegpt_pair_json(
                dataset_dir=dataset_dir,
                json_path_raw=json_path_raw,
                resolved_json_path=json_path,
                allowed_json_prefix=prefix,
            )
            if not allowed:
                _reject(reason, row)
                continue

            if request_timestamp_policy_requires_recorded(policy):
                try:
                    with open(json_path, "r") as f_json:
                        payload = json.load(f_json)
                except Exception:
                    _reject("request_json_parse_error", row)
                    continue
                if not isinstance(payload, dict):
                    _reject("request_json_invalid_payload", row)
                    continue
                if not request_json_has_recorded_timestamps(payload):
                    _reject("missing_recorded_request_timestamps", row)
                    continue

            if pair_key in pair_map:
                num_duplicate_pair_keys_overwritten += 1
            pair_map[pair_key] = json_path
            num_rows_kept += 1

    summary: Dict[str, object] = {
        "pair_manifest_csv": str(Path(pair_manifest_csv).resolve()),
        "request_timestamp_policy": policy,
        "allowed_json_prefix": str(prefix),
        "num_rows_total": int(num_rows_total),
        "num_rows_matched": int(num_rows_matched),
        "num_rows_kept": int(num_rows_kept),
        "num_rows_rejected": int(num_rows_total - num_rows_kept),
        "num_unique_pair_keys_kept": int(len(pair_map)),
        "num_duplicate_pair_keys_overwritten": int(num_duplicate_pair_keys_overwritten),
        "rejection_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
    }

    return PairManifestPolicyOutput(
        pair_map=pair_map,
        summary=summary,
        rejected_rows=rejected_rows,
    )
