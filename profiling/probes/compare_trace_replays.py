"""Verify that paired cache-off/on trace bundles replayed identical token marks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IDENTITY_FIELDS = (
    "input_lens", "output_lens",
    "prefix_tokens", "new_input_tokens", "planned_output_tokens",
    "source_ids", "prompt_sha256", "output_sha256",
    "forced_output_token_id", "request_seed", "decode_constraint",
)
PROTOCOL = "singleton_allowed_token_v1"


def _keyed_rows(requests: dict) -> dict[tuple[str, int], dict]:
    sessions = requests.get("session_ids") or []
    turns = requests.get("turn_idx") or []
    if len(sessions) != len(turns):
        raise ValueError("session_ids and turn_idx lengths differ")
    rows = {}
    for index, key in enumerate(zip(sessions, turns)):
        normalized = (str(key[0]), int(key[1]))
        if normalized in rows:
            raise ValueError(f"duplicate replay key: {normalized}")
        row = {}
        for field in IDENTITY_FIELDS:
            values = requests.get(field)
            if values is None or len(values) != len(sessions):
                raise ValueError(f"missing or ragged replay identity field: {field}")
            row[field] = values[index]
        rows[normalized] = row
    return rows


def _server_control(manifest: dict) -> dict:
    server = dict(manifest.get("server") or {})
    server.pop("active_gpu_uuids", None)
    server.pop("enable_prefix_caching", None)
    return server


def compare_bundle_data(
    off_manifest: dict, off_requests: dict,
    on_manifest: dict, on_requests: dict,
) -> dict:
    off_probe, on_probe = off_manifest["probe"], on_manifest["probe"]
    plan_field = (
        "trace_plan_sha256"
        if "trace_plan_sha256" in off_probe
        else "replay_plan_sha256"
    )
    if off_probe.get(plan_field) != on_probe.get(plan_field):
        raise ValueError("paired bundles use different trace plans")
    if off_probe.get("prefix_cache") is not False:
        raise ValueError("first bundle is not cache-off")
    if on_probe.get("prefix_cache") is not True:
        raise ValueError("second bundle is not cache-on")
    if (
        off_probe.get("decode_constraint") != PROTOCOL
        or on_probe.get("decode_constraint") != PROTOCOL
    ):
        raise ValueError("paired replay lacks deterministic decode protocol")
    for field in ("model", "hardware", "tp", "arch"):
        if off_manifest.get(field) != on_manifest.get(field):
            raise ValueError(f"paired replay control mismatch: {field}")
    if (off_manifest.get("versions") or {}).get("vllm") != (
        on_manifest.get("versions") or {}
    ).get("vllm"):
        raise ValueError("paired replay control mismatch: vllm version")
    if _server_control(off_manifest) != _server_control(on_manifest):
        raise ValueError("paired replay control mismatch: server")
    off_rows = _keyed_rows(off_requests)
    on_rows = _keyed_rows(on_requests)
    if set(off_rows) != set(on_rows):
        raise ValueError("paired replay has missing or extra request keys")
    mismatched = {
        field
        for key in off_rows
        for field in IDENTITY_FIELDS
        if off_rows[key][field] != on_rows[key][field]
    }
    if mismatched:
        raise ValueError(
            f"paired replay identity mismatch: {sorted(mismatched)}"
        )
    count = len(off_rows)
    if count == 0:
        raise ValueError("paired replay contains no requests")
    return {
        "status": "identical",
        plan_field: off_probe[plan_field],
        "requests": count,
        "cache_off_bundle": off_manifest["run_id"],
        "cache_on_bundle": on_manifest["run_id"],
    }


def _load_bundle(path: str) -> tuple[dict, dict]:
    root = Path(path)
    return (
        json.loads((root / "manifest.json").read_text()),
        json.loads((root / "requests.json").read_text()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_off_bundle")
    parser.add_argument("cache_on_bundle")
    args = parser.parse_args()
    off_manifest, off_requests = _load_bundle(args.cache_off_bundle)
    on_manifest, on_requests = _load_bundle(args.cache_on_bundle)
    print(json.dumps(compare_bundle_data(
        off_manifest, off_requests, on_manifest, on_requests
    ), indent=2))


if __name__ == "__main__":
    main()
