"""Verify that paired cache-off/on trace bundles replayed identical token marks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IDENTITY_FIELDS = (
    "session_ids", "turn_idx", "input_lens", "output_lens",
    "prefix_tokens", "new_input_tokens", "planned_output_tokens",
    "prompt_sha256", "output_sha256",
)


def compare_bundle_data(
    off_manifest: dict, off_requests: dict,
    on_manifest: dict, on_requests: dict,
) -> dict:
    off_probe, on_probe = off_manifest["probe"], on_manifest["probe"]
    if off_probe["trace_plan_sha256"] != on_probe["trace_plan_sha256"]:
        raise ValueError("paired bundles use different trace plans")
    if off_probe.get("prefix_cache") is not False:
        raise ValueError("first bundle is not cache-off")
    if on_probe.get("prefix_cache") is not True:
        raise ValueError("second bundle is not cache-on")
    mismatched = [
        field for field in IDENTITY_FIELDS
        if off_requests.get(field) != on_requests.get(field)
    ]
    if mismatched:
        raise ValueError(f"paired replay identity mismatch: {mismatched}")
    count = len(off_requests["session_ids"])
    if count == 0:
        raise ValueError("paired replay contains no requests")
    return {
        "status": "identical",
        "trace_plan_sha256": off_probe["trace_plan_sha256"],
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
