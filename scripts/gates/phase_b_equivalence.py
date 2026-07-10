"""Phase B equivalence gate: RunRecord rebuild vs reference artifacts (data-path §4B).

Rebuilds the two derived training artifacts from raw data into a scratch
directory and byte-compares every array against the reference on disk:

- GRU datasets:  results/experimental_continuous_v1/  (datasets/*.npz,
  splits/*.json, norm_params/*.json, manifest.json)
- Ledger cache:  feature-test/ledger_cache.npz

Modes:
- ``--mode baseline``  run BEFORE rewiring producers (B4): rebuilds with the
  code as it stands and compares to disk. Any diff here is data/code drift
  that predates the RunRecord rewiring (cf. D15) and is recorded as excluded
  context, not a failure of the rewiring.
- ``--mode new``       run AFTER rewiring producers (B5): same comparison; the
  gate passes when every array is byte-identical, or a remaining difference
  has the exact normalized rebuilt content recorded by the baseline.

Comparison mechanics: per-array ``a.tobytes() == b.tobytes()`` (NaN- and
-0.0-exact); never whole-file npz hashes (zip member mtimes differ). JSON is
compared parsed, with ``generated_at_utc`` masked and path values compared by
repo-relative suffix. Diff fingerprints capture both sides of that normalized
comparison, so a filename match cannot hide a changed baseline difference.

Run from repo root (takes minutes; re-parses every matched pair):
    uv run -m scripts.gates.phase_b_equivalence --mode baseline --record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MASKED_JSON_KEYS = {"generated_at_utc"}


def _hash_part(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big"))
    hasher.update(value)


def _update_array_hash(hasher, array: np.ndarray) -> None:
    _hash_part(hasher, array.dtype.str.encode())
    _hash_part(hasher, repr(array.shape).encode())
    if array.dtype != object:
        _hash_part(hasher, array.tobytes())
        return
    for value in array.reshape(-1):
        item = np.asarray(value)
        _hash_part(hasher, item.dtype.str.encode())
        _hash_part(hasher, repr(item.shape).encode())
        _hash_part(hasher, item.tobytes())


def _npz_content_sha256(path: Path) -> str:
    """Hash logical array values without NPZ zip metadata."""
    hasher = hashlib.sha256()
    with np.load(path, allow_pickle=True) as arrays:
        for key in sorted(arrays.files):
            _hash_part(hasher, key.encode())
            _update_array_hash(hasher, arrays[key])
    return hasher.hexdigest()


def _json_content_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_report(ref_npz: Path, new_npz: Path) -> dict:
    with np.load(ref_npz, allow_pickle=True) as ref, np.load(
        new_npz, allow_pickle=True
    ) as new:
        out: dict = {"file": str(ref_npz.relative_to(REPO_ROOT)), "keys": {}}
        keys_ref, keys_new = set(ref.files), set(new.files)
        out["missing_keys"] = sorted(keys_ref - keys_new)
        out["extra_keys"] = sorted(keys_new - keys_ref)
        for key in sorted(keys_ref & keys_new):
            a, b = ref[key], new[key]
            if a.dtype == object or b.dtype == object:
                if a.shape != b.shape:
                    out["keys"][key] = "shape_mismatch"
                    continue
                equal = all(
                    np.asarray(x).dtype == np.asarray(y).dtype
                    and np.asarray(x).shape == np.asarray(y).shape
                    and np.asarray(x).tobytes() == np.asarray(y).tobytes()
                    for x, y in zip(a.reshape(-1), b.reshape(-1))
                )
                out["keys"][key] = "equal" if equal else "diff"
            else:
                equal = (
                    a.dtype == b.dtype
                    and a.shape == b.shape
                    and a.tobytes() == b.tobytes()
                )
                out["keys"][key] = "equal" if equal else "diff"
    out["equal"] = (
        not out["missing_keys"]
        and not out["extra_keys"]
        and all(v == "equal" for v in out["keys"].values())
    )
    if not out["equal"]:
        out["reference_content_sha256"] = _npz_content_sha256(ref_npz)
        out["rebuilt_content_sha256"] = _npz_content_sha256(new_npz)
    return out


def _mask_json(value, base_dir: str):
    if isinstance(value, dict):
        return {
            k: ("<masked>" if k in MASKED_JSON_KEYS else _mask_json(v, base_dir))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_mask_json(v, base_dir) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        # Absolute paths differ between scratch and reference: compare suffix.
        return value.split(base_dir, 1)[-1] if base_dir in value else Path(value).name
    return value


def _json_report(ref_path: Path, new_path: Path) -> dict:
    ref = _mask_json(json.loads(ref_path.read_text()), "experimental_continuous_v1")
    new = _mask_json(json.loads(new_path.read_text()), Path(new_path).parts[-2])
    out = {
        "file": str(ref_path.relative_to(REPO_ROOT)),
        "equal": ref == new,
    }
    if not out["equal"]:
        out["reference_content_sha256"] = _json_content_sha256(ref)
        out["rebuilt_content_sha256"] = _json_content_sha256(new)
    return out


def _compare_tree(reference_dir: Path, rebuilt_dir: Path) -> list[dict]:
    reports: list[dict] = []
    for ref_npz in sorted((reference_dir / "datasets").glob("*.npz")):
        new_npz = rebuilt_dir / "datasets" / ref_npz.name
        if not new_npz.exists():
            reports.append({"file": str(ref_npz.relative_to(REPO_ROOT)), "equal": False, "missing": True})
            continue
        reports.append(_array_report(ref_npz, new_npz))
    for sub in ("splits", "norm_params"):
        for ref_json in sorted((reference_dir / sub).glob("*.json")):
            new_json = rebuilt_dir / sub / ref_json.name
            if not new_json.exists():
                reports.append({"file": str(ref_json.relative_to(REPO_ROOT)), "equal": False, "missing": True})
                continue
            reports.append(_json_report(ref_json, new_json))
    return reports


def _baseline_diff_fingerprints(
    records_dir: Path | None = None,
) -> dict[str, tuple[str, str]]:
    records_dir = records_dir or REPO_ROOT / "scripts" / "gates" / "records"
    records = sorted(records_dir.glob("phase_b_baseline_*.json"))
    if not records:
        raise ValueError("--mode new requires a recorded phase-B baseline")
    payload = json.loads(records[-1].read_text())
    fingerprints: dict[str, tuple[str, str]] = {}
    for row in payload.get("diffs", []):
        file = str(row["file"])
        reference_fingerprint = row.get("reference_content_sha256")
        fingerprint = row.get("rebuilt_content_sha256")
        if not isinstance(reference_fingerprint, str) or not isinstance(fingerprint, str):
            raise ValueError(
                "phase-B baseline lacks comparison-content fingerprints; "
                "rerun --mode baseline --record before --mode new"
            )
        if file in fingerprints:
            raise ValueError(f"phase-B baseline has duplicate diff entry: {file}")
        fingerprints[file] = (reference_fingerprint, fingerprint)
    return fingerprints


def _unexpected_diff_files(
    diff_rows: list[dict], baseline_fingerprints: dict[str, tuple[str, str]]
) -> list[str]:
    return sorted({
        str(row["file"])
        for row in diff_rows
        if baseline_fingerprints.get(str(row["file"]))
        != (
            row.get("reference_content_sha256"),
            row.get("rebuilt_content_sha256"),
        )
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "new"], required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--skip-gru", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    args = parser.parse_args()
    baseline_fingerprints = (
        _baseline_diff_fingerprints() if args.mode == "new" else {}
    )

    scratch = Path(args.scratch_dir) if args.scratch_dir else Path(
        tempfile.mkdtemp(prefix=f"phase_b_{args.mode}_")
    )
    scratch.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []

    if not args.skip_gru:
        from model.training_data.manifest import run_prepare_experimental_manifest

        gru_out = scratch / "experimental_continuous_v1"
        run_prepare_experimental_manifest(
            pair_manifest_csv=str(REPO_ROOT / "results" / "stage0" / "pair_manifest.csv"),
            out_dir=str(gru_out),
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
            min_traces_per_config=2,
        )
        reports.extend(
            _compare_tree(REPO_ROOT / "results" / "experimental_continuous_v1", gru_out)
        )

    if not args.skip_ledger:
        ledger_out = scratch / "ledger_cache.npz"
        subprocess.run(
            [
                "uv", "run", "python", "feature-test/build_ledger_cache.py",
                "--max-per-cell", "3",
                "--out", str(ledger_out),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        reports.append(
            _array_report(REPO_ROOT / "feature-test" / "ledger_cache.npz", ledger_out)
        )

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    n_equal = sum(1 for r in reports if r.get("equal"))
    diff_rows = [r for r in reports if not r.get("equal")]
    unexpected_diff_files = _unexpected_diff_files(
        diff_rows, baseline_fingerprints
    )
    passed = not diff_rows if args.mode == "baseline" else not unexpected_diff_files
    record = {
        "gate": "phase_b_equivalence",
        "mode": args.mode,
        "git_sha": git_sha,
        "scratch_dir": str(scratch),
        "n_files": len(reports),
        "n_equal": n_equal,
        "expected_baseline_diff_files": sorted(baseline_fingerprints),
        "unexpected_diff_files": unexpected_diff_files,
        "diffs": diff_rows,
        "pass": passed,
    }
    summary = {k: v for k, v in record.items() if k != "diffs"}
    print(json.dumps(summary, indent=2))
    for r in record["diffs"]:
        print(f"DIFF: {json.dumps(r)[:400]}")
    if args.record:
        records_dir = REPO_ROOT / "scripts" / "gates" / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        out = records_dir / f"phase_b_{args.mode}_{git_sha[:12]}.json"
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"record -> {out}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
