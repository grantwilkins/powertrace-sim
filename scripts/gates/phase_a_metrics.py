"""Phase A equivalence gate: metrics unification (archive/research_notes/data-path.md D11).

Regenerates the Azure facility metric CSVs through the unified
``model/classifiers/metrics.py`` implementations into a scratch directory and
compares them column-by-column against the committed ``results/eval_paper``
artifacts at ``HEAD``. The gate passes when the only changed columns are the intentional
ones (the LDC estimator moved from ``np.percentile`` to the exceedance-rank
convention). Everything else must match to full float precision as written.

Run from repo root:
    uv run -m scripts.gates.phase_a_metrics
    uv run -m scripts.gates.phase_a_metrics --record  # also write the JSON record

The heavier metric consumers (run_baselines_facility, generate_power_cdf_
comparison, hierarchy_figure) are covered by verbatim-move unit tests in
``model/tests/test_metrics.py`` and are not regenerated here; they need model
checkpoints and multi-minute runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INTENTIONAL_CHANGED_COLUMNS = {
    "azure_facility_metrics.csv": {"ldc_p95_kw", "ldc_p99_kw"},
    "azure_facility_ldc_15min.csv": set(),
    "azure_facility_site_traces_15min.csv": set(),
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _compare(reference: Path, regenerated: Path) -> dict:
    ref_rows = _read_csv(reference)
    new_rows = _read_csv(regenerated)
    result: dict = {"file": reference.name, "n_rows_ref": len(ref_rows), "n_rows_new": len(new_rows)}
    if len(ref_rows) != len(new_rows):
        result["status"] = "row_count_mismatch"
        return result
    changed: dict[str, dict] = {}
    for i, (r, n) in enumerate(zip(ref_rows, new_rows)):
        if set(r) != set(n):
            result["status"] = "column_set_mismatch"
            return result
        for col in r:
            if r[col] != n[col]:
                entry = changed.setdefault(col, {"n_rows_changed": 0, "example": None})
                entry["n_rows_changed"] += 1
                if entry["example"] is None:
                    entry["example"] = {"row": i, "before": r[col], "after": n[col]}
    result["changed_columns"] = changed
    allowed = INTENTIONAL_CHANGED_COLUMNS.get(reference.name, set())
    unexpected = sorted(set(changed) - allowed)
    result["unexpected_changes"] = unexpected
    result["status"] = "ok" if not unexpected else "unexpected_diff"
    return result


def _write_head_reference(name: str, out: Path) -> None:
    content = subprocess.run(
        ["git", "show", f"HEAD:results/eval_paper/{name}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    out.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the JSON gate record")
    parser.add_argument("--scratch-dir", default=None)
    args = parser.parse_args()

    from scripts.eval.azure_metrics import compute_azure_facility_metrics
    from scripts.eval.azure_defaults import build_default_paths

    defaults = build_default_paths()
    scratch = Path(args.scratch_dir) if args.scratch_dir else Path(tempfile.mkdtemp(prefix="phase_a_gate_"))
    scratch.mkdir(parents=True, exist_ok=True)
    baseline = scratch / "head_reference"
    baseline.mkdir(exist_ok=True)
    for name in INTENTIONAL_CHANGED_COLUMNS:
        _write_head_reference(name, baseline / name)

    compute_azure_facility_metrics(
        aggregated_root=defaults["aggregated_root"],
        node_traces_root=defaults["node_traces_root"],
        experimental_manifest=defaults["experimental_manifest"],
        metrics_csv=str(scratch / "azure_facility_metrics.csv"),
        ldc_csv=str(scratch / "azure_facility_ldc_15min.csv"),
        site_traces_15min_csv=str(scratch / "azure_facility_site_traces_15min.csv"),
        tp_gpus=4,
    )

    reports = [
        _compare(baseline / name, scratch / name)
        for name in INTENTIONAL_CHANGED_COLUMNS
    ]

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    record = {
        "gate": "phase_a_metrics",
        "git_sha": git_sha,
        "scratch_dir": str(scratch),
        "intentional_changes": {k: sorted(v) for k, v in INTENTIONAL_CHANGED_COLUMNS.items()},
        "reports": reports,
        "pass": all(r["status"] == "ok" for r in reports),
    }
    print(json.dumps(record, indent=2))
    if args.record:
        records_dir = REPO_ROOT / "scripts" / "gates" / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        out = records_dir / f"phase_a_metrics_{git_sha[:12]}.json"
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"record -> {out}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
