"""Regenerate every local artifact on the maintained paper allowlist."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path("results/clean_model/powertrace_v1.json")
MANIFEST = Path("results/paper/manifest.json")

COMMANDS = (
    ("prepare", "-m", "model.scripts.prepare_data"),
    (
        "train", "-m", "model.scripts.train",
        "--prepared-manifest", "results/clean_model/prepared_dataset.json",
        "--out-artifact", str(ARTIFACT),
    ),
    ("replay", "-m", "scripts.paper.replay"),
    ("timing", "power-test/plot_timing_parity.py"),
    ("qwen-transfer", "power-test/plot_transfer_traces.py"),
    ("burstgpt-transfer", "power-test/plot_burstgpt_transfer.py"),
    ("openhands-transfer", "power-test/plot_openhands_transfer.py"),
    (
        "facility", "-m", "scripts.eval.run_azure_pipeline",
        "--selected-artifact", str(ARTIFACT),
    ),
)

OUTPUTS = (
    "results/paper/selected_replay_manifest.json",
    "results/paper/selected_model_fidelity_table.tex",
    "results/paper/selected_representative_traces.json",
    "results/paper/timing_parity_manifest.json",
    "results/paper/appendix/transfer_trace_report.json",
    "results/paper/appendix/burstgpt_idle_transfer_report.json",
    "results/paper/appendix/openhands_platform_calibration.json",
    "results/azure_facility/node_traces/trace_summary.json",
    "results/eval_paper/azure_facility_metrics.csv",
    "results/eval_paper/azure_facility_ldc_15min.csv",
    "results/eval_paper/azure_facility_site_traces_15min.csv",
    "results/eval_paper/azure_oversubscription_capacity.json",
    "results/eval_paper/azure_facility_sizing_table.json",
    "results/paper/facility/azure_figure_1_diurnal_profile.pdf",
    "results/paper/facility/azure_figure_2_baseline_comparison_15min.pdf",
    "results/paper/facility/azure_figure_3_load_duration_curve.pdf",
    "results/paper/facility/azure_figure_4_rack_heatmap.pdf",
    "results/paper/facility/azure_figure_5_sizing_metrics.pdf",
    "results/paper/facility/azure_oversubscription_capacity.pdf",
    "results/paper/facility/azure_oversubscription_lines.pdf",
    "results/paper/facility/azure_figure_manifest.json",
)


def file_record(path: Path, *, root: Path = ROOT) -> dict[str, object]:
    absolute = root / path
    digest = hashlib.sha256(absolute.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": absolute.stat().st_size}


def write_manifest(root: Path = ROOT) -> Path:
    missing = [relative for relative in OUTPUTS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"paper regeneration left missing outputs: {missing}")
    artifact = root / ARTIFACT
    payload = {
        "schema_version": "powertrace-paper-artifacts-v1",
        "artifact": file_record(artifact.relative_to(root), root=root),
        "evidence_boundary": {
            "selected_replay_and_facility": "frozen release artifact",
            "appendix_transfer": "retrospective target-calibrated diagnostics",
            "external_score": "deferred and not included",
            "disaggregated_inference": "deferred and not included",
        },
        "outputs": [file_record(Path(relative), root=root) for relative in OUTPUTS],
    }
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    for name, *args in COMMANDS:
        print(f"[paper] {name}", flush=True)
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(write_manifest())


if __name__ == "__main__":
    main()
