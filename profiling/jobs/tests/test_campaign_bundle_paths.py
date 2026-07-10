"""
Claim:
Campaign orchestration writes live bundles under data/runs/<campaign_id>/<run_id>/,
keeps dry-run sample bundles outside live roots, and checkpoints the exact run
directory emitted by the completed entrypoint.

Plausible wrong implementations:
- Use RUNS directly as the live out-root, mixing different campaigns under one root.
- Let the dry-run sample writer reuse RUNS and pollute the live bundle tree.
- Mark checkpoints by selecting the newest matching glob, so a stale or parallel run
  can satisfy the current step.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from _sample_bundle import write_sample_bundle

ROOT = Path(__file__).resolve().parents[3]
RUN_CAMPAIGN = ROOT / "profiling" / "jobs" / "run_campaign.sh"


def _write_campaign(path: Path) -> None:
    path.write_text(json.dumps({
        "hardware": "H100",
        "model": "unit/model",
        "campaign_type": "tier1",
        "server": {"tp": 1},
        "probes": ["idle_hold"],
    }))


def test_sample_bundle_uses_dry_run_root_not_live_runs(monkeypatch, tmp_path):
    campaign = tmp_path / "sample_campaign.json"
    _write_campaign(campaign)
    live_root = tmp_path / "live"
    dry_root = tmp_path / "dry"
    monkeypatch.setenv("RUNS", str(live_root))
    monkeypatch.setenv("DRY_RUNS", str(dry_root))

    run_dir = write_sample_bundle(campaign)

    assert run_dir == dry_root / "sample_campaign" / "h100_tier1_tp1_SAMPLE"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "power.csv").exists()
    assert (run_dir / "engine.csv").exists()
    assert not live_root.exists()


def test_run_campaign_dry_run_separates_live_plan_from_sample_root(tmp_path):
    campaign = tmp_path / "unit_campaign.json"
    _write_campaign(campaign)
    live_parent = tmp_path / "live-runs"
    dry_parent = tmp_path / "dry-runs"
    env = dict(os.environ)
    env.update({
        "PYBIN": sys.executable,
        "RUNS": str(live_parent),
        "DRY_RUNS": str(dry_parent),
    })

    result = subprocess.run(
        ["bash", str(RUN_CAMPAIGN), str(campaign)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    live_root = live_parent / "unit_campaign"
    sample_dir = dry_parent / "unit_campaign" / "h100_tier1_tp1_SAMPLE"
    assert f"--out-root {live_root}" in result.stdout
    assert f"wrote sample bundle -> {sample_dir}" in result.stdout
    assert sample_dir.exists()
    assert not live_root.exists()


def test_run_campaign_checkpointing_does_not_use_bundle_globs():
    text = RUN_CAMPAIGN.read_text()
    assert 'checkpoint_bundle "$MARK" "$RUNS/*' not in text
    assert 'run_bundle_command "$MARK"' in text
