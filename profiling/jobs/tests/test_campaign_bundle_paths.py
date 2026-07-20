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
SUBMIT_CAMPAIGN = ROOT / "profiling" / "jobs" / "submit_campaign.sh"
CAMPAIGN_SBATCH = ROOT / "profiling" / "jobs" / "campaign.sbatch"
SERVER_LIFECYCLE = ROOT / "profiling" / "jobs" / "server_lifecycle.sh"


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
    assert "POWERTRACE_ACTIVE_GPU_UUIDS" in text
    assert "POWERTRACE_ACTIVE_GPU_INDICES" in text
    assert "CUDA_VISIBLE_DEVICES=$POWERTRACE_ACTIVE_GPU_INDICES" in text
    assert 'SERVER_MODEL="$($CCFG "$CAMPAIGN" --emit model)"' in text


def test_server_lifecycle_waits_for_openai_model_endpoint():
    text = SERVER_LIFECYCLE.read_text()
    assert "SERVER_MODEL" in text
    assert "POWERTRACE_BASE_URL" in text
    assert '"$base_url/v1/models"' in text
    assert '"$base_url/v1/completions"' in text
    assert "completion probe" in text


def test_sbatch_assigns_a_job_specific_api_port():
    text = CAMPAIGN_SBATCH.read_text()
    assert "SLURM_JOB_ID % 40000" in text
    assert 'POWERTRACE_BASE_URL="http://localhost:$POWERTRACE_PORT"' in text


def test_sealed_execute_requires_separate_output_root(tmp_path):
    campaign = tmp_path / "sealed.json"
    campaign.write_text(json.dumps({
        "hardware": "H100", "model": "unit/model", "campaign_type": "validate",
        "evidence_profile": "measured_ledger", "validation_role": "sealed",
        "server": {"tp": 1},
        "workload": {"dataset": "sharegpt", "num_prompts": 1,
                     "request_rate": 1.0, "seed": 7},
    }))
    env = dict(os.environ, PYBIN=sys.executable)
    env.pop("SEALED_RUNS", None)
    result = subprocess.run(
        ["bash", str(RUN_CAMPAIGN), str(campaign), "--execute"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "sealed campaigns require SEALED_RUNS" in result.stderr


def test_submit_script_refuses_implicit_a100_partition_for_h100():
    text = SUBMIT_CAMPAIGN.read_text()
    assert 'H100 campaign requires an explicit -p <H100_PARTITION>' in text
    assert 'H100 campaign cannot use the A100 ramr partition' in text


def test_submit_script_exports_current_repo_to_sbatch():
    submit = SUBMIT_CAMPAIGN.read_text()
    sbatch = CAMPAIGN_SBATCH.read_text()
    assert 'EXPORTS="ALL,CAMPAIGN=$CAMPAIGN_ABS,POWERTRACE_REPO=$REPO_ROOT"' in submit
    assert '--export="$EXPORTS"' in submit
    assert 'REPO="${POWERTRACE_REPO:-$HOME/powertrace-sim}"' in sbatch


def test_sealed_submission_explicitly_exports_private_run_root(tmp_path):
    campaign = tmp_path / "sealed.json"
    campaign.write_text(json.dumps({
        "hardware": "A100", "model": "unit/model", "campaign_type": "tier1",
        "validation_role": "sealed", "server": {"tp": 1},
        "probes": ["idle_hold"],
    }))
    scratch = tmp_path / "scratch"
    root = scratch / "ptsim"
    (root / "vllm-openai-v0.10.1.1.sandbox").mkdir(parents=True)
    model = root / "hf" / "hub" / "models--unit--model"
    model.mkdir(parents=True)
    (model / ".powertrace-stage-complete").write_text("complete\n")
    group_home = tmp_path / "group"
    group_home.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    sbatch.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "SCRATCH": str(scratch), "GROUP_HOME": str(group_home),
        "PATH": f"{fake_bin}:{env['PATH']}",
    })
    env.pop("SEALED_RUNS", None)

    result = subprocess.run(
        ["bash", str(SUBMIT_CAMPAIGN), str(campaign)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )

    sealed_root = root / "sealed-runs"
    export = (
        f"ALL,CAMPAIGN={campaign},POWERTRACE_REPO={ROOT},"
        f"SEALED_RUNS={sealed_root}"
    )
    assert f"--export={export}" in result.stdout
    assert sealed_root.stat().st_mode & 0o777 == 0o700


def test_batch_entrypoint_defaults_and_exports_sealed_root():
    text = CAMPAIGN_SBATCH.read_text()
    assert 'SEALED_RUNS="${SEALED_RUNS:-$ROOT/sealed-runs}"' in text
    assert 'chmod 700 "$SEALED_RUNS"' in text
    assert "export SEALED_RUNS" in text


def test_submit_script_owners_defaults_are_hardware_specific():
    text = SUBMIT_CAMPAIGN.read_text()
    assert 'A100) CONS="GPU_SKU:A100_SXM4&GPU_MEM:80GB"' in text
    assert 'H100) CONS="GPU_SKU:H100_SXM5&GPU_MEM:80GB"' in text


def test_offline_openhands_data_is_staged_and_exported():
    submit = SUBMIT_CAMPAIGN.read_text()
    sbatch = CAMPAIGN_SBATCH.read_text()
    assert "profiling/jobs/stage_openhands.sh" in submit
    assert "APPTAINERENV_OPENHANDS_DATASET_PATH" in sbatch
    assert 'TIME="$DEFAULT_TIME"' in submit
