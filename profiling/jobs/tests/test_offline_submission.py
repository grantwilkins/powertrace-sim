"""
Claim:
Scientific GPU jobs fail before submission or model loading when an immutable
local model, container, dataset, or trace-plan prerequisite is missing.

Plausible wrong implementations:
- Treat a config-only Hugging Face cache directory as a complete model.
- Discover missing ShareGPT or trace data only after loading the GPU model.
- Import the NumPy-heavy campaign stack with bare login-node Python.
- Submit part of the parallel suite before checking all shared inputs.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
JOBS = ROOT / "profiling" / "jobs"


def test_stage_marker_is_written_and_required_at_both_boundaries():
    stage = (JOBS / "stage_models.sh").read_text()
    submit = (JOBS / "submit_campaign.sh").read_text()
    sbatch = (JOBS / "campaign.sbatch").read_text()
    assert '".powertrace-stage-complete"' in stage
    assert ".powertrace-stage-complete" in submit
    assert ".powertrace-stage-complete" in sbatch
    assert '"*.py"' in stage and '"*.jinja"' in stage


def test_submit_parses_json_without_importing_scientific_stack():
    submit = (JOBS / "submit_campaign.sh").read_text()
    assert "python3 -m profiling.jobs.campaign_config" not in submit
    assert "json.load(open(sys.argv[1]))" in submit


def test_campaign_checks_data_before_live_orchestration():
    sbatch = (JOBS / "campaign.sbatch").read_text()
    launch = sbatch.index("bash profiling/jobs/run_campaign.sh")
    assert sbatch.index('test -s "$SHAREGPT_DATASET_PATH"') < launch
    assert sbatch.index('test -s "$TRACE_PLAN"') < launch
    assert "HF_DATASETS_OFFLINE=1" in sbatch


def test_server_lifecycle_exports_launch_and_ready_epochs():
    lifecycle = (JOBS / "server_lifecycle.sh").read_text()
    assert 'POWERTRACE_SERVER_LAUNCH_EPOCH_S="$(date +%s)"' in lifecycle
    assert 'export POWERTRACE_SERVER_LAUNCH_EPOCH_S' in lifecycle
    assert 'POWERTRACE_SERVER_READY_EPOCH_S="$(date +%s)"' in lifecycle
    assert 'export POWERTRACE_SERVER_READY_EPOCH_S' in lifecycle


def test_parallel_suite_preflights_every_shared_input_before_first_submit():
    script = (JOBS / "submit_expansion_jobs.sh").read_text()
    first_submit = script.index(
        "submit_a100 profiling/campaigns/validate_qwen3-8b_a100.json"
    )
    for requirement in (
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "moe_routing_samples.jsonl",
        "tracelab_code.json",
        "burstgpt_10min.json",
        'check_plan "$REPO/data/trace_plans/burstgpt_10min.json"',
        "google/gemma-4-26B-A4B-it",
    ):
        assert script.index(requirement) < first_submit
    assert script.count("submit_a100 profiling/campaigns/") == 8
    assert script.count("submit_h100 profiling/campaigns/") == 3
