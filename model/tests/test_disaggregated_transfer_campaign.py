"""
Claim:
The transfer campaign sends a pre-tokenized immutable request plan at absolute
deadlines phase-locked to the raw 250 ms meter, rejects queued cells, and
isolates NIXL handoff from frozen decoder compute.

Plausible wrong implementations:
- Regenerate a nominally repeated workload instead of reusing identical bytes.
- Sleep between requests and accumulate client/runtime drift.
- Round traffic to wall-clock quarters unrelated to the running meter.
- Accept a queue with a low median but a large p95.
- Treat decode first-byte latency as pure NIXL and double-count first-token work.
- Add state fields while dropping timed raw power semantics.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from profiling.client.power_logger import POWER_PROFILES, nvidia_smi_query_command
from profiling.disaggregated_prefill.planned_workload import (
    deadline_delay,
    make_plan,
    plan_sha256,
    validate_plan,
)
from profiling.disaggregated_prefill.transfer_campaign import (
    cells,
    fit_nixl_handoff_delay,
    phase_locked_origin,
    run_metadata,
    validate_meter_phase,
    validate_unsaturated,
)

ROOT = Path(__file__).resolve().parents[2]


def test_plan_hash_binds_prompt_lengths_outputs_and_offsets():
    requests = [
        {"prompt": "alpha", "prompt_len": 3, "output_len": 4},
        {"prompt": "beta", "prompt_len": 5, "output_len": 6},
    ]
    plan = make_plan(requests, interval_s=2.0, model="m", seed=17)

    assert [row["offset_s"] for row in plan["requests"]] == [0.0, 2.0]
    assert validate_plan(plan) == plan_sha256(plan)
    changed = json.loads(json.dumps(plan))
    changed["requests"][1]["prompt"] = "other"
    with pytest.raises(ValueError, match="hash"):
        validate_plan(changed)


def test_absolute_deadlines_do_not_accumulate_previous_lateness():
    assert deadline_delay(100.0, 2.0, now_monotonic=101.25) == 0.75
    assert deadline_delay(100.0, 4.0, now_monotonic=103.50) == 0.50
    assert deadline_delay(100.0, 4.0, now_monotonic=104.25) == 0.0


def test_transfer_cells_are_one_probe_calibration_and_exact_heldout_replay():
    assert cells() == [
        ("stage-probe", 0.2, 24, 7, "stage_probe", "probe"),
        ("rate-0p5-calibration", 0.5, 150, 11, "calibration", "calibration"),
        ("rate-0p5-heldout", 0.5, 150, 17, "evaluation", "heldout"),
        ("rate-0p5-replay", 0.5, 150, 17, "evaluation", "heldout"),
    ]


def test_traffic_origin_is_locked_to_the_running_meter_not_wall_quarters():
    origin = phase_locked_origin(
        first_midpoint_epoch_s=1000.113,
        now_epoch_s=1002.0,
        minimum_lead_s=30.0,
    )
    assert origin == pytest.approx(1032.113)
    validate_meter_phase(
        [origin - 0.25, origin, origin + 0.25, origin + 0.5],
        origin,
    )
    with pytest.raises(ValueError, match="phase"):
        validate_meter_phase(
            [origin + 0.08, origin + 0.33, origin + 0.58, origin + 0.83],
            origin,
        )


def test_queue_gate_rejects_a_quiet_median_with_excessive_tail(tmp_path):
    path = tmp_path / "engine_prefill.csv"
    rows = [
        "timestamp,num_requests_waiting,prompt_tokens_total",
        "10.0,0,0",
        "11.0,0,100",
        "12.0,0,200",
        "13.0,2,300",
        "14.0,2,400",
    ]
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="p95"):
        validate_unsaturated(path, traffic_start=10.0, traffic_end=14.0)


def test_nixl_delay_subtracts_frozen_first_iteration():
    assert fit_nixl_handoff_delay(
        [0.090, 0.092, 0.094],
        frozen_first_iteration_s=0.010,
    ) == pytest.approx(0.082)


def test_timed_state_profile_preserves_raw_power_and_query_bounds():
    fields = POWER_PROFILES["core_timed_state"]
    assert fields[:3] == ("timestamp", "query.start", "query.end")
    assert "power.draw" in fields
    assert "power.limit" in fields
    assert "clocks_event_reasons.hw_thermal_slowdown" in fields
    command = nvidia_smi_query_command(fields, "GPU-a,GPU-b")
    assert "--id=GPU-a,GPU-b" in command
    assert "power.draw" in command[1]


def test_transfer_metadata_freezes_six_scalars_and_fixed_arrivals():
    metadata = run_metadata("GPU-p", "GPU-d", "/image.sif")
    assert metadata["protocol"] == "cache_disabled_phase_transfer_v2"
    assert metadata["workload"]["arrival_process"] == "fixed_deadline"
    assert metadata["workload"]["output_tokens"] == 256
    assert metadata["power_measurement"]["profile"] == "core_timed_state"
    assert metadata["analysis_protocol"]["target_scalar_count"] == 6
    source = (ROOT / "profiling/disaggregated_prefill/transfer_campaign.py").read_text()
    assert "statistics.median(visible) < 2 * INTERVAL_S" in source
    assert "below two meter intervals" in source
    assert metadata["analysis_protocol"]["calibration_cell"] == (
        "rate-0p5-calibration"
    )


def test_transfer_batch_uses_a100_partition_and_modern_python():
    batch = (
        ROOT / "profiling/jobs/disaggregated_transfer_gpt_oss_20b.sbatch"
    ).read_text()
    assert "#SBATCH --partition=ramr" in batch
    assert "#SBATCH --gpus=2" in batch
    assert "ml devel python/3.12.1" in batch
    assert "sys.version_info >= (3, 10)" in batch


def test_transfer_runner_uses_prebuilt_plans_and_phase_locked_traffic():
    runner = (
        ROOT / "profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh"
    ).read_text()
    assert runner.index("planned_workload.py generate") < runner.index(
        "power_logger.py"
    )
    assert "--profile core_timed_state" in runner
    assert "transfer_campaign.py origin" in runner
    assert "planned_workload.py run" in runner
    assert '--request-plan "$PLAN"' in runner
    assert '--traffic-start "$DIR/traffic_start_epoch_s"' in runner
    assert '--plan "$PLAN"' in runner
    assert "benchmark_serving.py" not in runner
    assert '${METADATA_ARG:+"$METADATA_ARG"}' in runner
    assert '${PLAN_ARG:+"$PLAN_ARG"}' in runner
    assert 'MODE_ARG=()' not in runner
    assert 'PLAN_ARGS=()' not in runner
    assert '["sha256"]' in runner
    assert '[\\"sha256\\"]' not in runner


def test_transfer_runner_buffers_power_off_lustre():
    """4 Hz flushes to Lustre stall for seconds and break the sampling contract."""
    runner = (
        ROOT / "profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh"
    ).read_text()
    assert 'PWR="$L_SCRATCH/power-$TAG.csv"' in runner
    assert '> "$PWR" &' in runner
    assert 'cp "$PWR" "$DIR/power.csv"' in runner
    assert '> "$DIR/power.csv" &' not in runner
