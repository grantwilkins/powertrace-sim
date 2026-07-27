"""
Claim:
Campaign JSONs isolate model, arrival-rate, and arrival-pattern comparisons and
thread every declared workload parameter into the live command.

Plausible wrong implementations:
- Accept a zero or negative scalar rate even though only rate lists are checked.
- Ignore Gamma burstiness and silently run every workload as Poisson.
- Change prompts, server settings, or hardware alongside the intended comparison.
- Put multiple scientific comparisons into one sequential Slurm job.
"""

import json
from pathlib import Path

import pytest

import campaign_config as cc

CAMPAIGNS_DIR = Path(cc.REPO_ROOT) / "profiling" / "campaigns"
ALL_CAMPAIGNS = sorted(CAMPAIGNS_DIR.glob("*.json"))


def test_campaigns_dir_populated():
    assert ALL_CAMPAIGNS, "no campaign JSON files found"


@pytest.mark.parametrize("path", ALL_CAMPAIGNS, ids=lambda p: p.name)
def test_load_valid(path):
    c = cc.load_campaign(path)
    assert c["hardware"] in ("A100", "H100")
    assert c["model"]
    assert "tp" in c["server"]
    assert c["gpus_per_node"] == max(cc.tp_degrees(c))


def test_visible_gpu_count_is_derived_from_largest_tp(tmp_path):
    campaign = tmp_path / "tp4.json"
    campaign.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "tier1",
        "server": {"tp": 4}, "tp_pair": [4, 2],
        "probes": ["decode_staircase"],
    }))
    c = cc.load_campaign(campaign)
    assert c["gpus_per_node"] == 4
    assert "--gpus-per-node 4" in cc.probe_commands(c, 4)[0]


def test_job_port_is_shared_by_server_and_every_client(monkeypatch):
    monkeypatch.setenv("POWERTRACE_PORT", "23456")
    campaigns = [
        "sealed_burstgpt_qwen3-8b_a100.json",
        "sealed_openhands_qwen3-8b_a100.json",
        "sealed_qwen3-14b_a100.json",
    ]
    for name in campaigns:
        campaign = cc.load_campaign(CAMPAIGNS_DIR / name)
        tp = campaign["server"]["tp"]
        regime = cc.regimes(campaign)[0]
        assert "--port 23456" in cc.serve_command(campaign, tp, False)
        assert "--base-url http://localhost:23456/v1" in cc.run_command(
            campaign, tp, regime
        )


def test_invalid_job_port_is_rejected(monkeypatch):
    monkeypatch.setenv("POWERTRACE_PORT", "70000")
    with pytest.raises(cc.CampaignError, match="POWERTRACE_PORT"):
        cc.server_port()


def test_campaign_config_keeps_submission_metadata_stdlib_only():
    text = Path(cc.__file__).read_text()
    assert "evidence_contract" not in text
    assert cc.KNOWN_EVIDENCE_PROFILES == {"core", "measured_ledger"}


def test_known_probes_match_schedule_builders():
    """Drift guard: campaign_config.KNOWN_PROBES mirrors schedule.BUILDERS."""
    import schedule
    assert cc.KNOWN_PROBES == set(schedule.BUILDERS)


def test_rejects_unknown_probe(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "hardware": "H100", "model": "x", "campaign_type": "tier1",
        "server": {"tp": 8}, "probes": ["decode_staircase", "not_a_probe"],
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


def test_agentic_requires_sessions(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({
        "hardware": "H100", "model": "x", "campaign_type": "agentic",
        "server": {"tp": 1},
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


def test_agentic_campaign_loads():
    c = cc.load_campaign(CAMPAIGNS_DIR / "agentic_qwen3-8b.json")
    assert c["campaign_type"] == "agentic"
    assert "sessions" in c and c["sessions"]["n_sessions"] > 0
    assert cc.regimes(c) == [{"prefix_cache": False}, {"prefix_cache": True}]


def test_agentic_regime_flags_agree_per_index():
    """The server's --enable-prefix-caching and the run's --prefix-cache are both
    derived from the same regime index, so they can never disagree."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "agentic_qwen3-8b_a100.json")
    for idx, want in enumerate([False, True]):
        reg = cc.regimes(c)[idx]
        serve = cc.serve_command(c, 1, reg.get("prefix_cache"))
        run = cc.run_command(c, 1, reg)
        assert ("--enable-prefix-caching" in serve) is want
        assert ("--enable-prompt-tokens-details" in serve) is want
        assert ("--no-enable-prefix-caching" in serve) is not want
        assert ("--prefix-cache" in run) is want
    # replay campaign threads the corpus + gap params into the run command
    run0 = cc.run_command(c, 1, cc.regimes(c)[0])
    assert "--replay --corpus swe_smith" in run0
    assert "gap_params.json" in run0


def test_agentic_rejects_derived_prefix_cache_fields(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "agentic",
        "server": {"tp": 1, "enable_prefix_caching": True},
        "sessions": {"n_sessions": 4, "regimes": [{"prefix_cache": True}]},
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


def test_agentic_requires_regimes(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "agentic",
        "server": {"tp": 1}, "sessions": {"n_sessions": 4},
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


def test_trace_replay_command_binds_plan_cache_and_power_profile():
    off_campaign = cc.load_campaign(
        CAMPAIGNS_DIR / "trace_replay_qwen3-8b_a100_cache_off.json"
    )
    on_campaign = cc.load_campaign(
        CAMPAIGNS_DIR / "trace_replay_qwen3-8b_a100_cache_on.json"
    )
    assert off_campaign["server"] == on_campaign["server"]
    off = cc.regimes(off_campaign)[0]
    on = cc.regimes(on_campaign)[0]
    assert off == {"prefix_cache": False}
    assert on == {"prefix_cache": True}
    off_command = cc.run_command(off_campaign, 1, off)
    on_command = cc.run_command(on_campaign, 1, on)
    assert "--trace-plan data/trace_plans/tracelab_code.json" in off_command
    assert "--power-profile tp8_state" in off_command
    assert "--pre-idle-s 60.0" in off_command
    assert "--scheduling-policy sync" in off_command
    assert "--prefix-cache" not in off_command
    assert "--prefix-cache" in on_command


def test_async_scheduling_is_launched_and_recorded(tmp_path):
    path = tmp_path / "async.json"
    path.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "tier1",
        "server": {"tp": 1, "scheduling_policy": "async"},
        "probes": ["idle_hold"],
    }))
    campaign = cc.load_campaign(path)
    assert "--async-scheduling" in cc.serve_command(campaign, 1)
    assert "--scheduling-policy async" in cc.probe_commands(campaign, 1)[0]


def test_tp8_state_replay_uses_tp4_length_marks_without_repeating_tp4():
    tp8 = cc.load_campaign(
        CAMPAIGNS_DIR / "h100_tp8_state_diagnostic.json"
    )
    tp4 = cc.load_campaign(CAMPAIGNS_DIR / "h100_tp4_state_control.json")
    assert tp8["power_profile"] == tp4["power_profile"] == "tp8_state"
    assert cc.tp_degrees(tp8) == [8]
    assert cc.tp_degrees(tp4) == [4]
    assert tp8["trace"]["plan"].endswith("h100_tp4_state_marks.json")
    tp8_command = cc.run_command(tp8, 8, cc.regimes(tp8)[0])
    assert "--pre-idle-s 180.0" in tp8_command
    assert "--power-profile tp8_state" in tp8_command
    assert "--trace-plan data/trace_plans/h100_tp4_state_marks.json" in tp8_command
    assert "Do not rerun" in tp4["_note"]


@pytest.mark.parametrize(
    ("campaign_name", "plan_name", "idle_s"),
    [
        ("h100_405b_rate4_exact_replay.json", "h100_405b_rate4_200.json", 90.0),
        (
            "h100_qwen3_8b_idle_decomposition.json",
            "h100_qwen3_8b_rate4_200.json",
            60.0,
        ),
    ],
)
def test_conditional_exact_replays_bind_frozen_plans(
    campaign_name, plan_name, idle_s
):
    campaign = cc.load_campaign(CAMPAIGNS_DIR / campaign_name)
    command = cc.run_command(
        campaign, campaign["server"]["tp"], cc.regimes(campaign)[0]
    )
    assert f"--trace-plan data/trace_plans/{plan_name}" in command
    assert f"--pre-idle-s {idle_s}" in command
    assert "--scheduling-policy sync" in command


def test_validate_single_rate_becomes_one_explicit_regime():
    c = cc.load_campaign(CAMPAIGNS_DIR / "validate_qwen3-8b_a100.json")
    assert cc.regimes(c) == [{"request_rate": 4.0}]
    assert c["workload"]["burstiness"] == 1.0


def test_validate_multi_rate_regimes_preserve_one_seed(tmp_path):
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({
        "hardware": "H100", "model": "x", "campaign_type": "validate",
        "server": {"tp": 8}, "evidence_profile": "measured_ledger",
        "workload": {
            "dataset": "sharegpt", "num_prompts": 20,
            "request_rates": [1.0, 2.0], "seed": 17,
        },
    }))
    campaign = cc.load_campaign(path)
    assert cc.regimes(campaign) == [
        {"request_rate": 1.0}, {"request_rate": 2.0}
    ]
    commands = [
        cc.run_command(campaign, 8, regime) for regime in cc.regimes(campaign)
    ]
    assert "--request-rate 1.0 --burstiness 1.0 --seed 17" in commands[0]
    assert "--request-rate 2.0 --burstiness 1.0 --seed 17" in commands[1]
    assert all("--evidence-profile measured_ledger" in command for command in commands)


def test_validate_requires_workload(tmp_path):
    bad = tmp_path / "v.json"
    bad.write_text(json.dumps({
        "hardware": "H100", "model": "x", "campaign_type": "validate",
        "server": {"tp": 1},
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_rate", 0.0, "request_rate must be positive"),
        ("burstiness", -0.25, "burstiness must be positive"),
    ],
)
def test_validate_rejects_nonpositive_arrival_parameters(
    tmp_path, field, value, message
):
    workload = {
        "dataset": "sharegpt", "num_prompts": 20,
        "request_rate": 2.5, "burstiness": 1.0,
    }
    workload[field] = value
    path = tmp_path / "invalid_arrival.json"
    path.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "validate",
        "server": {"tp": 1}, "workload": workload,
    }))
    with pytest.raises(cc.CampaignError, match=message):
        cc.load_campaign(path)


def test_qwen_transfer_and_arrival_matrix_change_one_axis():
    a100 = cc.load_campaign(CAMPAIGNS_DIR / "validate_qwen3-8b_a100.json")
    h100 = cc.load_campaign(CAMPAIGNS_DIR / "validate_qwen3-8b.json")
    off_grid = cc.load_campaign(
        CAMPAIGNS_DIR / "arrival_rate_qwen3-8b_a100_r2p5.json"
    )
    bursty = cc.load_campaign(
        CAMPAIGNS_DIR / "arrival_pattern_qwen3-8b_a100_bursty.json"
    )
    smooth = cc.load_campaign(
        CAMPAIGNS_DIR / "arrival_pattern_qwen3-8b_a100_smooth.json"
    )

    assert a100["model"] == h100["model"]
    assert a100["server"] == h100["server"]
    assert a100["workload"] == h100["workload"]
    assert {a100["hardware"], h100["hardware"]} == {"A100", "H100"}

    rate_baseline = dict(a100["workload"])
    rate_off_grid = dict(off_grid["workload"])
    assert rate_baseline.pop("request_rate") == 4.0
    assert rate_off_grid.pop("request_rate") == 2.5
    assert rate_baseline == rate_off_grid

    for pattern, expected_shape in ((bursty, 0.25), (smooth, 4.0)):
        pattern_workload = dict(pattern["workload"])
        poisson_workload = dict(off_grid["workload"])
        assert pattern_workload.pop("burstiness") == expected_shape
        assert poisson_workload.pop("burstiness") == 1.0
        assert pattern_workload == poisson_workload


def test_arrival_matrix_is_one_regime_per_independent_job():
    paths = [
        "validate_qwen3-8b_a100.json",
        "validate_qwen3-8b.json",
        "arrival_rate_qwen3-8b_a100_r2p5.json",
        "arrival_pattern_qwen3-8b_a100_bursty.json",
        "arrival_pattern_qwen3-8b_a100_smooth.json",
    ]
    assert all(
        len(cc.regimes(cc.load_campaign(CAMPAIGNS_DIR / path))) == 1
        for path in paths
    )


def test_tier1_set_declares_both_anchors_with_correct_tp():
    llama = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    qwen = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_qwen3-235b.json")
    # Llama-3.1-70B anchor carries the TP4<->TP8 e_comm pair
    assert llama["model"].endswith("Llama-3.1-70B-Instruct")
    assert cc.tp_degrees(llama) == [8, 4]
    # Qwen3-235B MoE anchor is TP8-only (size-forced), no tp_pair
    assert qwen["model"] == "Qwen/Qwen3-235B-A22B"
    assert cc.tp_degrees(qwen) == [8]


def test_minimax_partial_tier1_probe_subset():
    m = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_minimax-m2.7.json")
    assert m["campaign_type"] == "tier1_partial"
    assert set(m["probes"]) == {"decode_staircase", "prefill_staircase", "context_holds"}


def test_serve_command_includes_knobs():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    cmd = cc.serve_command(c, 8)
    assert "vllm serve meta-llama/Llama-3.1-70B-Instruct" in cmd
    assert "--tensor-parallel-size 8" in cmd
    assert "--max-num-seqs 256" in cmd
    assert "--enable-chunked-prefill" in cmd


def test_prefill_probe_disables_chunked_prefill_at_run_time():
    """The prefill probe forces chunked-prefill OFF regardless of server default."""
    import schedule
    s = schedule.build_prefill_staircase()
    assert s.server_overrides["enable_chunked_prefill"] is False


def test_probe_serve_disables_chunked_prefill_for_prefill_staircase():
    """Each probe gets a server matching its needs: prefill -> chunked-prefill OFF."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    decode_serve = cc.probe_serve_command(c, "decode_staircase", 8)
    prefill_serve = cc.probe_serve_command(c, "prefill_staircase", 8)
    assert "--enable-chunked-prefill" in decode_serve       # default ON
    assert "--enable-chunked-prefill" not in prefill_serve   # OFF for prefill probe


def test_probe_serve_raises_max_model_len_for_context_holds():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    serve = cc.probe_serve_command(c, "context_holds", 8)
    # context holds keep a long-prefix tier while leaving tokenizer headroom.
    import re
    mml = int(re.search(r"--max-model-len (\d+)", serve).group(1))
    assert mml >= 131072


def test_probe_commands_nonempty_for_probe_campaign():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    cmds = cc.probe_commands(c, 8)
    assert len(cmds) == len(c["probes"])
    # Direct-script form (not `python -m`): probe drivers use top-level imports and
    # there is no profiling/__init__.py, so only the script-path form resolves.
    assert all(cmd.startswith("python3 profiling/probes/") and ".py " in cmd
               for cmd in cmds)


def test_known_probe_scripts_exist():
    for probe in cc.KNOWN_PROBES:
        assert (cc.REPO_ROOT / "profiling" / "probes" / f"{probe}.py").is_file()


def test_probe_commands_carry_out_root(monkeypatch):
    """Every emitted probe command carries --out-root, sourced from $RUNS so the
    sbatch can point bundles at $SCRATCH (never the repo-local default)."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    monkeypatch.setenv("RUNS", "/scratch/users/x/ptsim/runs")
    cmds = cc.probe_commands(c, 8)
    assert all("--out-root /scratch/users/x/ptsim/runs" in cmd for cmd in cmds)
    monkeypatch.delenv("RUNS", raising=False)
    assert "--out-root data/runs" in cc.probe_commands(c, 8)[0]


def test_tp_pair_second_leg_runs_declared_sync_subset():
    """The Llama TP pair repeats only probes that identify TP synchronization."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    primary, second = cc.tp_degrees(c)  # [8, 4]
    assert cc.probes_for_tp(c, primary) == c["probes"]
    assert cc.probes_for_tp(c, second) == [
        "decode_staircase", "decode_context_grid", "prefill_staircase"
    ]
    # probe_commands honours the per-leg selection
    assert len(cc.probe_commands(c, second)) == 3


def test_tp_pair_probes_default_when_absent():
    """Configs without an explicit tp_pair_probes get the decode+prefill default."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "a100_tier1_llama70b.json")
    assert c["tp_pair_probes"] == ["decode_staircase", "prefill_staircase"]


def test_transfer_campaign_profiles_map_to_their_ledger_axes():
    dense = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    moe = cc.load_campaign(CAMPAIGNS_DIR / "a100_iteration_gpt-oss-120b.json")
    assert dense["evidence_profile"] == "measured_ledger"
    assert moe["evidence_profile"] == "measured_ledger"
    assert "decode_context_grid" in dense["probes"]
    assert "decode_context_grid" in moe["probes"]


def test_gpt_oss_matrix_separates_tp_from_model_scale():
    small = cc.load_campaign(CAMPAIGNS_DIR / "a100_iteration_gpt-oss-20b.json")
    large = cc.load_campaign(CAMPAIGNS_DIR / "a100_iteration_gpt-oss-120b.json")
    assert cc.tp_degrees(small) == [2, 4]
    assert cc.probes_for_tp(small, 4) == [
        "decode_staircase", "decode_context_grid",
    ]
    assert large["server"]["tp"] == 4

    small_real = cc.load_campaign(CAMPAIGNS_DIR / "a100_hardcells_gpt-oss-20b.json")
    large_real = cc.load_campaign(CAMPAIGNS_DIR / "a100_hardcells_gpt-oss-120b.json")
    for campaign in (small_real, large_real):
        assert campaign["server"]["tp"] == 4
        assert campaign["workload"]["request_rates"] == [1.0, 2.0]
        assert campaign["workload"]["seed"] == 20260712


def test_llama_matrix_has_matched_scale_rates_and_within_model_tp_rates():
    small = cc.load_campaign(CAMPAIGNS_DIR / "h100_hardcells_llama70b.json")
    large = cc.load_campaign(CAMPAIGNS_DIR / "h100_hardcells_llama405b.json")
    assert small["workload"]["request_rates"] == [1.0, 2.0, 4.0]
    assert large["workload"]["request_rates"] == [1.0, 2.0]
    assert set(large["workload"]["request_rates"]) <= set(
        small["workload"]["request_rates"]
    )
    assert small["workload"]["seed"] == large["workload"]["seed"]
    assert cc.tp_degrees(small) == [8, 4]


def test_llama405b_campaigns_use_fp8_checkpoint():
    for name in ("h100_hardcells_llama405b.json", "h100_sealed_llama405b.json"):
        campaign = cc.load_campaign(CAMPAIGNS_DIR / name)
        server = campaign["server"]
        assert server["quantization"] == "compressed-tensors"
        assert server["dtype_hint"] == "fp8"

        serve = cc.serve_command(campaign, 8)
        run = cc.run_command(campaign, 8, cc.regimes(campaign)[0])
        assert campaign["model"] == "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8"
        assert "--quantization compressed-tensors" in serve
        assert "--quantization compressed-tensors" in run
        assert "--dtype-hint fp8" in run


def test_sealed_campaign_role_is_emitted_and_commands_keep_fresh_seed():
    campaign = cc.load_campaign(CAMPAIGNS_DIR / "h100_sealed_llama405b.json")
    assert campaign["validation_role"] == "sealed"
    command = cc.run_command(campaign, 8, cc.regimes(campaign)[0])
    assert "--validation-role sealed" in command
    assert "--seed 2026071201" in command


def test_final_sealed_campaigns_are_minimal_and_explicit():
    burst = cc.load_campaign(
        CAMPAIGNS_DIR / "sealed_burstgpt_qwen3-8b_a100.json"
    )
    burst_regimes = cc.regimes(burst)
    assert len(burst_regimes) == 3
    assert len({
        regime["plan"] for regime in burst_regimes
    }) == 3
    assert all(
        "--validation-role sealed" in cc.run_command(burst, 1, regime)
        for regime in burst_regimes
    )

    agentic = cc.load_campaign(
        CAMPAIGNS_DIR / "sealed_openhands_qwen3-8b_a100.json"
    )
    commands = [
        cc.run_command(agentic, 1, regime)
        for regime in cc.regimes(agentic)
    ]
    assert len(commands) == 6
    assert agentic["slurm_time"] == "48:00:00"
    assert "--dataset-revision aa8977805b4cefd317001d80ddf1ad52790e9d23" \
        in commands[0]
    assert "--pack-index 0 --pack-count 3" in commands[0]
    assert "--pack-index 2 --pack-count 3" in commands[-1]
    assert len(agentic["sessions"]["expected_plan_sha256"]) == 3
    assert agentic["sessions"]["max_turn_wait_s"] == 60
    assert agentic["sessions"]["max_session_wait_s"] == 120

    for name in (
        "sealed_qwen3-14b_a100.json",
        "sealed_qwen3-30b-a3b_h100.json",
    ):
        campaign = cc.load_campaign(CAMPAIGNS_DIR / name)
        assert campaign["validation_role"] == "sealed"
        assert len(cc.regimes(campaign)) == 1


def test_transfer_ci_campaigns_have_six_units_after_reuse():
    burst = cc.load_campaign(
        CAMPAIGNS_DIR / "transfer_ci_burstgpt_qwen3-8b_a100.json"
    )
    assert len(cc.regimes(burst)) == 4

    openhands = cc.load_campaign(
        CAMPAIGNS_DIR / "transfer_ci_openhands_qwen3-8b_a100.json"
    )
    assert len(cc.regimes(openhands)) == 3
    assert openhands["sessions"]["session_offset"] == 24
    assert "--session-offset 24" in cc.run_command(
        openhands, 1, cc.regimes(openhands)[0]
    )

    dense = cc.load_campaign(
        CAMPAIGNS_DIR / "transfer_ci_qwen3-14b_a100.json"
    )
    assert len(cc.regimes(dense)) == 5

    moe = cc.load_campaign(
        CAMPAIGNS_DIR / "transfer_ci_qwen3-30b-a3b_a100.json"
    )
    assert moe["server"]["tp"] == 2
    assert len(cc.regimes(moe)) == 6
def test_tp_pair_probes_rejects_unknown(tmp_path):
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({
        "hardware": "A100", "model": "x", "campaign_type": "tier1",
        "server": {"tp": 4}, "probes": ["decode_staircase"],
        "tp_pair": [4, 2], "tp_pair_probes": ["decode_staircase", "nope"],
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


def test_validate_run_command_carries_campaign_max_model_len():
    """run-cmd forwards the campaign's max_model_len so length pruning tracks the
    served context for whatever model size the campaign uses (same value the
    server is launched with -> they can never disagree)."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "validate_qwen3-8b_a100.json")
    mml = c["server"]["max_model_len"]
    cmd = cc.run_command(c, c["server"]["tp"])
    assert cmd.startswith("python3 profiling/probes/validate_run.py")
    assert f"--max-model-len {mml}" in cmd
    assert f"--max-model-len {mml}" in cc.serve_command(c, c["server"]["tp"])
    assert f"--dataset {c['workload']['dataset']}" in cmd
    assert "--burstiness 1.0" in cmd


def test_run_command_rejects_probe_campaign():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    with pytest.raises(cc.CampaignError):
        cc.run_command(c, 8)
