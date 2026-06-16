"""Unit tests for the campaign loader/validator (CAMPAIGN.md §5-F)."""

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


def test_validate_requires_workload(tmp_path):
    bad = tmp_path / "v.json"
    bad.write_text(json.dumps({
        "hardware": "H100", "model": "x", "campaign_type": "validate",
        "server": {"tp": 1},
    }))
    with pytest.raises(cc.CampaignError):
        cc.load_campaign(bad)


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
    # context holds need max_model_len >= the largest context (131072) + margin
    import re
    mml = int(re.search(r"--max-model-len (\d+)", serve).group(1))
    assert mml >= 131072


def test_probe_commands_nonempty_for_probe_campaign():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    cmds = cc.probe_commands(c, 8)
    assert len(cmds) == len(c["probes"])
    assert all(cmd.startswith("python -m profiling.probes.") for cmd in cmds)


def test_validate_run_command_carries_campaign_max_model_len():
    """run-cmd forwards the campaign's max_model_len so length pruning tracks the
    served context for whatever model size the campaign uses (same value the
    server is launched with -> they can never disagree)."""
    c = cc.load_campaign(CAMPAIGNS_DIR / "validate_qwen3-8b_a100.json")
    mml = c["server"]["max_model_len"]
    cmd = cc.run_command(c, c["server"]["tp"])
    assert cmd.startswith("python -m profiling.probes.validate_run")
    assert f"--max-model-len {mml}" in cmd
    assert f"--max-model-len {mml}" in cc.serve_command(c, c["server"]["tp"])
    assert f"--dataset {c['workload']['dataset']}" in cmd


def test_run_command_rejects_probe_campaign():
    c = cc.load_campaign(CAMPAIGNS_DIR / "h100_tier1_llama70b.json")
    with pytest.raises(cc.CampaignError):
        cc.run_command(c, 8)
