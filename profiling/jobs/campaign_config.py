"""Campaign JSON loader / validator + plan emitter (CAMPAIGN.md §5-F).

JSON (not YAML) to avoid a new ``pyyaml`` dependency and to match ``manifest.json``.
Pure: loads a campaign file, validates it, and emits the server-launch command and
per-probe invocation lines. ``run_campaign.sh`` consumes these; ``--dry-run``
prints the full plan without launching anything (the no-GPU verification path).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_EVIDENCE_PROFILES = {"core", "measured_ledger"}
KNOWN_POWER_PROFILES = {"core", "tp8_state"}

# Mirrors profiling/probes/schedule.BUILDERS; a test asserts they stay in sync.
KNOWN_PROBES = {
    "idle_hold", "decode_staircase", "prefill_staircase",
    "context_holds", "decode_context_grid", "transients", "mixed_grid",
}
CAMPAIGN_TYPES = {
    "tier1", "tier1_partial", "tier2", "validate", "agentic",
    "trace_replay", "roofline",
}
# The tp_pair second leg only repeats decode+prefill (CAMPAIGN.md §3): those two
# probes identify e_comm; re-running the full set at the 2nd TP is wasted budget.
DEFAULT_TP_PAIR_PROBES = ["decode_staircase", "prefill_staircase"]


def out_root() -> str:
    """Bundle output root. The sbatch exports ``RUNS=$SCRATCH/...``; off-cluster
    (dry run / tests) it falls back to the repo-local ``data/runs``. Never $HOME
    on Sherlock — the sbatch is responsible for pointing RUNS at $SCRATCH."""
    return os.environ.get("RUNS") or "data/runs"


def server_port() -> int:
    """Port shared by one campaign's server, readiness checks, and clients."""
    port = int(os.environ.get("POWERTRACE_PORT", "8000"))
    if not 1024 <= port <= 65535:
        raise CampaignError(f"POWERTRACE_PORT must be in [1024, 65535], got {port}")
    return port


def base_url() -> str:
    return f"http://localhost:{server_port()}/v1"


class CampaignError(ValueError):
    pass


def load_campaign(path) -> dict:
    """Load and validate a campaign JSON file."""
    c = json.loads(Path(path).read_text())
    c["_campaign_id"] = Path(path).stem
    _validate(c, path)
    return _with_defaults(c)


def _validate(c: dict, path) -> None:
    where = f"{path}: "
    for key in ("hardware", "model", "campaign_type", "server"):
        if key not in c:
            raise CampaignError(f"{where}missing required key '{key}'")
    if c["hardware"] not in ("A100", "H100"):
        raise CampaignError(f"{where}hardware must be A100 or H100")
    if c["campaign_type"] not in CAMPAIGN_TYPES:
        raise CampaignError(f"{where}unknown campaign_type '{c['campaign_type']}'")
    if "tp" not in c["server"]:
        raise CampaignError(f"{where}server.tp is required")
    if c.get("evidence_profile", "core") not in KNOWN_EVIDENCE_PROFILES:
        raise CampaignError(f"{where}unknown evidence_profile {c.get('evidence_profile')!r}")
    if c.get("power_profile", "core") not in KNOWN_POWER_PROFILES:
        raise CampaignError(f"{where}unknown power_profile {c.get('power_profile')!r}")
    if c.get("validation_role", "development") not in {"development", "sealed"}:
        raise CampaignError(f"{where}validation_role must be development or sealed")
    if c["server"].get("scheduling_policy", "sync") not in {"sync", "async"}:
        raise CampaignError(f"{where}server.scheduling_policy must be sync or async")

    if c["campaign_type"] == "validate":
        if "workload" not in c:
            raise CampaignError(f"{where}validate campaigns need a 'workload' block")
        rates = c["workload"].get("request_rates")
        rate = c["workload"].get("request_rate")
        if (rates is None) == (rate is None):
            raise CampaignError(
                f"{where}validate workload needs exactly one of request_rate/request_rates"
            )
        if rates is not None and not (
            isinstance(rates, list) and rates
            and all(math.isfinite(float(value)) and float(value) > 0 for value in rates)
        ):
            raise CampaignError(f"{where}workload.request_rates must be positive")
        if rate is not None and not (
            math.isfinite(float(rate)) and float(rate) > 0
        ):
            raise CampaignError(f"{where}workload.request_rate must be positive")
        burstiness = float(c["workload"].get("burstiness", 1.0))
        if not math.isfinite(burstiness) or burstiness <= 0:
            raise CampaignError(f"{where}workload.burstiness must be positive")
    elif c["campaign_type"] in {"agentic", "trace_replay"}:
        block = "sessions" if c["campaign_type"] == "agentic" else "trace"
        ss = c.get(block)
        if not ss:
            raise CampaignError(
                f"{where}{c['campaign_type']} campaigns need a '{block}' block"
            )
        if c["campaign_type"] == "trace_replay" and not ss.get("plan"):
            raise CampaignError(f"{where}trace_replay trace.plan is required")
        if float(ss.get("pre_idle_s", 0.0)) < 0.0:
            raise CampaignError(f"{where}{block}.pre_idle_s must be non-negative")
        regs = ss.get("regimes")
        if not (isinstance(regs, list) and regs
                and all(isinstance(r, dict) and "prefix_cache" in r for r in regs)):
            raise CampaignError(
                f"{where}{block} needs a non-empty 'regimes' list, each "
                f"with a 'prefix_cache' boolean")
        if "enable_prefix_caching" in c["server"] or "prefix_cache" in ss:
            raise CampaignError(
                f"{where}{c['campaign_type']} prefix-caching is derived per regime; "
                f"drop server.enable_prefix_caching and {block}.prefix_cache")
    elif c["campaign_type"] == "roofline":
        probes = c.get("probes", [])
        if not probes:
            raise CampaignError(f"{where}roofline campaigns need a 'probes' list")
        unknown = set(probes) - KNOWN_PROBES
        if unknown:
            raise CampaignError(f"{where}unknown probe(s): {sorted(unknown)}")
        rf = c.get("roofline")
        if not isinstance(rf, dict):
            raise CampaignError(f"{where}roofline campaigns need a 'roofline' block")
        ss = rf.get("sessions")
        if not isinstance(ss, dict):
            raise CampaignError(f"{where}roofline.sessions is required")
        if c["server"].get("enable_prefix_caching") or ss.get("prefix_cache"):
            raise CampaignError(f"{where}first roofline campaign is cache-off only")
    else:
        probes = c.get("probes", [])
        if not probes:
            raise CampaignError(f"{where}{c['campaign_type']} needs a 'probes' list")
        unknown = set(probes) - KNOWN_PROBES
        if unknown:
            raise CampaignError(f"{where}unknown probe(s): {sorted(unknown)}")

    tp_pair = c.get("tp_pair")
    if tp_pair is not None:
        if not (isinstance(tp_pair, list) and len(tp_pair) == 2):
            raise CampaignError(f"{where}tp_pair must be a 2-element list")

    tpp = c.get("tp_pair_probes")
    if tpp is not None:
        if not isinstance(tpp, list):
            raise CampaignError(f"{where}tp_pair_probes must be a list")
        unknown = set(tpp) - KNOWN_PROBES
        if unknown:
            raise CampaignError(f"{where}unknown tp_pair_probes: {sorted(unknown)}")


def _defaults(d: dict, values: dict) -> None:
    for key, value in values.items():
        d.setdefault(key, value)


def _with_defaults(c: dict) -> dict:
    _defaults(c["server"], {
        "max_num_seqs": 256, "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True, "enable_prefix_caching": False,
        "kv_cache_dtype": "auto", "max_model_len": 131072,
        "quantization": None, "dtype_hint": None, "extra_args": [], "extra_env": {},
        "scheduling_policy": "sync",
        "weight_footprint_bytes": None,
        "embedding_bytes_per_param": None,
        "fp8_flop_frac": None,
    })
    # submit_campaign.sh allocates exactly the largest TP degree.  The logger
    # must describe that visible set, not the physical node's installed GPUs.
    c["gpus_per_node"] = max(tp_degrees(c))
    c.setdefault("probes", [])
    c.setdefault("evidence_profile", "core")
    c.setdefault("power_profile", "core")
    c.setdefault("validation_role", "development")
    c.setdefault("tp_pair_probes", list(DEFAULT_TP_PAIR_PROBES))
    if c["campaign_type"] == "validate":
        c["workload"].setdefault("burstiness", 1.0)
    if c["campaign_type"] == "roofline":
        r = c.setdefault("roofline", {})
        _defaults(r, {"window_s": 5.0, "contexts": [2048, 8192, 32768, 65536]})
        _defaults(r.setdefault("context_holds", {}), {"batch": 8, "output_len": 256})
        _defaults(r.setdefault("mixed_grid", {}), {
            "n_points": 24, "seed": 0, "prefill_min": 512,
            "prefill_max": 65536, "output_len": 512, "hold_s": 45.0,
        })
        _defaults(r.setdefault("sessions", {}), {
            "n_sessions": 16, "seed": 0, "concurrency": "auto",
            "min_turns": 8, "max_turns": 16, "prefix_tokens": 4096,
            "user_tokens_mean": 1024, "assistant_tokens_mean": 512,
            "gap_mean_s": 3.0, "gap_sigma": 0.8,
        })
    if c["campaign_type"] == "trace_replay":
        c["trace"].setdefault("pre_idle_s", 0.0)
    return c


def _env_prefix(env: dict) -> str:
    if not env:
        return ""
    parts = [f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items())]
    return "env " + " ".join(parts) + " "


def serve_command(c: dict, tp: int, prefix_cache=None) -> str:
    """The ``vllm serve`` command for this campaign at a given TP.

    ``prefix_cache`` overrides ``server.enable_prefix_caching`` when given — agentic
    campaigns pass the per-regime value so the launched server always matches the
    run's regime (the two can never disagree).
    """
    s = c["server"]
    pc = s["enable_prefix_caching"] if prefix_cache is None else prefix_cache
    # Pin the served name to the HF id: with HF_HUB_OFFLINE, vLLM 0.19 otherwise
    # registers the model under its local snapshot PATH, and the benchmark client
    # (which requests --model <hf id>) then 404s ("model not found").
    parts = [f"vllm serve {c['model']}", f"--served-model-name {c['model']}",
             f"--port {server_port()}",
             f"--tensor-parallel-size {tp}",
             f"--max-num-seqs {s['max_num_seqs']}",
             f"--max-num-batched-tokens {s['max_num_batched_tokens']}",
             f"--kv-cache-dtype {s['kv_cache_dtype']}",
             f"--max-model-len {s['max_model_len']}"]
    if s["enable_chunked_prefill"]:
        parts.append("--enable-chunked-prefill")
    if pc:
        parts.append("--enable-prefix-caching")
    if s["scheduling_policy"] == "async":
        parts.append("--async-scheduling")
    if s.get("quantization"):
        parts.append(f"--quantization {s['quantization']}")
    parts.extend(s["extra_args"])
    return _env_prefix(s.get("extra_env", {})) + " ".join(parts)


def _server_record_flags(s: dict) -> list[str]:
    flags = [f"--scheduling-policy {s['scheduling_policy']}"]
    if s.get("quantization"):
        flags.append(f"--quantization {s['quantization']}")
    if s.get("dtype_hint"):
        flags.append(f"--dtype-hint {s['dtype_hint']}")
    if s.get("weight_footprint_bytes") is not None:
        flags.append(f"--weight-footprint-bytes {s['weight_footprint_bytes']}")
    if s.get("embedding_bytes_per_param") is not None:
        flags.append(
            f"--embedding-bytes-per-param {s['embedding_bytes_per_param']}"
        )
    if s.get("fp8_flop_frac") is not None:
        flags.append(f"--fp8-flop-frac {s['fp8_flop_frac']}")
    return flags


def regimes(c: dict) -> list[dict]:
    """Prefix-cache regimes to run: one per regime for agentic, a single pass else."""
    if c["campaign_type"] in {"agentic", "trace_replay"}:
        block = "sessions" if c["campaign_type"] == "agentic" else "trace"
        return list(c[block]["regimes"])
    if c["campaign_type"] == "validate":
        workload = c["workload"]
        rates = workload.get("request_rates", [workload.get("request_rate")])
        return [{"request_rate": float(rate)} for rate in rates]
    return [{}]


def probes_for_tp(c: dict, tp: int) -> list[str]:
    """Probes to run at a given TP.

    The primary TP (``server.tp``) runs the full ``probes`` list. The tp_pair
    second leg runs only ``tp_pair_probes`` (decode+prefill by default) — those
    identify e_comm and re-running the full set there is wasted budget
    (CAMPAIGN.md §3). The subset is intersected with ``probes`` so the leg never
    runs a probe the campaign didn't declare.
    """
    if int(tp) == int(c["server"]["tp"]):
        return list(c["probes"])
    subset = c.get("tp_pair_probes") or DEFAULT_TP_PAIR_PROBES
    return [p for p in subset if p in c["probes"]]


def tp_degrees(c: dict) -> list[int]:
    """The TP degrees to sweep: server.tp plus any tp_pair second leg."""
    tps = [int(c["server"]["tp"])]
    pair = c.get("tp_pair")
    if pair:
        for t in pair:
            if int(t) not in tps:
                tps.append(int(t))
    return tps


def _build_probe_schedule(probe: str, c: dict):
    """Build the canonical schedule for a probe, including campaign knobs."""
    sys.path.insert(0, str(REPO_ROOT / "profiling" / "probes"))
    import schedule  # noqa: E402

    mns = int(c["server"]["max_num_seqs"])
    if c["campaign_type"] == "roofline":
        r = c["roofline"]
        if probe == "context_holds":
            ch = r["context_holds"]
            return schedule.build_context_holds(
                contexts=tuple(int(x) for x in r["contexts"]),
                batch=int(ch["batch"]),
                output_len=int(ch["output_len"]),
            )
        if probe == "mixed_grid":
            mg = r["mixed_grid"]
            return schedule.build_mixed_grid(
                n_points=int(mg["n_points"]),
                seed=int(mg["seed"]),
                decode_range=(1, mns),
                prefill_range=(int(mg["prefill_min"]), int(mg["prefill_max"])),
                hold_s=float(mg["hold_s"]),
                output_len=int(mg["output_len"]),
            )

    if probe == "decode_staircase":
        return schedule.build_decode_staircase(mns)
    if probe == "mixed_grid":
        return schedule.build_mixed_grid(decode_range=(1, mns))
    return schedule.BUILDERS[probe]()


def _schedule_overrides(probe: str, c: dict) -> dict:
    """Server overrides a probe requires (chunked-prefill OFF, long max_model_len).

    Built from the canonical ``schedule`` builders so the orchestrator launches a
    server matching each probe's needs, not a single shared server.
    """
    return _build_probe_schedule(probe, c).server_overrides


def probe_serve_command(c: dict, probe: str, tp: int) -> str:
    """``vllm serve`` command for a specific probe (merges its server overrides)."""
    ov = _schedule_overrides(probe, c)
    s = dict(c["server"])
    if "enable_chunked_prefill" in ov:
        s["enable_chunked_prefill"] = ov["enable_chunked_prefill"]
    if "max_model_len" in ov:
        s["max_model_len"] = max(int(s["max_model_len"]), int(ov["max_model_len"]))
    if "env" in ov:
        env = dict(s.get("extra_env", {}))
        env.update(ov["env"])
        s["extra_env"] = env
    return serve_command(dict(c, server=s), tp)


def _probe_extra_args(c: dict, probe: str) -> str:
    if c["campaign_type"] != "roofline":
        return ""
    r = c["roofline"]
    if probe == "context_holds":
        ch = r["context_holds"]
        contexts = " ".join(str(int(x)) for x in r["contexts"])
        return (
            f" --contexts {contexts}"
            f" --batch {int(ch['batch'])}"
            f" --output-len {int(ch['output_len'])}"
        )
    if probe == "mixed_grid":
        mg = r["mixed_grid"]
        return (
            f" --n-points {int(mg['n_points'])}"
            f" --seed {int(mg['seed'])}"
            f" --prefill-min {int(mg['prefill_min'])}"
            f" --prefill-max {int(mg['prefill_max'])}"
            f" --output-len {int(mg['output_len'])}"
            f" --hold-s {float(mg['hold_s'])}"
        )
    return ""


def probe_commands(c: dict, tp: int) -> list[str]:
    """Direct-script probe invocations for one TP.

    Emits ``python3 profiling/probes/<probe>.py …`` (run with cwd at the repo
    root) rather than ``python -m …``: the probe drivers use top-level imports
    (``from _cli import …``) and there is no ``profiling/__init__.py``, so only
    the direct-script form resolves — the same form the validate pipeline runs.
    ``run_campaign.sh`` prepends the container ``$APP`` prefix.
    """
    s = c["server"]
    common = (
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp} "
        f"--base-url {base_url()} "
        f"--gpus-per-node {c['gpus_per_node']} "
        f"--max-num-seqs {s['max_num_seqs']} "
        f"--max-model-len {s['max_model_len']} "
        f"--kv-cache-dtype {s['kv_cache_dtype']} "
        f"--out-root {out_root()} --evidence-profile {c['evidence_profile']}"
        f" --power-profile {c['power_profile']}"
        f" --validation-role {c['validation_role']}"
    )
    flags = _server_record_flags(s)
    if flags:
        common += " " + " ".join(flags)
    # MoE models: prefer the vendor-published active-param count ("A<N>B") over the
    # analytic estimate (good only to ~10-30%); n_active scales the FLOPs work rate.
    if c.get("n_active_override"):
        common += f" --n-active-override {c['n_active_override']}"
    return [f"python3 profiling/probes/{probe}.py {common}{_probe_extra_args(c, probe)}"
            for probe in probes_for_tp(c, tp)]


def validate_command(c: dict, tp: int, regime=None) -> str:
    """`validate_run` invocation for a validate campaign (real dataset traffic).

    Carries the campaign's ``server.max_model_len`` so the dataset length pruner
    tracks the SAME served context the server uses — automatically correct for any
    model size, never the fixed 1024/2048.
    """
    s, w = c["server"], c["workload"]
    request_rate = (regime or {}).get("request_rate", w.get("request_rate"))
    cmd = (
        f"python3 profiling/probes/validate_run.py "
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp} "
        f"--base-url {base_url()} "
        f"--gpus-per-node {c['gpus_per_node']} "
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']} "
        f"--kv-cache-dtype {s['kv_cache_dtype']} --out-root {out_root()} "
        f"--dataset {w.get('dataset', 'sharegpt')} "
        f"--num-prompts {w.get('num_prompts')} --request-rate {request_rate} "
        f"--burstiness {w['burstiness']} --seed {w.get('seed', 0)} "
        f"--validation-role {c['validation_role']} "
        f"--evidence-profile {c['evidence_profile']}"
        f" --power-profile {c['power_profile']}"
    )
    if w.get("dataset_path"):
        cmd += f" --dataset-path {w['dataset_path']}"
    if float(w.get("pre_idle_s", 0.0)):
        cmd += f" --pre-idle-s {float(w['pre_idle_s'])}"
    flags = _server_record_flags(s)
    if flags:
        cmd += " " + " ".join(flags)
    # Keep MoE active-param count identical to the probe (training) bundles so the
    # held-out validate test isn't graded against a different arch for the same model.
    if c.get("n_active_override"):
        cmd += f" --n-active-override {c['n_active_override']}"
    return cmd


def agentic_command(c: dict, tp: int, regime: dict) -> str:
    """`agentic_run` invocation for one prefix-cache regime (multi-turn sessions).

    Replay campaigns (``sessions.corpus`` set) stream real traces; otherwise the
    synthetic generator runs. ``--prefix-cache`` comes from the regime, matching
    the server launched for the same regime index.
    """
    s, ss = c["server"], c["sessions"]
    parts = [
        "python3 profiling/probes/agentic_run.py",
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp}",
        f"--base-url {base_url()}",
        f"--gpus-per-node {c['gpus_per_node']}",
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']}",
        f"--out-root {out_root()}",
        f"--evidence-profile {c['evidence_profile']}",
        f"--power-profile {c['power_profile']}",
        f"--validation-role {c['validation_role']}",
        f"--n-sessions {ss.get('n_sessions', 8)} --seed {ss.get('seed', 0)}",
    ]
    parts.extend(_server_record_flags(s))
    if c.get("n_active_override"):
        parts.append(f"--n-active-override {c['n_active_override']}")
    if ss.get("concurrency") is not None:
        parts.append(f"--concurrency {ss['concurrency']}")
    if ss.get("corpus"):
        parts.append(f"--replay --corpus {ss['corpus']}")
        if ss.get("gap_params"):
            parts.append(f"--gap-params {ss['gap_params']}")
        if ss.get("dataset_revision"):
            parts.append(f"--dataset-revision {ss['dataset_revision']}")
        parts.append(
            f"--pack-index {int(regime.get('pack_index', ss.get('pack_index', 0)))} "
            f"--pack-count {int(ss.get('pack_count', 1))}"
        )
    else:
        parts.append(f"--gap-mean-s {ss.get('gap_mean_s', 3.0)}")
    if float(ss.get("pre_idle_s", 0.0)):
        parts.append(f"--pre-idle-s {float(ss['pre_idle_s'])}")
    if regime.get("prefix_cache"):
        parts.append("--prefix-cache")
    return " ".join(parts)


def trace_replay_command(c: dict, tp: int, regime: dict) -> str:
    """Exact direct-token replay invocation for one cache regime."""
    s, trace = c["server"], c["trace"]
    parts = [
        "python3 profiling/probes/trace_replay_run.py",
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp}",
        f"--base-url {base_url()}",
        f"--gpus-per-node {c['gpus_per_node']}",
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']}",
        f"--kv-cache-dtype {s['kv_cache_dtype']} --out-root {out_root()}",
        f"--evidence-profile {c['evidence_profile']}",
        f"--power-profile {c['power_profile']}",
        f"--validation-role {c['validation_role']}",
        f"--trace-plan {regime.get('plan', trace['plan'])}",
        f"--concurrency {int(trace.get('concurrency', 64))}",
        f"--cache-block-tokens {int(trace.get('cache_block_tokens', 16))}",
        f"--pre-idle-s {float(trace.get('pre_idle_s', 0.0))}",
    ]
    parts.extend(_server_record_flags(s))
    if c.get("n_active_override"):
        parts.append(f"--n-active-override {c['n_active_override']}")
    if regime.get("prefix_cache"):
        parts.append("--prefix-cache")
    return " ".join(parts)


def roofline_agentic_command(c: dict, tp: int) -> str:
    """Synthetic long-session workload for the cache-off roofline campaign."""
    s, ss = c["server"], c["roofline"]["sessions"]
    parts = [
        "python3 profiling/probes/agentic_run.py",
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp}",
        f"--base-url {base_url()}",
        f"--gpus-per-node {c['gpus_per_node']}",
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']}",
        f"--kv-cache-dtype {s['kv_cache_dtype']} --out-root {out_root()}",
    ]
    parts.extend(_server_record_flags(s))
    for flag, key in (
        ("n-sessions", "n_sessions"), ("seed", "seed"), ("concurrency", "concurrency"),
        ("min-turns", "min_turns"), ("max-turns", "max_turns"),
        ("prefix-tokens", "prefix_tokens"), ("user-tokens-mean", "user_tokens_mean"),
        ("assistant-tokens-mean", "assistant_tokens_mean"),
        ("gap-mean-s", "gap_mean_s"), ("gap-sigma", "gap_sigma"),
    ):
        parts.append(f"--{flag} {ss[key]}")
    if c.get("n_active_override"):
        parts.append(f"--n-active-override {c['n_active_override']}")
    return " ".join(parts)


def roofline_analyze_command(c: dict, tp: int) -> str:
    """Analyze the bundles written under the roofline run root."""
    return (
        "python3 -m scripts.eval.occupancy_roofline "
        f"--run-root {out_root()} "
        f"--label {c['_campaign_id']} "
        f"--model {c['model']} "
        f"--hardware {c['hardware']} "
        f"--tp {tp} "
        f"--window-s {float(c['roofline']['window_s'])}"
    )


def run_command(c: dict, tp: int, regime=None) -> str:
    """The non-probe entrypoint command for validate / agentic campaigns."""
    t = c["campaign_type"]
    if t == "validate":
        return validate_command(c, tp, regime)
    if t == "agentic":
        return agentic_command(c, tp, regime or {})
    if t == "trace_replay":
        return trace_replay_command(c, tp, regime or {})
    raise CampaignError(
        f"run_command is only for validate/agentic/trace_replay, not '{t}'"
    )


def render_plan(c: dict) -> str:
    lines = [
        f"# Campaign: {c['campaign_type']} | {c['model']} | {c['hardware']}",
        f"# TP degrees: {tp_degrees(c)}",
        f"# evidence: {c['evidence_profile']} | power: {c['power_profile']} "
        f"| role: {c['validation_role']}",
    ]
    for tp in tp_degrees(c):
        lines.append(f"\n## TP={tp}")
        if c["campaign_type"] == "validate":
            w = c["workload"]
            lines.append(f"SERVE: {serve_command(c, tp)}")
            for regime in regimes(c):
                lines.append(
                    f"VALIDATE: benchmark_serving --dataset {w.get('dataset')} "
                    f"--num-prompts {w.get('num_prompts')} "
                    f"--request-rate {regime['request_rate']} "
                    f"--burstiness {w['burstiness']} "
                    f"--seed {w.get('seed', 0)} "
                    f"--pre-idle-s {float(w.get('pre_idle_s', 0.0))}"
                )
        elif c["campaign_type"] in {"agentic", "trace_replay"}:
            for r in regimes(c):
                pc = bool(r.get("prefix_cache"))
                lines.append(f"SERVE[prefix_cache={pc}]: {serve_command(c, tp, pc)}")
                lines.append(
                    f"{c['campaign_type'].upper()}[prefix_cache={pc}]: "
                    f"{run_command(c, tp, r)}")
        elif c["campaign_type"] == "roofline":
            lines.append(f"# output root: {out_root()}")
            for probe, cmd in zip(probes_for_tp(c, tp), probe_commands(c, tp)):
                lines.append(f"SERVE[{probe}]: {probe_serve_command(c, probe, tp)}")
                lines.append(f"PROBE: {cmd}")
            lines.append(f"SERVE[long_agentic]: {serve_command(c, tp, False)}")
            lines.append(f"AGENTIC[long_cache_off]: {roofline_agentic_command(c, tp)}")
            lines.append(f"ANALYZE: {roofline_analyze_command(c, tp)}")
        else:
            # one server per probe (probes need different launch flags)
            for probe, cmd in zip(probes_for_tp(c, tp), probe_commands(c, tp)):
                lines.append(f"SERVE[{probe}]: {probe_serve_command(c, probe, tp)}")
                lines.append(f"PROBE: {cmd}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--emit", default="plan",
                    choices=["plan", "json", "model", "tps", "type", "serve",
                             "probes", "probe-names", "probe-serves", "run-cmd",
                             "regimes", "roofline-agentic", "analyze-cmd", "role"])
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--regime-idx", type=int, default=0,
                    help="prefix-cache regime index (agentic; see --emit regimes)")
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    if args.emit == "json":
        print(json.dumps(c, indent=2))
    elif args.emit == "model":
        print(c["model"])
    elif args.emit == "type":
        print(c["campaign_type"])
    elif args.emit == "role":
        print(c["validation_role"])
    elif args.emit == "tps":
        print("\n".join(str(t) for t in tp_degrees(c)))
    elif args.emit == "regimes":
        print(len(regimes(c)))
    elif args.emit == "run-cmd":
        if args.tp is None:
            raise CampaignError("--emit run-cmd requires --tp")
        print(run_command(c, args.tp, regimes(c)[args.regime_idx]))
    elif args.emit == "roofline-agentic":
        if args.tp is None:
            raise CampaignError("--emit roofline-agentic requires --tp")
        print(roofline_agentic_command(c, args.tp))
    elif args.emit == "analyze-cmd":
        if args.tp is None:
            raise CampaignError("--emit analyze-cmd requires --tp")
        print(roofline_analyze_command(c, args.tp))
    elif args.emit == "serve":
        if args.tp is None:
            raise CampaignError("--emit serve requires --tp")
        print(serve_command(c, args.tp, regimes(c)[args.regime_idx].get("prefix_cache")))
    elif args.emit == "probes":
        if args.tp is None:
            raise CampaignError("--emit probes requires --tp")
        print("\n".join(probe_commands(c, args.tp)))
    elif args.emit == "probe-names":
        if args.tp is None:
            raise CampaignError("--emit probe-names requires --tp")
        print("\n".join(probes_for_tp(c, args.tp)))
    elif args.emit == "probe-serves":
        if args.tp is None:
            raise CampaignError("--emit probe-serves requires --tp")
        print("\n".join(probe_serve_command(c, p, args.tp)
                        for p in probes_for_tp(c, args.tp)))
    else:
        print(render_plan(c))


if __name__ == "__main__":
    sys.exit(main())
