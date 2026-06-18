"""Campaign JSON loader / validator + plan emitter (CAMPAIGN.md §5-F).

JSON (not YAML) to avoid a new ``pyyaml`` dependency and to match ``manifest.json``.
Pure: loads a campaign file, validates it, and emits the server-launch command and
per-probe invocation lines. ``run_campaign.sh`` consumes these; ``--dry-run``
prints the full plan without launching anything (the no-GPU verification path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Mirrors profiling/probes/schedule.BUILDERS; a test asserts they stay in sync.
KNOWN_PROBES = {
    "idle_hold", "decode_staircase", "prefill_staircase",
    "context_holds", "transients", "mixed_grid",
}
CAMPAIGN_TYPES = {"tier1", "tier1_partial", "tier2", "validate", "agentic"}
REPO_ROOT = Path(__file__).resolve().parents[2]

# The tp_pair second leg only repeats decode+prefill (CAMPAIGN.md §3): those two
# probes identify e_comm; re-running the full set at the 2nd TP is wasted budget.
DEFAULT_TP_PAIR_PROBES = ["decode_staircase", "prefill_staircase"]


def out_root() -> str:
    """Bundle output root. The sbatch exports ``RUNS=$SCRATCH/...``; off-cluster
    (dry run / tests) it falls back to the repo-local ``data/runs``. Never $HOME
    on Sherlock — the sbatch is responsible for pointing RUNS at $SCRATCH."""
    return os.environ.get("RUNS") or "data/runs"


class CampaignError(ValueError):
    pass


def load_campaign(path) -> dict:
    """Load and validate a campaign JSON file."""
    c = json.loads(Path(path).read_text())
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

    if c["campaign_type"] == "validate":
        if "workload" not in c:
            raise CampaignError(f"{where}validate campaigns need a 'workload' block")
    elif c["campaign_type"] == "agentic":
        ss = c.get("sessions")
        if not ss:
            raise CampaignError(f"{where}agentic campaigns need a 'sessions' block")
        regs = ss.get("regimes")
        if not (isinstance(regs, list) and regs
                and all(isinstance(r, dict) and "prefix_cache" in r for r in regs)):
            raise CampaignError(
                f"{where}agentic sessions need a non-empty 'regimes' list, each "
                f"with a 'prefix_cache' boolean")
        if "enable_prefix_caching" in c["server"] or "prefix_cache" in ss:
            raise CampaignError(
                f"{where}agentic prefix-caching is derived per regime; drop "
                f"server.enable_prefix_caching and sessions.prefix_cache")
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


def _with_defaults(c: dict) -> dict:
    s = c["server"]
    s.setdefault("max_num_seqs", 256)
    s.setdefault("max_num_batched_tokens", 8192)
    s.setdefault("enable_chunked_prefill", True)
    s.setdefault("enable_prefix_caching", False)
    s.setdefault("kv_cache_dtype", "auto")
    s.setdefault("max_model_len", 131072)
    s.setdefault("dtype_hint", None)
    s.setdefault("extra_args", [])
    s.setdefault("extra_env", {})
    c.setdefault("gpus_per_node", 8)
    c.setdefault("probes", [])
    c.setdefault("tp_pair_probes", list(DEFAULT_TP_PAIR_PROBES))
    return c


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
             f"--tensor-parallel-size {tp}",
             f"--max-num-seqs {s['max_num_seqs']}",
             f"--max-num-batched-tokens {s['max_num_batched_tokens']}",
             f"--kv-cache-dtype {s['kv_cache_dtype']}",
             f"--max-model-len {s['max_model_len']}"]
    if s["enable_chunked_prefill"]:
        parts.append("--enable-chunked-prefill")
    if pc:
        parts.append("--enable-prefix-caching")
    parts.extend(s["extra_args"])
    return " ".join(parts)


def regimes(c: dict) -> list[dict]:
    """Prefix-cache regimes to run: one per regime for agentic, a single pass else."""
    if c["campaign_type"] == "agentic":
        return list(c["sessions"]["regimes"])
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


def _schedule_overrides(probe: str, c: dict) -> dict:
    """Server overrides a probe requires (chunked-prefill OFF, long max_model_len).

    Built from the canonical ``schedule`` builders so the orchestrator launches a
    server matching each probe's needs, not a single shared server.
    """
    sys.path.insert(0, str(REPO_ROOT / "profiling" / "probes"))
    import schedule  # noqa: E402

    mns = int(c["server"]["max_num_seqs"])
    if probe == "decode_staircase":
        s = schedule.build_decode_staircase(mns)
    elif probe == "mixed_grid":
        s = schedule.build_mixed_grid(decode_range=(1, mns))
    else:
        s = schedule.BUILDERS[probe]()
    return s.server_overrides


def probe_serve_command(c: dict, probe: str, tp: int) -> str:
    """``vllm serve`` command for a specific probe (merges its server overrides)."""
    ov = _schedule_overrides(probe, c)
    s = dict(c["server"])
    if "enable_chunked_prefill" in ov:
        s["enable_chunked_prefill"] = ov["enable_chunked_prefill"]
    if "max_model_len" in ov:
        s["max_model_len"] = max(int(s["max_model_len"]), int(ov["max_model_len"]))
    return serve_command(dict(c, server=s), tp)


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
        f"--gpus-per-node {c['gpus_per_node']} "
        f"--max-num-seqs {s['max_num_seqs']} "
        f"--max-model-len {s['max_model_len']} "
        f"--kv-cache-dtype {s['kv_cache_dtype']} "
        f"--out-root {out_root()}"
    )
    if s.get("dtype_hint"):
        common += f" --dtype-hint {s['dtype_hint']}"
    # MoE models: prefer the vendor-published active-param count ("A<N>B") over the
    # analytic estimate (good only to ~10-30%); n_active scales the FLOPs work rate.
    if c.get("n_active_override"):
        common += f" --n-active-override {c['n_active_override']}"
    return [f"python3 profiling/probes/{probe}.py {common}"
            for probe in probes_for_tp(c, tp)]


def validate_command(c: dict, tp: int) -> str:
    """`validate_run` invocation for a validate campaign (real dataset traffic).

    Carries the campaign's ``server.max_model_len`` so the dataset length pruner
    tracks the SAME served context the server uses — automatically correct for any
    model size, never the fixed 1024/2048.
    """
    s, w = c["server"], c["workload"]
    cmd = (
        f"python3 profiling/probes/validate_run.py "
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp} "
        f"--gpus-per-node {c['gpus_per_node']} "
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']} "
        f"--kv-cache-dtype {s['kv_cache_dtype']} --out-root {out_root()} "
        f"--dataset {w.get('dataset', 'sharegpt')} "
        f"--num-prompts {w.get('num_prompts')} --request-rate {w.get('request_rate')}"
    )
    if w.get("dataset_path"):
        cmd += f" --dataset-path {w['dataset_path']}"
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
        f"--gpus-per-node {c['gpus_per_node']}",
        f"--max-model-len {s['max_model_len']} --max-num-seqs {s['max_num_seqs']}",
        f"--out-root {out_root()}",
        f"--n-sessions {ss.get('n_sessions', 8)} --seed {ss.get('seed', 0)}",
    ]
    if c.get("n_active_override"):
        parts.append(f"--n-active-override {c['n_active_override']}")
    if ss.get("concurrency") is not None:
        parts.append(f"--concurrency {ss['concurrency']}")
    if ss.get("corpus"):
        parts.append(f"--replay --corpus {ss['corpus']} --gap-params {ss['gap_params']}")
    else:
        parts.append(f"--gap-mean-s {ss.get('gap_mean_s', 3.0)}")
    if regime.get("prefix_cache"):
        parts.append("--prefix-cache")
    return " ".join(parts)


def run_command(c: dict, tp: int, regime=None) -> str:
    """The non-probe entrypoint command for validate / agentic campaigns."""
    t = c["campaign_type"]
    if t == "validate":
        return validate_command(c, tp)
    if t == "agentic":
        return agentic_command(c, tp, regime or {})
    raise CampaignError(f"run_command is only for validate/agentic, not '{t}'")


def render_plan(c: dict) -> str:
    lines = [
        f"# Campaign: {c['campaign_type']} | {c['model']} | {c['hardware']}",
        f"# TP degrees: {tp_degrees(c)}",
    ]
    for tp in tp_degrees(c):
        lines.append(f"\n## TP={tp}")
        if c["campaign_type"] == "validate":
            w = c["workload"]
            lines.append(f"SERVE: {serve_command(c, tp)}")
            lines.append(
                f"VALIDATE: benchmark_serving --dataset {w.get('dataset')} "
                f"--num-prompts {w.get('num_prompts')} "
                f"--request-rate {w.get('request_rate')}"
            )
        elif c["campaign_type"] == "agentic":
            ss = c["sessions"]
            for r in regimes(c):
                pc = bool(r.get("prefix_cache"))
                lines.append(f"SERVE[prefix_cache={pc}]: {serve_command(c, tp, pc)}")
                lines.append(
                    f"AGENTIC[prefix_cache={pc}]: {run_command(c, tp, r)}")
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
                    choices=["plan", "json", "tps", "type", "serve", "probes",
                             "probe-names", "probe-serves", "run-cmd", "regimes"])
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--regime-idx", type=int, default=0,
                    help="prefix-cache regime index (agentic; see --emit regimes)")
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    if args.emit == "json":
        print(json.dumps(c, indent=2))
    elif args.emit == "type":
        print(c["campaign_type"])
    elif args.emit == "tps":
        print("\n".join(str(t) for t in tp_degrees(c)))
    elif args.emit == "regimes":
        print(len(regimes(c)))
    elif args.emit == "run-cmd":
        if args.tp is None:
            raise CampaignError("--emit run-cmd requires --tp")
        print(run_command(c, args.tp, regimes(c)[args.regime_idx]))
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
