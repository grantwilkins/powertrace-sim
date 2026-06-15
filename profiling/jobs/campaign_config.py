"""Campaign JSON loader / validator + plan emitter (CAMPAIGN.md §5-F).

JSON (not YAML) to avoid a new ``pyyaml`` dependency and to match ``manifest.json``.
Pure: loads a campaign file, validates it, and emits the server-launch command and
per-probe invocation lines. ``run_campaign.sh`` consumes these; ``--dry-run``
prints the full plan without launching anything (the no-GPU verification path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Mirrors profiling/probes/schedule.BUILDERS; a test asserts they stay in sync.
KNOWN_PROBES = {
    "idle_hold", "decode_staircase", "prefill_staircase",
    "context_holds", "transients", "mixed_grid",
}
CAMPAIGN_TYPES = {"tier1", "tier1_partial", "tier2", "validate", "agentic"}
REPO_ROOT = Path(__file__).resolve().parents[2]


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
        if "sessions" not in c:
            raise CampaignError(f"{where}agentic campaigns need a 'sessions' block")
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
    return c


def serve_command(c: dict, tp: int) -> str:
    """The ``vllm serve`` command for this campaign at a given TP."""
    s = c["server"]
    parts = [f"vllm serve {c['model']}", f"--tensor-parallel-size {tp}",
             f"--max-num-seqs {s['max_num_seqs']}",
             f"--max-num-batched-tokens {s['max_num_batched_tokens']}",
             f"--kv-cache-dtype {s['kv_cache_dtype']}",
             f"--max-model-len {s['max_model_len']}"]
    if s["enable_chunked_prefill"]:
        parts.append("--enable-chunked-prefill")
    if s["enable_prefix_caching"]:
        parts.append("--enable-prefix-caching")
    parts.extend(s["extra_args"])
    return " ".join(parts)


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
    """`python -m profiling.probes.<probe>` invocations for one TP."""
    s = c["server"]
    cmds = []
    common = (
        f"--model {c['model']} --hardware {c['hardware']} --tp {tp} "
        f"--gpus-per-node {c['gpus_per_node']} "
        f"--max-num-seqs {s['max_num_seqs']} "
        f"--max-model-len {s['max_model_len']} "
        f"--kv-cache-dtype {s['kv_cache_dtype']}"
    )
    if s.get("dtype_hint"):
        common += f" --dtype-hint {s['dtype_hint']}"
    for probe in c["probes"]:
        cmds.append(f"python -m profiling.probes.{probe} {common}")
    return cmds


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
            s = c["sessions"]
            lines.append(f"SERVE: {serve_command(c, tp)}")
            lines.append(
                f"AGENTIC: session_runner n_sessions={s.get('n_sessions')} "
                f"prefix_cache={s.get('prefix_cache')} "
                f"(run with prefix_cache on AND off)"
            )
        else:
            # one server per probe (probes need different launch flags)
            for probe, cmd in zip(c["probes"], probe_commands(c, tp)):
                lines.append(f"SERVE[{probe}]: {probe_serve_command(c, probe, tp)}")
                lines.append(f"PROBE: {cmd}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--emit", default="plan",
                    choices=["plan", "json", "tps", "serve", "probes", "probe-serves"])
    ap.add_argument("--tp", type=int, default=None)
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    if args.emit == "json":
        print(json.dumps(c, indent=2))
    elif args.emit == "tps":
        print("\n".join(str(t) for t in tp_degrees(c)))
    elif args.emit == "serve":
        if args.tp is None:
            raise CampaignError("--emit serve requires --tp")
        print(serve_command(c, args.tp))
    elif args.emit == "probes":
        if args.tp is None:
            raise CampaignError("--emit probes requires --tp")
        print("\n".join(probe_commands(c, args.tp)))
    elif args.emit == "probe-serves":
        if args.tp is None:
            raise CampaignError("--emit probe-serves requires --tp")
        print("\n".join(probe_serve_command(c, p, args.tp) for p in c["probes"]))
    else:
        print(render_plan(c))


if __name__ == "__main__":
    sys.exit(main())
