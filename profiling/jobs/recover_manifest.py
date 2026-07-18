"""Synthesize a manifest.json for an orphaned bundle (crash/timeout before write).

When a probe run is killed (time limit, preemption) AFTER collecting per-level
data but BEFORE the manifest is written, the bundle has power.csv + engine.csv +
levels/level_00N.json but no manifest.json, so the ledger builders skip it. The
collected levels are still good data. This tool reconstructs a manifest by:

  * borrowing arch / server / tp / hardware / model from a DONOR run of the same
    model that completed (its manifest), and
  * rebuilding probe.levels[] (t_start_epoch / t_end_epoch / params) from each
    level JSON's request_timestamps + duration.

Usage (stdlib only):
    python3 profiling/jobs/recover_manifest.py --bundle <orphan_run_dir> \
        --donor <run_dir_with_manifest> [--probe-type prefill_staircase]
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="orphan run dir (no manifest)")
    ap.add_argument("--donor", required=True, help="run dir of same model WITH a manifest")
    ap.add_argument("--probe-type", default=None,
                    help="probe type (default: infer from bundle dir name)")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    donor_m = json.loads((Path(args.donor) / "manifest.json").read_text())

    ptype = args.probe_type
    if ptype is None:
        name = bundle.name
        for p in ("prefill_staircase", "decode_staircase", "idle_hold",
                  "context_holds", "mixed_grid", "transients"):
            if p in name:
                ptype = p
                break
    if ptype is None:
        raise SystemExit("could not infer --probe-type from bundle name")

    level_files = sorted((bundle / "levels").glob("level_*.json"))
    if not level_files:
        raise SystemExit(f"no level files in {bundle}/levels")

    levels = []
    for i, lf in enumerate(level_files):
        d = json.loads(lf.read_text())
        ts = d.get("request_timestamps") or []
        if not ts:
            print(f"  skip {lf.name}: no request_timestamps")
            continue
        t0, t1 = float(min(ts)), float(max(ts))
        dur = float(d.get("duration", t1 - t0))
        in_lens = d.get("input_lens") or [0]
        out_lens = d.get("output_lens") or [1]
        levels.append({
            "level": i,
            "label": f"{ptype}_{int(in_lens[0])}",
            "concurrency": int(d.get("max_concurrency") or 1),
            "num_prompts": int(d.get("num_prompts") or len(ts)),
            "params": {"input_len": int(in_lens[0]),
                       "output_len": int(out_lens[0]),
                       "prefix_len": 0, "ignore_eos": True},
            "t_start_epoch": t0,
            "t_end_epoch": max(t1, t0 + dur),
            "summary": {"completed": int(d.get("completed") or len(ts)),
                        "duration": dur},
            "recovered": True,
        })

    if not levels:
        raise SystemExit("no recoverable levels (none had request_timestamps)")

    manifest = {
        "manifest_version": 1,
        "run_id": bundle.name,
        "recovered_from_donor": Path(args.donor).name,
        "model": donor_m["model"],
        "arch": donor_m["arch"],
        "hardware": donor_m["hardware"],
        "tp": donor_m["tp"],
        "gpus_per_node": donor_m.get("gpus_per_node", 8),
        "server": donor_m.get("server", {}),
        "versions": donor_m.get("versions", {}),
        "clock": donor_m.get("clock", {}),
        "probe": {
            "type": ptype,
            "levels": levels,
            "window": {"start_epoch": levels[0]["t_start_epoch"],
                       "end_epoch": levels[-1]["t_end_epoch"]},
        },
    }
    out = bundle / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}: {len(levels)} levels "
          f"({', '.join(str(l['params']['input_len']) for l in levels)}) "
          f"model={manifest['model']} tp={manifest['tp']}")


if __name__ == "__main__":
    main()
