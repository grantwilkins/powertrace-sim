"""Gate check: does the arrival-only simulation place work correctly on the
250 ms grid the power model consumes?

For every evaluation run, bin the MEASURED token-completion events
(arrival + time-to-first-token + cumulative inter-token latencies) and the
SIMULATED events onto one uniform 250 ms grid on the shared run clock, and
compare at 1 s aggregation (the power comparison resolution): per-run
Pearson correlation, normalized RMSE of the decode token rate, and total
token conservation.

The comparison deliberately does NOT use the ledger cache's channel stream:
that stream silently drops bins that have no power samples, so its time
axis is compressed by logger gaps and cannot be aligned bin-for-bin (the
`keep_power_gaps` builder option exists for exactly this reason). Both
series here come from the same clock, so alignment is exact by
construction and the comparison isolates simulator placement fidelity.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_timing import MAX_NUM_SEQS  # noqa: E402
from iteration_time import launch_overhead_s, transformer_bw_scale  # noqa: E402
from scheduler_sim import EngineConfig, simulate_requests  # noqa: E402
from model.training_data.arch import get_arch  # noqa: E402

BASE = Path(__file__).resolve().parent
EVAL_ROLES = ("test_indomain", "holdout_model", "holdout_twin", "holdout_rate")


def main():
    data = dict(np.load(BASE / "timing_dataset.npz", allow_pickle=False))
    manifest = json.loads((BASE / "split_manifest_fp8.json").read_text())
    fitted = json.loads((BASE / "fitted_efficiencies.json").read_text())
    roles = {int(k): v for k, v in manifest["roles"].items()}
    dt = 0.25

    rows = []
    for rid, role in sorted(roles.items()):
        if role not in EVAL_ROLES:
            continue
        hardware = str(data["run_hardware"][rid])
        model = str(data["run_model"][rid])
        tp = int(data["run_tp"][rid])
        arch = get_arch(model)
        params = fitted[hardware]
        idx = np.flatnonzero(data["req_run_id"] == rid)
        order = idx[np.argsort(data["arrival_time_s"][idx], kind="stable")]
        requests = [(float(data["arrival_time_s"][i]),
                     int(data["input_tokens"][i]),
                     int(data["output_tokens"][i])) for i in order]
        simulated = simulate_requests(
            requests, arch=arch, hardware=hardware, tp=tp,
            eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
            transformer_bw_scale=transformer_bw_scale(arch, params, hardware),
            t_launch_s=launch_overhead_s(
                arch, base_s=params["base_overhead_s"],
                per_message_s=params["per_message_s"][str(tp)]),
            t_sample_s=params["per_token_sample_s"],
            engine=EngineConfig(max_num_seqs=MAX_NUM_SEQS.get(model, 256)))

        offsets_arr = data["itl_offsets"]
        measured_events = np.concatenate([
            data["arrival_time_s"][i] + data["ttft_s"][i]
            + np.r_[0.0, np.cumsum(data["itl_values"][offsets_arr[i]:offsets_arr[i + 1]])]
            for i in order])
        sim_events = np.concatenate([
            sim["arrival_s"] + sim["ttft_s"] + params["first_token_overhead_s"]
            + np.r_[0.0, np.cumsum(sim["itl_s"])]
            for sim in simulated])
        n_bins = int(np.ceil(max(measured_events.max(), sim_events.max()) / dt)) + 1

        def bin_events(event_times):
            binned = np.zeros(n_bins)
            bins = np.floor(event_times / dt).astype(int)
            keep = (bins >= 0) & (bins < n_bins)
            np.add.at(binned, bins[keep], 1.0 / dt)
            return binned

        measured_dec = bin_events(measured_events)
        sim_dec = bin_events(sim_events)

        # Aggregate both to 1 s (the power comparison resolution).
        n1 = n_bins // 4
        meas_1s = measured_dec[:n1 * 4].reshape(n1, 4).mean(axis=1)
        sim_1s = sim_dec[:n1 * 4].reshape(n1, 4).mean(axis=1)
        if n1 < 30 or meas_1s.std() == 0:
            continue
        corr = float(np.corrcoef(meas_1s, sim_1s)[0, 1])
        nrmse = float(np.sqrt(np.mean((sim_1s - meas_1s) ** 2))
                      / max(meas_1s.mean(), 1e-9))
        conservation = float(abs(sim_1s.sum() - meas_1s.sum())
                             / max(meas_1s.sum(), 1e-9))
        rows.append({"role": role, "hardware": hardware, "model": model,
                     "tp": tp, "rate": float(data["run_rate"][rid]),
                     "corr_1s": corr, "nrmse_mean_1s": nrmse,
                     "token_conservation_err": conservation})

    groups = defaultdict(list)
    for r in rows:
        groups[(r["role"], r["hardware"])].append(r)
    print(f"{'role':14s} {'hw':4s} runs  corr_med  corr_p10  nrmse_med  conserve_med")
    for (role, hw), values in sorted(groups.items()):
        med = lambda k: float(np.median([v[k] for v in values]))
        p10 = float(np.percentile([v["corr_1s"] for v in values], 10))
        print(f"{role:14s} {hw:4s} {len(values):4d}  {med('corr_1s'):8.3f} "
              f"{p10:8.3f} {med('nrmse_mean_1s'):9.3f} "
              f"{med('token_conservation_err'):11.4f}")
    out = BASE / "bin_placement_validation.json"
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nwrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
