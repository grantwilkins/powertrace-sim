"""DIAGNOSTIC ONLY — attempted fit of a mixture-of-experts
routing-correlation constant from the gpt-oss-20b iteration staircase.

VERDICT (2026-07-16): NOT IDENTIFIABLE from latency deltas; this script
reports the attempt and deliberately writes nothing. The staircase deltas
are nearly pure batch-linear, because the expert weight sweep sits below
the operator roofline max at these batch sizes and is therefore invisible
in latency, while the per-token cost (fitted on dense models) conflates
with any routing parameter — the fit slams its bound (gamma -> 0.05,
delta rmse_log 2.3). Separating routing correlation from per-token
overhead requires router/expert counters, exactly as the power-side
analysis concluded (FEATURE_TEST_LEARNINGS.md section 6, item 2). The
probes DID establish the engine-configuration conflict (async scheduling
in legacy gpt-oss serving vs none in the probes) documented in
fit_efficiencies.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iteration_time import HARDWARE_PROFILES, iteration_work  # noqa: E402
from model.training_data.arch import get_arch  # noqa: E402

BASE = Path(__file__).resolve().parent
GAMMA_BOUNDS = (0.05, 1.0)


def staircase_deltas(calibration):
    rows = [r for r in calibration["rows"]
            if r.get("model") == "gpt-oss-20b" and r["probe"] == "decode_staircase"]
    by_tp = {}
    for row in rows:
        by_tp.setdefault(int(row["tp"]), []).append(row)
    deltas = []
    for tp, levels in sorted(by_tp.items()):
        levels.sort(key=lambda r: r["batch"])
        base = next(r for r in levels if r["batch"] == 1)
        for row in levels:
            if row["batch"] <= 1:
                continue
            deltas.append({
                "tp": tp,
                "batch": float(row.get("effective_decode_batch") or row["batch"]),
                "context": float(row["context_tokens_mean"]),
                "base_batch": 1.0,
                "base_context": float(base["context_tokens_mean"]),
                "delta_s": (row["measured"]["median_itl_ms"]
                            - base["measured"]["median_itl_ms"]) / 1e3,
            })
    return deltas


def main():
    calibration = json.loads((BASE / "probe_calibration.json").read_text())
    fitted = json.loads((BASE / "fitted_efficiencies.json").read_text())
    deltas = staircase_deltas(calibration)
    if not deltas:
        raise SystemExit("no gpt-oss-20b staircase rows found")
    arch = get_arch("gpt-oss-20b")
    params = fitted["A100"]
    profile = HARDWARE_PROFILES["A100"]

    def predicted_delta(gamma, d):
        def step_s(batch, context):
            work = iteration_work(
                {**arch, "moe_routing_gamma": gamma},
                decode_batch=batch, context_mean=context)
            compute = work["gemm_flops"] / (params["eff_flops"]
                                            * profile["peak_flops_s"] * d["tp"])
            attn_c = work["attn_flops"] / (params["eff_flops"]
                                           * profile["peak_flops_s"] * d["tp"])
            rate = params["eff_bw"] * profile["hbm_bytes_s"] * d["tp"]
            gemm = max(compute, work["gemm_bytes"] / rate)
            attn = max(attn_c, work["attn_bytes"] / rate)
            return (gemm + attn
                    + params["per_token_sample_s"] * work["sampled_tokens"])
        return step_s(d["batch"], d["context"]) - step_s(d["base_batch"],
                                                         d["base_context"])

    def loss(gamma):
        residuals = [np.log(max(predicted_delta(gamma, d), 1e-6))
                     - np.log(max(d["delta_s"], 1e-6)) for d in deltas]
        return float(np.mean(np.asarray(residuals) ** 2))

    result = minimize_scalar(loss, bounds=GAMMA_BOUNDS, method="bounded")
    gamma = float(result.x)
    uniform = np.sqrt(loss(1.0))
    at_bound = gamma <= GAMMA_BOUNDS[0] * 1.01 or gamma >= GAMMA_BOUNDS[1] * 0.99
    print(f"best gamma={gamma:.4f} (delta-fit rmse_log "
          f"{np.sqrt(result.fun):.3f}; uniform routing gives {uniform:.3f}; "
          f"{len(deltas)} deltas)")
    print("verdict: NOT IDENTIFIABLE — "
          + ("bound-slammed, " if at_bound else "")
          + "residual far above measurement noise; nothing written. "
          "Router/expert counters are required (see module docstring).")


if __name__ == "__main__":
    main()
