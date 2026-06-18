"""Fit & evaluate the explainable power model on the new gemma profiling data.

Pipeline
  1. Load staircase probes as measured-state steady points + validate runs as
     measured-state per-second bins (dataio).
  2. Fit the first-principles model ladder (physics.MODELS) on the probe levels.
  3. Report in-sample R^2 + per-level energy error for each ladder rung.
  4. Transferability: leave-one-model-family-out (fit dense -> predict MoE and
     vice versa) using ONLY arch arithmetic to bridge.
  5. Grade on the held-out validate runs (realistic ShareGPT-like workload):
     bin R^2 + run-mean energy error.

Everything is reported per hardware (A100 here). No coefficient is reused from
feature-test; we refit from the clean probe data so the result is honest.

Run (on a compute node with the module loaded):
    ml python/3.12.1 && $SCRATCH/npm-venv/bin/python new-profiling-model/fit_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataio
import physics
from nnls import nnls

RUNS = Path("/scratch/users/gfw/ptsim/runs")

STAIRCASE = [
    "a100_decode_staircase_tp2_1781723125",   # gemma-4-31B (dense)
    "a100_decode_staircase_tp2_1781723242",   # gemma-4-26B-A4B (MoE)
    "a100_prefill_staircase_tp2_1781724228",  # gemma-4-26B-A4B (MoE)
]
VALIDATE = [
    "a100_validate_tp2_1781730630",           # gemma-4-31B (dense)
    "a100_validate_tp2_1781730636",           # gemma-4-26B-A4B (MoE)
]


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(
        r2=r2(y, p),
        rmse=float(np.sqrt(np.mean((y - p) ** 2))),
        mape=float(np.mean(np.abs(p - y) / np.maximum(y, 1.0)) * 100),
    )


def window_metrics(y, p, win):
    """Metrics on non-overlapping window means (energy fidelity)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = (y.size // win) * win
    if n < win:
        return metrics(y, p)
    ym = y[:n].reshape(-1, win).mean(1)
    pm = p[:n].reshape(-1, win).mean(1)
    return metrics(ym, pm)


def fit_with_multiplier(points, feats):
    """NNLS fit + per-family multiplier on the dynamic (non-idle) part.

    Mirrors feature-test/final_model: the static idle/link terms are shared and
    multiplier-exempt; one scalar per family rescales the dynamic sum to absorb
    kernel-efficiency differences. Multiplier defaults to 1 for unseen families.
    """
    X = physics.design(points, feats)
    y = np.array([p["power"] for p in points])
    fam = np.array([p["family"] for p in points])
    idle_mask = np.array([f in physics.IDLE_FEATS for f in feats])
    coef, _ = nnls(X, y)
    dyn = X[:, ~idle_mask] @ coef[~idle_mask]
    idle = X[:, idle_mask] @ coef[idle_mask]
    mult = {}
    for fi in np.unique(fam):
        m = (fam == fi) & (dyn > 1.0)
        if m.sum() == 0:
            continue
        mult[fi] = float(np.clip(
            np.sum(dyn[m] * (y[m] - idle[m])) / max(np.sum(dyn[m] ** 2), 1e-9),
            0.5, 2.0))
    return dict(coef=coef, mult=mult, feats=feats, idle_mask=idle_mask)


def predict(fit, points):
    X = physics.design(points, fit["feats"])
    coef, idle_mask = fit["coef"], fit["idle_mask"]
    dyn = X[:, ~idle_mask] @ coef[~idle_mask]
    idle = X[:, idle_mask] @ coef[idle_mask]
    fam = np.array([p["family"] for p in points])
    mvec = np.array([fit["mult"].get(f, 1.0) for f in fam])
    return idle + mvec * dyn


def level_energy_errors(points, pred):
    """Per-point |pred-meas|/meas %, the per-level energy error."""
    y = np.array([p["power"] for p in points])
    return np.abs(pred - y) / np.maximum(y, 1.0) * 100


def run_mean_errors(points, pred):
    y = np.array([p["power"] for p in points])
    rid = np.array([p["run_id"] for p in points])
    out = {}
    for r in np.unique(rid):
        m = rid == r
        out[r] = abs(np.mean(pred[m]) - np.mean(y[m])) / max(np.mean(y[m]), 1.0) * 100
    return out


def make_plots(fit_pts, val_pts, grade_fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = HERE / "results"
    OUT.mkdir(exist_ok=True)

    # (1) decode staircase: measured vs predicted power per level, per model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, fam, title in zip(axes, ["dense-32b", "moe-8b"],
                              ["gemma-4-31B (dense)", "gemma-4-26B-A4B (MoE)"]):
        pts = [p for p in fit_pts if p["family"] == fam
               and p["probe"] == "decode_staircase"]
        pts = sorted(pts, key=lambda p: p["dec_tok"])
        x = [p["dec_tok"] for p in pts]
        ax.plot(x, [p["power"] for p in pts], "ko-", label="measured", ms=5)
        for name in ["F1_phase", "S_sat"]:
            pred = predict(grade_fits[name], pts)
            ax.plot(x, pred, "o--", ms=4, alpha=0.8, label=name)
        ax.set_title(title)
        ax.set_xlabel("decode throughput (tok/s)")
        ax.set_ylabel("node power (W)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Decode staircase: measured-state fit (idle anchor + roofline)")
    fig.tight_layout()
    fig.savefig(OUT / "decode_staircase_fit.png", dpi=140)
    plt.close(fig)

    # (2) held-out validate traces
    fig, axes = plt.subplots(len(val_pts), 1, figsize=(10, 3.0 * len(val_pts)))
    if len(val_pts) == 1:
        axes = [axes]
    for ax, (r, pts) in zip(axes, val_pts.items()):
        if not pts:
            continue
        y = np.array([p["power"] for p in pts])
        t = np.arange(len(y))
        ax.plot(t, y, "k-", lw=1.0, label="measured")
        for name in ["F1_phase", "S_sat"]:
            pred = predict(grade_fits[name], pts)
            r2v = r2(y, pred)
            ax.plot(t, pred, lw=1.2, alpha=0.8, label=f"{name} (R2={r2v:.2f})")
        ax.set_title(f"{pts[0]['model']} validate (held out)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("node power (W)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "validate_traces.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT}/decode_staircase_fit.png, validate_traces.png")


def main():
    print("=" * 78)
    print("LOADING measured-state data")
    print("=" * 78)
    probe_pts = []
    idle_pts = []
    # idle anchors from runs with clean no-traffic edges (decode + validate).
    # The prefill run's pre-roll is contaminated (KV pre-alloc), so skip it.
    for r in ["a100_decode_staircase_tp2_1781723125",
              "a100_decode_staircase_tp2_1781723242"] + VALIDATE:
        ip = dataio.idle_point(RUNS / r)
        if ip is not None and 120 < ip["power"] < 260:
            idle_pts.append(ip)
    print(f"  idle anchors: {len(idle_pts)} "
          f"(node W: {[round(p['power']) for p in idle_pts]})")
    for r in STAIRCASE:
        pts = dataio.load_staircase(RUNS / r)
        probe_pts += pts
        m0 = pts[0] if pts else {}
        print(f"  {r}: {len(pts)} levels  ({m0.get('model','?')}, {m0.get('probe','?')})")
    val_pts = {}
    for r in VALIDATE:
        pts = dataio.load_validate_bins(RUNS / r)
        val_pts[r] = pts
        m0 = pts[0] if pts else {}
        print(f"  {r}: {len(pts)} bins   ({m0.get('model','?')}, validate)")

    # probe summary table
    print("\nProbe steady-state points (measured state -> measured power):")
    print(f"  {'run':<38}{'lvl':>4}{'conc':>6}{'batch':>8}{'dec/s':>9}"
          f"{'pre/s':>9}{'power':>8}")
    for p in probe_pts:
        print(f"  {p['run_id']:<38}{str(p['level']):>4}{str(p['concurrency']):>6}"
              f"{p['batch']:>8.1f}{p['dec_tok']:>9.0f}{p['pre_tok']:>9.0f}"
              f"{p['power']:>8.0f}")

    families = sorted({p["family"] for p in probe_pts})
    print(f"\nFamilies: {families}")

    # ---------------------------------------------------------------- ladder
    print("\n" + "=" * 78)
    print("MODEL LADDER  (fit on probe levels, pooled across both gemma models)")
    print("=" * 78)
    decode_pts = [p for p in probe_pts if p["probe"] == "decode_staircase"]
    fit_pts = idle_pts + probe_pts  # idle anchor + decode + prefill
    y = np.array([p["power"] for p in fit_pts])
    print(f"  fitting on {len(fit_pts)} probe points "
          f"(power range {y.min():.0f}-{y.max():.0f} W)\n")
    results = {}
    for name, feats in physics.MODELS.items():
        fit = fit_with_multiplier(fit_pts, feats)
        pred = predict(fit, fit_pts)
        m = metrics(y, pred)
        ee = level_energy_errors(fit_pts, pred)
        results[name] = (fit, m)
        print(f"  {name:<14} R2={m['r2']:.4f}  RMSE={m['rmse']:6.1f}W  "
              f"energy-err: med={np.median(ee):4.1f}% max={np.max(ee):4.1f}%  "
              f"({len(feats)} terms)")

    # ---------------------------------------------------------------- transfer
    print("\n" + "=" * 78)
    print("TRANSFERABILITY  (leave-one-family-out, bridged by arch arithmetic)")
    print("  zero-shot  = held family unseen, multiplier = 1.0")
    print("  Tier-2     = + 1 efficiency multiplier calibrated on held family")
    print("              (the realistic campaign workflow)")
    print("=" * 78)
    for name in physics.MODELS:
        feats = physics.MODELS[name]
        line = [f"  {name:<12}"]
        for held in families:
            tr = [p for p in fit_pts if p["family"] != held]
            te = [p for p in fit_pts if p["family"] == held]
            te_probe = [p for p in te if p["source"] == "probe"]
            if not tr or not te_probe:
                continue
            # zero-shot: constants from other family, multiplier 1
            fit = fit_with_multiplier(tr, feats)
            fit["mult"].pop(held, None)
            ee0 = level_energy_errors(te_probe, predict(fit, te_probe))
            # Tier-2: reuse the same constants, calibrate ONE multiplier on the
            # held family's own probe points (1 scalar), then score.
            X = physics.design(te_probe, feats)
            y = np.array([p["power"] for p in te_probe])
            idle = X[:, fit["idle_mask"]] @ fit["coef"][fit["idle_mask"]]
            dyn = X[:, ~fit["idle_mask"]] @ fit["coef"][~fit["idle_mask"]]
            mok = dyn > 1.0
            mlt = float(np.clip(np.sum(dyn[mok] * (y[mok] - idle[mok]))
                                / max(np.sum(dyn[mok] ** 2), 1e-9), 0.5, 2.0))
            fit["mult"][held] = mlt
            ee2 = level_energy_errors(te_probe, predict(fit, te_probe))
            line.append(f"{held}: 0shot {np.median(ee0):4.1f}% -> Tier2 "
                        f"{np.median(ee2):4.1f}% (m={mlt:.2f})")
        print("   ".join(line))

    # ---------------------------------------------------------------- grade
    print("\n" + "=" * 78)
    print("HELD-OUT GRADE  (fit on ALL probes, predict the validate workloads)")
    print("=" * 78)
    grade_fits = {}
    for name in ["F0_floor", "F1_phase", "S_sat", "B_best"]:
        feats = physics.MODELS[name]
        fit = fit_with_multiplier(fit_pts, feats)  # probes only -> validate held out
        grade_fits[name] = fit
        # validate families ARE seen in probes, but multiplier is from probe data
        print(f"\n  -- {name} --")
        for r, pts in val_pts.items():
            if not pts:
                continue
            pred = predict(fit, pts)
            y = np.array([p["power"] for p in pts])
            m = metrics(y, pred)
            rme = list(run_mean_errors(pts, pred).values())[0]
            # windowed energy fidelity (energy = integrated power; 1 s noise
            # averages out, so this is the metric that matters for energy use)
            w30 = window_metrics(y, pred, 30)
            print(f"     {pts[0]['model']:<26} bin R2={m['r2']:.3f}  "
                  f"30s-win R2={w30['r2']:.3f} MAPE={w30['mape']:.1f}%  "
                  f"run-energy-err={rme:4.1f}%  (n={len(pts)} bins)")

    # ------------------------------------------------------------ figures
    make_plots(fit_pts, val_pts, grade_fits)

    # ---------------------------------------------------------------- export
    OUT = HERE / "results"
    OUT.mkdir(exist_ok=True)
    export = {}
    for name, (fit, m) in results.items():
        export[name] = dict(
            r2=m["r2"], rmse=m["rmse"],
            coefficients={lab: float(c) for lab, c in
                          zip(physics.labels(fit["feats"]), fit["coef"])},
            family_multipliers=fit["mult"],
        )
    (OUT / "ladder_fit.json").write_text(json.dumps(export, indent=2))
    print(f"\nWrote {OUT}/ladder_fit.json")


if __name__ == "__main__":
    main()
