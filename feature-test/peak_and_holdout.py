"""Peak fix + unseen-model generalization.

Structural change: the saturating DVFS/activity terms are driven by
DECODE-ONLY compute utilization, so prefill energy cannot hide there and must
flow through its own linear coefficients (e_f_prefill, e_w_prefill). This is
what lets the model produce the prefill power peaks on a saturated node.

Generalization demo: hold out the dense-405b family entirely (the only FP8
model), fit on 8B/70B classes, predict every 405B run.

Run: uv run python feature-test/peak_and_holdout.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

import fit_map_priors as fmp  # noqa: E402
from final_model import LAGS  # noqa: E402
from fit_models import FEATURES, load_cache, metrics  # noqa: E402
from model.classifiers.physics import (  # noqa: E402
    FEATURE_ORDER,
    SCHEMA_VERSION,
    predict_mean_node_power,
)

OUT = Path("feature-test/results")

FEATS2 = ["tp", "tp_link", "busy_tp", "sat_bw_a", "sat_bw_b", "sat_bw_c",
          "flops_pre", "flops_dec", "w_read_pre", "kv_write", "comm"]
IDLE2 = {"tp", "tp_link"}


def setup_features(d):
    # Roofline-consistent decode memory power: a saturating function of
    # achieved DRAM bandwidth utilization (weights + KV reads vs per-GPU BW).
    # Linear at low utilization (8B/70B), saturating at high (405B) — replaces
    # the linear weight-bytes term + binary decode indicator, which
    # double-count when both extrapolate to big models.
    FEATURES["sat_bw_a"] = ("p_bw_u0.05 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.05)))
    FEATURES["sat_bw_b"] = ("p_bw_u0.15 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.15)))
    FEATURES["sat_bw_c"] = ("p_bw_u0.4 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.4)))
    fmp.PRIORS["sat_bw_a"] = (80.0, 0.70)
    fmp.PRIORS["sat_bw_b"] = (80.0, 0.70)
    fmp.PRIORS["sat_bw_c"] = (80.0, 0.70)
    fmp.PRIORS["busy_tp"] = (30.0, 0.70)
    # Decode is memory-bound GEMV: marginal compute energy must be small.
    # Without this, e_f_decode inflates to ~7 pJ/FLOP in-sample (absorbed by
    # family multipliers) and detonates on 10x-FLOPs held-out models.
    fmp.PRIORS["flops_dec"] = (5.0e-13, 0.40)
    # Multipliers pinned near 1 so shared coefficients stay honest.
    fmp.PHI_PRIOR_SD = 0.05
    # FP8 arithmetic costs ~half BF16 energy per FLOP: physics, not fitted.
    for k in ("flops_pre", "flops_dec"):
        d[k] = d[k] * (1.0 - 0.5 * d["fp8"])
    fmp.FEATS = FEATS2  # map_fit/laplace read module globals


def design2(d, hw, run_ids):
    lag = LAGS[hw]
    cols = [np.asarray(FEATURES[f][1](d), dtype=np.float64) for f in FEATS2]
    return np.column_stack([lag(c, run_ids) for c in cols])


def _ledger_hash(path="feature-test/ledger_cache.npz"):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-cache", default="feature-test/ledger_cache.npz")
    parser.add_argument("--run-index")
    parser.add_argument("--out-dir", default=str(OUT))
    return parser


def _load_run_index(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") == "ledger-run-index-v1":
        rows = payload.get("runs", [])
    elif payload.get("ledger_schema_version") == 2:
        rows = [
            {
                "run_id": int(entry["run_index"]),
                "source_id": f"bundle:{entry['run_id']}",
                "source_layout": "bundle",
                "paths": {"run_dir": entry["run_dir"]},
                "sha256": entry["sha256"],
            }
            for entry in payload.get("runs", [])
        ]
    else:
        raise ValueError("Unsupported ledger run index")
    entries = {int(entry["run_id"]): entry for entry in rows}
    if len(entries) != len(rows):
        raise ValueError("Ledger run index contains duplicate run_id values")
    return entries


def _training_sources(run_ids, run_index):
    ids = [int(value) for value in np.unique(run_ids)]
    missing = [run_id for run_id in ids if run_id not in run_index]
    if missing:
        raise ValueError(f"Ledger run index is missing run_id values: {missing}")
    return [run_index[run_id] for run_id in ids]


def _git_revision():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return commit, dirty


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(args.ledger_cache)
    run_index_path = Path(args.run_index or ledger_path.with_suffix(".runs.json"))
    run_index = _load_run_index(run_index_path)
    d = load_cache(ledger_path)
    setup_features(d)
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    fam_names = [str(x) for x in d["family_names"]]
    mn = [str(x) for x in d["model_names"]]
    idle_cols = np.array([f in IDLE2 for f in FEATS2])
    artifact_hardware = {}
    artifact_validation = {}

    for hw_i, hw in enumerate([str(x) for x in d["hw_names"]]):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        X = design2(d, hw, run_ids)
        th, mult, sigma = fmp.fit_two_stage(X[hw_mask], y[hw_mask], fams[hw_mask], idle_cols)
        pred = fmp.predict_map(X[hw_mask], th, mult, fams[hw_mask], idle_cols)
        ins = metrics(y[hw_mask], pred)
        post_sd = fmp.laplace_sd(X[hw_mask], y[hw_mask], th, mult, fams[hw_mask], idle_cols, sigma)
        busy_hw = hw_mask & (d["batch"] > 0)
        cap_w = float(np.quantile(y[busy_hw] / d["tp"][busy_hw], 0.995))
        sources = _training_sources(run_ids[hw_mask], run_index)
        artifact_hardware[hw] = {
            "coefficients": {f: float(np.exp(th[j])) for j, f in enumerate(FEATS2)},
            "priors": {
                f: {"mean": float(fmp.PRIORS[f][0]), "sd_log": float(fmp.PRIORS[f][1])}
                for f in FEATS2
            },
            "posterior_sd_log": {f: float(post_sd[j]) for j, f in enumerate(FEATS2)},
            "family_multipliers": {fam_names[k]: float(v) for k, v in mult.items()},
            "hbm_bandwidth_bytes_s": 2.0e12 if hw == "A100" else 3.35e12,
            "lag": {
                "moving_average_bins": 1 if hw == "A100" else 2,
                "ema_alpha": 0.6 if hw == "A100" else 0.7,
            },
            "cap_w_per_gpu": cap_w,
            "cap_provenance": "training busy-bin per-GPU power p99.5",
            "training_source_ids": [entry["source_id"] for entry in sources],
        }
        exported_pred = np.minimum(pred, cap_w * d["tp"][hw_mask])
        artifact_validation[hw] = {"training_fit": metrics(y[hw_mask], exported_pred)}

        print(f"\n######## {hw} peak-fix model (decode-only DVFS terms)")
        print(f"  in-sample: R2={ins['r2']:.4f} RMSE={ins['rmse']:.1f}W MAPE={ins['mape']:.1f}%")
        for j, f in enumerate(FEATS2):
            shrink = post_sd[j] / fmp.PRIORS[f][1]
            drift = (th[j] - np.log(fmp.PRIORS[f][0])) / fmp.PRIORS[f][1]
            status = ("DATA-IDENTIFIED" if shrink < 0.3 else
                      "partial" if shrink < 0.7 else "PRIOR-DOMINATED")
            print(f"    {FEATURES[f][0]:30s} {np.exp(th[j]):10.3g} shrink={shrink:5.2f} "
                  f"drift={drift:+5.1f}  {status}")

        # peak fidelity on saturated TP8 runs
        sat = hw_mask & (d["rate"] >= 2.0) & (d["tp"] == 8) & (d["batch"] > 2)
        pr_all = fmp.predict_map(X, th, mult, fams, idle_cols)
        for q in (0.5, 0.9, 0.99):
            print(f"  saturated busy bins P{int(q*100)}: measured {np.quantile(y[sat], q):.0f}W "
                  f"predicted {np.quantile(pr_all[sat], q):.0f}W")

        # ---- 405B holdout (H100 only)
        if hw != "H100":
            continue
        fi_405 = fam_names.index("dense-405b")
        tr = hw_mask & (fams != fi_405)
        te = hw_mask & (fams == fi_405)
        # cold start: no information from the held-out family may enter
        th_h, mult_h, sigma_h = fmp.fit_two_stage(X[tr], y[tr], fams[tr], idle_cols)
        mult_h.pop(fi_405, None)
        pred_h = fmp.predict_map(X, th_h, mult_h, fams, idle_cols)
        # physical clamp: sustained per-GPU power cap, estimated from TRAINING
        # families only (p99.5 of busy per-GPU power); GPUs enforce power
        # limits, so no prediction may exceed it.
        busy_tr = tr & (d["batch"] > 0)
        cap_w = float(np.quantile(y[busy_tr] / d["tp"][busy_tr], 0.995))
        pred_h = np.minimum(pred_h, cap_w * d["tp"])
        print(f"  power cap from training families: {cap_w:.0f} W/GPU")
        mh = metrics(y[te], pred_h[te])
        errs, rm, rp = [], [], []
        for r in np.unique(run_ids[te]):
            mm = te & (run_ids == r)
            a, b = float(np.mean(y[mm])), float(np.mean(pred_h[mm]))
            rm.append(a)
            rp.append(b)
            errs.append(abs(b - a) / a * 100)
        print("\n  === 405B HOLDOUT (trained only on 8B/70B classes) ===")
        print(f"  bin-level: R2={mh['r2']:.4f} RMSE={mh['rmse']:.1f}W MAPE={mh['mape']:.1f}%")
        print(f"  run means: median err {np.median(errs):.1f}%  worst {max(errs):.1f}%  ({len(errs)} runs)")
        post_sd_h = fmp.laplace_sd(
            X[tr], y[tr], th_h, mult_h, fams[tr], idle_cols, sigma_h
        )
        artifact_hardware[hw].update({
            "coefficients": {f: float(np.exp(th_h[j])) for j, f in enumerate(FEATS2)},
            "posterior_sd_log": {
                f: float(post_sd_h[j]) for j, f in enumerate(FEATS2)
            },
            "family_multipliers": {
                fam_names[k]: float(v) for k, v in mult_h.items()
            },
            "cap_w_per_gpu": cap_w,
            "cap_provenance": "dense-8b/70b training busy-bin per-GPU power p99.5",
            "training_source_ids": [
                entry["source_id"] for entry in _training_sources(run_ids[tr], run_index)
            ],
        })
        artifact_validation[hw] = {
            "training_fit": metrics(y[tr], pred_h[tr]),
            "dense_405b_holdout": {
            "bin_metrics": mh,
            "run_mean_median_error_pct": float(np.median(errs)),
            "run_mean_worst_error_pct": float(max(errs)),
            "num_runs": int(len(errs)),
            },
        }

        # figure: trace overlay + run-mean scatter
        is405 = np.array([mn[i] == "llama-3-405b" for i in d["model_idx"]])
        cand = te & is405 & (d["rate"] == 2.0)
        rid = np.unique(run_ids[cand])[0]
        mm = run_ids == rid
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
        t = np.arange(int(mm.sum()))
        axes[0].plot(t, y[mm], color="k", lw=0.7, label="measured (never trained on)")
        axes[0].plot(t, pred_h[mm], color="#d62728", lw=1.1,
                     label="predicted from 8B/70B physics")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("node power (W)")
        axes[0].set_title("llama-3-405B-FP8, 8×H100 @ 2 req/s — zero-shot")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].scatter(rm, rp, s=22, color="#d62728")
        lim = [min(rm) * 0.9, max(rm) * 1.08]
        axes[1].plot(lim, lim, "k--", lw=1)
        axes[1].plot(lim, [x * 1.1 for x in lim], "k:", lw=0.7)
        axes[1].plot(lim, [x * 0.9 for x in lim], "k:", lw=0.7)
        axes[1].set_xlabel("measured run mean (W)")
        axes[1].set_ylabel("predicted run mean (W)")
        axes[1].set_title(f"all {len(errs)} held-out runs (±10%)")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "holdout_405b.png", dpi=140)
        fig.savefig(out / "holdout_405b.pdf")
        print(f"  wrote {out}/holdout_405b.png/.pdf")

    fit_revision, fit_revision_dirty = _git_revision()
    source_index = [run_index[key] for key in sorted(run_index)]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "fit_revision": fit_revision,
        "fit_revision_dirty": fit_revision_dirty,
        "dt_s": float(np.asarray(d["dt_s"]).item()),
        "feature_order": list(FEATURE_ORDER),
        "feature_equations": {
            "tp": "tp",
            "tp_link": "tp * I(tp > 1)",
            "busy_tp": "tp * I(pre_tok + dec_tok > 0)",
            "sat_bw_a": "tp * (1 - exp(-util_mem / 0.05))",
            "sat_bw_b": "tp * (1 - exp(-util_mem / 0.15))",
            "sat_bw_c": "tp * (1 - exp(-util_mem / 0.4))",
            "flops_pre": "2 * n_active * pre_tok * dtype_scale",
            "flops_dec": "2 * n_active * dec_tok * dtype_scale",
            "w_read_pre": "prefill weight bytes / s",
            "kv_write": "KV write bytes / s",
            "comm": "node-total TP communication bytes / s",
        },
        "architecture_schema": {
            "required": ["family", "n_active", "fp8"],
            "fp8_dtype_scale": 0.5,
        },
        "architectures": {
            str(model): json.loads(str(arch_json))
            for model, arch_json in zip(d["model_names"], d["model_arch_json"])
        },
        "family_multiplier_policy": "fitted known family; unseen family defaults to 1.0",
        "training_bundle_ids": [
            entry["source_id"] for entry in source_index
            if entry["source_layout"] == "bundle"
        ],
        "training_legacy_pair_ids": [
            entry["source_id"] for entry in source_index
            if entry["source_layout"] == "sharegpt"
        ],
        "training_source_index": source_index,
        "training_ledger_artifact": {
            "path": str(ledger_path),
            "sha256": _ledger_hash(ledger_path),
            "run_index_path": str(run_index_path),
            "run_index_sha256": _ledger_hash(run_index_path),
        },
        "residual": None,
        "hardware": artifact_hardware,
        "validation": artifact_validation,
    }

    # The exported H100 transfer artifact must reproduce the exact held-out
    # prediction printed above through the production kernel, run by run.
    if "H100" in artifact_hardware:
        checks = []
        fi_405 = fam_names.index("dense-405b")
        te = (d["hw_idx"] == 1) & (fams == fi_405)
        X = design2(d, "H100", run_ids)
        coef = np.asarray([artifact_hardware["H100"]["coefficients"][f] for f in FEATS2])
        reference = fmp.predict_map(X, np.log(coef), {}, fams, idle_cols)
        reference = np.minimum(
            reference,
            artifact_hardware["H100"]["cap_w_per_gpu"] * d["tp"],
        )
        for run_id in np.unique(run_ids[te]):
            mask = run_ids == run_id
            ledger = {key: d[key][mask] for key in (
                "pre_tok", "dec_tok", "w_read", "kv_read", "w_read_pre", "kv_write", "comm"
            )}
            predicted = predict_mean_node_power(
                ledger,
                {
                    "family": "dense-405b",
                    "n_active": float(d["n_active"][mask][0]),
                    "fp8": int(d["fp8"][mask][0]),
                },
                tp=int(d["tp"][mask][0]),
                hardware="H100",
                artifact=artifact,
            )
            checks.append(float(np.max(np.abs(predicted - reference[mask]))))
        artifact["validation"]["H100"]["kernel_max_abs_diff_w"] = max(checks)
        if max(checks) > 1e-8:
            raise AssertionError("Exported H100 artifact does not reproduce the holdout")

    artifact_path = out / "physics_artifact_v1.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"  wrote {artifact_path}")


if __name__ == "__main__":
    main()
