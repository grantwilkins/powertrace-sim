"""Frozen feature-test evaluator for deterministic physics and causal residuals."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from artifact_export import benchmark_deployments, selected_physics_artifact, selected_provenance_ids  # noqa: E402
from evaluation_core import bootstrap_intervals, causal_design, model_scalar_count, stratified_summaries  # noqa: E402
from evaluation_core import summaries as _summaries, trace_metrics  # noqa: E402
from evaluation_core import write_csv as _write_csv  # noqa: E402
from baselines import (fit_causal_activity_ridge, fit_same_configuration_physics_oracle,  # noqa: E402
                       predict_causal_activity_ridge, predict_same_configuration_physics_oracle)  # noqa: E402
from feature_metrics import secondary_trace_metrics  # noqa: E402
from gates import PREFERENCE as PREFERENCE  # noqa: E402
from gates import choose_hardware_candidate  # noqa: E402
from gates import choose_source_candidate as choose_source_candidate  # noqa: E402
from gates import energy_limit, passes_b2_comparison, passes_correction_safety, passes_primary  # noqa: E402
from gmm_bigru_baseline import fit_s0_gmm_bigru, predict_s0_gmm_bigru  # noqa: E402
from model.classifiers.physics import load_selected_physics_artifact, physics_design, physics_feature_order  # noqa: E402
TAPS_S = (0, 1, 2, 4, 8)
RIDGES = (0.001, 0.01, 0.1, 1.0, 10.0)
CANDIDATES = ("B0", "B1", "B2", "B3", "B4", "M0", "M0b", "M0bR", "M0dR", "M0c", "M4A", "M1", "M2", "M3")
# Board power limits from the NVIDIA datasheets (A100 SXM4 80GB: 400 W;
# H100 SXM5: 700 W). The M0c cap is this physical limit, not a fitted
# quantile of training power.
TDP_W_PER_GPU = {"A100": 400.0, "H100": 700.0}
_METER_KERNEL_PATH = Path(__file__).resolve().parent / "meter_kernel.json"
_METER_KERNEL_CACHE: dict | None = None


def meter_kernel(hardware: str) -> dict:
    """Per-hardware meter response identified once from S0 training steps."""
    global _METER_KERNEL_CACHE
    if _METER_KERNEL_CACHE is None:
        _METER_KERNEL_CACHE = json.loads(_METER_KERNEL_PATH.read_text())
    return _METER_KERNEL_CACHE[hardware]
def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assign_repeats(run_rows: list[dict]) -> dict[int, int]:
    """Assign repeat within config/rate by sorted stable source identity."""
    groups: dict[tuple, list[dict]] = {}
    for row in run_rows:
        key = (row["config_id"], float(row["rate"]))
        groups.setdefault(key, []).append(row)
    return {int(row["run_id"]): repeat for rows in groups.values()
            for repeat, row in enumerate(sorted(rows, key=lambda x: x["source_id"]))}

def _run_rows(d, index: dict) -> list[dict]:
    first = {int(r): int(np.flatnonzero(d["run_id"] == r)[0]) for r in np.unique(d["run_id"])}
    models, hardware = list(map(str, d["model_names"])), list(map(str, d["hw_names"]))
    rows = []
    for source in index["runs"]:
        run_id = int(source["run_id"])
        if run_id not in first:
            continue
        i = first[run_id]
        rows.append({
            **source, "model": models[int(d["model_idx"][i])],
            "hardware": hardware[int(d["hw_idx"][i])], "tp": int(d["tp"][i]),
            "rate": float(d["rate"][i]),
        })
    repeats = assign_repeats(rows)
    for row in rows:
        row["repeat"] = repeats[int(row["run_id"])]
    return rows
def build_splits(rows: list[dict]) -> list[dict]:
    """Build the predeclared S0/S1/S2/S3 conditional-timing masks."""
    splits = []
    for hw in ("A100", "H100"):
        pool = [r for r in rows if r["hardware"] == hw]
        splits.append(_split(f"S0_{hw}", pool, pool, transfer=False))
    source = [r for r in rows if r["hardware"] == "A100" and r["model"] == "gpt-oss-20b" and r["tp"] in (1, 2)]
    target = [r for r in rows if r["hardware"] == "A100" and r["model"] == "gpt-oss-120b" and r["tp"] in (4, 8)]
    splits.append(_split("S1_A100_gpt_oss", source, target, transfer=True))
    source = [r for r in rows if r["hardware"] == "H100" and r["model"] == "llama-3-8b"]
    target = [r for r in rows if r["hardware"] == "H100" and r["model"] == "llama-3-70b" and r["tp"] in (4, 8)]
    splits.append(_split("S2a_H100_llama70", source, target, transfer=True))
    source = [r for r in rows if r["hardware"] == "H100" and r["model"] in ("llama-3-8b", "llama-3-70b")]
    target = [r for r in rows if r["hardware"] == "H100" and r["model"] == "llama-3-405b" and r["tp"] == 8]
    splits.append(_split("S2b_H100_llama405", source, target, transfer=True))
    for hw in ("A100", "H100"):
        pool = [r for r in rows if r["hardware"] == hw]
        for tp in sorted({r["tp"] for r in pool}):
            source = [r for r in pool if r["tp"] != tp]
            target = [r for r in pool if r["tp"] == tp]
            if len({r["tp"] for r in source}) >= 2:
                splits.append(_split(f"S3_{hw}_hold_tp{tp}", source, target, transfer=True))
    return [s for s in splits if s["train"] and s["development"] and s["test"]]
def _split(name: str, source: list[dict], target: list[dict], *, transfer: bool) -> dict:
    train = [r["run_id"] for r in source if r["repeat"] == 0]
    dev = [r["run_id"] for r in source if r["repeat"] == 1]
    refit = [r["run_id"] for r in source] if transfer else train + dev
    test = [r["run_id"] for r in target] if transfer else [r["run_id"] for r in target if r["repeat"] == 2]
    return {"name": name, "mode": "conditional-timing transfer" if transfer else "held-repeat",
            "train": train, "development": dev, "refit": refit, "test": test}
def _physical_design(d, kind="M0", alpha=1.0, hardware="A100", residence=False) -> np.ndarray:
    profiles = {
        "A100": (2e12, 312e12, 600e9),
        "H100": (3.35e12, 990e12, 900e9),
    }
    bandwidth, peak, link = profiles[hardware]
    ledger = {key: d[key] for key in (
        "pre_tok", "dec_tok", "w_read", "w_read_dec", "kv_read",
        "w_read_pre", "kv_write", "comm", "A_t",
    )}
    arch = {key: d[key] for key in ("n_active", "w_bytes", "fp8")}
    if "fp8_flop_frac" in d:
        arch["fp8_flop_frac"] = d["fp8_flop_frac"]
    moving_average_s = (
        float(meter_kernel(hardware)["moving_average_s"]) if kind == "M0c"
        else float(d["dt_s"]) * (2 if hardware == "H100" else 1)
    )
    design, _ = physics_design(
        ledger, arch,
        tp=d["tp"], hardware_profile={
            "hbm_bandwidth_bytes_s": bandwidth,
            "compute_peak_flops_s": peak,
            "link_bandwidth_bytes_s": link,
            "residence_bytes_per_gpu": 80e9,
        }, mean_kind=kind, residence=residence, dt_s=float(d["dt_s"]),
        moving_average_s=moving_average_s,
        ema_alpha=alpha, run_ids=d["run_id"],
    )
    return design


def _nnls_columns(x: np.ndarray, y: np.ndarray, *, drop=()) -> np.ndarray:
    """RMS-scaled non-negative fit that skips empty or dropped columns."""
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    keep = scale > 0
    for column in drop:
        keep[column] = False
    coefficients = np.zeros(x.shape[1])
    if keep.any():
        scaled, _ = nnls(x[:, keep] / scale[keep], y)
        coefficients[keep] = scaled / scale[keep]
    return coefficients


def _prefill_influence(pre_tok, run_ids, window: int) -> np.ndarray:
    """Bins whose meter window overlaps any prefill activity, per run."""
    flags = (np.asarray(pre_tok, dtype=np.float64) > 0.0).astype(np.float64)
    out = np.zeros(flags.shape[0], dtype=bool)
    starts = np.r_[0, np.flatnonzero(np.diff(run_ids)) + 1]
    ends = np.r_[starts[1:], flags.shape[0]]
    for start, end in zip(starts, ends):
        padded = np.r_[np.zeros(window - 1), flags[start:end]]
        csum = np.r_[0.0, np.cumsum(padded)]
        out[start:end] = (csum[window:] - csum[:-window]) > 0
    return out


def _m0c_coefficients(d, split: dict, *, hardware: str, residence: bool,
                      design: np.ndarray) -> np.ndarray:
    """Phase-anchored staged NNLS for the M0c concave mean.

    Decode-only and idle bins identify the floors and the memory response;
    prefill-influenced bins identify the compute response from the stage-one
    residual; the final pass refits the non-compute columns on every training
    bin with the compute response frozen, so the memory curve sees the full
    load range while compute/memory exchange stays impossible. Declared
    assumption with published support (Splitwise ISCA'24 Fig. 8, POLCA):
    decode power is memory-bound, prefill is compute-bound. The influence
    mask covers the meter's moving-average support; the EMA tail beyond it
    is a documented approximation.
    """
    names = physics_feature_order("M0c", residence=residence)
    compute = np.array([name.startswith("compute_ramp") for name in names])
    train = np.isin(d["run_id"], split["refit"])
    y = d["power"].astype(float)
    window = max(1, int(round(
        float(meter_kernel(hardware)["moving_average_s"]) / float(d["dt_s"]))))
    influence = _prefill_influence(d["pre_tok"], d["run_id"], window)
    stage_one = train & ~influence
    stage_two = train & influence
    coefficients = np.zeros(design.shape[1])
    base = design[stage_one][:, ~compute]
    drop = (1,) if np.allclose(base[:, 0], base[:, 1]) else ()
    coefficients[~compute] = _nnls_columns(base, y[stage_one], drop=drop)
    if stage_two.any():
        residual = y[stage_two] - design[stage_two] @ coefficients
        coefficients[compute] = _nnls_columns(design[stage_two][:, compute], residual)
        refined = y[train] - design[train][:, compute] @ coefficients[compute]
        coefficients[~compute] = _nnls_columns(
            design[train][:, ~compute], refined, drop=drop)
    return coefficients


def _physics_fit(d, split: dict, *, kind="M0", alpha=1.0, hardware="A100",
                 residence=False, cap_quantile=0.995, design=None) -> dict:
    train = np.isin(d["run_id"], split["refit"])
    y = d["power"].astype(float)
    x = (_physical_design(d, kind, alpha, hardware, residence)
         if design is None else design)
    if kind == "M0c":
        coefficients = _m0c_coefficients(
            d, split, hardware=hardware, residence=residence, design=x)
        moving_average_s = float(meter_kernel(hardware)["moving_average_s"])
    else:
        scale = np.sqrt(np.mean(x[train] ** 2, axis=0))
        keep = scale > 0
        if np.allclose(x[train, 0], x[train, 1]):
            keep[1] = False
        scaled, _ = nnls(x[train][:, keep] / scale[keep], y[train])
        coefficients = np.zeros(x.shape[1])
        coefficients[keep] = scaled / scale[keep]
        moving_average_s = float(d["dt_s"]) * (2 if hardware == "H100" else 1)
    if cap_quantile == "tdp":
        cap_w_per_gpu = TDP_W_PER_GPU[hardware]
    else:
        busy = (d["pre_tok"] + d["dec_tok"]) > 0
        cap_w_per_gpu = float(np.quantile(
            y[train & busy] / d["tp"][train & busy], cap_quantile))
    return {
        "physics": coefficients, "mean_kind": kind, "lag_alpha": float(alpha),
        "hardware": hardware, "residence": bool(residence),
        "moving_average_s": moving_average_s,
        "cap_quantile": cap_quantile if cap_quantile == "tdp" else float(cap_quantile),
        "cap_w_per_gpu": cap_w_per_gpu,
    }


def _residual_fits(d, split: dict, candidate: str, physics: dict, ridges, design=None):
    train = np.isin(d["run_id"], split["refit"])
    y = d["power"].astype(float)
    x = (_physical_design(d, physics["mean_kind"], physics["lag_alpha"],
                          physics["hardware"], physics["residence"])
         if design is None else design)
    h, names, scale = causal_design(d, candidate, set(split["refit"]))
    gram = h[train].T @ h[train]
    rhs = h[train].T @ (y[train] - x[train] @ physics["physics"])
    eye = np.eye(h.shape[1])
    fits = [{"candidate": candidate, **physics,
             "residual": np.linalg.solve(gram + float(alpha) * eye, rhs),
             "residual_names": names, "scale": scale, "ridge": float(alpha)}
            for alpha in ridges]
    return fits, h, x


def _predict(d, fit: dict) -> np.ndarray:
    if fit["candidate"] == "B0":
        return np.full(d["power"].size, fit["mean"])
    pred = _physical_design(d, fit["mean_kind"], fit["lag_alpha"],
                            fit["hardware"], fit["residence"]) @ fit["physics"]
    if "residual" in fit:
        h, _, _ = causal_design(d, fit["candidate"], set(), scale=fit["scale"])
        pred += h @ fit["residual"]
    return np.minimum(pred, fit["cap_w_per_gpu"] * d["tp"])


def _evaluate(d, split, candidate, fit, rows_by_id, pred=None) -> list[dict]:
    pred = _predict(d, fit) if pred is None else pred
    output = []
    for run_id in split["test"]:
        mask = d["run_id"] == run_id
        meta = rows_by_id[run_id]
        cap = None
        if candidate == "B4":
            config = int(d["model_idx"][mask][0]) * 100 + int(d["tp"][mask][0])
            cap = fit["configurations"][str(config)]["cap_w_per_gpu"] * meta["tp"]
        elif "cap_w_per_gpu" in fit:
            cap = fit["cap_w_per_gpu"] * meta["tp"]
        output.append({"split": split["name"], "candidate": candidate, "run_id": run_id,
                       "source_id": meta["source_id"], "hardware": meta["hardware"], "model": meta["model"],
                       "tp": meta["tp"], "rate": meta["rate"],
                       **trace_metrics(d["power"][mask], pred[mask]),
                       **secondary_trace_metrics(d["power"][mask], pred[mask],
                                                 dt_s=float(d["dt_s"]), cap_w=cap)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-cache", default="feature-test/ledger_cache_250ms.npz")
    parser.add_argument("--run-index", default="feature-test/ledger_cache_250ms.runs.json")
    parser.add_argument("--out-dir", default="results/feature_test_v1")
    args = parser.parse_args(argv)
    cache, index_path, out = Path(args.ledger_cache), Path(args.run_index), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, index = dict(np.load(cache, allow_pickle=False)), json.loads(index_path.read_text())
    if not np.isclose(float(d["dt_s"]), 0.25):
        raise ValueError("Frozen evaluator requires a native 250 ms ledger")
    rows = _run_rows(d, index)
    candidate_seconds = {candidate: 0.0 for candidate in CANDIDATES}
    splits, per_run, fits, source_scores, physics_bases = build_splits(rows), [], {}, {}, {}
    rows_by_id = {r["run_id"]: r for r in rows}
    for split in splits:
        dev_split = {**split, "refit": split["train"], "test": split["development"]}
        hardware = rows_by_id[split["train"][0]]["hardware"]
        alphas = (0.5, 0.55, 0.6)
        source = np.isin(d["run_id"], split["train"])
        residence_ok = np.unique(d["w_bytes"][source]).size >= 2
        mean_specs = {
            "B3": ("M0", 0.6 if hardware == "A100" else 0.7, False, (0.995,)),
            "M0": ("M0", None, False, (0.995,)),
            "M0b": ("M0b", None, False, (0.995,)),
            "M0bR": ("M0b", None, residence_ok, (0.995,)),
            "M0dR": ("M0d", None, residence_ok, (0.95, 0.975, 0.995)),
            "M0c": ("M0c", float(meter_kernel(hardware)["ema_alpha"]),
                    residence_ok, ("tdp",)),
            "M4A": ("M4A", 0.5, residence_ok, (0.995,)),
        }
        dev_means, final_means, final_predictions, dev_scores = {}, {}, {}, {}
        for name, (kind, fixed_alpha, residence, caps) in mean_specs.items():
            candidate_started = time.perf_counter()
            choices = []
            for alpha in ((fixed_alpha,) if fixed_alpha is not None else alphas):
                design = _physical_design(d, kind, alpha, hardware, residence)
                base = _physics_fit(d, dev_split, kind=kind, alpha=alpha,
                                    hardware=hardware, residence=residence,
                                    cap_quantile=caps[0], design=design)
                raw = design @ base["physics"]
                train = np.isin(d["run_id"], dev_split["refit"])
                busy = (d["pre_tok"] + d["dec_tok"]) > 0
                for cap in caps:
                    cap_w = (TDP_W_PER_GPU[hardware] if cap == "tdp"
                             else float(np.quantile(
                                 d["power"][train & busy] / d["tp"][train & busy], cap)))
                    trial = {**base, "candidate": name, "cap_quantile": cap,
                             "cap_w_per_gpu": cap_w}
                    pred = np.minimum(raw, trial["cap_w_per_gpu"] * d["tp"])
                    score = _summaries(_evaluate(
                        d, dev_split, name, trial, rows_by_id, pred))[0]
                    failed = score["energy_error_pct_median"] > energy_limit(split["name"])
                    choices.append((failed, score["energy_error_pct_median"] if failed else 0.0,
                                    -score["acf_r2_median"], score["nrmse_range_median"],
                                    alpha, cap, trial, score))
            chosen = min(choices, key=lambda x: x[:6])
            dev_means[name], dev_scores[name] = chosen[6], chosen[7]
            if chosen[4] != alpha:
                design = _physical_design(d, kind, chosen[4], hardware, residence)
            final_means[name] = _physics_fit(
                d, split, kind=kind, alpha=chosen[4], hardware=hardware,
                residence=residence, cap_quantile=chosen[5],
                design=design,
            )
            final_predictions[name] = np.minimum(
                design @ final_means[name]["physics"],
                final_means[name]["cap_w_per_gpu"] * d["tp"],
            )
            candidate_seconds[name] += time.perf_counter() - candidate_started
        residual_mean = min(
            ("M0", "M0b", "M0bR", "M0c"),
            key=lambda name: (dev_scores[name]["energy_error_pct_median"] > energy_limit(split["name"]),
                              dev_scores[name]["energy_error_pct_median"] if
                              dev_scores[name]["energy_error_pct_median"] > energy_limit(split["name"]) else 0.0,
                              -dev_scores[name]["acf_r2_median"], dev_scores[name]["nrmse_range_median"]),
        )
        physics_bases[split["name"]] = residual_mean
        dev_physics, final_physics = dev_means[residual_mean], final_means[residual_mean]
        residual_design = _physical_design(
            d, final_physics["mean_kind"], final_physics["lag_alpha"],
            hardware, final_physics["residence"],
        )
        config = d["model_idx"].astype(int) * 100 + d["tp"].astype(int)
        busy = (d["pre_tok"] + d["dec_tok"]) > 0
        for candidate in CANDIDATES:
            if candidate in ("B2", "B4") and not split["name"].startswith("S0"):
                continue
            if candidate == "B2":
                candidate_started = time.perf_counter()
                fit = fit_s0_gmm_bigru(
                    d["run_id"], config, d["A_t"], d["delta_A_t"], d["power"],
                    split["train"], split["development"],
                )
                pred = predict_s0_gmm_bigru(
                    d["run_id"], config, d["A_t"], d["delta_A_t"], split["test"], fit,
                )
                fit["training_source_ids"] = sorted(
                    rows_by_id[r]["source_id"] for r in split["train"] + split["development"]
                )
                fits[(split["name"], candidate)] = fit
                per_run += _evaluate(d, split, candidate, fit, rows_by_id, pred)
                candidate_seconds[candidate] += time.perf_counter() - candidate_started
                continue
            candidate_started = time.perf_counter()
            ridge = None
            if candidate == "B0":
                dev_fit = {"candidate": "B0", "mean": float(np.mean(
                    d["power"][np.isin(d["run_id"], dev_split["refit"])]))}
                dev_scores[candidate] = _summaries(
                    _evaluate(d, dev_split, candidate, dev_fit, rows_by_id))[0]
            if candidate == "B1":
                trials = []
                for alpha in RIDGES:
                    trial = fit_causal_activity_ridge(
                        d["run_id"], d["A_t"], d["delta_A_t"], d["power"],
                        dev_split["refit"], dt_s=float(d["dt_s"]), ridge=alpha)
                    pred = predict_causal_activity_ridge(
                        d["run_id"], d["A_t"], d["delta_A_t"], trial)
                    score = _summaries(_evaluate(
                        d, dev_split, candidate, trial, rows_by_id, pred))[0]
                    trials.append((-score["acf_r2_median"], score["nrmse_range_median"], alpha, score))
                _, _, ridge, dev_scores[candidate] = min(trials)
            if candidate == "B4":
                design = _physical_design(d, "M0", 0.6, hardware)
                dev_fit = fit_same_configuration_physics_oracle(
                    design, d["power"], d["tp"], busy, d["run_id"], config,
                    dev_split["refit"])
                known = np.isin(config, [int(x) for x in dev_fit["configurations"]])
                pred = np.full(d["power"].shape, np.nan)
                pred[known] = predict_same_configuration_physics_oracle(
                    design[known], d["tp"][known], config[known], dev_fit)
                dev_scores[candidate] = _summaries(_evaluate(
                    d, dev_split, candidate, dev_fit, rows_by_id, pred))[0]
            if candidate in ("M1", "M2", "M3"):
                trials = []
                dev_fits, dev_h, dev_x = _residual_fits(
                    d, dev_split, candidate, dev_physics, RIDGES, residual_design
                )
                for trial in dev_fits:
                    pred = np.minimum(
                        dev_x @ trial["physics"] + dev_h @ trial["residual"],
                        trial["cap_w_per_gpu"] * d["tp"],
                    )
                    score = _summaries(
                        _evaluate(d, dev_split, candidate, trial, rows_by_id, pred)
                    )[0]
                    failed = score["energy_error_pct_median"] > energy_limit(split["name"])
                    trials.append((failed, score["energy_error_pct_median"] if failed else 0.0,
                                   -score["acf_r2_median"], score["nrmse_range_median"],
                                   trial["ridge"], score))
                chosen_residual = min(trials, key=lambda x: x[:5])
                ridge, dev_scores[candidate] = chosen_residual[4:]
                del dev_fits, dev_h, dev_x
            if candidate == "B0":
                fit = {"candidate": "B0", "mean": float(np.mean(
                    d["power"][np.isin(d["run_id"], split["refit"])]))}
            elif candidate == "B1":
                fit = fit_causal_activity_ridge(
                    d["run_id"], d["A_t"], d["delta_A_t"], d["power"],
                    split["refit"], dt_s=float(d["dt_s"]), ridge=ridge)
                pred = predict_causal_activity_ridge(
                    d["run_id"], d["A_t"], d["delta_A_t"], fit)
            elif candidate == "B4":
                fit = fit_same_configuration_physics_oracle(
                    design, d["power"], d["tp"], busy, d["run_id"], config,
                    split["refit"])
                known = np.isin(config, [int(x) for x in fit["configurations"]])
                pred = np.full(d["power"].shape, np.nan)
                pred[known] = predict_same_configuration_physics_oracle(
                    design[known], d["tp"][known], config[known], fit)
            elif candidate in ("M1", "M2", "M3"):
                final_fits, final_h, final_x = _residual_fits(
                    d, split, candidate, final_physics, (ridge,), residual_design
                )
                fit = final_fits[0]
            else:
                fit = {"candidate": candidate, **final_means[candidate]}
            fit["training_source_ids"] = sorted(rows_by_id[r]["source_id"] for r in split["refit"])
            fits[(split["name"], candidate)] = fit
            if candidate in ("B1", "B4"):
                per_run += _evaluate(d, split, candidate, fit, rows_by_id, pred)
            elif candidate in ("M1", "M2", "M3"):
                pred = np.minimum(
                    final_x @ fit["physics"] + final_h @ fit["residual"],
                    fit["cap_w_per_gpu"] * d["tp"],
                )
                per_run += _evaluate(d, split, candidate, fit, rows_by_id, pred)
                del final_fits, final_h, final_x, pred
            else:
                pred = final_predictions.get(candidate)
                per_run += _evaluate(d, split, candidate, fit, rows_by_id, pred)
            candidate_seconds[candidate] += time.perf_counter() - candidate_started
        source_scores[split["name"]] = dev_scores
    selected_by_hardware = {}
    for hardware in ("A100", "H100"):
        names = [s["name"] for s in splits if rows_by_id[s["train"][0]]["hardware"] == hardware]
        selected_by_hardware[hardware] = choose_hardware_candidate(
            names, source_scores, physics_bases)
    selected_by_split = {
        split["name"]: selected_by_hardware[rows_by_id[split["train"][0]]["hardware"]]
        for split in splits
    }
    aggregate = _summaries(per_run)
    aggregate_by_key = {(row["split"], row["candidate"]): row for row in aggregate}
    data_efficiency = []
    for hardware, candidate in selected_by_hardware.items():
        full_fit = fits.get((f"S0_{hardware}", candidate))
        if full_fit is None or "physics" not in full_fit or "residual" in full_fit:
            continue
        split = next(item for item in splits if item["name"] == f"S0_{hardware}")
        one_fit = {"candidate": candidate, **_physics_fit(
            d, {**split, "refit": split["train"]}, kind=full_fit["mean_kind"],
            alpha=full_fit["lag_alpha"], hardware=hardware,
            residence=full_fit["residence"], cap_quantile=full_fit["cap_quantile"],
        )}
        one = _summaries(_evaluate(d, split, candidate, one_fit, rows_by_id))[0]
        full = aggregate_by_key[(split["name"], candidate)]
        energy_delta = one["energy_error_pct_median"] - full["energy_error_pct_median"]
        acf_delta = full["acf_r2_median"] - one["acf_r2_median"]
        data_efficiency.append({"hardware": hardware, "candidate": candidate,
                                "one_repeat_energy_delta_pct_points": energy_delta,
                                "one_repeat_acf_r2_decrease": acf_delta,
                                "passes": energy_delta <= 2.0 and acf_delta <= 0.05})
    gates = []
    for r in aggregate:
        name = r["split"]
        bias_ok = True
        if name.startswith("S3"):
            values = [x for x in per_run if x["split"] == name and x["candidate"] == r["candidate"]]
            rates = sorted({x["rate"] for x in values})
            bias_ok = all(abs(np.median([x["mean_bias_pct"] for x in values if x["rate"] == rate])) <= 10
                          for rate in (rates[0], rates[-1]))
        correction_ok = (
            passes_correction_safety(r, aggregate_by_key[(name, physics_bases[name])])
            if r["candidate"] in ("M1", "M2", "M3") else True
        )
        b2 = aggregate_by_key.get((name, "B2"))
        b2_ok = not name.startswith("S0") or b2 is None or passes_b2_comparison(r, b2)
        passes = passes_primary(r, name) and bias_ok and correction_ok and b2_ok
        gates.append({**r, "low_high_rate_bias_passes": bias_ok,
                      "correction_safety_passes": correction_ok,
                      "b2_comparison_passes": b2_ok, "passes": bool(passes)})
    choices = set(selected_by_hardware.values())
    selected = next(iter(choices)) if len(choices) == 1 else "hardware_local"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    split_doc = {"ledger_sha256": _sha256(cache), "run_index_sha256": _sha256(index_path), "code_revision": revision,
                 "dt_s": 0.25, "splits": splits}
    _json(out / "split_manifest.json", split_doc)
    exact_state = all(k in d for k in ("A_t", "delta_A_t", "running_requests", "waiting_requests"))
    exact_marks = all(k in d for k in ("arrivals", "input_tokens_arriving", "output_tokens_requested"))
    limitations = (["current cache lacks waiting/backlog and true A_t"] if not exact_state else []) + \
                  (["M2 request marks are executed-work proxies"] if not exact_marks else [])
    _json(out / "candidate_manifest.json", {"candidates": CANDIDATES, "taps_s": TAPS_S, "ridge_strengths": RIDGES,
          "exact_request_state": exact_state, "exact_offered_marks": exact_marks,
          "limitations": limitations})
    _write_csv(out / "per_run_metrics.csv", per_run)
    _write_csv(out / "aggregate_metrics.csv", aggregate)
    _write_csv(out / "stratified_metrics.csv", stratified_summaries(per_run))
    _write_csv(out / "bootstrap_intervals.csv", bootstrap_intervals(per_run))
    _write_csv(out / "data_efficiency.csv", data_efficiency)
    _write_csv(out / "transfer_scorecard.csv", gates)
    complexity = [{"candidate": candidate,
                   "learned_scalars_max": max((model_scalar_count(fit) for (split, name), fit in fits.items()
                                                if name == candidate), default=0),
                   "fit_and_evaluate_s": candidate_seconds[candidate]}
                  for candidate in CANDIDATES]
    _write_csv(out / "model_complexity.csv", complexity)
    unsupported = (["true A_t and waiting backlog unavailable"] if not exact_state else []) + \
                  (["offered request marks unavailable"] if not exact_marks else []) + \
                  ["arrival-only scheduler not validated", "PP/EP/DP/CP not validated"]
    failures = [{"scope": "ledger", "status": "unsupported_by_evidence", "reason": x} for x in unsupported]
    deployments, deployment_specs = {}, {}
    deployable = {"B3", "M0", "M0b", "M0bR", "M0dR", "M0c", "M4A"}
    for hardware, candidate in selected_by_hardware.items():
        if candidate not in deployable:
            failures.append({"scope": hardware, "status": "no_deployable_artifact",
                             "reason": f"selected residual mode {candidate} lacks production support"})
            continue
        selection_ids, excluded_ids = selected_provenance_ids(
            hardware, splits, rows_by_id)
        artifact = selected_physics_artifact(
            hardware, candidate, fits[(f"S0_{hardware}", candidate)], rows,
            selection_source_ids=selection_ids,
            excluded_target_source_ids=excluded_ids,
        )
        artifact_path = out / f"selected_physics_{hardware}.json"
        _json(artifact_path, artifact)
        load_selected_physics_artifact(artifact_path)
        deployments[hardware] = artifact_path.name
        deployment_specs[hardware] = (candidate, artifact, artifact_path)
    benchmarks = benchmark_deployments(deployment_specs, d, splits, fits)
    _write_csv(out / "deployment_benchmark.csv", benchmarks)
    _write_csv(out / "support_failures.csv", failures)
    serial_fits = {f"{s}:{c}": {k: (v.tolist() if isinstance(v, np.ndarray) else [x.tolist() for x in v] if k == "scale" else v)
                                for k, v in fit.items()} for (s, c), fit in fits.items()
                   if c == selected or selected_by_split.get(s) == c}
    chosen_gates = [g for g in gates if selected_by_split[g["split"]] == g["candidate"]]
    _json(out / "selected_model.json", {
          "selection_basis": "source_development_only_after_retrospective_model_design",
          "sealed_external_validation": False,
          "selected_candidate": selected, "selected_by_hardware": selected_by_hardware,
          "validation_role_by_split": selected_by_split,
          "deployment_artifacts": deployments,
          "deployment_benchmarks": benchmarks,
          "data_efficiency": data_efficiency,
          "selected_target_gates_pass": all(g["passes"] for g in chosen_gates),
          "roles": {"transfer": selected_by_split},
          "validation_fits": serial_fits,
          "failed_or_unsupported_gates": [g for g in gates if not g["passes"]] + failures})
    (out / "README.md").write_text(
        "# Feature test v1\n\nDeterministic 250 ms conditional-timing evaluation. "
        f"Source-only hardware modes: {selected_by_hardware}. "
        f"All selected gates pass: {all(g['passes'] for g in chosen_gates)}. "
        "Exact failures and artifacts are in `selected_model.json`; "
        "arrival-only and non-TP parallelism remain unsupported.\n")
if __name__ == "__main__":
    main()
