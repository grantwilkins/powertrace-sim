#!/usr/bin/env python3
"""
A2 feature sufficiency figure: held-out predictive information retention for
{A}, {dA}, F2, F3, F6 versus regime labels z_t.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.classifiers.gru import GRUClassifier

SUBSET_ORDER = ["A", "ΔA", "F2", "F3", "F6"]


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _parse_include_configs(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None:
        return None
    out = {tok.strip() for tok in str(raw).split(",") if tok.strip()}
    return out if out else None


def _resolve_device(device: str) -> torch.device:
    value = str(device).strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_config_id(config_id: str) -> Tuple[str, str, int]:
    m = re.match(r"^(?P<model>.+)_(?P<hw>A100|H100)_tp(?P<tp>\d+)$", str(config_id).strip())
    if m is None:
        raise ValueError(f"Invalid config_id format: {config_id}")
    model = str(m.group("model"))
    hw = str(m.group("hw"))
    tp = int(m.group("tp"))
    return model, hw, tp


def map_model_name_for_npz(model_name: str) -> str:
    model = str(model_name).strip()
    if model.startswith("deepseek-r1-distill-"):
        return model.replace("deepseek-r1-distill-", "deepseek-r1-", 1)
    return model


def map_config_to_npz_stem(config_id: str) -> str:
    model, hw, _tp = parse_config_id(config_id)
    mapped_model = map_model_name_for_npz(model)
    return f"vllm-benchmark_{mapped_model}_{hw.lower()}"


def compute_t_arrive_log(power_timestamps: np.ndarray, request_timestamps: np.ndarray) -> np.ndarray:
    n_power = int(power_timestamps.size)
    out = np.zeros((n_power,), dtype=np.float64)

    if n_power < 2 or request_timestamps.size == 0:
        return out

    dt = float(np.median(np.diff(power_timestamps)))
    arrivals = np.sort(np.asarray(request_timestamps, dtype=np.float64).reshape(-1))

    for i, t in enumerate(power_timestamps):
        if i == 0:
            interval_start = float(t - (dt / 2.0))
        else:
            interval_start = float((t + power_timestamps[i - 1]) / 2.0)

        if i == (n_power - 1):
            interval_end = float(t + (dt / 2.0))
        else:
            interval_end = float((t + power_timestamps[i + 1]) / 2.0)

        hits = arrivals[(arrivals >= interval_start) & (arrivals < interval_end)]
        if hits.size == 0:
            continue

        first_arrival = float(hits[0])
        idx = int(np.searchsorted(arrivals, first_arrival))
        if idx > 0:
            inter_arrival = max(0.0, first_arrival - float(arrivals[idx - 1]))
            out[i] = float(np.log1p(inter_arrival))
    return out


def histogram_requests(
    bin_ts: np.ndarray,
    req_ts: np.ndarray,
    in_tok: np.ndarray,
    out_tok: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(bin_ts, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    dt = float(np.median(np.diff(ts))) if ts.size > 1 else 0.25
    if ts.size > 1 and not np.all(np.diff(ts) > 0):
        ts = np.unique(ts)
        if ts.size <= 1:
            if ts.size == 0:
                ts = np.array([0.0, dt], dtype=np.float64)
            else:
                ts = np.array([ts[0], ts[0] + dt], dtype=np.float64)
    elif ts.size <= 1:
        if ts.size == 0:
            ts = np.array([0.0, dt], dtype=np.float64)
        else:
            ts = np.array([ts[0], ts[0] + dt], dtype=np.float64)

    edges = np.append(ts, ts[-1] + dt)
    req = np.asarray(req_ts, dtype=np.float64).reshape(-1)
    nin = np.asarray(in_tok, dtype=np.float64).reshape(-1)
    nout = np.asarray(out_tok, dtype=np.float64).reshape(-1)

    n = int(min(req.size, nin.size, nout.size))
    req = req[:n]
    nin = nin[:n]
    nout = nout[:n]

    cnt, _ = np.histogram(req, edges)
    tok_in, _ = np.histogram(req, edges, weights=nin)
    tok_out, _ = np.histogram(req, edges, weights=nout)
    return cnt.astype(np.float64), tok_in.astype(np.float64), tok_out.astype(np.float64)


def build_trace_feature_subsets(
    *,
    timestamps: np.ndarray,
    active_requests: np.ndarray,
    prefill_tokens: np.ndarray,
    decode_tokens: np.ndarray,
    request_timestamps: np.ndarray,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
) -> Dict[str, np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    active = np.asarray(active_requests, dtype=np.float64).reshape(-1)
    prefill = np.asarray(prefill_tokens, dtype=np.float64).reshape(-1)
    decode = np.asarray(decode_tokens, dtype=np.float64).reshape(-1)

    n = int(min(ts.size, active.size, prefill.size, decode.size))
    ts = ts[:n]
    active = active[:n]
    prefill = prefill[:n]
    decode = decode[:n]

    if n <= 1:
        return {
            "A": np.zeros((0, 1), dtype=np.float64),
            "ΔA": np.zeros((0, 1), dtype=np.float64),
            "F2": np.zeros((0, 2), dtype=np.float64),
            "F3": np.zeros((0, 3), dtype=np.float64),
            "F6": np.zeros((0, 6), dtype=np.float64),
        }

    req_ts = np.asarray(request_timestamps, dtype=np.float64).reshape(-1)
    nin = np.asarray(input_tokens, dtype=np.float64).reshape(-1)
    nout = np.asarray(output_tokens, dtype=np.float64).reshape(-1)
    rn = int(min(req_ts.size, nin.size, nout.size))
    req_ts = req_ts[:rn]
    nin = nin[:rn]
    nout = nout[:rn]

    valid_req = np.isfinite(req_ts) & np.isfinite(nin) & np.isfinite(nout)
    req_ts = req_ts[valid_req]
    nin = nin[valid_req]
    nout = nout[valid_req]

    cnt, tok_in, tok_out = histogram_requests(ts, req_ts, nin, nout)
    t_arrive_log = compute_t_arrive_log(ts, req_ts)

    m = int(min(n, cnt.size, tok_in.size, tok_out.size, t_arrive_log.size))
    active = active[:m]
    prefill = prefill[:m]
    decode = decode[:m]
    cnt = cnt[:m]
    tok_in = tok_in[:m]
    tok_out = tok_out[:m]
    t_arrive_log = t_arrive_log[:m]

    delta = active[1:] - active[:-1]

    f6_full = np.stack([cnt, tok_in, tok_out, active, prefill, decode], axis=1)

    return {
        "A": active[1:].reshape(-1, 1),
        "ΔA": delta.reshape(-1, 1),
        "F2": np.stack([active[1:], delta], axis=1),
        "F3": np.stack([active[1:], delta, t_arrive_log[1:]], axis=1),
        "F6": f6_full[1:, :],
    }


def _prepare_gmm_arrays(payload: Mapping[str, object]) -> Dict[str, np.ndarray]:
    means_sorted = np.asarray(payload.get("means", []), dtype=np.float64).reshape(-1)
    variances_sorted = np.asarray(payload.get("variances", []), dtype=np.float64).reshape(-1)
    weights_sorted = np.asarray(payload.get("weights", []), dtype=np.float64).reshape(-1)
    k = int(payload.get("k", means_sorted.size))

    if means_sorted.size != k or variances_sorted.size != k or weights_sorted.size != k:
        raise ValueError("Invalid GMM payload: means/variances/weights lengths must match k")

    variances_sorted = np.clip(variances_sorted, a_min=1e-12, a_max=None)
    weights_sorted = np.clip(weights_sorted, a_min=1e-12, a_max=None)
    weights_sorted = weights_sorted / np.sum(weights_sorted)

    order = np.asarray(payload.get("order", []), dtype=np.int64).reshape(-1)
    label_map = np.asarray(payload.get("label_map", []), dtype=np.int64).reshape(-1)

    if order.size == k and label_map.size == k:
        means_raw = np.empty((k,), dtype=np.float64)
        vars_raw = np.empty((k,), dtype=np.float64)
        weights_raw = np.empty((k,), dtype=np.float64)
        means_raw[order] = means_sorted
        vars_raw[order] = variances_sorted
        weights_raw[order] = weights_sorted
        return {
            "means_raw": means_raw,
            "variances_raw": vars_raw,
            "weights_raw": weights_raw,
            "label_map": label_map.astype(np.int64),
        }

    return {
        "means_raw": means_sorted,
        "variances_raw": variances_sorted,
        "weights_raw": weights_sorted,
        "label_map": np.arange(k, dtype=np.int64),
    }


def predict_regime_labels_from_gmm(power_values: np.ndarray, gmm_payload: Mapping[str, object]) -> np.ndarray:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    gmm = _prepare_gmm_arrays(gmm_payload)
    means = gmm["means_raw"]
    variances = gmm["variances_raw"]
    weights = gmm["weights_raw"]

    y2 = y.reshape(-1, 1)
    log_prob = (
        -0.5 * np.log(2.0 * np.pi * variances.reshape(1, -1))
        -0.5 * ((y2 - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1)
        + np.log(weights.reshape(1, -1))
    )
    raw_labels = np.argmax(log_prob, axis=1).astype(np.int64)
    label_map = gmm["label_map"]
    if np.max(raw_labels, initial=-1) >= label_map.size:
        raise ValueError("Internal label mapping mismatch while predicting z_t")
    return label_map[raw_labels].astype(np.int64)


def compute_ce_null(labels: np.ndarray, num_classes: int) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0 or int(num_classes) <= 0:
        return float("nan")
    counts = np.bincount(np.clip(y, a_min=0, a_max=int(num_classes) - 1), minlength=int(num_classes)).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return float("nan")
    p = counts / total
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def compute_info_retention(ce_subset: float, ce_null: float, ce_f6: float) -> Tuple[float, float]:
    ce_subset_f = float(ce_subset)
    ce_null_f = float(ce_null)
    ce_f6_f = float(ce_f6)
    if (not np.isfinite(ce_subset_f)) or (not np.isfinite(ce_null_f)) or ce_null_f <= 0.0:
        return float("nan"), float("nan")

    ir_abs = 1.0 - (ce_subset_f / ce_null_f)

    gap = ce_null_f - ce_f6_f
    if (not np.isfinite(gap)) or abs(gap) <= 1e-12:
        ir_vs_f6 = float("nan")
    else:
        ir_vs_f6 = (ce_null_f - ce_subset_f) / gap

    if ir_abs < 0.0 and ir_abs > -1e-12:
        ir_abs = 0.0
    return float(ir_abs), float(ir_vs_f6)


def bootstrap_median_ci(values: Iterable[float], n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    med = float(np.median(arr))
    if int(n_bootstrap) <= 0:
        return med, float("nan"), float("nan")

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(arr.size), size=(int(n_bootstrap), int(arr.size)))
    samples = arr[idx]
    meds = np.median(samples, axis=1)
    lo = float(np.percentile(meds, 2.5))
    hi = float(np.percentile(meds, 97.5))
    return med, lo, hi


def _extract_trace_row(arr: np.ndarray, idx: int) -> np.ndarray:
    row = np.asarray(arr[idx], dtype=np.float64).reshape(-1)
    return row


def _load_npz_trace_count(data: Mapping[str, np.ndarray]) -> int:
    return int(np.asarray(data["tensor_parallelism"]).reshape(-1).size)


def _resolve_run_manifest_configs(run_manifest_path: str, include_configs: Optional[set[str]]) -> Dict[str, Dict[str, object]]:
    with open(run_manifest_path, "r") as f:
        payload = json.load(f)

    cfgs = payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("run manifest missing 'configs' dict")

    out: Dict[str, Dict[str, object]] = {}
    for config_id, entry in cfgs.items():
        if include_configs is not None and str(config_id) not in include_configs:
            continue
        if not isinstance(entry, dict):
            continue
        out[str(config_id)] = dict(entry)
    return out


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru_sharegpt_all"
            / "kauto_max12_f2"
            / "run_manifest.json"
        ),
        "training_data_dir": str(repo_root / "model" / "training_data"),
        "gmm_dir": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru_sharegpt_all"
            / "kauto_max12_f2"
            / "gmms"
        ),
        "out_figure": str(repo_root / "figures" / "feature_sufficiency_curve.pdf"),
        "out_per_config_csv": str(repo_root / "results" / "eval_paper" / "feature_sufficiency_per_config.csv"),
        "out_summary_csv": str(repo_root / "results" / "eval_paper" / "feature_sufficiency_summary.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "feature_sufficiency_manifest.json"),
    }


def _config_seed(base_seed: int, config_id: str, subset: str) -> int:
    h = sum(ord(c) for c in f"{config_id}:{subset}")
    return int(base_seed + h)


def _split_trace_indices(n: int, seed: int, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    n_train = int(max(1, min(n - 1, round(train_ratio * n))))
    train_idx = np.sort(idx[:n_train])
    test_idx = np.sort(idx[n_train:])
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _train_gru_ce(
    *,
    train_traces: Sequence[Tuple[np.ndarray, np.ndarray]],
    test_traces: Sequence[Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    num_classes: int,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> float:
    if len(train_traces) == 0 or len(test_traces) == 0:
        return float("nan")

    def _to_float_tensor(x_np: np.ndarray) -> torch.Tensor:
        x_arr = np.asarray(x_np, dtype=np.float32)
        try:
            return torch.from_numpy(x_arr)
        except Exception:
            # Fallback for environments where torch numpy bindings are unavailable.
            return torch.tensor(x_arr.tolist(), dtype=torch.float32)

    def _to_long_tensor(y_np: np.ndarray) -> torch.Tensor:
        y_arr = np.asarray(y_np, dtype=np.int64)
        try:
            return torch.from_numpy(y_arr)
        except Exception:
            # Fallback for environments where torch numpy bindings are unavailable.
            return torch.tensor(y_arr.tolist(), dtype=torch.long)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = GRUClassifier(Dx=int(input_dim), K=int(num_classes), H=int(hidden_dim), num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    order_rng = np.random.default_rng(int(seed) + 1)

    for _epoch in range(int(max(1, epochs))):
        model.train()
        order = order_rng.permutation(len(train_traces))
        for j in order:
            x_np, y_np = train_traces[int(j)]
            if x_np.shape[0] <= 0:
                continue
            x_t = _to_float_tensor(x_np).to(device).unsqueeze(0)
            y_t = _to_long_tensor(y_np).to(device).unsqueeze(0)

            logits = model(x_t)
            loss = criterion(logits.reshape(-1, int(num_classes)), y_t.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for x_np, y_np in test_traces:
            if x_np.shape[0] <= 0:
                continue
            x_t = _to_float_tensor(x_np).to(device).unsqueeze(0)
            y_t = _to_long_tensor(y_np).to(device).unsqueeze(0)
            logits = model(x_t)
            loss = criterion_sum(logits.reshape(-1, int(num_classes)), y_t.reshape(-1))
            total_loss += float(loss.item())
            total_steps += int(y_np.size)

    if total_steps <= 0:
        return float("nan")
    return float(total_loss / float(total_steps))


def _plot_summary(summary_rows: Sequence[Mapping[str, object]], out_path: str) -> None:
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42, "ps.fonttype": 42})

    x = np.arange(len(SUBSET_ORDER), dtype=np.int64)

    med_vs = np.asarray([float(next(r for r in summary_rows if r["subset"] == s)["median_ir_vs_f6"]) for s in SUBSET_ORDER], dtype=np.float64)
    lo_vs = np.asarray([float(next(r for r in summary_rows if r["subset"] == s)["ci95_low_ir_vs_f6"]) for s in SUBSET_ORDER], dtype=np.float64)
    hi_vs = np.asarray([float(next(r for r in summary_rows if r["subset"] == s)["ci95_high_ir_vs_f6"]) for s in SUBSET_ORDER], dtype=np.float64)

    med_abs = np.asarray([float(next(r for r in summary_rows if r["subset"] == s)["median_ir_abs"]) for s in SUBSET_ORDER], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(x, med_vs, marker="o", linewidth=2.1, color="#2c3e50", label="IR_vs_F6 (primary)")
    ax.fill_between(x, lo_vs, hi_vs, color="#2c3e50", alpha=0.15)

    ax.plot(x, med_abs, marker="s", linewidth=1.7, linestyle="--", color="#1f77b4", alpha=0.9, label="IR_abs")

    ax.axhline(0.95, linestyle="--", linewidth=1.0, color="#e74c3c", alpha=0.9)
    ax.text(x[-1] + 0.10, 0.95, "95%", color="#e74c3c", va="center", ha="left", fontsize=9)

    for i, y in enumerate(med_vs):
        if np.isfinite(y):
            ax.text(i, y + 0.03, f"{(100.0 * y):.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(SUBSET_ORDER)
    ax.set_ylabel("Information Retention")
    ax.set_xlabel("Feature Subset")
    ax.set_title("Feature Sufficiency: Held-out Predictive Information Retention")

    finite_vals = np.concatenate([med_vs[np.isfinite(med_vs)], hi_vs[np.isfinite(hi_vs)], med_abs[np.isfinite(med_abs)]])
    if finite_vals.size > 0:
        y_min = float(min(-0.05, np.min(finite_vals) - 0.08))
        y_max = float(max(1.05, np.max(finite_vals) + 0.08))
    else:
        y_min, y_max = -0.05, 1.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.30, len(SUBSET_ORDER) - 0.70 + 0.35)
    ax.legend(loc="best", frameon=False)

    _ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_feature_sufficiency_figure(
    *,
    run_manifest: str,
    training_data_dir: str,
    gmm_dir: str,
    bootstrap_samples: int,
    seed: int,
    out_figure: str,
    out_per_config_csv: str,
    out_summary_csv: str,
    out_json: str,
    include_configs: Optional[str] = None,
    device: str = "auto",
    epochs: int = 8,
    hidden_dim: int = 32,
    lr: float = 1e-3,
) -> Dict[str, object]:
    include_set = _parse_include_configs(include_configs)
    run_configs = _resolve_run_manifest_configs(run_manifest, include_set)
    torch_device = _resolve_device(device)

    skipped: Dict[str, str] = {}
    selected_configs: List[Dict[str, object]] = []
    per_config_rows: List[Dict[str, object]] = []
    subset_values_vs_f6: Dict[str, List[float]] = {s: [] for s in SUBSET_ORDER}
    subset_values_abs: Dict[str, List[float]] = {s: [] for s in SUBSET_ORDER}

    for config_id in sorted(run_configs.keys()):
        entry = run_configs[config_id]
        try:
            model_name, hw, tp = parse_config_id(config_id)
        except ValueError as exc:
            skipped[config_id] = f"invalid_config_id:{exc}"
            continue

        mapped_model = map_model_name_for_npz(model_name)
        npz_name = f"vllm-benchmark_{mapped_model}_{hw.lower()}.npz"
        npz_path = str(Path(training_data_dir) / npz_name)
        if not os.path.exists(npz_path):
            skipped[config_id] = "missing_npz"
            continue

        k = int(entry.get("k", 10))
        gmm_name = f"{_safe_slug(config_id)}_k{k}.json"
        gmm_path = str(Path(gmm_dir) / gmm_name)
        if not os.path.exists(gmm_path):
            skipped[config_id] = "missing_gmm"
            continue

        try:
            with open(gmm_path, "r") as f:
                gmm_payload = json.load(f)

            traces: List[Dict[str, object]] = []
            with np.load(npz_path, allow_pickle=True) as data:
                n_traces_all = _load_npz_trace_count(data)
                tp_all = np.asarray(data["tensor_parallelism"], dtype=np.int64).reshape(-1)

                idxs = np.where(tp_all == int(tp))[0]
                if idxs.size < 2:
                    skipped[config_id] = "insufficient_tp_traces"
                    continue

                for idx in idxs.tolist():
                    ts_all = _extract_trace_row(data["timestamps"], idx)
                    pw_all = _extract_trace_row(data["power_traces"], idx)
                    active_all = _extract_trace_row(data["active_requests"], idx)
                    prefill_all = _extract_trace_row(data["prefill_tokens"], idx)
                    decode_all = _extract_trace_row(data["decode_tokens"], idx)

                    ts_mask = np.isfinite(ts_all) & (ts_all > 0.0)
                    ts = ts_all[ts_mask]
                    power = pw_all[ts_mask]
                    active = active_all[ts_mask]
                    prefill = prefill_all[ts_mask]
                    decode = decode_all[ts_mask]
                    if ts.size <= 3 or power.size <= 3:
                        continue

                    req_ts_all = _extract_trace_row(data["request_timestamps"], idx)
                    in_all = _extract_trace_row(data["input_tokens"], idx)
                    out_all = _extract_trace_row(data["output_tokens"], idx)
                    rn = int(min(req_ts_all.size, in_all.size, out_all.size))
                    req_ts = req_ts_all[:rn]
                    nin = in_all[:rn]
                    nout = out_all[:rn]
                    req_mask = np.isfinite(req_ts) & (req_ts > 0.0)
                    req_ts = req_ts[req_mask]
                    nin = nin[req_mask]
                    nout = nout[req_mask]

                    z = predict_regime_labels_from_gmm(power[1:], gmm_payload)
                    feats = build_trace_feature_subsets(
                        timestamps=ts,
                        active_requests=active,
                        prefill_tokens=prefill,
                        decode_tokens=decode,
                        request_timestamps=req_ts,
                        input_tokens=nin,
                        output_tokens=nout,
                    )

                    lengths = [z.size] + [int(feats[s].shape[0]) for s in SUBSET_ORDER]
                    L = int(min(lengths))
                    if L <= 4:
                        continue

                    z = z[:L].astype(np.int64)
                    if np.unique(z).size < 2:
                        continue

                    traces.append(
                        {
                            "z": z,
                            "A": np.asarray(feats["A"][:L], dtype=np.float64),
                            "ΔA": np.asarray(feats["ΔA"][:L], dtype=np.float64),
                            "F2": np.asarray(feats["F2"][:L], dtype=np.float64),
                            "F3": np.asarray(feats["F3"][:L], dtype=np.float64),
                            "F6": np.asarray(feats["F6"][:L], dtype=np.float64),
                        }
                    )

            if len(traces) < 2:
                skipped[config_id] = "insufficient_valid_traces"
                continue

            split_seed = _config_seed(seed, config_id, "split")
            train_idx, test_idx = _split_trace_indices(len(traces), seed=split_seed, train_ratio=0.7)
            if train_idx.size == 0 or test_idx.size == 0:
                skipped[config_id] = "split_failed"
                continue

            num_classes = int(max(k, np.max(np.concatenate([tr["z"] for tr in traces], axis=0)) + 1))
            y_test = np.concatenate([traces[int(i)]["z"] for i in test_idx.tolist()], axis=0)
            ce_null = compute_ce_null(y_test, num_classes=num_classes)
            if not np.isfinite(ce_null) or ce_null <= 0.0:
                skipped[config_id] = "invalid_ce_null"
                continue

            ce_by_subset: Dict[str, float] = {}
            subset_row_cache: Dict[str, Dict[str, object]] = {}

            for subset in SUBSET_ORDER:
                train_traces_subset: List[Tuple[np.ndarray, np.ndarray]] = []
                test_traces_subset: List[Tuple[np.ndarray, np.ndarray]] = []

                for i in train_idx.tolist():
                    tr = traces[int(i)]
                    x = np.asarray(tr[subset], dtype=np.float64)
                    y = np.asarray(tr["z"], dtype=np.int64)
                    if x.shape[0] != y.size or x.shape[0] <= 0:
                        continue
                    train_traces_subset.append((x, y))

                for i in test_idx.tolist():
                    tr = traces[int(i)]
                    x = np.asarray(tr[subset], dtype=np.float64)
                    y = np.asarray(tr["z"], dtype=np.int64)
                    if x.shape[0] != y.size or x.shape[0] <= 0:
                        continue
                    test_traces_subset.append((x, y))

                if len(train_traces_subset) == 0 or len(test_traces_subset) == 0:
                    ce_by_subset[subset] = float("nan")
                    continue

                input_dim = int(train_traces_subset[0][0].shape[1])
                ce_subset = _train_gru_ce(
                    train_traces=train_traces_subset,
                    test_traces=test_traces_subset,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    seed=_config_seed(seed, config_id, subset),
                    device=torch_device,
                    hidden_dim=int(hidden_dim),
                    epochs=int(epochs),
                    lr=float(lr),
                )
                ce_by_subset[subset] = float(ce_subset)
                subset_row_cache[subset] = {
                    "input_dim": int(input_dim),
                    "n_train_traces": int(len(train_traces_subset)),
                    "n_test_traces": int(len(test_traces_subset)),
                }

            ce_f6 = float(ce_by_subset.get("F6", float("nan")))
            if not np.isfinite(ce_f6):
                skipped[config_id] = "invalid_ce_f6"
                continue

            config_rows: List[Dict[str, object]] = []
            ok = True
            for subset in SUBSET_ORDER:
                ce_subset = float(ce_by_subset.get(subset, float("nan")))
                if not np.isfinite(ce_subset):
                    ok = False
                    break
                ir_abs, ir_vs_f6 = compute_info_retention(ce_subset=ce_subset, ce_null=ce_null, ce_f6=ce_f6)
                row = {
                    "config_id": config_id,
                    "subset": subset,
                    "ce_subset": float(ce_subset),
                    "ce_null": float(ce_null),
                    "ce_f6": float(ce_f6),
                    "ir_abs": float(ir_abs),
                    "ir_vs_f6": float(ir_vs_f6),
                    "info_retained_abs_pct": float(100.0 * ir_abs) if np.isfinite(ir_abs) else float("nan"),
                    "info_retained_vs_f6_pct": float(100.0 * ir_vs_f6) if np.isfinite(ir_vs_f6) else float("nan"),
                    "n_test_samples": int(y_test.size),
                    "n_classes": int(num_classes),
                    "input_dim": int(subset_row_cache[subset]["input_dim"]),
                    "n_train_traces": int(subset_row_cache[subset]["n_train_traces"]),
                    "n_test_traces": int(subset_row_cache[subset]["n_test_traces"]),
                }
                config_rows.append(row)

            if not ok or len(config_rows) != len(SUBSET_ORDER):
                skipped[config_id] = "insufficient_subset_coverage"
                continue

            per_config_rows.extend(config_rows)
            for row in config_rows:
                subset_values_abs[str(row["subset"])].append(float(row["ir_abs"]))
                subset_values_vs_f6[str(row["subset"])].append(float(row["ir_vs_f6"]))

            selected_configs.append(
                {
                    "config_id": config_id,
                    "mapped_model": mapped_model,
                    "hardware": hw,
                    "tensor_parallelism": int(tp),
                    "npz_path": npz_path,
                    "gmm_path": gmm_path,
                    "num_tp_traces_in_npz": int(len([1 for _ in idxs])),
                    "num_total_traces_in_npz": int(n_traces_all),
                    "num_valid_traces_used": int(len(traces)),
                    "num_train_traces": int(train_idx.size),
                    "num_test_traces": int(test_idx.size),
                }
            )
        except Exception as exc:
            skipped[config_id] = f"processing_error:{type(exc).__name__}:{exc}"
            continue

    if len(per_config_rows) == 0:
        raise RuntimeError("No valid per-config held-out predictive rows were produced.")

    per_config_rows = sorted(per_config_rows, key=lambda r: (str(r["config_id"]), SUBSET_ORDER.index(str(r["subset"]))))

    summary_rows: List[Dict[str, object]] = []
    for i, subset in enumerate(SUBSET_ORDER):
        values_abs = [float(v) for v in subset_values_abs[subset] if np.isfinite(v)]
        values_vs = [float(v) for v in subset_values_vs_f6[subset] if np.isfinite(v)]

        med_abs, lo_abs, hi_abs = bootstrap_median_ci(values_abs, n_bootstrap=bootstrap_samples, seed=(int(seed) + 11 + i))
        med_vs, lo_vs, hi_vs = bootstrap_median_ci(values_vs, n_bootstrap=bootstrap_samples, seed=(int(seed) + 101 + i))

        summary_rows.append(
            {
                "subset": subset,
                "median_ir_vs_f6": float(med_vs),
                "ci95_low_ir_vs_f6": float(lo_vs),
                "ci95_high_ir_vs_f6": float(hi_vs),
                "median_ir_abs": float(med_abs),
                "ci95_low_ir_abs": float(lo_abs),
                "ci95_high_ir_abs": float(hi_abs),
                "median_info_retained_vs_f6_pct": float(100.0 * med_vs) if np.isfinite(med_vs) else float("nan"),
                "median_info_retained_abs_pct": float(100.0 * med_abs) if np.isfinite(med_abs) else float("nan"),
                "n_configs": int(len(values_vs)),
            }
        )

    _write_csv(
        out_per_config_csv,
        per_config_rows,
        fieldnames=[
            "config_id",
            "subset",
            "ce_subset",
            "ce_null",
            "ce_f6",
            "ir_abs",
            "ir_vs_f6",
            "info_retained_abs_pct",
            "info_retained_vs_f6_pct",
            "n_test_samples",
            "n_classes",
            "input_dim",
            "n_train_traces",
            "n_test_traces",
        ],
    )
    _write_csv(
        out_summary_csv,
        summary_rows,
        fieldnames=[
            "subset",
            "median_ir_vs_f6",
            "ci95_low_ir_vs_f6",
            "ci95_high_ir_vs_f6",
            "median_ir_abs",
            "ci95_low_ir_abs",
            "ci95_high_ir_abs",
            "median_info_retained_vs_f6_pct",
            "median_info_retained_abs_pct",
            "n_configs",
        ],
    )
    _plot_summary(summary_rows, out_figure)

    manifest = {
        "schema_version": "feature-sufficiency-predictive-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "run_manifest": run_manifest,
            "training_data_dir": training_data_dir,
            "gmm_dir": gmm_dir,
            "include_configs": sorted(list(include_set)) if include_set is not None else None,
        },
        "estimator": {
            "metric": "heldout_predictive_ce_reduction",
            "ir_abs": "1 - CE_subset / CE_null",
            "ir_vs_f6": "(CE_null - CE_subset) / (CE_null - CE_F6)",
            "num_classes_source": "gmm_k_with_observed_max_fallback",
        },
        "sequence_model": {
            "model": "GRUClassifier",
            "hidden_dim": int(hidden_dim),
            "num_layers": 1,
            "epochs": int(epochs),
            "lr": float(lr),
            "device": str(torch_device),
            "train_test_split": "trace-level 70/30",
        },
        "aggregation": {
            "mode": "per_config_median_with_bootstrap_ci95",
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
            "subset_order": list(SUBSET_ORDER),
            "primary_series": "ir_vs_f6",
        },
        "summary": {
            "num_run_manifest_configs": int(len(run_configs)),
            "num_selected_configs": int(len(selected_configs)),
            "num_skipped_configs": int(len(skipped)),
            "num_per_config_rows": int(len(per_config_rows)),
        },
        "selected_configs": selected_configs,
        "skipped_configs": skipped,
        "outputs": {
            "figure": out_figure,
            "per_config_csv": out_per_config_csv,
            "summary_csv": out_summary_csv,
            "manifest_json": out_json,
        },
    }
    _write_json(out_json, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = _build_default_paths()
    p = argparse.ArgumentParser(
        description=(
            "Generate A2 feature sufficiency figure using held-out predictive "
            "information retention (IR_abs and IR_vs_F6)."
        )
    )
    p.add_argument("--run-manifest", default=defaults["run_manifest"])
    p.add_argument("--training-data-dir", default=defaults["training_data_dir"])
    p.add_argument("--gmm-dir", default=defaults["gmm_dir"])
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-figure", default=defaults["out_figure"])
    p.add_argument("--out-per-config-csv", default=defaults["out_per_config_csv"])
    p.add_argument("--out-summary-csv", default=defaults["out_summary_csv"])
    p.add_argument("--out-json", default=defaults["out_json"])
    p.add_argument("--include-configs", default=None, help="Optional comma-separated config_ids to include.")
    p.add_argument("--device", default="auto", help="Torch device (default: auto)")
    p.add_argument("--epochs", type=int, default=8, help="GRU training epochs per subset/config")
    p.add_argument("--hidden-dim", type=int, default=32, help="GRU hidden dimension")
    p.add_argument("--lr", type=float, default=1e-3, help="GRU learning rate")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run_feature_sufficiency_figure(
        run_manifest=str(args.run_manifest),
        training_data_dir=str(args.training_data_dir),
        gmm_dir=str(args.gmm_dir),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
        out_figure=str(args.out_figure),
        out_per_config_csv=str(args.out_per_config_csv),
        out_summary_csv=str(args.out_summary_csv),
        out_json=str(args.out_json),
        include_configs=args.include_configs,
        device=str(args.device),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
    )

    print("[feature_sufficiency_figure] Summary:")
    print(f"  num_run_manifest_configs: {manifest['summary']['num_run_manifest_configs']}")
    print(f"  num_selected_configs: {manifest['summary']['num_selected_configs']}")
    print(f"  num_skipped_configs: {manifest['summary']['num_skipped_configs']}")
    print(f"  per_config_csv: {manifest['outputs']['per_config_csv']}")
    print(f"  summary_csv: {manifest['outputs']['summary_csv']}")
    print(f"  figure: {manifest['outputs']['figure']}")
    print(f"  manifest: {manifest['outputs']['manifest_json']}")


if __name__ == "__main__":
    main()
