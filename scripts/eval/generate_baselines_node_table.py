#!/usr/bin/env python3
"""
Generate paper-ready node-level baseline comparison table.

Reads node baseline comparison metrics (typically baselines_node_level.csv),
aggregates across selected configurations, and writes:
  - CSV table values
  - JSON summary (with numeric details)
  - LaTeX snippet

Optional: recompute baselines_node_level.csv before table generation by calling
scripts/eval/run_baselines_node.py logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.scripts.request_data_policy import (
    DEFAULT_ALLOWED_JSON_PREFIX,
    DEFAULT_REQUEST_TIMESTAMP_POLICY,
    REQUEST_TIMESTAMP_POLICIES,
)

CONFIG_ID_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")

METHOD_LABEL_MAP = {
    "tdp": "TDP",
    "mean": "Mean",
    "marginal_gmm": "Marginal GMM",
    "splitwise_lut": "Splitwise",
    "ours": "Ours",
}


def _ensure_dir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_config_id(config_id: str) -> Optional[Dict[str, object]]:
    match = CONFIG_ID_RE.match(str(config_id).strip())
    if match is None:
        return None
    family = str(match.group(1))
    size_b = int(match.group(2))
    hardware = str(match.group(3))
    tp = int(match.group(4))
    return {
        "model_family": family,
        "model_size_b": size_b,
        "hardware": hardware,
        "tp": tp,
    }


def _is_moe_config(config_id: str) -> bool:
    parsed = _parse_config_id(config_id)
    if parsed is None:
        return False
    family = str(parsed["model_family"]).lower()
    size_b = int(parsed["model_size_b"])
    if "deepseek-r1-distill" in family:
        return False
    if "gpt-oss" in family and size_b >= 20:
        return True
    return False


def _is_dense_config(config_id: str) -> bool:
    return not _is_moe_config(config_id)


def _is_representative_dense_config(config_id: str) -> bool:
    parsed = _parse_config_id(config_id)
    if parsed is None:
        return False
    if not _is_dense_config(config_id):
        return False
    return int(parsed["model_size_b"]) == 70 and int(parsed["tp"]) == 4


def _parse_csv_list(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw:
        return []
    out: List[str] = []
    for token in raw:
        if token is None:
            continue
        for part in str(token).split(","):
            p = part.strip()
            if p:
                out.append(p)
    deduped: List[str] = []
    seen = set()
    for item in out:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _load_rows(node_metrics_csv: str) -> List[Dict[str, object]]:
    if not os.path.exists(node_metrics_csv):
        raise FileNotFoundError(f"Node metrics CSV not found: {node_metrics_csv}")
    with open(node_metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Node metrics CSV missing header")
        required_cols = {
            "config_id",
            "method",
            "status",
            "ks_stat",
            "acf_r2",
            "nrmse",
            "delta_energy_pct",
        }
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Node metrics CSV missing required columns: {sorted(missing)}"
            )
        return [dict(row) for row in reader]


def _select_configs(
    *,
    rows: Sequence[Mapping[str, object]],
    config_ids: Sequence[str],
    arch_filter: str,
    representative_only: bool,
) -> List[str]:
    available = sorted(
        {
            str(row.get("config_id", "")).strip()
            for row in rows
            if str(row.get("status", "")).strip() == "evaluated"
        }
    )
    available = [c for c in available if c]
    if len(available) == 0:
        raise ValueError("No evaluated configs found in node metrics CSV")

    if len(config_ids) > 0:
        selected = [c for c in config_ids if c in set(available)]
        missing = [c for c in config_ids if c not in set(available)]
        if missing:
            raise ValueError(
                f"Requested config_ids not found in evaluated rows: {missing}"
            )
        return selected

    arch = str(arch_filter).strip().lower()
    if arch not in {"dense", "moe", "all"}:
        raise ValueError("arch_filter must be one of {'dense','moe','all'}")

    selected = list(available)
    if arch == "dense":
        selected = [c for c in selected if _is_dense_config(c)]
    elif arch == "moe":
        selected = [c for c in selected if _is_moe_config(c)]

    if len(selected) == 0:
        raise ValueError(f"No configs remain after arch_filter='{arch_filter}'")

    if representative_only and arch in {"dense", "all"}:
        representative = [c for c in selected if _is_representative_dense_config(c)]
        if len(representative) > 0:
            selected = representative
    return selected


def _to_float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _choose_third_method_key(
    *,
    rows_for_selected_configs: Sequence[Mapping[str, object]],
    candidates: Sequence[str],
) -> str:
    available_methods = {
        str(row.get("method", "")).strip()
        for row in rows_for_selected_configs
        if str(row.get("status", "")).strip() == "evaluated"
    }
    for key in candidates:
        if key in available_methods:
            return key
    raise ValueError(
        "None of the requested third-method candidates are present in evaluated rows: "
        f"candidates={list(candidates)}, available={sorted(available_methods)}"
    )


def _aggregate_method_metrics(
    *,
    rows: Sequence[Mapping[str, object]],
    selected_configs: Sequence[str],
    method_key: str,
    delta_energy_mode: str,
) -> Dict[str, float]:
    selected_set = set(str(x) for x in selected_configs)
    picked = [
        row
        for row in rows
        if str(row.get("status", "")).strip() == "evaluated"
        and str(row.get("config_id", "")).strip() in selected_set
        and str(row.get("method", "")).strip() == str(method_key)
    ]
    if len(picked) == 0:
        raise ValueError(f"No evaluated rows for method='{method_key}' in selection")

    ks_vals = [_to_float_or_nan(r.get("ks_stat")) for r in picked]
    acf_vals = [_to_float_or_nan(r.get("acf_r2")) for r in picked]
    nrmse_vals = [_to_float_or_nan(r.get("nrmse")) for r in picked]
    delta_vals_raw = [_to_float_or_nan(r.get("delta_energy_pct")) for r in picked]

    dem = str(delta_energy_mode).strip().lower()
    if dem == "abs":
        delta_vals = [abs(v) if np.isfinite(v) else float("nan") for v in delta_vals_raw]
    elif dem == "signed":
        delta_vals = list(delta_vals_raw)
    else:
        raise ValueError("delta_energy_mode must be one of {'abs','signed'}")

    return {
        "ks_stat": _nanmean(ks_vals),
        "acf_r2": _nanmean(acf_vals),
        "nrmse": _nanmean(nrmse_vals),
        "delta_energy_pct": _nanmean(delta_vals),
        "num_configs": float(len(picked)),
    }


def _fmt(value: Optional[float], decimals: int, dash: str = "---") -> str:
    if value is None:
        return dash
    if not np.isfinite(float(value)):
        return dash
    return f"{float(value):.{int(decimals)}f}"


def _build_table_rows(
    *,
    method_metrics: Mapping[str, Mapping[str, float]],
    method_labels: Mapping[str, str],
    decimals: int,
    force_mean_delta_energy_zero: bool,
    hide_constant_acf: bool,
) -> List[Dict[str, str]]:
    order = ["tdp", "mean", "third", "ours"]
    rows: List[Dict[str, str]] = []
    for key in order:
        m = method_metrics[key]
        label = str(method_labels[key])
        acf_val: Optional[float] = float(m["acf_r2"])
        if hide_constant_acf and key in {"tdp", "mean"}:
            acf_val = None
        delta_val = float(m["delta_energy_pct"])
        if force_mean_delta_energy_zero and key == "mean":
            delta_val = 0.0
        rows.append(
            {
                "method": label,
                "ks": _fmt(float(m["ks_stat"]), decimals),
                "acf_r2": _fmt(acf_val, decimals),
                "nrmse": _fmt(float(m["nrmse"]), decimals),
                "delta_e_pct": _fmt(delta_val, decimals),
            }
        )
    return rows


def _build_latex_table(*, rows: Sequence[Mapping[str, str]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{0.4em}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} &",
        r"\textbf{KS~$\downarrow$} &",
        r"\textbf{ACF $R^2$~$\uparrow$} &",
        r"\textbf{NRMSE~$\downarrow$} &",
        r"\textbf{$\Delta$E (\%)~$\downarrow$} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['ks']} & {row['acf_r2']} & {row['nrmse']} & {row['delta_e_pct']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: str, rows: Sequence[Mapping[str, str]]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "ks", "acf_r2", "nrmse", "delta_e_pct"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "node_metrics_csv": str(repo_root / "results" / "eval_paper" / "baselines_node_level.csv"),
        "out_csv": str(repo_root / "results" / "eval_paper" / "baselines_node_table.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "baselines_node_table.json"),
        "out_tex": str(repo_root / "results" / "eval_paper" / "baselines_node_table.tex"),
    }


def generate_baselines_node_table(
    *,
    node_metrics_csv: str,
    out_csv: str,
    out_json: str,
    out_tex: str,
    config_ids: Sequence[str],
    arch_filter: str,
    representative_only: bool,
    third_method_candidates: Sequence[str],
    third_method_label: Optional[str],
    decimals: int = 2,
    delta_energy_mode: str = "abs",
    force_mean_delta_energy_zero: bool = False,
    hide_constant_acf: bool = True,
    caption: str = (
        "Baseline comparison at node level, averaged across representative dense configurations."
    ),
    label: str = "tab:baselines-node",
    recompute_node_metrics: bool = False,
    run_manifest: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1_gru_all/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
) -> Dict[str, object]:
    if recompute_node_metrics:
        from scripts.eval.run_baselines_node import run_baselines_node

        run_baselines_node(
            run_manifest=run_manifest,
            experimental_manifest=experimental_manifest,
            throughput_db=throughput_db,
            pair_manifest_csv=pair_manifest_csv,
            ar1_params_dir=ar1_params_dir,
            out_csv=node_metrics_csv,
            config_ids=list(config_ids) if len(config_ids) > 0 else None,
            num_seeds=int(num_seeds),
            base_seed=int(base_seed),
            device=str(device),
            acf_max_lag=int(acf_max_lag),
            decode_mode=str(decode_mode),
            median_filter_window=int(median_filter_window),
            ours_std_scale=float(ours_std_scale),
            ours_logit_temperature=float(ours_logit_temperature),
            splitwise_perf_model_csv=str(splitwise_perf_model_csv),
            splitwise_source_model=str(splitwise_source_model),
            splitwise_source_hardware=str(splitwise_source_hardware),
            splitwise_source_tp=int(splitwise_source_tp),
            request_timestamp_policy=str(request_timestamp_policy),
            allowed_json_prefix=str(allowed_json_prefix),
        )

    rows_all = _load_rows(node_metrics_csv)
    selected_configs = _select_configs(
        rows=rows_all,
        config_ids=config_ids,
        arch_filter=arch_filter,
        representative_only=bool(representative_only),
    )
    selected_set = set(selected_configs)
    rows_selected = [
        r for r in rows_all if str(r.get("config_id", "")).strip() in selected_set
    ]
    third_key = _choose_third_method_key(
        rows_for_selected_configs=rows_selected,
        candidates=third_method_candidates,
    )

    metrics_tdp = _aggregate_method_metrics(
        rows=rows_all,
        selected_configs=selected_configs,
        method_key="tdp",
        delta_energy_mode=delta_energy_mode,
    )
    metrics_mean = _aggregate_method_metrics(
        rows=rows_all,
        selected_configs=selected_configs,
        method_key="mean",
        delta_energy_mode=delta_energy_mode,
    )
    metrics_third = _aggregate_method_metrics(
        rows=rows_all,
        selected_configs=selected_configs,
        method_key=third_key,
        delta_energy_mode=delta_energy_mode,
    )
    metrics_ours = _aggregate_method_metrics(
        rows=rows_all,
        selected_configs=selected_configs,
        method_key="ours",
        delta_energy_mode=delta_energy_mode,
    )

    method_metrics = {
        "tdp": metrics_tdp,
        "mean": metrics_mean,
        "third": metrics_third,
        "ours": metrics_ours,
    }
    method_labels = {
        "tdp": METHOD_LABEL_MAP["tdp"],
        "mean": METHOD_LABEL_MAP["mean"],
        "third": str(third_method_label)
        if third_method_label is not None
        else METHOD_LABEL_MAP.get(third_key, third_key),
        "ours": METHOD_LABEL_MAP["ours"],
    }
    table_rows = _build_table_rows(
        method_metrics=method_metrics,
        method_labels=method_labels,
        decimals=int(decimals),
        force_mean_delta_energy_zero=bool(force_mean_delta_energy_zero),
        hide_constant_acf=bool(hide_constant_acf),
    )
    latex = _build_latex_table(rows=table_rows, caption=caption, label=label)

    _write_csv(out_csv, table_rows)
    _write_json(
        out_json,
        {
            "inputs": {
                "node_metrics_csv": str(Path(node_metrics_csv).resolve()),
                "recompute_node_metrics": bool(recompute_node_metrics),
                "selected_configs": list(selected_configs),
                "arch_filter": str(arch_filter),
                "representative_only": bool(representative_only),
                "third_method_key": str(third_key),
            },
            "formatting": {
                "decimals": int(decimals),
                "delta_energy_mode": str(delta_energy_mode),
                "force_mean_delta_energy_zero": bool(force_mean_delta_energy_zero),
                "hide_constant_acf": bool(hide_constant_acf),
                "caption": str(caption),
                "label": str(label),
            },
            "method_metrics": {
                method_labels["tdp"]: {
                    k: float(v) for k, v in method_metrics["tdp"].items()
                },
                method_labels["mean"]: {
                    k: float(v) for k, v in method_metrics["mean"].items()
                },
                method_labels["third"]: {
                    k: float(v) for k, v in method_metrics["third"].items()
                },
                method_labels["ours"]: {
                    k: float(v) for k, v in method_metrics["ours"].items()
                },
            },
            "table_rows": [dict(r) for r in table_rows],
        },
    )
    _ensure_dir_for_file(out_tex)
    with open(out_tex, "w") as f:
        f.write(latex)

    return {
        "node_metrics_csv": node_metrics_csv,
        "out_csv": out_csv,
        "out_json": out_json,
        "out_tex": out_tex,
        "selected_configs": selected_configs,
        "third_method_key": third_key,
        "method_labels": method_labels,
        "table_rows": table_rows,
        "latex": latex,
        "method_metrics": method_metrics,
    }


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate the node-level baseline comparison table "
            "(TDP / Mean / Marginal GMM-or-Splitwise / Ours)."
        )
    )
    parser.add_argument("--node-metrics-csv", default=defaults["node_metrics_csv"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    parser.add_argument("--out-tex", default=defaults["out_tex"])
    parser.add_argument(
        "--config-ids",
        nargs="*",
        default=None,
        help="Optional explicit config IDs (comma-separated or space-separated).",
    )
    parser.add_argument(
        "--arch-filter",
        default="dense",
        choices=["dense", "moe", "all"],
    )
    parser.add_argument(
        "--no-representative-only",
        action="store_true",
        help="Use all configs matching --arch-filter instead of representative dense subset.",
    )
    parser.add_argument(
        "--third-method-candidates",
        nargs="*",
        default=["splitwise_lut"],
        help=(
            "Candidate method keys for the third row; first present in CSV is used. "
            "Example: splitwise_lut"
        ),
    )
    parser.add_argument(
        "--third-method-label",
        default=None,
        help="Optional override for third-row display label.",
    )
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument(
        "--delta-energy-mode",
        default="abs",
        choices=["abs", "signed"],
        help="abs: report absolute delta-energy percent; signed: preserve sign.",
    )
    parser.add_argument(
        "--force-mean-delta-energy-zero",
        action="store_true",
        help="Force Mean row delta-energy cell to 0.00 (for paper-style table formatting).",
    )
    parser.add_argument(
        "--show-constant-acf",
        action="store_true",
        help="Show constant-method ACF values if finite instead of '---'.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Baseline comparison at node level, averaged across representative dense configurations."
        ),
    )
    parser.add_argument("--label", default="tab:baselines-node")

    parser.add_argument(
        "--recompute-node-metrics",
        action="store_true",
        help="Recompute baselines_node_level.csv before generating table.",
    )
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json")
    parser.add_argument("--experimental-manifest", default="results/experimental_continuous_v1_gru_all/manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument(
        "--request-timestamp-policy",
        default=DEFAULT_REQUEST_TIMESTAMP_POLICY,
        choices=list(REQUEST_TIMESTAMP_POLICIES),
    )
    parser.add_argument("--allowed-json-prefix", default=DEFAULT_ALLOWED_JSON_PREFIX)
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument("--decode-mode", default="stochastic", choices=["stochastic", "argmax"])
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--ours-std-scale", type=float, default=1.0)
    parser.add_argument("--ours-logit-temperature", type=float, default=1.0)
    parser.add_argument("--splitwise-perf-model-csv", default="data/perf_model.csv")
    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)

    args = parser.parse_args()
    config_ids = _parse_csv_list(args.config_ids)
    third_method_candidates = _parse_csv_list(args.third_method_candidates)
    if len(third_method_candidates) == 0:
        raise ValueError("third_method_candidates cannot be empty")

    result = generate_baselines_node_table(
        node_metrics_csv=str(args.node_metrics_csv),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        out_tex=str(args.out_tex),
        config_ids=config_ids,
        arch_filter=str(args.arch_filter),
        representative_only=not bool(args.no_representative_only),
        third_method_candidates=third_method_candidates,
        third_method_label=(
            str(args.third_method_label) if args.third_method_label is not None else None
        ),
        decimals=int(args.decimals),
        delta_energy_mode=str(args.delta_energy_mode),
        force_mean_delta_energy_zero=bool(args.force_mean_delta_energy_zero),
        hide_constant_acf=not bool(args.show_constant_acf),
        caption=str(args.caption),
        label=str(args.label),
        recompute_node_metrics=bool(args.recompute_node_metrics),
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        throughput_db=str(args.throughput_db),
        pair_manifest_csv=str(args.pair_manifest_csv),
        ar1_params_dir=str(args.ar1_params_dir),
        num_seeds=int(args.num_seeds),
        base_seed=int(args.base_seed),
        device=str(args.device),
        acf_max_lag=int(args.acf_max_lag),
        decode_mode=str(args.decode_mode),
        median_filter_window=int(args.median_filter_window),
        ours_std_scale=float(args.ours_std_scale),
        ours_logit_temperature=float(args.ours_logit_temperature),
        splitwise_perf_model_csv=str(args.splitwise_perf_model_csv),
        splitwise_source_model=str(args.splitwise_source_model),
        splitwise_source_hardware=str(args.splitwise_source_hardware),
        splitwise_source_tp=int(args.splitwise_source_tp),
        request_timestamp_policy=str(args.request_timestamp_policy),
        allowed_json_prefix=str(args.allowed_json_prefix),
    )

    print("[generate_baselines_node_table] Done")
    print(f"  node_metrics_csv : {result['node_metrics_csv']}")
    print(f"  out_csv          : {result['out_csv']}")
    print(f"  out_json         : {result['out_json']}")
    print(f"  out_tex          : {result['out_tex']}")
    print(f"  selected_configs : {len(result['selected_configs'])}")
    print(f"  third_method_key : {result['third_method_key']}")
    for row in result["table_rows"]:
        print(
            f"    {row['method']}: KS={row['ks']}, ACF_R2={row['acf_r2']}, "
            f"NRMSE={row['nrmse']}, dE={row['delta_e_pct']}"
        )


if __name__ == "__main__":
    main()
