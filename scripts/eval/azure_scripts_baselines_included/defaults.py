#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict


DEFAULT_CONFIG_ID = "deepseek-r1-distill-70b_H100_tp4"
DEFAULT_METHODS_GENERATION = ("ours", "splitwise_lut", "splitwise_strict")
DEFAULT_TRACE_KINDS = (
    "tdp_baseline",
    "mean_baseline",
    "splitwise_lut",
    "splitwise_strict",
    "ours",
)


def build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "repo_root": str(repo_root),
        "run_manifest": str(
            repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2" / "run_manifest.json"
        ),
        "experimental_manifest": str(repo_root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "throughput_db": str(repo_root / "model" / "config" / "throughput_database.json"),
        "pair_manifest_csv": str(repo_root / "results" / "stage0" / "pair_manifest.csv"),
        "splitwise_perf_model_csv": str(repo_root / "data" / "perf_model.csv"),
        "ar1_params_dir": str(
            repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2_ar1_thresh" / "ar1_params"
        ),
        "node_stream_dir": str(repo_root / "data" / "azure_facility" / "node_streams"),
        "node_traces_root": str(repo_root / "results" / "azure_facility_baselines_included" / "node_traces"),
        "aggregated_root": str(repo_root / "results" / "azure_facility_baselines_included" / "aggregated"),
        "metrics_csv": str(repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_metrics.csv"),
        "ldc_csv": str(repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_ldc_15min.csv"),
        "site_traces_15min_csv": str(
            repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_site_traces_15min.csv"
        ),
        "parsed_requests_csv": str(
            repo_root / "data" / "azure_trace" / "parsed" / "day_2024-05-16_requests.csv"
        ),
        "figures_out_dir": str(repo_root / "figures" / "azure_baselines_included"),
        "sizing_out_csv": str(repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_sizing_table.csv"),
        "sizing_out_json": str(repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_sizing_table.json"),
        "sizing_out_tex": str(repo_root / "results" / "eval_paper_baselines_included" / "azure_facility_sizing_table.tex"),
    }
