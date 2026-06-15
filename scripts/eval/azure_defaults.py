#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict

from model.utils.config import parse_csv_list, safe_float
from model.utils.io import (
    ensure_dir,
    ensure_dir_for_file,
    load_json,
    resolve_existing_path,
    write_json,
)

DEFAULT_CONFIG_ID = "llama-3-70b_A100_tp8"
DEFAULT_METHODS_GENERATION = ("ours", "splitwise_strict")
DEFAULT_TRACE_KINDS = (
    "tdp_baseline",
    "mean_baseline",
    "splitwise_strict",
    "ours",
)
DEFAULT_OVERSUB_METHODS = (
    "mean_baseline",
    "splitwise_strict",
    "ours",
)
DEFAULT_NON_GPU_OVERHEAD_W = 1000.0
DEFAULT_PUE = 1.3
DEFAULT_SPLITWISE_SOURCE_MODEL = "llama-3-70b"
DEFAULT_SPLITWISE_SOURCE_HARDWARE = "a100-80gb"
DEFAULT_SPLITWISE_SOURCE_TP = 8
MODEL_NAME_MAP = {
    "llama-3": "Llama-3.1",
    "deepseek-r1-distill": "DeepSeek-R1-Distill",
    "gpt-oss": "gpt-oss",
}


def build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "repo_root": str(repo_root),
        "run_manifest": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru"
            / "k10_f2"
            / "run_manifest.json"
        ),
        "experimental_manifest": str(
            repo_root / "results" / "experimental_continuous_v1" / "manifest.json"
        ),
        "throughput_db": str(repo_root / "model" / "throughput_database.json"),
        "pair_manifest_csv": str(
            repo_root / "results" / "stage0" / "pair_manifest.csv"
        ),
        "splitwise_perf_model_csv": str(repo_root / "data" / "perf_model.csv"),
        "ar1_params_dir": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru"
            / "k10_f2_ar1_thresh"
            / "ar1_params"
        ),
        "node_stream_dir": str(repo_root / "data" / "azure_facility" / "node_streams"),
        "node_traces_root": str(
            repo_root / "results" / "azure_facility" / "node_traces"
        ),
        "aggregated_root": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "metrics_csv": str(
            repo_root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        ),
        "ldc_csv": str(
            repo_root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"
        ),
        "site_traces_15min_csv": str(
            repo_root
            / "results"
            / "eval_paper"
            / "azure_facility_site_traces_15min.csv"
        ),
        "parsed_requests_csv": str(
            repo_root
            / "data"
            / "azure_trace"
            / "parsed"
            / "day_2024-05-16_requests.csv"
        ),
        "figures_out_dir": str(repo_root / "figures"),
        "oversub_capacity_plot": str(
            repo_root / "figures" / "azure_oversubscription_capacity.pdf"
        ),
        "oversub_lines_plot": str(
            repo_root / "figures" / "azure_oversubscription_lines.pdf"
        ),
        "oversub_csv": str(
            repo_root / "results" / "eval_paper" / "azure_oversubscription_capacity.csv"
        ),
        "oversub_json": str(
            repo_root
            / "results"
            / "eval_paper"
            / "azure_oversubscription_capacity.json"
        ),
        "sizing_out_csv": str(
            repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.csv"
        ),
        "sizing_out_json": str(
            repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.json"
        ),
        "sizing_out_tex": str(
            repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.tex"
        ),
    }
