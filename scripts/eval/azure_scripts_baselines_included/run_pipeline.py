#!/usr/bin/env python3
"""
Run the full isolated Azure pipeline with Splitwise baselines included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_scripts_baselines_included.azure_aggregate import aggregate_all_methods
from scripts.eval.azure_scripts_baselines_included.azure_figures import generate_azure_figures
from scripts.eval.azure_scripts_baselines_included.azure_generate_traces import generate_node_traces
from scripts.eval.azure_scripts_baselines_included.azure_metrics import compute_azure_facility_metrics
from scripts.eval.azure_scripts_baselines_included.defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_METHODS_GENERATION,
    DEFAULT_TRACE_KINDS,
    build_default_paths,
)
from scripts.eval.azure_scripts_baselines_included.generate_azure_facility_sizing_table import (
    generate_azure_facility_sizing_table,
)


def run_pipeline(**kwargs) -> Dict[str, object]:
    trace_summary = generate_node_traces(
        run_manifest=kwargs["run_manifest"],
        experimental_manifest=kwargs["experimental_manifest"],
        throughput_db=kwargs["throughput_db"],
        ar1_params_dir=kwargs["ar1_params_dir"],
        node_stream_dir=kwargs["node_stream_dir"],
        out_root=kwargs["node_traces_root"],
        config_id=kwargs["config_id"],
        methods=kwargs["methods"],
        duration_s=kwargs["duration_s"],
        dt=kwargs["dt"],
        rows=kwargs["rows"],
        racks_per_row=kwargs["racks_per_row"],
        nodes_per_rack=kwargs["nodes_per_rack"],
        batch_size=kwargs["batch_size"],
        base_seed=kwargs["base_seed"],
        device=kwargs["device"],
        decode_mode=kwargs["decode_mode"],
        median_filter_window=kwargs["median_filter_window"],
        ours_std_scale=kwargs["ours_std_scale"],
        ours_logit_temperature=kwargs["ours_logit_temperature"],
        splitwise_perf_model_csv=kwargs["splitwise_perf_model_csv"],
        splitwise_source_model=kwargs["splitwise_source_model"],
        splitwise_source_hardware=kwargs["splitwise_source_hardware"],
        splitwise_source_tp=kwargs["splitwise_source_tp"],
        splitwise_calibration_mode=kwargs["splitwise_calibration_mode"],
        pair_manifest_csv=kwargs["pair_manifest_csv"],
        tp_gpus=kwargs["tp_gpus"],
        n_gpus_per_node=kwargs["n_gpus_per_node"],
        non_gpu_overhead_w=kwargs["non_gpu_overhead_w"],
        require_recorded_timestamps=not kwargs["allow_synthetic_request_timestamps"],
    )

    agg_summary = aggregate_all_methods(
        node_traces_root=kwargs["node_traces_root"],
        aggregated_root=kwargs["aggregated_root"],
        methods=kwargs["methods"],
        dt=kwargs["dt"],
        rows=kwargs["rows"],
        racks_per_row=kwargs["racks_per_row"],
        nodes_per_rack=kwargs["nodes_per_rack"],
        non_gpu_overhead_w=kwargs["non_gpu_overhead_w"],
        pue=kwargs["pue"],
    )

    metrics_summary = compute_azure_facility_metrics(
        aggregated_root=kwargs["aggregated_root"],
        node_traces_root=kwargs["node_traces_root"],
        experimental_manifest=kwargs["experimental_manifest"],
        metrics_csv=kwargs["metrics_csv"],
        ldc_csv=kwargs["ldc_csv"],
        site_traces_15min_csv=kwargs["site_traces_15min_csv"],
        config_id=kwargs["config_id"],
        trace_kinds=kwargs["trace_kinds"],
        rows=kwargs["rows"],
        racks_per_row=kwargs["racks_per_row"],
        nodes_per_rack=kwargs["nodes_per_rack"],
        tp_gpus=kwargs["tp_gpus"] if kwargs["tp_gpus"] is not None else 4,
        gpu_tdp_w=kwargs["gpu_tdp_w"],
        non_gpu_overhead_w=kwargs["non_gpu_overhead_w"],
        pue=kwargs["pue"],
    )

    figures_manifest = generate_azure_figures(
        parsed_requests_csv=kwargs["parsed_requests_csv"],
        aggregated_root=kwargs["aggregated_root"],
        metrics_csv=kwargs["metrics_csv"],
        ldc_csv=kwargs["ldc_csv"],
        site_traces_15min_csv=kwargs["site_traces_15min_csv"],
        out_dir=kwargs["figures_out_dir"],
        trace_kinds=kwargs["trace_kinds"],
        arrival_bin_seconds=kwargs["arrival_bin_seconds"],
        power_resolution_seconds=900,
        heatmap_downsample_seconds=kwargs["heatmap_downsample_seconds"],
        peak_window_hours=kwargs["peak_window_hours"],
        rows=kwargs["rows"],
        racks_per_row=kwargs["racks_per_row"],
        dry_run=kwargs["dry_run_figures"],
    )

    sizing_summary = generate_azure_facility_sizing_table(
        metrics_csv=kwargs["metrics_csv"],
        out_csv=kwargs["sizing_out_csv"],
        out_json=kwargs["sizing_out_json"],
        out_tex=kwargs["sizing_out_tex"],
        method_order=kwargs["method_order"],
        power_factor=kwargs["power_factor"],
        hide_redundant_cells=kwargs["hide_redundant_cells"],
    )

    return {
        "trace_summary": trace_summary,
        "aggregation_summary": agg_summary,
        "metrics_summary": metrics_summary,
        "figures_manifest": figures_manifest,
        "sizing_summary": sizing_summary,
    }


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Run isolated Azure pipeline with Splitwise baselines included.")

    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    parser.add_argument("--splitwise-perf-model-csv", default=defaults["splitwise_perf_model_csv"])
    parser.add_argument("--ar1-params-dir", default=defaults["ar1_params_dir"])
    parser.add_argument("--node-stream-dir", default=defaults["node_stream_dir"])
    parser.add_argument("--node-traces-root", default=defaults["node_traces_root"])
    parser.add_argument("--aggregated-root", default=defaults["aggregated_root"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument("--site-traces-15min-csv", default=defaults["site_traces_15min_csv"])
    parser.add_argument("--parsed-requests-csv", default=defaults["parsed_requests_csv"])
    parser.add_argument("--figures-out-dir", default=defaults["figures_out_dir"])
    parser.add_argument("--sizing-out-csv", default=defaults["sizing_out_csv"])
    parser.add_argument("--sizing-out-json", default=defaults["sizing_out_json"])
    parser.add_argument("--sizing-out-tex", default=defaults["sizing_out_tex"])

    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS_GENERATION))
    parser.add_argument("--trace-kinds", default=",".join(DEFAULT_TRACE_KINDS))
    parser.add_argument("--method-order", default=",".join(DEFAULT_TRACE_KINDS))

    parser.add_argument("--duration-s", type=float, default=86400.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode-mode", default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--ours-std-scale", type=float, default=1.0)
    parser.add_argument("--ours-logit-temperature", type=float, default=1.0)

    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)
    parser.add_argument("--splitwise-calibration-mode", default="train_phase_matched_v1")

    parser.add_argument("--tp-gpus", type=int, default=None)
    parser.add_argument("--n-gpus-per-node", type=int, default=None)
    parser.add_argument("--gpu-tdp-w", type=float, default=700.0)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=1000.0)
    parser.add_argument("--pue", type=float, default=1.3)

    parser.add_argument("--arrival-bin-seconds", type=int, default=300)
    parser.add_argument("--heatmap-downsample-seconds", type=int, default=300)
    parser.add_argument("--peak-window-hours", type=float, default=3.0)
    parser.add_argument("--dry-run-figures", action="store_true")

    parser.add_argument("--power-factor", type=float, default=0.9)
    parser.add_argument("--hide-redundant-cells", action="store_true")
    parser.add_argument("--allow-synthetic-request-timestamps", action="store_true")

    args = parser.parse_args()

    result = run_pipeline(
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        throughput_db=str(args.throughput_db),
        pair_manifest_csv=str(args.pair_manifest_csv),
        splitwise_perf_model_csv=str(args.splitwise_perf_model_csv),
        ar1_params_dir=str(args.ar1_params_dir),
        node_stream_dir=str(args.node_stream_dir),
        node_traces_root=str(args.node_traces_root),
        aggregated_root=str(args.aggregated_root),
        metrics_csv=str(args.metrics_csv),
        ldc_csv=str(args.ldc_csv),
        site_traces_15min_csv=str(args.site_traces_15min_csv),
        parsed_requests_csv=str(args.parsed_requests_csv),
        figures_out_dir=str(args.figures_out_dir),
        sizing_out_csv=str(args.sizing_out_csv),
        sizing_out_json=str(args.sizing_out_json),
        sizing_out_tex=str(args.sizing_out_tex),
        config_id=str(args.config_id),
        methods=str(args.methods),
        trace_kinds=str(args.trace_kinds),
        method_order=str(args.method_order),
        duration_s=float(args.duration_s),
        dt=float(args.dt),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        batch_size=int(args.batch_size),
        base_seed=int(args.base_seed),
        device=str(args.device),
        decode_mode=str(args.decode_mode),
        median_filter_window=int(args.median_filter_window),
        ours_std_scale=float(args.ours_std_scale),
        ours_logit_temperature=float(args.ours_logit_temperature),
        splitwise_source_model=str(args.splitwise_source_model),
        splitwise_source_hardware=str(args.splitwise_source_hardware),
        splitwise_source_tp=int(args.splitwise_source_tp),
        splitwise_calibration_mode=str(args.splitwise_calibration_mode),
        tp_gpus=(int(args.tp_gpus) if args.tp_gpus is not None else None),
        n_gpus_per_node=(int(args.n_gpus_per_node) if args.n_gpus_per_node is not None else None),
        gpu_tdp_w=float(args.gpu_tdp_w),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        pue=float(args.pue),
        arrival_bin_seconds=int(args.arrival_bin_seconds),
        heatmap_downsample_seconds=int(args.heatmap_downsample_seconds),
        peak_window_hours=float(args.peak_window_hours),
        dry_run_figures=bool(args.dry_run_figures),
        power_factor=float(args.power_factor),
        hide_redundant_cells=bool(args.hide_redundant_cells),
        allow_synthetic_request_timestamps=bool(args.allow_synthetic_request_timestamps),
    )

    print("[azure_bi_pipeline] Done")
    print(f"  trace_manifest : {result['trace_summary']['trace_manifest_csv']}")
    print(f"  metrics_csv    : {result['metrics_summary']['metrics_csv']}")
    print(f"  figures_manifest: {result['figures_manifest']['output_paths']['manifest']}")
    print(f"  sizing_tex     : {result['sizing_summary']['out_tex']}")


if __name__ == "__main__":
    main()
