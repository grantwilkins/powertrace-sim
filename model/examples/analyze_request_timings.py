import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from model.simulators.arrival_simulator import (
    FastWorkloadGenerator,
    ServeGenRequest,
    ServeGenWorkloadGenerator,
    ServingConfig,
    ServingSystemSimulator,
    create_deepseek_config,
    create_llama_config,
)


def build_config(model: str, hardware: str, tp: int) -> ServingConfig:
    model_l = model.lower()
    if "llama" in model_l:
        size_b = int(model_l.split("-")[-1].replace("b", ""))
        return create_llama_config(size_b, tp=tp, hardware=hardware)
    if "deepseek" in model_l:
        size_b = int(model_l.split("-")[-1].replace("b", ""))
        return create_deepseek_config(size_b, tp=tp, hardware=hardware)
    raise ValueError(f"Unknown model spec: {model}")


def compute_request_metrics(requests: List[ServeGenRequest]) -> Dict[str, np.ndarray]:
    if not requests:
        raise ValueError("No processed requests provided")

    ttft = np.array(
        [
            (r.prefill_end - r.prefill_start)
            if r.prefill_end is not None and r.prefill_start is not None
            else np.nan
            for r in requests
        ],
        dtype=float,
    )

    tpot = np.array(
        [
            ((r.decode_end - r.decode_start) / max(1, r.output_tokens))
            if r.decode_end is not None and r.decode_start is not None
            else np.nan
            for r in requests
        ],
        dtype=float,
    )

    latency = np.array(
        [
            (r.decode_end - r.arrival_time) if r.decode_end is not None else np.nan
            for r in requests
        ],
        dtype=float,
    )

    # Drop NaNs if any
    ttft = ttft[~np.isnan(ttft)]
    tpot = tpot[~np.isnan(tpot)]
    latency = latency[~np.isnan(latency)]

    return {"ttft": ttft, "tpot": tpot, "latency": latency}


def summarize(metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if metrics.get("ttft") is not None and metrics["ttft"].size > 0:
        # milliseconds
        ttft_ms = metrics["ttft"] * 1_000.0
        out["ttft_ms"] = {
            "p50": float(np.percentile(ttft_ms, 50)),
            "p95": float(np.percentile(ttft_ms, 95)),
        }
    if metrics.get("tpot") is not None and metrics["tpot"].size > 0:
        # milliseconds
        tpot_ms = metrics["tpot"] * 1_000.0
        out["tpot_ms"] = {
            "p50": float(np.percentile(tpot_ms, 50)),
            "p95": float(np.percentile(tpot_ms, 95)),
        }
    if metrics.get("latency") is not None and metrics["latency"].size > 0:
        # seconds (already)
        lat = metrics["latency"]
        out["latency_s"] = {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        }
    return out


def maybe_write_csv(path: str, requests: List[ServeGenRequest]) -> None:
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "request_id",
                "arrival_time_s",
                "input_tokens",
                "output_tokens",
                "ttft_s",
                "tpot_s",
                "latency_s",
            ]
        )
        for r in requests:
            ttft_s = (
                (r.prefill_end - r.prefill_start)
                if r.prefill_end is not None and r.prefill_start is not None
                else np.nan
            )
            tpot_s = (
                ((r.decode_end - r.decode_start) / max(1, r.output_tokens))
                if r.decode_end is not None and r.decode_start is not None
                else np.nan
            )
            latency_s = (
                (r.decode_end - r.arrival_time) if r.decode_end is not None else np.nan
            )
            w.writerow(
                [
                    r.request_id,
                    r.arrival_time,
                    r.input_tokens,
                    r.output_tokens,
                    ttft_s,
                    tpot_s,
                    latency_s,
                ]
            )


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    # Minimal readable text table in stdout
    def fmt(v: float, unit: str = "") -> str:
        if unit == "ms":
            return f"{v:.1f}"
        return f"{v:.2f}"

    print()
    print("Metric                 | Value")
    print("-----------------------+-----------------")
    if "ttft_ms" in summary:
        s = summary["ttft_ms"]
        print("TTFT (milliseconds)")
        print(f"  P50                 | {fmt(s['p50'], 'ms')}")
        print(f"  P95                 | {fmt(s['p95'], 'ms')}")
        print("-----------------------+-----------------")
    if "tpot_ms" in summary:
        s = summary["tpot_ms"]
        print("TPOT (milliseconds)")
        print(f"  P50                 | {fmt(s['p50'], 'ms')}")
        print(f"  P95                 | {fmt(s['p95'], 'ms')}")
        print("-----------------------+-----------------")
    if "latency_s" in summary:
        s = summary["latency_s"]
        print("Request Latency (seconds)")
        print(f"  P50                 | {fmt(s['p50'])}")
        print(f"  P95                 | {fmt(s['p95'])}")
        print(f"  P99                 | {fmt(s['p99'])}")
        print("-----------------------+-----------------")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TTFT, TPOT, and request latency using arrival simulator",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3-70b",
        help="Model spec, e.g., llama-3-8b or deepseek-r1-70b",
    )
    parser.add_argument(
        "--hardware", type=str, default="A100", help="Hardware type, e.g., A100 or H100"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument(
        "--workload",
        type=str,
        default="language",
        help="Workload category: language|reason|multimodal",
    )
    parser.add_argument(
        "--duration", type=float, default=3600.0, help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Mean request arrival rate (requests/sec)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--use-fast",
        action="store_true",
        help="Use fast synthetic workload generator (no ServeGen)",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional path to write per-request metrics CSV",
    )
    args = parser.parse_args()

    config = build_config(args.model, args.hardware, args.tp)

    # Generate requests
    if args.use_fast:
        reqs = FastWorkloadGenerator().generate_requests(
            duration=args.duration,
            rate_requests_per_sec=args.rate,
            seed=args.seed,
        )
    else:
        # Fall back to fast mode if ServeGen is not available at runtime
        try:
            reqs = ServeGenWorkloadGenerator().generate_requests(
                category=args.workload,
                model_type="m-large"
                if "70b" in args.model or "405b" in args.model
                else "m-mid",
                duration=args.duration,
                time_window=None,
                rate_requests_per_sec=args.rate,
                seed=args.seed,
            )
        except Exception:
            reqs = FastWorkloadGenerator().generate_requests(
                duration=args.duration,
                rate_requests_per_sec=args.rate,
                seed=args.seed,
            )

    # Simulate processing
    simulator = ServingSystemSimulator(config)
    processed = simulator.simulate_request_processing(reqs)

    # Compute metrics and summarize
    metrics = compute_request_metrics(processed)
    summary = summarize(metrics)
    print_summary_table(summary)

    # Optional CSV output
    if args.csv_out:
        maybe_write_csv(args.csv_out, processed)
        print(f"Wrote per-request metrics to: {args.csv_out}")


if __name__ == "__main__":
    main()
