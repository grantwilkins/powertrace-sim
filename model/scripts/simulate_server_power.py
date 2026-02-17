import argparse
import json
import os

import numpy as np

from model.simulators.arrival_simulator import ServingConfig
from model.simulators.server_power_simulator import ServerPowerSimulator


def _infer_perf_db_model_name(model_name: str) -> str:
    """
    performance_database.json uses family keys (e.g., 'llama-3.1', 'deepseek-r1-distill').
    Prefer explicit mapping here to avoid surprises.
    """
    name = (model_name or "").lower()
    if "llama" in name:
        return "llama-3.1"
    if "deepseek" in name:
        return "deepseek-r1-distill"
    raise ValueError(
        f"Unknown model family for perf DB mapping: model_name={model_name!r}. "
        "Pass a model name containing 'llama' or 'deepseek', or update mapping."
    )


def _save_csv(path: str, timestamps: np.ndarray, watts: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp_s,watts\n")
        for t, p in zip(timestamps, watts):
            f.write(f"{float(t)},{float(p)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a single server power trace using learned weights + performance database."
    )
    parser.add_argument("--model-name", type=str, required=True, help="e.g. llama-3-8b")
    parser.add_argument("--model-size-b", type=int, required=True, help="e.g. 8, 70")
    parser.add_argument(
        "--hardware", type=str, required=True, choices=["A100", "H100", "a100", "h100"]
    )
    parser.add_argument("--tp", type=int, required=True, choices=[1, 2, 4, 8])
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--rate-qps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dt", type=float, default=1.0)
    parser.add_argument(
        "--use-fast-workload",
        action="store_true",
        help="Use the lightweight synthetic workload generator (does not require ServeGen).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["language", "reason", "multimodal"],
        help="ServeGen category override (otherwise inferred from model name).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="m-mid",
        choices=["m-small", "m-mid", "m-large"],
        help="ServeGen model_type bucket.",
    )
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional path to write a small metadata JSON next to the CSV.",
    )
    args = parser.parse_args()

    hardware = args.hardware.upper()
    perf_db_model_name = _infer_perf_db_model_name(args.model_name)

    cfg = ServingConfig(
        model_name=args.model_name,
        model_size_b=int(args.model_size_b),
        hardware=hardware,
        tensor_parallelism=int(args.tp),
        perf_db_model_name=perf_db_model_name,
    )

    sim = ServerPowerSimulator(cfg)
    category = args.category
    if category is None:
        category = "language" if "llama" in args.model_name.lower() else "reason"
    res = sim.simulate_server_power(
        category=category,
        model_type=args.model_type,
        duration=float(args.duration_s),
        rate_requests_per_sec=float(args.rate_qps),
        seed=int(args.seed),
        output_dt=float(args.output_dt),
        return_profile=False,
        use_fast_workload=bool(args.use_fast_workload),
    )

    ts = np.asarray(res["timestamps"], dtype=float)
    watts = np.asarray(res["watts"], dtype=float)
    _save_csv(args.out_csv, ts, watts)

    if args.metadata_json:
        meta = {
            "model_name": args.model_name,
            "model_size_b": int(args.model_size_b),
            "hardware": hardware,
            "tp": int(args.tp),
            "duration_s": float(args.duration_s),
            "rate_qps": float(args.rate_qps),
            "seed": int(args.seed),
            "output_dt": float(args.output_dt),
            "perf_db_model_name": perf_db_model_name,
            "resolved_weights_path": res.get("weights_path"),
        }
        os.makedirs(os.path.dirname(args.metadata_json) or ".", exist_ok=True)
        with open(args.metadata_json, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
