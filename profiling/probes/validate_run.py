"""CLI: validate workload — real dataset (ShareGPT) + synthetic Poisson arrivals.

Emits the same §2 bundle as the probe/agentic runners. The dataset path comes from
``--dataset-path`` or ``$SHAREGPT_DATASET_PATH``. Run as
``python -m profiling.probes.validate_run --model Qwen/Qwen3-8B --hardware A100 --tp 1``.
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "client"))

from _cli import base_parser, server_cfg  # noqa: E402
import validate_runner  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--dataset", default="sharegpt")
    p.add_argument("--dataset-path", default=os.environ.get("SHAREGPT_DATASET_PATH"))
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--request-rate", type=float, default=4.0)
    args = p.parse_args()
    if not args.dataset_path:
        raise SystemExit(
            "validate_run: --dataset-path or $SHAREGPT_DATASET_PATH is required")

    workload = {"dataset": args.dataset, "num_prompts": args.num_prompts,
                "request_rate": args.request_rate}
    print(validate_runner.run(
        workload, model=args.model, hardware=args.hardware, tp=args.tp,
        gpus_per_node=args.gpus_per_node, server_cfg=server_cfg(args),
        out_root=args.out_root, dataset_path=args.dataset_path, base_url=args.base_url,
        weight_footprint_bytes=args.weight_footprint_bytes,
        dtype_hint=args.dtype_hint, n_active_override=args.n_active_override))


if __name__ == "__main__":
    main()
