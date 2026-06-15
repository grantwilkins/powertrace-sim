"""Write a sample synthetic §2 bundle for a campaign (dry-run review artifact).

No server, no GPU, no model download: emits the manifest + empty logger CSVs so a
reviewer can see exactly what each run will produce before launch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "profiling" / "client"))
sys.path.insert(0, str(REPO_ROOT / "profiling" / "jobs"))

import campaign_config  # noqa: E402
import metrics_logger  # noqa: E402
import power_logger  # noqa: E402
import run_manifest  # noqa: E402


def _stub_arch(model: str) -> dict:
    # offline placeholder so dry-run needs no network/model download
    return {
        "family": "sample", "n_active": 0.0, "w_bytes": 0.0, "d_model": 0,
        "n_layers": 0, "n_kv": 0, "head_dim": 0, "moe_frac": 0.0, "n_experts": 1,
        "top_k": 1, "swa_window": 0.0, "fp8": 0, "swa_global_ratio": 0.0,
        "linear_attention": 0, "n_linear_layers": 0,
        "_note": f"stub arch for dry-run; real values extracted from {model} config.json at launch",
    }


def main():
    campaign_path = sys.argv[1]
    c = campaign_config.load_campaign(campaign_path)
    tp = int(c["server"]["tp"])
    run_id = f"{c['hardware'].lower()}_{c['campaign_type']}_tp{tp}_SAMPLE"
    run_dir = REPO_ROOT / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # empty logger CSVs with the real headers
    (run_dir / "power.csv").write_text(", ".join(power_logger.EXTENDED_FIELDS) + "\n")
    (run_dir / "engine.csv").write_text(",".join(metrics_logger.ENGINE_HEADER) + "\n")
    (run_dir / "requests.json").write_text(json.dumps(
        {k: [] for k in ("input_lens", "output_lens", "ttfts", "itls",
                         "request_timestamps")}))

    manifest = run_manifest.build_manifest(
        run_id=run_id,
        probe={"type": ",".join(c["probes"]) or c["campaign_type"], "levels": []},
        model=c["model"], arch=_stub_arch(c["model"]), hardware=c["hardware"],
        tp=tp, gpus_per_node=c["gpus_per_node"], server=c["server"],
        versions={"vllm": "DRY-RUN", "git_sha": "DRY-RUN", "gpu_driver": "DRY-RUN"},
        clock=run_manifest.capture_clock(),
    )
    run_manifest.write_manifest(str(run_dir / "manifest.json"), manifest)
    print(f"wrote sample bundle -> {run_dir}")


if __name__ == "__main__":
    main()
