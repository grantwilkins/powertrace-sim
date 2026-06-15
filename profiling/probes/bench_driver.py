"""Drive the repo's vendored ``benchmark_serving.py`` as the probe traffic source.

Why this script and not stock ``vllm bench serve``: this vendored copy is vLLM's
serving benchmark **plus the epoch-timestamp patch** — ``backend_request_func.py``
sets ``output.request_timestamp = time.time()`` (absolute wall-clock epoch) at send
time, and ``benchmark_serving.py`` saves it as ``request_timestamps``. That epoch
field is what lets us align per-request timing with the nvidia-smi power log (also
epoch) and the /metrics engine log. Stock ``vllm bench serve`` only saves
``start_times`` from ``perf_counter`` (monotonic, NOT epoch), which cannot be
aligned to nvidia-smi without a separately-measured offset.

A "level" = one steady-state operating point. We run it as
``--request-rate inf --max-concurrency N`` (the Maximum-Throughput pattern) with
``--ignore-eos`` so output length is fixed and the state is held, sized by
``--num-prompts`` to last ~hold_s.

``build_command`` / ``merge_request_arrays`` / ``parse_level_result`` are pure and
unit-tested; ``run_level`` is the live subprocess layer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BENCH_SCRIPT = Path(__file__).resolve().parents[1] / "client" / "benchmark_serving.py"

# Per-request arrays the reconstruction ledger needs (parse_request_json contract).
REQUEST_ARRAY_KEYS = (
    "input_lens", "output_lens", "ttfts", "itls", "request_timestamps",
)
# Per-level aggregates kept in the manifest for slicing / sanity.
SUMMARY_KEYS = (
    "duration", "completed", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
)


def build_command(model: str, base_url: str, tp: int, level, result_path) -> list[str]:
    """Construct the ``benchmark_serving.py`` argv for one probe level (pure)."""
    r = level.request
    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        "--model", model,
        "--backend", "vllm",
        "--base-url", base_url,
        "--dataset-name", "random",
        "--random-input-len", str(r.input_len),
        "--random-output-len", str(r.output_len),
        "--random-prefix-len", str(r.prefix_len),
        "--request-rate", "inf",
        "--max-concurrency", str(level.concurrency),
        "--num-prompts", str(level.num_prompts),
        "--tensor-parallel-size", str(tp),
        "--save-result", "--save-detailed",
        "--result-filename", str(result_path),
    ]
    if r.ignore_eos:
        cmd.append("--ignore-eos")
    return cmd


def parse_level_result(path) -> dict:
    """Read one level's result JSON into {arrays, summary} (epoch timestamps kept)."""
    data = json.loads(Path(path).read_text())
    arrays = {k: list(data.get(k, [])) for k in REQUEST_ARRAY_KEYS}
    summary = {k: data.get(k) for k in SUMMARY_KEYS}
    return {"arrays": arrays, "summary": summary}


def empty_level_result() -> dict:
    """Result for an idle (concurrency 0) level — no traffic."""
    return {"arrays": {k: [] for k in REQUEST_ARRAY_KEYS},
            "summary": {k: 0 for k in SUMMARY_KEYS}}


def merge_request_arrays(level_results: list[dict]) -> dict:
    """Concatenate per-request arrays across levels into one requests.json dict.

    Each request carries its own absolute epoch ``request_timestamp``, so the
    ledger places it correctly in time regardless of level ordering; the manifest
    level windows let downstream slice by probe level.
    """
    merged = {k: [] for k in REQUEST_ARRAY_KEYS}
    for lr in level_results:
        for k in REQUEST_ARRAY_KEYS:
            merged[k].extend(lr["arrays"][k])
    return merged


def run_level(model, base_url, tp, level, result_path) -> dict:
    """Live: run benchmark_serving.py for one level, return parsed result."""
    if level.concurrency <= 0:
        import time
        time.sleep(level.hold_seconds)
        return empty_level_result()
    cmd = build_command(model, base_url, tp, level, result_path)
    subprocess.run(cmd, check=True)
    return parse_level_result(result_path)
