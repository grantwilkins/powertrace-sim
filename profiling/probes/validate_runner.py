"""Validate runner: real-dataset workload under the continuous loggers -> §2 bundle.

The ``validate`` campaign drives the vendored ``benchmark_serving.py`` against a
**real dataset** (ShareGPT prompts) with a **synthetic Poisson arrival schedule**
(``--request-rate``), and emits the SAME four-file bundle as the probes
(``power.csv`` + ``engine.csv`` + ``requests.json`` + ``manifest.json``) by reusing
``probe_runner.logging_session`` and ``run_manifest``. The probes use
``--dataset-name random`` (synthetic content); this is the real-data path: real
prompts/tasks, only the arrival process is synthetic.

``build_validate_command`` is pure / unit-tested; ``run`` is the live orchestration
(one ``benchmark_serving`` subprocess for the whole workload, wrapped by the
continuous power + /metrics loggers).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLIENT = _HERE.parents[0] / "client"
sys.path.insert(0, str(_HERE))     # bench_driver, probe_runner
sys.path.insert(0, str(_CLIENT))   # client loggers / manifest / arch

import bench_driver  # noqa: E402
import probe_runner  # noqa: E402

BENCH_SCRIPT = _CLIENT / "benchmark_serving.py"


def build_validate_command(model, base_url, tp, workload, dataset_path,
                           result_path, max_model_len=None) -> list[str]:
    """``benchmark_serving.py`` argv for one real-dataset validate workload (pure).

    Real prompts (``workload['dataset']`` + ``dataset_path``) paced by a synthetic
    Poisson arrival process (``--request-rate``). ``base_url`` carries the ``/v1``
    the loggers derive ``/metrics`` from; ``benchmark_serving`` composes its URL as
    ``base_url + endpoint``, so we strip ``/v1`` and pass ``--endpoint`` explicitly
    to avoid a doubled path. ``--save-detailed`` is what writes the per-request
    epoch ``request_timestamps`` that align this run to power/engine.

    ``max_model_len`` is forwarded so dataset length pruning tracks the served
    context window (else benchmark_serving drops every prompt over 1024 tokens).
    """
    root = base_url.rsplit("/v1", 1)[0].rstrip("/")
    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        "--model", model,
        "--backend", "vllm",
        "--base-url", root,
        "--endpoint", "/v1/completions",
        "--dataset-name", str(workload["dataset"]),
        "--dataset-path", str(dataset_path),
        "--request-rate", str(workload["request_rate"]),
        "--num-prompts", str(workload["num_prompts"]),
        "--tensor-parallel-size", str(tp),
        "--save-result", "--save-detailed",
        "--result-filename", str(result_path),
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(int(max_model_len))]
    return cmd


def build_validate_window(workload, t_start_epoch, t_end_epoch, command,
                          summary) -> dict:
    """Manifest entry locating the workload in absolute time (pure)."""
    return {
        "level": 0,
        "label": f"{workload['dataset']}_rate{workload['request_rate']}",
        "concurrency": -1,                       # open-loop arrival, not fixed conc
        "num_prompts": int(workload["num_prompts"]),
        "t_start_epoch": float(t_start_epoch),
        "t_end_epoch": float(t_end_epoch),
        "params": {"dataset": workload["dataset"],
                   "request_rate": workload["request_rate"],
                   "num_prompts": int(workload["num_prompts"])},
        "command": command,
        "summary": summary,
    }


def run(workload, *, model, hardware, tp, gpus_per_node, server_cfg, out_root,
        dataset_path, base_url="http://localhost:8000/v1",
        weight_footprint_bytes=None, dtype_hint=None, n_active_override=None,
        run_id=None):
    """Drive ``benchmark_serving`` over a real dataset; write the §2 bundle."""
    run_manifest = probe_runner._client_mod("run_manifest")
    arch_extract = probe_runner._client_mod("arch_extract")

    run_id = run_id or f"{hardware.lower()}_validate_tp{tp}_{int(time.time())}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "levels").mkdir(exist_ok=True)
    result_path = run_dir / "levels" / "validate.json"

    arch = arch_extract.extract_arch(
        arch_extract.load_config(model), dtype_hint=dtype_hint,
        weight_footprint_bytes=weight_footprint_bytes,
        n_active_override=n_active_override,
    )

    command = build_validate_command(model, base_url, tp, workload, dataset_path,
                                     result_path, max_model_len=server_cfg.get("max_model_len"))
    window_start = time.time()
    with probe_runner.logging_session(run_dir, base_url) as clock:
        t0 = time.time()
        subprocess.run(command, check=True)
        t1 = time.time()
    window_end = time.time()

    level = bench_driver.parse_level_result(result_path)
    (run_dir / "requests.json").write_text(
        json.dumps(bench_driver.merge_request_arrays([level])))

    manifest = run_manifest.build_manifest(
        run_id=run_id,
        probe={"type": "validate",
               "window": {"start_epoch": window_start, "end_epoch": window_end},
               "levels": [build_validate_window(
                   workload, t0, t1, command, level["summary"])]},
        model=model, arch=arch, hardware=hardware, tp=tp,
        gpus_per_node=gpus_per_node, server=dict(server_cfg),
        versions=run_manifest.collect_versions(), clock=clock,
    )
    run_manifest.write_manifest(str(run_dir / "manifest.json"), manifest)
    return run_dir
