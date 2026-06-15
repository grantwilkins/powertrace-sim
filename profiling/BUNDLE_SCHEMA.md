# Bundle schema — the alignment contract

Every run emits one bundle `data/runs/<run_id>/` with four files. The design goal
of this doc: make explicit that we capture **all** the fields needed to align
per-request latencies ↔ engine state ↔ power, since without a common time base the
data is unusable. **All three time series are in absolute epoch seconds**, which is
what makes alignment possible.

```
data/runs/<run_id>/
├── power.csv        nvidia-smi @ 4 Hz   (epoch wall time)
├── engine.csv       vLLM /metrics @ 4 Hz (epoch, time.time())
├── requests.json    per-request latencies + epoch arrival times
├── manifest.json    arch, server, clock, and per-level epoch WINDOWS
└── levels/level_NNN.json   raw per-level benchmark output (kept for audit)
```

## 1. `power.csv` — power & DVFS (source: `power_logger.nvidia_smi_command`)

Per-GPU rows, 4 Hz. Header (units stripped from values):

| column | role |
|---|---|
| `timestamp` | **epoch wall time** (nvidia-smi local time → UTC via `power_timestamp_to_epoch`) — the alignment key |
| `power.draw` | per-GPU watts — the regression target (summed across the TP group) |
| `clocks.sm`, `clocks.mem` | DVFS state (largest previously-unmodeled term) |
| `utilization.gpu`, `utilization.memory` | occupancy cross-checks |
| `memory.used` | KV/weight footprint sanity |
| `temperature.gpu` | thermal context |

## 2. `engine.csv` — measured engine state (source: `metrics_logger`)

4 Hz scrape of vLLM `/metrics`. First column `timestamp` = `time.time()` epoch.

| column | type | role |
|---|---|---|
| `timestamp` | epoch | alignment key |
| `num_requests_running` | gauge | true decode batch size |
| `num_requests_waiting` | gauge | queue depth |
| `gpu_cache_usage_perc` | gauge | KV-cache occupancy (→ KV-read proxy, Phase-2) |
| `prompt_tokens_total` | counter | prefill token rate (diff on edges) |
| `generation_tokens_total` | counter | decode token rate (diff on edges) |
| `iteration_tokens_total_{sum,count}` | counter | iteration/effective-batch |
| `request_prefill_time_seconds_sum` | counter | prefill-iteration rate (→ `pre_iter`, Phase-2) |
| `request_decode_time_seconds_sum` | counter | decode-time accounting |

Counters are cumulative → difference across bin **edges** (never `last−first` within a
bin). Gauges → bin-mean.

## 3. `requests.json` — per-request latencies + epoch timestamps

Source: vendored `benchmark_serving.py --save-detailed` (one run per probe level,
concatenated). This is the reconstruction-ledger contract (`parse_request_json`):

| field | role |
|---|---|
| `request_timestamps` | **absolute epoch send time per request** (`backend_request_func.py` `time.time()`). THE field that aligns requests to power/engine; stock `vllm bench serve` lacks it (it saves monotonic `start_times`). |
| `ttfts` | time-to-first-token per request (s) — prefill/queue boundary |
| `itls` | inter-token latencies per request (list) — decode duration = Σ itls |
| `input_lens` | prompt tokens (incl. prefix) per request |
| `output_lens` | generated tokens per request |

`tpot`/`e2e` are derivable from the above and not separately stored.

## 4. `manifest.json` — the binding metadata (source: `run_manifest`)

```jsonc
{
  "run_id": "...", "model": "...", "hardware": "H100", "tp": 8, "gpus_per_node": 8,
  "arch": { /* extract_arch: n_active, w_bytes, n_layers, n_kv, head_dim,
              moe_frac, n_experts, top_k, swa_window, swa_global_ratio,
              linear_attention, n_linear_layers, fp8 */ },
  "server": { "max_num_seqs": 256, "enable_chunked_prefill": false,
              "enable_prefix_caching": false, "kv_cache_dtype": "auto",
              "max_model_len": 131072 },
  "versions": { "vllm": "...", "git_sha": "...", "gpu_driver": "..." },
  "clock": { "power_epoch_offset_s": 0.0, "engine_epoch_offset_s": 0.0,
             "monotonic_start": 12345.6 },
  "probe": {
    "type": "decode_staircase",
    "window": { "start_epoch": 1781..., "end_epoch": 1781... },   // whole run
    "levels": [
      { "level": 4, "label": "decode_N16", "concurrency": 16, "num_prompts": 32,
        "t_start_epoch": 1781..., "t_end_epoch": 1781...,          // ← per-level WINDOW
        "params": { "input_len": 8, "output_len": 2048, "prefix_len": 0,
                    "ignore_eos": true },
        "command": ["python", "benchmark_serving.py", "..."],      // reproducibility
        "summary": { "duration": 44.8, "completed": 32,
                     "output_throughput": 2200.0, ... } }
    ]
  }
}
```

## Alignment recipe (how the three streams are joined)

1. All timestamps are **epoch seconds**. Bin power & engine on a common 1 s grid.
2. `power.csv` is nvidia-smi local time coerced to UTC, so it can be skewed from
   `time.time()` by a whole/half hour; the reconstruction builder cancels this with
   the `%1800` fold (`build_run_bins`). New whole-pipeline runs stay aligned because
   `requests.json` timestamps and `engine.csv` both use `time.time()`.
3. **Per-level windows** (`t_start_epoch`/`t_end_epoch`) let any consumer slice the
   power/engine series by probe level — e.g. fit the power cap only on the saturation
   level, or `e_kv` only on `context_holds` levels — without re-deriving boundaries.
4. `requests.json` provides the per-request work (ttft/itl/lens) at each request's
   epoch arrival, which `build_ledger_bundle` turns into per-bin work rates.

## Multi-turn / agentic extension (planned)

The agentic workload emits the **same four files**; `requests.json` gains per-turn
fields so a session is reconstructable:

| extra field | role |
|---|---|
| `session_ids` | which conversation each request belongs to |
| `turn_idx` | turn number within the session (context grows with it) |
| `post_gap_s` | think-time / tool-execution idle after this turn (the agentic idle the model must capture) |
| `prefix_cache` | whether prefix caching was on (a cache hit skips prefill) |

`input_lens` already grows per turn (full prior context + new message), so KV/prefill
scaling falls out. The manifest's `probe.type` becomes `agentic`, and
`server.enable_prefix_caching` records the on/off regime.
