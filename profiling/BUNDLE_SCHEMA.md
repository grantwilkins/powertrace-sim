# Bundle schema — the alignment contract

Every campaign live run emits one bundle `data/runs/<campaign_id>/<run_id>/` with
four files. The design goal of this doc: make explicit that we capture **all** the
fields needed to align per-request latencies ↔ engine state ↔ power, since without
a common time base the data is unusable. Requests and engine samples are Unix
epoch; raw power uses local wall time plus the manifest's measured UTC offset.
Bundle ingestion converts power to Unix epoch before any model view is built.

```
data/runs/<campaign_id>/<run_id>/
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
| `timestamp` | nvidia-smi local wall time; converted using `manifest.clock.local_utc_offset_s` |
| `index`, `uuid` | stable device identity; required on every bundle row |
| `power.draw` | per-GPU watts — the regression target (summed across the TP group) |
| `clocks.sm`, `clocks.mem` | DVFS state (largest previously-unmodeled term) |
| `utilization.gpu`, `utilization.memory` | occupancy cross-checks |
| `memory.used` | KV/weight footprint sanity |
| `temperature.gpu` | thermal context |

The `tp8_state` power profile adds `pstate`, `power.limit`, and NVIDIA clock-event
reasons for software power capping, hardware slowdown, and software/hardware
thermal slowdown. A diagnostic bundle is rejected if any requested column is
absent; it is not silently reduced to the core profile.

## 2. `engine.csv` — measured engine state (source: `metrics_logger`)

4 Hz scrape of vLLM `/metrics`. First column `timestamp` = `time.time()` epoch.

| column | type | role |
|---|---|---|
| `timestamp` | epoch | alignment key |
| `num_requests_running` | gauge | total running requests across phases; **not** phase-specific decode batch |
| `num_requests_waiting` | gauge | queue depth |
| `gpu_cache_usage_perc` | gauge | KV-cache occupancy diagnostic |
| `prompt_tokens_total` | counter | prefill token rate (diff on edges) |
| `generation_tokens_total` | counter | decode token rate (diff on edges) |
| `iteration_tokens_total_{sum,count}` | counter | iteration/effective-batch |
| `request_prefill_time_seconds_sum` | optional counter | retained when the installed vLLM version exposes it; not required or consumed |
| `request_decode_time_seconds_sum` | optional counter | retained when exposed; not required or consumed |
| `num_preemptions_total`, prefix-cache counters | optional counters | retained for audit when exposed; not part of the current ledger |

Counters are cumulative → difference across bin **edges** (never `last−first` within a
bin). Gauges → bin-mean.

The `measured_ledger` evidence profile requires only the stock gauges and token /
iteration counters above. `feature-test/build_ledger_bundle.py --state-source
measured_engine` consumes them through a hybrid projection. It never interprets
`num_requests_running` as decode batch and never fabricates missing collective,
router, or expert counters. Every required sample must be finite; scrape gaps
invalidate the bundle rather than being interpolated.

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
  "evidence_profile": "measured_ledger",
  "validation_role": "development | sealed",
  "arch": { /* extract_arch: n_active, w_bytes, n_layers, n_kv, head_dim,
              moe_frac, n_experts, top_k, swa_window, swa_global_ratio,
              linear_attention, n_linear_layers, fp8 */ },
  "server": { "max_num_seqs": 256, "enable_chunked_prefill": false,
              "enable_prefix_caching": false, "kv_cache_dtype": "auto",
              "max_model_len": 131072,
              "active_gpu_uuids": ["GPU-...", "GPU-..."] },
  "versions": { "vllm": "...", "git_sha": "...", "gpu_driver": "..." },
  "clock": { "local_utc_offset_s": -25200.0,
             "power_timestamp_basis": "local_wall_time",
             "engine_timestamp_basis": "unix_epoch",
             "request_timestamp_basis": "unix_epoch",
             "reference_epoch_s": 1781..., "monotonic_start": 12345.6 },
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

1. Convert each raw power timestamp to Unix epoch by subtracting
   `clock.local_utc_offset_s`; requests and engine timestamps are already epoch.
2. Require both `index` and `uuid`, validate their stable one-to-one mapping, and
   group one 4 Hz sample across at most 50 ms of per-GPU capture skew. Every sample
   must contain the same UUID set with size exactly `gpus_per_node`; a mismatch
   aborts ingestion and rows are never combined into fixed-size anonymous blocks.
   For a smaller TP leg in a max-TP allocation, require
   `server.active_gpu_uuids` to match the first `tp` device columns summed into
   the target.
3. **Per-level windows** (`t_start_epoch`/`t_end_epoch`) let any consumer slice the
   power/engine series by probe level — e.g. fit the power cap only on the saturation
   level, or `e_kv` only on `context_holds` levels — without re-deriving boundaries.
4. `requests.json` provides the per-request work (ttft/itl/lens) at each request's
   epoch arrival, which `build_ledger_bundle` turns into per-bin work rates.

## Maintained projection into the model

```text
power.csv ───────────────┐
engine.csv ──┐           ├─> RunRecord ─> measured hybrid bins ─> ledger_cache.npz
requests.json┼─ manifest ┘                         │
             └─ hashes/clock/arch                  └─> selected physics inputs
```

| persisted ledger column | exact origin | current status |
|---|---|---|
| `power` | summed topology-validated GPU power | regression target |
| `pre_tok`, `dec_tok` | counter differences on bin edges | selected-model input |
| `A_t`, `running_requests`, `waiting_requests` | piecewise-linear time means | selected-model input |
| `batch`, `pre_active`, `iters`, `kv_read` and derived work | request TTFT/ITL plus manifest architecture | selected-model input; reconstructed |
| `engine_iteration_tokens_rate`, `engine_iterations_rate`, `engine_tokens_per_iteration` | iteration histogram sum/count differences | persisted diagnostic/candidate input |
| `engine_gpu_cache_usage` | time-mean stock cache gauge | persisted diagnostic/candidate input |

The bundle sidecar records this per-field lineage. The last two rows are present
only for `--state-source measured_engine`; the frozen selected artifact does not
consume them automatically. This is deliberate: capture and persistence are now
ready, while feature selection remains governed by the existing evaluation gates.

The normalized `RunRecord` retains every power column, every list-valued request
column (including agentic extensions), every engine column, and source hashes.
GRU and physics views intentionally project only the fields they consume.

## Multi-turn and exact-trace extensions

The agentic and `trace_replay` workloads emit the **same four files**.
`trace_replay` consumes a hash-bound plan and uses deterministic direct token IDs,
so cache-off/on runs have identical prompts and arrival marks without retaining
private trace text. `requests.json` gains:

| extra field | role |
|---|---|
| `session_ids` | which conversation each request belongs to |
| `turn_idx` | turn number within the session (context grows with it) |
| `post_gap_s` | think-time / tool-execution idle after this turn (the agentic idle the model must capture) |
| `prefix_cache` | whether prefix caching was on (a cache hit skips prefill) |
| `planned_ready_epoch`, `arrival_delay_s` | requested release and observed send delay; release precedes concurrency acquisition |
| `prefix_tokens`, `new_input_tokens`, `planned_output_tokens` | exact replay marks |
| `cached_prompt_tokens` | server-reported prompt tokens served from cache |
| `expected_cached_tokens` | plan expectation; measured value must be within one declared cache block |
| `reasoning_tokens` | server-reported reasoning subset of completion tokens |
| `source_ids` | source-local row/session provenance |
| `prompt_sha256`, `output_sha256` | canonical prompt/output-token identity checks across paired regimes |
| `forced_output_token_id`, `request_seed`, `decode_constraint` | deterministic singleton-token replay protocol |

`input_lens` already grows per turn (full prior context + new message), so KV/prefill
scaling falls out. Reasoning is charged once as ordinary decode through
`output_lens`; `reasoning_tokens` is a slice label, not additional work. The
manifest records `probe.type`, the trace source/revision/hash/seed,
`server.enable_prefix_caching`, and the top-level evidence/validation roles.
Real agentic replay uses `probe.replay_plan_sha256`; direct-token replay uses
`probe.trace_plan_sha256`. Both hashes exclude the cache treatment so the
off/on pair can be compared.

Synthetic real-text validation records both `request_rate` and `burstiness` in
the level parameters and benchmark command. Interarrival times use a Gamma
distribution with mean `1 / request_rate` and shape `burstiness`: shape 1 is
Poisson, smaller values are burstier, and larger values are smoother. These
fields describe offered arrivals, not achieved throughput.
