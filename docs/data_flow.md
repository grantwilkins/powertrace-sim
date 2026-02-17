# Powertrace-Sim Data Flow (Current → Target)

This repo grew “organically” and currently has multiple partially-overlapping ways to:
- collect serving latency + token stats (TTFT/TPOT/etc),
- collect GPU power traces,
- convert those into training datasets,
- train a model and then generate simulated power traces.

The goal of this doc is to establish **one consistent data flow** with **as few sources of truth as possible**.

---

## Definitions (terms)

- **TTFT**: time-to-first-token (seconds). Measured on the client side.
- **E2E**: end-to-end latency per request (seconds). Measured on the client side.
- **TPOT**: time-per-output-token excluding first token (seconds/token).
  - Canonical formula: `tpot_s = (e2e_s - ttft_s) / max(output_tokens - 1, 1)`
- **Request timestamp**: epoch seconds at which the client sends the request.

---

## Current Data Flow (as of Jan 2026)

```text
job-scripts/*.sh + server/serve-*.sh
  └─ start vLLM OpenAI-compatible server

client/benchmark_serving.py + client/backend_request_func.py
  └─ send requests at a chosen rate (Poisson/gamma)
  └─ record per-request TTFT and ITLs (and sometimes request timestamps)
  └─ write vllm-*.json results

nvidia-smi sampling (via job scripts)
  └─ write power CSV (timestamp + per-GPU power rows)

model/training_data/utils/prepare_training_data.py
  └─ parse benchmark JSON + parse power CSV
  └─ **synthesize** request timestamps (Poisson) instead of using recorded timestamps
  └─ derive prefill/decode timing using ad-hoc rules
  └─ write dataset NPZ used by training

model/train_entry.py → model/classifiers/train.py
  └─ train GRU classifier
  └─ write weights .pt

model/utils/extract_performance_stats.py → model/config/performance_database.json
  └─ fit TTFT/TPOT distributions from benchmark JSONs
  └─ optionally attach GMM params from NPZ training data

model/simulators/server_power_simulator.py (+ datacenter simulator)
  └─ combine: (perf DB TTFT/TPOT) + (GMM params) + (weights) to simulate power
```

### Main current issues
- **Multiple definitions / derivations** of TTFT/TPOT/decode time.
- **Timestamp alignment is not authoritative**: training prep may ignore recorded request timestamps.
- Model/hardware/tp metadata is often inferred from filenames rather than recorded explicitly.

---

## Target Data Flow (recommended)

We want exactly **three canonical artifact types**:

1. **Run Bundle** (single source of truth for raw observations)
2. **Training Dataset** (single source of truth for supervised learning inputs)
3. **Model Profile** (single source of truth for inference/simulation configuration)

```text
bench collection
  └─ RunBundle/
       run.json
       requests.jsonl
       power_raw.csv
       power_agg.csv

data prep
  └─ Dataset/
       dataset.npz
       dataset.json  (manifest + provenance)

training + fitting
  └─ ModelProfile/
       profile.json  (weights + GMM + TTFT/TPOT models)
       weights.pt

inference / simulation
  └─ consumes ModelProfile (+ optionally RunBundle trace replay)
```

---

## Canonical Schemas (v0)

These are “contracts”. Code should **emit** these and consumers should **prefer** these.

### 1) Run Bundle: `run.json` (RunMetadata)

Minimal fields:

```json
{
  "run_id": "2026-01-29T12-34-56Z_llama-3-8b_h100_tp1_qps10",
  "created_at": "2026-01-29T12:34:56Z",
  "model_name": "llama-3-8b",
  "hardware": "h100",
  "tensor_parallelism": 1,
  "request_rate_qps": 10.0,
  "dataset_name": "sharegpt",
  "backend": "vllm",
  "notes": ""
}
```

### 2) Run Bundle: `requests.jsonl` (RequestTiming)

One JSON object per line, **per request**:

```json
{
  "request_id": "req_000001",
  "request_timestamp_s": 1738157696.123,
  "input_tokens": 512,
  "output_tokens": 256,
  "ttft_s": 0.412,
  "e2e_s": 2.901,
  "tpot_s": 0.0098,
  "success": true,
  "error": ""
}
```

Notes:
- `request_timestamp_s` is epoch seconds at request send time.
- `tpot_s` is derived from `ttft_s`, `e2e_s`, `output_tokens`.

### 3) Dataset: `dataset.json` (DatasetManifest)

```json
{
  "dataset_id": "dataset_2026-01-29_llama-3-8b_h100_tp1",
  "created_at": "2026-01-29T13:00:00Z",
  "source_run_ids": ["..."],
  "timestamp_alignment": {
    "source": "recorded_request_timestamp_s",
    "fallback": "synthetic_poisson"
  },
  "feature_spec": {
    "dt_s": 0.25,
    "Dx": 6,
    "columns": ["cnt", "tok_in", "tok_out", "active_requests", "prefill_tokens", "decode_tokens"]
  }
}
```

### 4) Model Profile: `profile.json` (single inference source of truth)

```json
{
  "profile_id": "llama-3-8b_h100_tp1",
  "created_at": "2026-01-29T14:00:00Z",
  "model_name": "llama-3-8b",
  "hardware": "h100",
  "tensor_parallelism": 1,
  "weights": {
    "path": "weights.pt",
    "sha256": "..."
  },
  "power_states": {
    "mu": [ ... ],
    "sigma": [ ... ]
  },
  "latency_models": {
    "ttft": { "type": "gamma_glm", "...": "..." },
    "tpot": { "type": "gaussian", "...": "..." }
  },
  "feature_spec": {
    "dt_s": 0.25,
    "Dx": 6
  }
}
```

