# Supported Workflows (Bench → Train → Infer)

This repo supports multiple experiments, but these are the **intended** workflows to keep stable.

## 1) Bench collection (generate latency artifacts)

Run the benchmark client and save both the summary JSON and per-request JSONL:

```bash
python client/benchmark_serving.py \
  --backend vllm \
  --model <served_model_id> \
  --dataset-name sharegpt \
  --dataset-path <path> \
  --request-rate 10 \
  --num-prompts 1000 \
  --save-result \
  --save-detailed \
  --save-requests-jsonl
```

Outputs:
- `vllm-...json` (run summary + optionally per-request arrays)
- `vllm-...requests.jsonl` (preferred ingestion format for data prep/perf stats)

## 2) Build training dataset (power + requests → NPZ)

Use recorded request timestamps if present, and prefer JSONL sidecar:

```bash
python model/training_data/utils/prepare_training_data.py \
  --data_root_dir <root_dir_with_power_and_results> \
  --save_path <out>.npz \
  --timestamp_source recorded_scaled_or_poisson \
  --prefer_requests_jsonl \
  --decode_time_source e2e_minus_ttft
```

Notes:
- `--timestamp_source poisson` is legacy behavior (synthetic timestamps).
- `recorded_scaled_or_poisson` is usually best when power and request clocks differ.

## 3) Build performance database (TTFT/TPOT models)

This reads per-request JSONL sidecars when present (falls back to legacy benchmark JSON):

```bash
python model/utils/extract_performance_stats.py \
  --data_root_dir <root_dir_with_results_json> \
  --output_file model/config/performance_database.json
```

## 4) Inference: generate a simulated server power trace

Uses `model/config/performance_database.json` + `model/best_weights/*`:

```bash
python model/scripts/simulate_server_power.py \
  --model-name llama-3-8b \
  --model-size-b 8 \
  --hardware H100 \
  --tp 1 \
  --duration-s 600 \
  --rate-qps 2.0 \
  --use-fast-workload \
  --out-csv training_results/server_power.csv \
  --metadata-json training_results/server_power.meta.json
```
