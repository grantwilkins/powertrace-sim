# Minimal A100 Transfer-Interval Campaign

This campaign adds configuration-level uncertainty intervals to the transfer
section without repeating the ShareGPT profiling campaign. Every row uses the
same A100 generation, fixed TP, runtime, telemetry, and scoring protocol. The
interval is the bootstrap interval over six independent windows, packs, or load
points; it is not a per-sample or per-second confidence interval.

## Existing and additional units

| Transfer axis | Existing evidence | Reused in interval | New executions | Final units |
|---|---:|---:|---:|---:|
| BurstGPT arrivals, Qwen3-8B A100 TP1 | 3 windows | 2 populated windows | 4 windows | 6 |
| OpenHands waits, Qwen3-8B A100 TP1 | 3 packs, each cache-off/on | 3 cache-off packs | 3 offset cache-off packs | 6 |
| Dense checkpoint, Qwen3-14B A100 TP1 | 1 rate point | 1 | 5 rates | 6 |
| MoE checkpoint, Qwen3-30B-A3B | 1 H100 TP2 point | 0 for A100 interval | 6 A100 TP2 rates | 6 |

The seven-request middle BurstGPT window is retained as descriptive evidence
but excluded from the interval. The new OpenHands packs use
`session_offset=24`, so each pack starts after the 24 sessions used in the
existing pack. Report OpenHands with the frozen simulator and the same
idle-only adjustment as the other workload-transfer rows; do not fit dynamic
gains to these six evaluation packs.

The dense and MoE rows vary only request rate over the same 600-prompt ShareGPT
sample. This tests transfer over offered load while holding token marks,
hardware generation, runtime, and model parallelism fixed.

## Estimated A100 time

Existing bundle durations imply about 90 seconds of model restart overhead per
regime. The prospective budget is:

| Campaign | New regimes | GPUs | Estimated wall time | Estimated GPU-hours |
|---|---:|---:|---:|---:|
| BurstGPT | 4 | 1 | 1.2 h | 1.2 |
| OpenHands | 3 | 1 | 0.35 h | 0.35 |
| Qwen3-14B dense | 5 | 1 | 0.95 h | 0.95 |
| Qwen3-30B-A3B MoE | 6 | 2 | 1.0 h | 2.0 |
| **Total** | **18** | | **about 3.5 serial allocation-hours** | **about 4.5 A100 GPU-hours** |

Reserve 5 A100 GPU-hours to cover queueing-independent variation in model
startup and overloaded high-rate points. The JSON wall-time limits total 5.75
allocation-hours and are deliberately conservative.

## Sherlock preparation

Stage the three required checkpoints and the pinned OpenHands source:

```bash
bash profiling/jobs/stage_models.sh \
  Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B

bash profiling/jobs/stage_openhands.sh \
  aa8977805b4cefd317001d80ddf1ad52790e9d23
```

Build four populated BurstGPT windows while excluding the two existing windows
that will be reused:

```bash
SOURCE_SHA256="$(sha256sum \
  "$GROUP_HOME/gfw/BurstGPT_without_fails_2.csv" | cut -d' ' -f1)"

uv run python profiling/agentic_traces/build_transfer_ci_plans.py \
  "$GROUP_HOME/gfw/BurstGPT_without_fails_2.csv" \
  --revision "$SOURCE_SHA256"
```

Freeze the three exact OpenHands plan hashes after the model tokenizer and
pinned dataset are staged:

```bash
bash profiling/jobs/transfer_ci_a100.sh --freeze-openhands
```

Review all emitted server and workload commands without launching:

```bash
bash profiling/jobs/transfer_ci_a100.sh --dry-run
```

After predictions and the idle-only scoring rule are frozen, submit all four
checkpoint-resumable jobs:

```bash
bash profiling/jobs/transfer_ci_a100.sh --submit
```

Do not add cache-on duplicates or hardware/TP sweeps to this campaign. Those
change the scientific question and are not needed for the transfer interval.
