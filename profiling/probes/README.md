# Profiling probes & campaign runner (Tier 0 + Tier 1)

State-space profiling for the first-principles power model, implementing
`profiling/CAMPAIGN.md` §5-A/B/E/F. This is **additive**: the existing
`profiling/jobs/*.sh` rate×TP pipeline and `feature-test/build_ledger_cache.py`
are left untouched and remain the known-good path.

> Status: built and unit-tested, **not yet run on a GPU**. The acceptance gate is
> the test suite + the ledger equivalence test + the `--dry-run` plan. Live
> behaviour (probe → server → loggers) is validated on first launch.

## What it does

Instead of measuring *workloads*, probes pin the engine at chosen state-space
points (idle → bandwidth-saturation → compute-saturation → power-cap) and log the
true engine state from vLLM `/metrics` alongside extended `nvidia-smi` power. Each
run emits one self-describing **bundle** `data/runs/<run_id>/`:

| file | source |
|---|---|
| `power.csv` | extended `nvidia-smi` @ 4 Hz (`profiling/client/power_logger.py`) |
| `engine.csv` | vLLM `/metrics` @ 4 Hz (`profiling/client/metrics_logger.py`) |
| `requests.json` | per-request latencies + **epoch** timestamps, from the vendored `benchmark_serving.py` via `bench_driver` |
| `manifest.json` | run/probe/arch/server/clock + per-level **epoch windows** (`run_manifest`) |

**Traffic generation:** probes shell out to the repo's vendored `benchmark_serving.py`
(`--max-concurrency N --request-rate inf --ignore-eos --save-detailed`), *not* a
bespoke client — it already emits per-request epoch `request_timestamps`
(`backend_request_func.py`), which is the field that aligns requests to the
nvidia-smi power log. See **`profiling/BUNDLE_SCHEMA.md`** for the full alignment
contract (every field, in every file, and how the three time series join).

`feature-test/build_ledger_bundle.py` turns a bundle into the **same**
`ledger_cache.npz` schema the model already trains on, so `fit_map_priors.py` /
`peak_and_holdout.py` run unchanged. It uses the **reconstruction path**
(ttft/itl → state), which is proven bit-for-bit equivalent to the known-good
`build_ledger_cache.build_run_bins`. `engine.csv` is collected by the scraper but
**consuming it as the state source is deferred to Phase-2** — a first draft of
that consumer produced biased data (token-rate undercount + clock misalignment +
zeroed KV columns), so it's a documented `NotImplementedError` stub
(`bins_from_engine_csv`) to be implemented against real bundles and validated
against reconstruction before it's trusted.

## Layout

```
profiling/client/   arch_extract.py  power_logger.py  metrics_logger.py  run_manifest.py
profiling/probes/   schedule.py (pure)   probe_runner.py  bench_driver.py (live)
                    decode_staircase.py  prefill_staircase.py  context_holds.py
                    transients.py  mixed_grid.py
profiling/jobs/     campaign_config.py   run_campaign.sh   server_lifecycle.sh
profiling/campaigns/  *.json
feature-test/       build_ledger_bundle.py  (new; build_ledger_cache.py untouched)
```

`schedule.py` holds every number that determines the measured operating point and
is fully unit-tested offline; the live layer only executes a schedule.

## Probes (Tier 1)

`idle_hold` · `decode_staircase` (saturation, `e_w_decode`, power cap) ·
`prefill_staircase` (`e_f_prefill`, attention-L²; chunked-prefill **OFF**) ·
`context_holds` (`e_kv`) · `transients` (lag-filter α) · `mixed_grid`
(prefill×decode interaction).

## Model lineup & effort

| model | type | TP | effort |
|---|---|---|---|
| Llama-3.1-70B | dense | 4↔8 | Tier-1 anchor (e_comm pair) |
| Qwen3-235B-A22B | MoE | 8 | Tier-1 anchor |
| MiniMax-M2.7 | MoE + **linear attn** | 8 | partial Tier-1 (attn/KV probes) |
| Gemma-3-27B / 12B | dense + **SWA 5:1** | 1 | Tier-2 calibration |
| Qwen3-8B/14B/32B, Qwen3-30B-A3B | dense/MoE | 1–2 | validate-first, escalate >6% |
| Llama-4-Scout | MoE | 4 | validate-first |

TP policy: lowest TP that fits (small models at TP8 are idle-dominated). See
`profiling/campaigns/*.json`.

> **MiniMax-M2.7 caveat:** lightning/linear attention breaks the softmax
> `e_f_prefill`/`e_kv` assumptions. The infra collects it (probes + an
> `arch_extract` `linear_attention` flag + a provisional recurrent-KV work rate),
> but the actual linear-attention **fit term** is downstream follow-up — until
> then its data is collected, not fit.

## Usage

```bash
# Dry run (default): print the full server+probe plan, write a sample bundle,
# launch nothing.
bash profiling/jobs/run_campaign.sh profiling/campaigns/h100_tier1_llama70b.json

# Execute on GPUs (one server per probe, since probes need different launch flags)
bash profiling/jobs/run_campaign.sh profiling/campaigns/h100_tier1_llama70b.json --execute

# Agentic multi-turn workload (run with prefix-cache on AND off); emits the same bundle
uv run python -m profiling.probes.agentic_run --model Qwen/Qwen3-8B --hardware H100 --tp 1 \
    --n-sessions 8 --gap-mean-s 3.0 --prefix-cache --enable-prefix-caching

# Build the training ledger from collected bundles
uv run python feature-test/build_ledger_bundle.py --runs-glob 'data/runs/*'
```

> **Live-execution wiring note:** `run_campaign.sh --execute` currently auto-drives
> *probe* campaigns. `validate` and `agentic` campaigns render their plan in dry-run
> and are launched via their own entrypoints (the `benchmark_serving`/`agentic_run`
> commands shown in the plan); wiring them into `--execute` is a small follow-up.

## Tests (acceptance gate)

```bash
uv run -m pytest -x profiling feature-test
```

- `feature-test/tests/test_build_ledger_bundle.py::test_phase1_matches_old_builder`
  — **load-bearing**: the new reconstruction path reproduces
  `build_ledger_cache.build_run_bins` bit-for-bit (rtol 1e-9) on old-format data.
- `profiling/client/tests/test_arch_extract.py` — reproduces the curated arch dict
  for the existing 6 models before the extractor is trusted on new ones.
- `profiling/probes/tests/test_schedule.py`, `profiling/jobs/tests/test_campaign_config.py`,
  loggers/manifest tests.

Phase-2 (implement + validate the measured `engine.csv` state consumer against
reconstruction) runs post-launch once real bundles exist.
