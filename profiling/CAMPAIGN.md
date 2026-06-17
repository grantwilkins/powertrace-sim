# PowerTrace Profiling Campaign

How to gather the cleanest data for the first-principles power model in
`feature-test/`, in the format that makes training trivial.

---

## 1. Principle: measure states, not workloads

The model is `P_node(t) = standing(hardware) + Σ energy_coeff · work_rate(t)`, where
each `work_rate` (FLOPs/s, bytes/s, bandwidth utilization) is **computed** from the
per-bin engine state and the model's architecture. So training data quality is
about **coverage of the per-GPU operating range**, not workload realism or volume.

Two empirical findings from `feature-test/` drive every choice below:

1. **Sample count barely matters.** 1 run/operating-point ≈ 3 runs; the first 30 s
   of a run ≈ the full 10 min. Repetition and long runs re-measure states we've
   already seen. → **short holds at distinct states**, not long Poisson sweeps.
2. **Coverage of the per-GPU range is everything.** A model that never pushes the
   GPU past ~40 % bandwidth util leaves the saturation curve and power ceiling
   unmeasured; held-out error jumps 5 % → 15 %. → **stress idle → saturation →
   power-cap, once per GPU type.**

Corollary: the expensive characterization is **per hardware platform**, done once
(~2–3 GPU-h). A new *model* on known hardware is nearly free (config arithmetic +
one short calibration). The current grid (every model × every TP × 6 rates × 5
repeats ≈ 8–10 GPU-h/config) spends almost its entire budget re-measuring known
states. This campaign replaces it.

Second principle: **measure engine state, don't reconstruct it.** Today we infer
batch size / phase from client-side `ttft`+`itl`. vLLM already exposes the true
state at `/metrics` (`num_requests_running`, `gpu_cache_usage_perc`, token
counters). Scraping it turns every bin into a *labeled* `(state → power)` sample
and removes the most error-prone step in the pipeline.

---

## 2. The data contract (what every run emits)

One self-describing bundle per run: `data/runs/<run_id>/`

| file | source | contents (per row unless noted) |
|---|---|---|
| `power.csv` | `nvidia-smi` @ 4 Hz | timestamp, **per-GPU**: `power.draw`, `clocks.sm`, `clocks.mem`, `utilization.gpu`, `utilization.memory`, `memory.used`, `temperature.gpu` |
| `engine.csv` | vLLM `/metrics` @ 4 Hz | timestamp, `num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc`, `prompt_tokens_total`, `generation_tokens_total`, `iteration_tokens_total_{sum,count}`, `request_{prefill,decode}_time_seconds_sum` |
| `requests.json` | `benchmark_serving.py` | `input_lens`, `output_lens`, `ttfts`, `itls`, `request_timestamps` + aggregates (unchanged) |
| `manifest.json` | new emitter | everything below |

`manifest.json` is what kills the fragile timestamp-pairing in `stage0`:

```jsonc
{
  "run_id": "h100_decode_staircase_tp8_n16_20260615-2210",
  "probe": {"type": "decode_staircase", "level": 16, "params": {...}},
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "arch": {"n_active": 7.05e10, "weight_bytes": 1.41e11, "n_layers": 80,
           "n_kv_heads": 8, "head_dim": 128, "d_model": 8192, "dtype": "bf16",
           "n_experts": 1, "top_k": 1, "sliding_window": 0},   // from config.json
  "hardware": "H100", "tp": 8, "gpus_per_node": 8,
  "server": {"max_num_seqs": 256, "max_num_batched_tokens": 8192,
             "enable_chunked_prefill": true, "enable_prefix_caching": false,
             "kv_cache_dtype": "auto", "max_model_len": 131072},
  "versions": {"vllm": "0.x.y", "git_sha": "…", "gpu_driver": "…"},
  "clock": {"power_epoch_offset_s": 0.0, "engine_epoch_offset_s": 0.0,
            "monotonic_start": 12345.6}
}
```

The ledger builder then reads one directory → aligned `(power, state, work_rate)`
rows. No cross-file inference, no guessing.

---

## 3. The probe campaign

### Tier 0 — Instrumentation (build once)

- High-frequency `/metrics` scraper aligned to the power logger (§5-A).
- Extended `nvidia-smi` field set incl. `clocks.sm` (DVFS is the largest
  unmodeled term; it is a free field).
- Manifest emitter + standardized bundle layout.

### Tier 1 — Hardware characterization (per GPU type, ~once, 2–3 GPU-h)

Closed-loop probes that **pin the engine at chosen state-space points**. All use
the `random` dataset with `--ignore-eos` and a fixed `--max-concurrency` so the
state is controlled, not emergent. Each level held ~45 s (≈180 power samples).

| probe | how | identifies |
|---|---|---|
| **idle hold** | no traffic, 60 s | `p_idle` |
| **decode staircase** | concurrency `N ∈ {1,2,4,…,max_num_seqs}`, prompt=8 tok, output=2048, `ignore_eos` | decode memory power + saturation curve (`a1,a2,a3`), `e_w_decode`, **power cap** |
| **prefill staircase** | input_len `∈ {256,1k,4k,16k,64k}`, `max_tokens=1`, **chunked-prefill OFF**, concurrency 1 | `e_f_prefill`, attention-L² term, TTFT-vs-length curve |
| **context-decode holds** | long primed prefix, then steady decode at fixed batch, context `∈ {2k,8k,32k,128k}` | `e_kv` (KV-read energy) — the term short prompts can't identify |
| **mixed grid** (Latin hypercube ~16 pts) | decode floor of `N` seqs + controlled prefill injection | prefill×decode interaction |
| **transient steps** | idle→batch `N`→idle, sharp | lag-filter constant `α` (rise/fall) |
| **TP pair** | repeat decode+prefill staircase at a 2nd TP degree | `e_comm` (NVLink) |

This single sweep covers idle → bandwidth-saturation → compute-saturation →
power-cap, which §1 shows is the whole requirement. Run it on **one anchor model**
(70B-class is ideal: spans a wide per-GPU range) plus the TP-pair repeat.

### Tier 2 — Per-model calibration (per new model, ~30–45 min)

Condensed decode + prefill staircase at the **deployment TP only**. Fits the
single per-model efficiency multiplier and the TTFT/ITL curves for the scheduler
twin. Hardware constants are reused; other TPs are predicted (validated to ~6 %
for MoE 20B→120B and ~6 % for dense 8B/70B→405B in `feature-test/`). For a model
whose per-GPU range stays inside the anchor's, even this is optional.

### Tier 3 — Validation workloads (held out, never used for fitting)

These test the scheduler twin end-to-end; they are the realism the model is
graded against, not trained on.

- **Chat**: ShareGPT (existing pipeline, unchanged).
- **Long-context** (new, §5-C): inputs 16k–128k, realistic outputs.
- **Agentic** (new, §5-D): multi-turn sessions, growing context, tool-gap idle,
  prefix-cache on **and** off.

---

## 4. vLLM knobs to control and record

The model assumes a known scheduler; these flags change the work→power mapping and
must be **fixed within a campaign and recorded in every manifest**:

- `--max-num-seqs` — sets the decode-batch ceiling = the saturation knee. Pick the
  production value; it bounds the decode staircase.
- `--max-num-batched-tokens` + `--enable-chunked-prefill` — controls how prefill
  interleaves with decode. **Disable chunked prefill for the prefill staircase** so
  prefill occupies clean, multi-bin, pure-prefill dwells (the bins that identify
  `e_f_prefill`). Keep it at production setting for Tier 3.
- `--enable-prefix-caching` — a cache hit skips prefill compute entirely, which
  roughly halves agentic prefill power. **Off for Tier 1 probes** (we want raw
  prefill cost); **both on and off for agentic Tier 3** (both regimes occur in
  production).
- `--kv-cache-dtype`, `--max-model-len` — set `max-model-len` high enough for the
  128k context holds (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`); record KV dtype (it
  scales `e_kv`).

---

## 5. Extensions to build (scoped)

Effort tags: **S** ≤ ½ day, **M** ~1–2 days, **L** ~3–5 days.

### A. `/metrics` scraper sidecar — `profiling/client/metrics_logger.py` (**S**)
Poll `http://host:port/metrics` at 4 Hz, write `engine.csv`. Record a one-time
clock offset against the power logger (read both clocks at start). Reuse the
Prometheus parsing already in `client_async.py`. **Highest value-per-effort item.**

### B. Probe drivers — `profiling/probes/` (**M**)
Closed-loop drivers over the OpenAI-compatible endpoint, each emitting the §2
bundle. Reuse `RandomDataset` + `--ignore-eos`; the new capability is *holding a
fixed concurrency for a fixed wall-time per level*:
- `decode_staircase.py`, `prefill_staircase.py`, `context_holds.py`,
  `transients.py`, `mixed_grid.py`
- shared `probe_runner.py`: launches server with a given config, waits healthy,
  starts both loggers, drives the level schedule, writes manifests.

### C. Long-context support (**S**, mostly config)
`RandomDataset` already takes `--random-input-len`; the gap is operational:
server launched with large `--max-model-len`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
and a `--random-prefix-len` sweep for the context-decode holds. Add a
`long_context.py` probe wrapping these. No core dataset change needed.

### D. Agentic session client — `profiling/probes/{agentic,session_driver,session_runner,agentic_run}.py` (**L**, scaffolding built)
The one genuinely new workload generator. A *session* = a sequence of turns where
each turn's prompt = full prior context + new user/tool message, so context grows
monotonically; between turns, a **tool-execution gap** (idle, lognormal duration).
Parameters: turns/session, context-growth/turn, gap distribution, optional
parallel tool fan-out, prefix-cache on/off. Two modes: (i) **synthetic**
(parameterized), (ii) **replay** (feed real agent transcripts: ordered
`(prompt, output_len, post_gap_s)` triples). Emits the same bundle, so the ledger
and model consume it unchanged — agentic is just a schedule with idle gaps and
large recurring prefills.

**Status:** the live multi-turn sender (`session_driver.send_session`), plan
generators (`agentic.build_synthetic_sessions` / `from_transcript`), bundle emitter
(`session_runner`), CLI (`agentic_run`), and campaign wiring (`jobs/campaign_config.py`
`agentic` type, `run_campaign.sh`) are built and unit-tested. What remains to make
agentic numbers *admissible and trustworthy* is scoped in **§9** — the synthetic
`_filler` content path does not satisfy the real-data constraint, prefix-caching is
unmodeled in both ledger and twin, and the grading loop is not closed.

### E. Run manifest + bundling — `profiling/client/run_manifest.py` (**S**)
Emit `manifest.json`; pull arch fields from the served model's `config.json`
automatically. Replaces timestamp-pairing in `stage0`.

### F. Orchestrator refresh — `profiling/jobs/` (**S**)
Replace the rate×TP×iteration bash loops with a thin runner over a campaign YAML
(`campaigns/h100_tier1.yaml`, `tier2_<model>.yaml`, `tier3_validation.yaml`) that
lists probes and server configs. Keeps the robust server-lifecycle handling that
already exists in the current job scripts.

---

## 6. Output → training handoff

The ledger builder in `feature-test/build_ledger_cache.py` is updated to read the
§2 bundle directly:
- **state** comes from `engine.csv` (measured) instead of being reconstructed from
  `ttft`/`itl` — `requests.json` becomes validation/cross-check, not the source.
- **work rates** are computed from measured state + `manifest.arch`.
- **power** is aligned via `manifest.clock` offsets (no timestamp folding).

Output: the same `(power, work_rates, run_id, arch)` per-bin table the current
model trains on — so `fit_map_priors.py` / `peak_and_holdout.py` run unchanged. The
probe `type` and `level` ride along in the manifest for slicing (e.g. fit the
power cap only on saturation probes, `e_kv` only on context holds).

---

## 7. Budget

| | scope | GPU time |
|---|---|---|
| Tier 1 | once per GPU type (A100, H100, …) | ~2–3 h |
| Tier 2 | once per new model | ~30–45 min |
| Tier 3 | per workload class (chat / long-ctx / agentic), held out | ~30 min each |

A new model on already-characterized hardware: **~30 min** (Tier 2 only), vs the
current **8–10 h**. Long-context and agentic coverage — absent today — are added
as ~30-min validation passes plus the one-time §5-C/§5-D builds.

---

## 8. Build order

1. **A + E** (metrics scraper + manifest) — retrofit onto the *existing* job
   scripts immediately; every future run is then richer and self-describing.
2. **B** (probe drivers) + ledger-builder update (§6) — unlocks Tier 1; re-derive
   the hardware constants from clean probes and compare to the current
   `feature-test/` fit.
3. **C** (long-context) — closes the `e_kv` / attention-L² identifiability gap.
4. **D** (agentic client) — the new frontier workload; validate the scheduler twin
   on growing-context, gap-driven traffic with prefix caching on and off. The
   generator scaffolding is built; **§9** is the remaining readiness plan (real
   traces, measured-state ledger, prefix-cache modeling, grading).

---

## 9. Agentic readiness plan

The model is validated on a single A100 setup via the normal (probe + `validate`)
path. Agentic Tier-3 (§3, §5-D) is the next workload class. The session generator,
live sender, bundle emitter, and campaign wiring already exist and are unit-tested;
GPU cost is small (A100 is characterized — this is Tier-3 validation, ~30 min/run).
The remaining work is **almost entirely software**: making the data admissible, the
ledger honest, and the grading loop closed. Five gaps, in priority order.

### What's built vs. what's missing

Built: `agentic.py` (synthetic + `from_transcript` plans), `session_driver.py`
(live multi-turn sender — context grows by appending real replies, lognormal
tool-gap idle, rich `requests.json` superset), `session_runner.py` + `agentic_run.py`
(emit the §2 bundle, record the `prefix_cache` regime), `jobs/campaign_config.py`
`agentic` type + `run_campaign.sh`, `campaigns/agentic_qwen3-8b.json`.

Missing: real content, measured-state consumption, both cache regimes on the target
hardware, prefix-cache modeling in the twin, and an agentic grade step.

### Twin verification (done — informs Gaps 2 & 4)

The twin's request path (`model/pipeline/request_builder.py` → `inference.py`)
consumes **only** `input_lens`, `output_lens`, `request_timestamps`; it ignores the
agentic extensions (`session_ids`, `turn_idx`, `post_gap_s`, `prefix_cache`).
Consequence:

- **Idle tool-gaps** — handled implicitly: a gap is just an interval with no new
  arrival, so idle falls out for free. Low risk.
- **Growing context** — handled: each turn's `input_len` carries the full grown
  prompt, so per-turn KV/prefill scale is captured. Low risk.
- **Prefix-cache hits** — **not modeled**: the twin sees the full `input_len` and
  simulates full prefill on every turn even when the engine skipped it. So
  **prefix-cache-*on* agentic runs are systematically over-predicted on prefill
  power.** Same root cause as Gap 2 (a cache hit means prefill work didn't happen,
  but `input_lens` still says it did). Prefix-caching is the *single* unmodeled
  dynamic, on both the ledger (training) and inference (grading) sides.

### The gaps

| # | item | effort | blocking? |
|---|---|---|---|
| 1 | **Real agent-trace loader** (SWE-agent / tool-use trajectories) → ordered turns with **real message strings** + observed `post_gap_s`; thread real text through `session_driver.send_session` (replace `_filler` on the replay path); extend `from_transcript` to carry text, not just `(in_tok, out_tok, gap)`. | M | Yes — satisfies the real-data constraint |
| 2 | **`bins_from_engine_csv`** (currently `NotImplementedError` in `feature-test/build_ledger_bundle.py`) so the ledger uses *measured* state from `engine.csv` (already collected) instead of reconstructing it; route agentic bundles through it so cache-skipped prefill is not fabricated. | M | Yes — for cache-on data |
| 3 | **A100 agentic configs**, paired `prefix_cache` true/false; confirm `run_campaign.sh` relaunches the server with matching `--enable-prefix-caching` per regime. (Existing `agentic_qwen3-8b.json` is H100, cache-on only.) | S | Yes |
| 4 | **Twin prefix-cache modeling** — *deferred.* Verified needed for cache-*on* grading (see above). Until built, grade **cache-off only** and treat cache-on as a known over-prediction. | (verified, deferred) | — |
| 5 | **Agentic grade harness** — agentic bundle → twin → predicted-vs-measured power, sliced by regime, against the ~6 % held-out bar (cf. `validate_runner` for ShareGPT). | S | No |

### Decisions taken

- **Data source: real agent traces (SWE-agent / tool-use).** Synthetic `_filler`
  remains only as a smoke-test. The replay path must send real text, not
  token-count-matched filler. **Open input:** which corpus and whether it carries
  real inter-turn gap timestamps (no agent-transcript dataset is staged in-repo
  today — only the WildChat+OpenCodeInstruct mix from `mix_dataset.py`, used by
  `validate`). The choice sets the parser shape and whether `post_gap_s` is observed
  or must be synthesized.
- **Twin (Gap 4): verify-only for now.** Verification is done (above); modeling is a
  flagged follow-up, not part of this pass.

### Recommended order

1. **Gaps 1 + 3** — real-trace data flowing on already-characterized A100, both
   prefix-cache regimes (~30 min/run × 2).
2. **Gap 2** — measured-state ledger, so cache-on runs are honest.
3. **Gap 5** — close the grading loop (cache-off is fully gradable without Gap 4).
4. **Gap 4** — twin prefix-cache modeling unblocks cache-*on* grading; deferred.
