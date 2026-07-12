# PowerTrace Profiling Campaign

> Execution lives in `profiling/MODEL_READINESS_RUNBOOK.md`. This document is
> background and design rationale; do not use its generic tiers or historical
> extension list as a launch checklist.

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

Second principle: **claim only what the maintained data path measures.** Stock
vLLM exposes total running/waiting requests, KV-cache occupancy, executed
prompt/generation token counters, and the iteration-token histogram. It does not
expose phase-specific batch, collective duration, NVLink traffic, router choices,
or expert touches. The measured ledger is therefore intentionally hybrid:
stock counters replace exactly measured token and total-active fields; request
TTFT/ITL reconstruction remains the source for phase batch and KV geometry.
TP synchronization and MoE effects are identified by matched probes, not by
invented runtime columns.

---

## 2. The data contract (what every run emits)

One self-describing bundle per campaign run:
`data/runs/<campaign_id>/<run_id>/`

| file | source | contents (per row unless noted) |
|---|---|---|
| `power.csv` | `nvidia-smi` @ 4 Hz | timestamp, **per-GPU**: `index`, `uuid`, `power.draw`, `clocks.sm`, `clocks.mem`, `utilization.gpu`, `utilization.memory`, `memory.used`, `temperature.gpu` |
| `engine.csv` | stock vLLM `/metrics` @ 4 Hz | required: timestamp, `num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc`, `prompt_tokens_total`, `generation_tokens_total`, `iteration_tokens_total_{sum,count}`; optional version-dependent counters are retained but not assumed |
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
             "kv_cache_dtype": "auto", "max_model_len": 131072,
             "active_gpu_uuids": ["GPU-...", "GPU-..."]},
  "versions": {"vllm": "0.x.y", "git_sha": "…", "gpu_driver": "…"},
  "clock": {"local_utc_offset_s": -25200.0,
            "power_timestamp_basis": "local_wall_time",
            "engine_timestamp_basis": "unix_epoch", "monotonic_start": 12345.6}
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

`feature-test/build_ledger_bundle.py --state-source measured_engine` is the
maintained handoff. It passes each bundle through `RunRecord`, reconstructs the
uniform power grid once, then maps fields as follows:

| ledger field | source | consumed by selected physics path? |
|---|---|---|
| `power` | topology-validated `power.csv` | target |
| `pre_tok`, `dec_tok` | edge differences of stock vLLM token counters | yes |
| `A_t`, running, waiting | time means of stock vLLM total-request gauges | yes |
| `batch`, `pre_active`, prefill iterations, `kv_read` | request TTFT/ITL + architecture reconstruction | yes |
| iteration rate, tokens/iteration, GPU-cache usage | stock vLLM diagnostics persisted in the measured cache | available for the next candidate; not silently consumed by the frozen artifact |
| collective duration, NVLink bytes, expert touches | unavailable in stock vLLM | no; infer axes only from matched probe differences |

Every live run preflights `/metrics`, requires complete finite coverage of every
required column, validates counter monotonicity,
cadence, GPU identity/topology, and power/engine epoch alignment, and records the
observed evidence contract in `manifest.instrumentation`. Missing required stock
fields fail the run rather than falling back to reconstruction.

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
2. **B** (probe drivers) + measured-ledger path (§6) — unlocks Tier 1; re-derive
   the hardware constants from clean probes and compare to the current
   `feature-test/` fit.
3. **C** (long-context) — closes the `e_kv` / attention-L² identifiability gap.
4. **D** (agentic client) — the new frontier workload; validate the scheduler twin
   on growing-context, gap-driven traffic with prefix caching on and off. The
   generator scaffolding is built; **§9** is the remaining readiness plan (real
   traces, measured-state ledger, prefix-cache modeling, grading).

---

## 9. Model-readiness campaign generated from the v2 failures

The executable configs deliberately cover only cells that resolve a named
ambiguity in `FEATURE_TEST_LEARNINGS.md`:

| order | configs | comparison | question answered |
|---|---|---|---|
| 1 | `a100_tier1_llama70b.json` | controlled operating-point sweep | establishes the A100 hardware response |
| 2 | `a100_iteration_gpt-oss-20b.json` | 20B TP2↔TP4 at identical batch × context points | isolates TP on one model |
| 3 | `a100_iteration_gpt-oss-{20b,120b}.json` TP4 legs | 20B↔120B at identical TP, batch, and context | isolates model scale without changing TP |
| 4 | `h100_tier1_llama70b.json` | dense TP4↔TP8 decode, prefill, and context grid | establishes the H100 response surface and TP contrast on the same model |
| 5 | `h100_hardcells_llama{70b,405b}.json` | matched ShareGPT marks/rates, fixed seed | localizes the 405B TP8 bias using the exact stock evidence available in serving |
| 6 | `a100_hardcells_gpt-oss-{20b,120b}.json` | matched TP4 ShareGPT marks/rates, fixed seed | checks model scale under realistic scheduling without a TP confound |
| 7 | `h100_sealed_llama405b.json`, `a100_sealed_gpt-oss-120b.json` | fresh rates and seeds on restricted storage | final score only after equations and artifacts freeze |

`decode_context_grid` is a Cartesian grid over batch `{1,4,16}` and context
`{2k,8k,32k}`. The orthogonality is load-bearing: it separates iteration
granularity from context/KV traffic instead of replacing one confound with
another. Validation campaigns support multiple declared rates and explicit
seeds. Sealed execution requires `SEALED_RUNS`; development output cannot
accidentally land in the sealed location.

Do not add custom collective/router counters to this campaign unless a concrete
vLLM patch is implemented, versioned, tested, and its column is traced through
`RunRecord` and the ledger cache first. Until then, TP and MoE labels describe
experimental contrasts, never directly observed per-iteration mechanisms.

Agentic remains a later Tier-3 workload. Prefix-cache-on grading is not ready:
the arrival-only twin still treats the full prompt as executed prefill. This does
not block the model-readiness campaign above, which is cache-off throughout.
