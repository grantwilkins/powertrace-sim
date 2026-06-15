# feature-test: first-principles LLM inference power model

Goal: using only existing profiling data, drive error metrics as low as possible
for a **first-principles, explainable** node-power model that works **across
tensor-parallel degrees** and **splits energy over prefill and decode**.

## Final model

Per hardware platform (A100, H100), node power is a non-negative sum of
physically interpretable terms, passed through a causal lag filter, with one
efficiency multiplier per model family:

```
P_node(t) = p_idle·TP + p_link·TP·1[TP>1]                      # standing power
          + p_active·TP·1[busy] + Σ p_sat·TP·(1−e^(−util/u0))  # DVFS / clocks
          + e_f_pre·FLOPs_pre/s + e_f_dec·FLOPs_dec/s          # compute
          + e_w_pre·Wbytes_pre/s + e_w_dec·Wbytes_dec/s        # weight traffic
          + e_kv_r·KVread/s + e_kv_w·KVwrite/s + e_comm·NVLink/s
P_meas(t) = Lag(P_node)(t)        # A100: EMA(0.6); H100: MA(2)+EMA(0.7)
P_final   = idle + m_family · dynamic     # m ∈ [0.93, 1.11], default 1.0
```

All work rates are **computed, not learned**, from per-request timing
(arrival, TTFT, decode time) + architecture descriptors (params, dtype,
layers, KV geometry, MoE experts/top-k). ~10 fitted constants per hardware.

## Results (450 runs, 273k one-second bins, 7 models, TP 1–8)

| metric | A100 | H100 |
|---|---|---|
| CV R² (5-fold by run, 1 s bins) | **0.961** | **0.961** |
| CV RMSE | 121 W | 206 W |
| R² @ 5 s windows | 0.983 | 0.981 |
| R² @ 30 s windows | **0.991** | **0.991** |
| MAPE @ 30 s windows | 5.4% | 6.1% |
| LOMO run-mean err (median, mult=1) | 3.9–13.5% | 6.6–11.3% |
| LOTO run-mean err (median) | 3.4–10.8% | 2.6–16.9% |

Baseline (naive 4-term linear roofline, M0): R² 0.937/0.917, RMSE 154/300 W.
Final model cuts RMSE ~22%/31% and halves 30 s-window MAPE.

Fitted constants are physical: p_idle ≈ 40 W/GPU on both platforms (true idle);
NVLink standing power 19 W (A100) / 58 W (H100) per GPU when sharded;
e_w ≈ 0.15 nJ/B on both (HBM-class); e_f_dec ≈ 1.1–1.5 pJ/FLOP.
Family multipliers: dense-8b 0.93–0.97, dense-70b 0.97–1.00, moe 1.06–1.11,
405b-fp8 1.05 — deviations from 1 localize kernel-efficiency differences.

## Phase attribution (from `attribution_*.csv`)

Energy splits per family/TP via feature-zeroing: e.g. llama-70b H100 TP8 =
33% idle + 2.6% prefill + 58% decode + 6% shared DVFS. Prefill is 1–4% for
ShareGPT (short prompts) — the machinery matters for agentic workloads where
prefill share grows with context length.

## What moved the needle (model ladder, see `results/model_ladder_metrics.csv`)

1. **M0→M2 phase split + comm**: small gains; exposed that TTFT-based prefill
   timing was wrong (TTFT includes queueing → prefill burst placed at *end*
   of TTFT window with duration n_in/λ_prefill, from throughput DB).
2. **M3 active floor** (+0.3 R² pt A100): GPUs burn ~20–35 W/GPU just being
   busy (clock boost above idle).
3. **M5/M7 DVFS saturating bases** (+0.2 pt): concave power-vs-utilization.
4. **Lag filter** (+1.4–3.4 pts, biggest single win): nvidia-smi power is a
   moving average; matching it matters at 1 s bins.
5. **M8 NVLink standing term**: separated idle/link/active cleanly
   (p_idle 80–126 W conflation → 40 W idle + link + floor).
6. **Per-family multipliers** (+0.2 pt, removes family bias): the gray-box
   "shrinkage" layer; all within ±11% of 1.
7. MoE fix: prefill chunks touch **all** experts (batch-based expert-touch
   only applies to decode).

## MAP fit with physical priors (`fit_map_priors.py`)

Alternative estimator: coefficients `c = exp(θ)` with log-normal priors centered
on datasheet physics, Student-t (ν=4) heteroscedastic likelihood
(σ_i = a + b·P̂), per-family multipliers with tight priors toward 1. Laplace
approximation yields per-coefficient posterior sd → each constant is labeled
DATA-IDENTIFIED / partial / PRIOR-DOMINATED, plus a *drift* diagnostic
(MAP-vs-prior distance in prior σ units).

Results: H100 CV R² 0.964 / MAPE 14.2% (vs 0.961 / 16.3% NNLS); A100 R² 0.959 /
MAPE 12.5% (vs 0.961 / 13.8%) — the t-likelihood trades a little squared error
for better relative error. The bigger value is diagnostic:

- `e_f_prefill`: PRIOR-DOMINATED with drift −9…−10σ — the data *actively*
  pushes prefill toward 0 because its energy is absorbed by collinear
  activity terms. Unresolvable from ShareGPT; needs the prefill staircase.
- `e_kv` (+7…+11σ) and `e_comm` (+8…+10σ): "identified" at implausible values —
  they are proxies for a missing per-token/batch power term, i.e. a
  model-misspecification flag, not a measurement of HBM/NVLink energy.
- Standing/decode terms (p_idle, p_link, e_w_decode, e_f_decode on A100):
  identified at physical values with low drift — trustworthy.

Every flagged coefficient maps to one probe in the redesigned profiling
pipeline (prefill staircase → e_f_pre/e_w_pre; long-context decode → e_kv;
fixed-workload TP pair → e_comm). See `results/map_identifiability_*.csv`.

## Peak fidelity & zero-shot generalization (`peak_and_holdout.py`)

Two coupled findings. (1) The missed power peaks are prefill bursts on a
saturated node: top-decile power bins carry 33–38% more prefill tok/s, and
H100 saturated runs reach P99 = 4.3 kW vs the ~3.0 kW decode plateau.
(2) Driving the DVFS terms with decode-only utilization and replacing the
{linear weight-bytes + decode-indicator} pair with a **saturating function of
DRAM bandwidth utilization** (roofline-consistent: linear at 8B/70B's low BW
utilization, saturating at 405B's high utilization) fixes both peaks and
extrapolation:

| | peaks (H100 sat. P90/P99) | 405B zero-shot (run-mean med.) | in-sample R² |
|---|---|---|---|
| MAP M8 (additive) | 2.9 kW cap vs 4.0/4.3 | ~10% | 0.964 |
| decode-DVFS + linear bytes | 3.8/4.0 kW | 28.8% (double-counts) | 0.967 |
| + saturating BW power | 4.0/4.5 kW | unstable (see below) | 0.961 |
| **final (see below)** | 4.0/4.5 kW (A100), over on H100 P99 | **9.6%** (worst 12.9%), dynamics intact | 0.956–0.957 |

Four corrections found by auditing a suspicious-looking 405B holdout:
(1) **warm-start leakage** — initializing the holdout fit from full-data
coefficients leaked held-out information through optimizer path dependence
(reported 3.5%; honest cold start gave +42/+81% at rates 2/4); holdout fits
must be cold-started. (2) **power cap** — uncapped extrapolation predicted
7 kW on a ~4 kW node; a sustained per-GPU cap (508 W/GPU = p99.5 of busy
per-GPU power, training families only) is a real device constraint. But a cap
that *binds everywhere* flattens all dynamics — it must be a guardrail, which
required (3) **fixing the inflation it was hiding**: e_f_decode ran to
7 pJ/FLOP in-sample (absorbed by family multipliers ≈0.85; held-out models
get 1.0 and detonate on 10× FLOPs). Fix: tight prior on decode compute energy
(memory-bound GEMV ⇒ small marginal compute), multipliers pinned to ±5%, and
(4) **FP8 FLOPs cost ~half BF16 energy** — encoded as a constant, not fitted.
Final zero-shot on never-seen 405B-FP8: median 9.6%, worst 12.9%, bin R²
0.77, full trace dynamics preserved (`results/holdout_405b.png`). The 405B
raw data itself was audited clean: 0 corrupted CSV blocks, balanced per-GPU
power, idle floor ~1.1 kW matching every other H100 config.

## MoE scalability: fit 20B, predict 120B (`scalability_moe.py`)

Hardest within-family test: fit on **gpt-oss-20B only** (TP1/TP2, A100), cold
start, then predict gpt-oss-120B (TP4/TP8) — extrapolating size (20B->120B),
expert count (32->128), and GPU count simultaneously. Only the architecture
descriptor (active params, weight bytes, experts, layers) changes; the A100
hardware constants are reused.

| | run-mean err (median) | bin R² |
|---|---|---|
| gpt-oss-20B (fit family) | 1.7% | 0.93 |
| gpt-oss-120B (**zero-shot**) | **5.5%** | 0.90 |

TP8 predictions are essentially exact (0.9–4.6% across all rates); TP4 mid-rates
overpredict ~18% (one TP setting under-resolved by 20B's TP1/2-only data). The
zero-shot 120B trace keeps full dynamics (`results/scalability_moe.png`). This
shows the MoE expert-traffic term (`weights_read = Sw · iters · min(1, batch·Ktop/Nexp)`)
transfers across a 4.75× weight-size jump — the size dependence is genuinely
carried by the architecture arithmetic, not memorized per model.

## Known limits / residual error

- Intrinsic per-bin power noise is ~30–44 W (std); remaining bin-level RMSE is
  systematic temporal mismatch: client-side timing jitter, preemption, and
  asymmetric power dynamics. This is the stochastic layer's job (existing
  GMM/AR machinery), not the mean model's.
- llama-3-8b A100 JSONs lack `request_timestamps` (excluded; arch covered by
  the deepseek distill).
- H100 LOTO tp1 (16.9%) extrapolates from only 8B-class runs at TP1.
- e_kv is small and collinear with batch at ShareGPT context lengths; don't
  quote it as HBM energy until long-context probes exist.

## Files

- `build_ledger_cache.py` — parse runs → per-second work ledger (`ledger_cache.npz`)
- `fit_models.py` — model ladder M0–M8 + lag variants, CV/LOMO/LOTO harness
- `diagnose_residuals.py` — residual breakdown + noise-floor estimate
- `final_model.py` — final fit, metrics, attribution, coefficient export
- `results/` — metrics CSVs, `final_coefficients.json`, attribution tables, figures
