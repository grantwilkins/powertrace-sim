# new-profiling-model: a measured-state power model for the gemma-4 data

Goal: from the new gemma-4 profiling runs (`profiling/CAMPAIGN.md`), build the
**most accurate, explainable, and transferable** node-power model possible —
lowest energy error and highest R² — without reward-hacking. This folder is the
experiment; the headline is that the `feature-test/` pipeline gave bad results
on the gemma data for a diagnosable reason, and fixing it from first principles
restores ≤2% held-out energy error.

## Data

Profiled on A100 (ramr, tp=2), bf16, vLLM 0.19.1.dev:

| model | type | probes |
|---|---|---|
| `gemma-4-31B-it` | dense (30 B active) | decode staircase, validate |
| `gemma-4-26B-A4B-it` | MoE (128 experts, top-8, 4 B active) | decode + prefill staircase, validate |

Bundles live in `$SCRATCH/ptsim/runs/`. Each is the §2 contract: `power.csv`
(nvidia-smi @4 Hz, per-GPU), `engine.csv` (vLLM `/metrics` @4 Hz), `requests.json`,
`manifest.json`, and `levels/` for the staircases.

## Why `feature-test/` gave bad results

`feature-test/build_ledger_bundle.py` runs **every** bundle through the
`ttft`/`itl` **reconstruction** path — the code path built for ShareGPT-style
validate runs. Two consequences on the gemma data:

1. **It throws away the measured state.** The staircases are closed-loop
   probes: the engine is pinned at a known concurrency for ~100–700 s and vLLM
   `/metrics` records the *true* state the whole time. Reconstructing that state
   from client-side request timing is unnecessary and biased. In particular the
   reconstruction never sees the real **saturation knee** — the engine sustains
   only ~82 sequences when 128 are requested and ~105 when 256 are — which is
   the single most informative point on the decode curve.
2. **`dec_iter = dec_tok / batch` blows up.** When an interpolated batch is
   near zero but decode tokens are flowing (a binning artifact), the weight-read
   rate explodes; on the gemma-31B validate run this produced predictions of
   >1 MW and R² of −8×10⁷.

There was also a deeper modelling problem inherited from the old fit: **no idle
anchor.** The lowest staircase point (N=1) is already busy, so the standing
(idle) term is unidentified and the fit over-loads it — fine for the
high-power dense model, catastrophic for the low-power MoE at small batch.

## The fix (this folder), from first principles

`dataio.py` reads each probe **level as one steady-state `(measured state →
measured power)` sample** and each validate run as per-second **measured-state**
bins. Three changes, each physically motivated:

- **Measured iteration rate.** `iters/s = d(iteration_tokens_total_count)/dt`
  (a logged-but-unused counter). Every forward pass streams the weights once, so
  weight traffic = `w_eff · iters/s` — fully measured, bounded, no division by
  batch. KV reads use `dec_tok/s · ctx_eff · kv_tok` (analytic; `gpu_cache_usage`
  is all-NaN in these runs).
- **True idle anchor.** Mean node power while `num_requests_running ≈ 0` —
  a clean **~185 W node (≈93 W/GPU)**, consistent across both models. This pins
  standing power and is what makes the model transfer between dense and MoE.
- **SWA-aware KV + MoE expert-touch** carried by arch arithmetic, so the same
  hardware constants describe both architectures (only the descriptor changes).

`physics.py` is the explainable feature set (the same terms as `feature-test`),
fit non-negative (`nnls.py`, pure numpy — scipy/sklearn won't build on the
cluster's python module). A key empirical finding: on these A100s the **SM clock
is pinned at 1410 MHz** (max boost) across the entire staircase and `util.gpu`
is saturated at ~100% — so DVFS terms carry no signal. The physics is pure
roofline-energy at fixed clock; the power curve is the **memory system filling
up** (weights at low batch, KV reads at high batch).

## Results

Fit on the staircase probes + idle anchors (27 points), graded on the **held-out
validate runs** (realistic ShareGPT-like workload, never used for fitting).
Energy is integrated power, so the metric that matters is windowed fidelity
(1 s power carries ~30–44 W meter noise that averages out):

| recommended **F1_phase** | run-energy-err | 30 s-window R² | 30 s MAPE | 1 s bin R² |
|---|---|---|---|---|
| gemma-4-31B (dense) | **1.9 %** | 0.84 | 4.0 % | 0.60¹ |
| gemma-4-26B-A4B (MoE) | **1.9 %** | 0.95 | 1.9 % | 0.93 |

¹ The dense run is a short 79-bin trace whose ±100 W prefill-burst + meter jitter
is comparable to its dynamic range, so 1 s R² is noise-limited; the mean,
envelope and 30 s energy are all correct (see `results/validate_traces.png`).
`F0_floor` (6 terms) is nearly identical and marginally better on dense
(window R² 0.88, energy err 0.0 %).

Model ladder (in-sample R² on probe levels):

| model | terms | R² | energy err (med) |
|---|---|---|---|
| R0_roofline (pure energy) | 5 | 0.72 | 12.9 % |
| F1_phase (**recommended**) | 10 | 0.90 | 7.9 % |
| S_sat (saturating util) | 7 | **0.93** | 5.7 % |

**Recommended model — F1_phase** (every coefficient physical):

```
P_node = p_idle·TP + p_link·TP·1[TP>1]            # standing  ~93 W/GPU (= measured idle)
       + p_active_floor·TP·1[busy]                # busy step ~118 W/GPU
       + e_f_pre·FLOPs_pre/s + e_f_dec·FLOPs_dec/s # compute   0.19 / 3.4 pJ/FLOP
       + e_w_dec·Wbytes_dec/s + e_kv·KV/s + e_comm·NVLink/s
P_final = idle + m_family·dynamic                  # m: dense 1.09, moe 0.90  (±10%)
```

`S_sat` (concave memory/compute fill) has the highest *in-sample* R² (0.93) but
**over-predicts the dense plateau** on held-out data (negative 30 s-window R²,
10.9 % energy err) — a cautionary case where the best in-sample number is the
wrong model. The computed memory-bandwidth utilisation is in fact non-monotonic
with batch (weight reads/s fall as KV reads/s rise), so a saturating-util form
fights the data; the linear roofline-energy form (F0/F1) is both simpler and
more robust. This is why we did **not** chase in-sample R² — it would have
selected the wrong model.

### Transferability (the hardest test: 2 models, predict one from the other)

Leave-one-family-out, bridged only by arch arithmetic:

| | zero-shot (m=1) | Tier-2 (1 calibrated multiplier) |
|---|---|---|
| dense → predict MoE | 38–47 % | **10–13 %** |
| MoE → predict dense | 16–22 % | **10–18 %** |

Zero-shot cross-architecture (dense↔MoE) is genuinely hard with only one model
of each type. The **Tier-2 workflow from the campaign** — reuse hardware
constants, calibrate one efficiency multiplier from a short staircase of the new
model — brings it to ~10–13 %. Once both models are profiled (the real
deployment state), the pooled fit gives the ≤2 % held-out numbers above.

## Files

- `dataio.py` — measured-state bundle reader (idle anchor, measured iters, SWA/MoE arch arithmetic)
- `physics.py` — explainable feature set + model ladder
- `nnls.py` — non-negative least squares (pure numpy)
- `fit_eval.py` — fit ladder, transferability (zero-shot + Tier-2), held-out grade, plots
- `results/` — `ladder_fit.json` (coefficients), `decode_staircase_fit.png`, `validate_traces.png`

## Reproduce (Sherlock)

```bash
# one-time CPU venv on scratch (numpy/pandas/matplotlib; scipy/sklearn omitted)
srun -p dev --time=00:15:00 --mem=8GB bash $SCRATCH/setup_venv.sh
# run
srun -p dev --time=00:10:00 --mem=4GB $SCRATCH/runpy.sh new-profiling-model/fit_eval.py
```

## Honest limits

- **TP=2 only** here, so `p_idle` vs `p_link` is degenerate (their sum, 93 W/GPU,
  is identified; the split is not). A TP-pair probe would separate them.
- Two models (one dense, one MoE) is thin for proving cross-architecture
  transfer; the Tier-2 number is the trustworthy one until more models land.
- Prefill staircase levels are short (8–25 s) and noisy; they weakly constrain
  `e_f_prefill`. A longer prefill hold (chunked-prefill off) would tighten it.
- The dense-31B validate is a short run; its bin R² is noise-limited, not biased.
