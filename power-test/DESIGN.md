# power-test: static power surface on iteration-granularity coordinates

Plan items 2 and 3 of `archive/research_notes/PIPELINE_PLAN.md`. Inputs are the simulated ledger cache
(feature-test/ledger_cache_sim_250ms.npz, arrival-only channels on the
simulator clock) and the measured node power joined onto that same clock.
Everything here is dense-first; gpt-oss cells are reported but excluded from
the hardware fit and carry the MoE routing block (`docs/plans/TODO.md` item 1).

## Evidence base (all verified against code/data 2026-07-16)

- Join: timing-clock zero = earliest validated request epoch; power sample
  time on that clock is p_ts - p_ts[0] - delta with
  delta = (t0_req - p_ts[0]) - round((t0_req - p_ts[0])/1800)*1800.
  Measured over all 160 train runs: fold factor K = 0 everywhere, delta in
  [8.2, 18.0] s. 0.72% of train bins have no power sample. All 900 source
  files exist; run_id keys timing_dataset.npz = timing_dataset.runs.json =
  sim cache.
- Meter: H100 reading = pure 1.0 s trailing moving average (probe-fitted,
  cited arXiv:2312.02741, independently confirmed); A100 near-instant at
  250 ms bins. The shipped EMA constants 0.75/0.8 are a laundering
  artifact of a ~0.3 s work-onset delay and MUST NOT be reused. No step
  probe has ever run, so no first-order device time constant exists for
  either hardware: that stage is OMITTED here and stays blocked on the
  probe (a first-order stage without probe data would be an invented
  constant).
- Surface: the concave basis is tp*min(u, knot) ramps with non-negative
  coefficients (concave monotone by construction). Communication is
  dropped: collinear with compute (r ~ 0.99) and carries zero identified
  energy in every prior fit. FP8 compute scale is fractional,
  1 - 0.5*clip(fp8_flop_frac, 0, 1). A board-power value is not treated as a
  sample-by-sample meter cap; clipping requires an explicit run-level
  operating limit.

## Contracts

### 1. power-test/join_power.py (CLI)

Reads feature-test/ledger_cache_sim_250ms.npz and
timing-test/timing_dataset.runs.json. For every run_id (all 450):
verify the power CSV sha256 against runs.json; parse with
model.training_data.power_parsing.parse_power_csv_per_gpu
(gpus_per_node=8); TP-sum with tp_sum_power; parse the request JSON with
parse_request_json to get t0_req = min over validated request timestamps;
compute delta and K as above; RAISE if K != 0 or the fold gate
(min arrival in [-2, 600] s) fails; map samples to bins with
b = floor((p_ts - p_ts[0] - delta) / 0.25) on the run's sim grid (bin
count = that run's row count in the sim cache); per-bin mean; NaN where a
bin has no sample.

Writes power-test/sim_ledger_power_250ms.npz: identical to the sim cache
with `power` filled (NaN preserved for empty bins) and a boolean
`power_valid` column added. Writes
power-test/sim_ledger_power_250ms.provenance.json: schema_version,
per-run {run_id, delta_s, k_fold, n_bins, n_empty_power_bins,
sha256_verified: true}. Prints a coverage summary per role.

### 2. power-test/power_surface.py (module)

`surface_design(d, hardware)` -> (matrix [n_bins, k], names). `d` is a
dict of per-bin arrays from the joined cache plus per-bin arch columns
(n_active, w_bytes, fp8, fp8_flop_frac, tp). Columns (each times tp, all
coefficients later constrained >= 0):

- `tp` (always-on idle), `tp_link` = tp * 1[tp > 1] (fabric floor),
  `resident_weights = w_bytes / 80e9`, and `busy_tp = tp * busy`
  (busy is the simulated duty fraction in [0, 1]).
- One-hinge concave compute coordinates `u_compute` and
  `min(u_compute, 0.4)`, where
  u_compute = dtype_scale * 2 * n_active * (pre_tok + dec_tok)
  / (tp * compute_peak_flops_s); dtype_scale = 1 - 0.5 *
  clip(fp8_flop_frac, 0, 1) when present else 0.5 if fp8 else 1.0.
- One-hinge concave memory coordinates `u_memory` and
  `min(u_memory, 0.4)`, where u_memory =
  (w_read + kv_read + kv_write) / (tp * hbm_bandwidth_bytes_s).
  In the simulated ledger, `w_read` is the timing model's weight bytes for
  each exact engine iteration spread over that iteration's wall-time; it is
  not reconstructed from token completions or average batch.
- `iter_rate` = tp * engine_iterations_rate / 1000 (linear per-iteration
  launch/sync power; linear is concave; 1000/s is a conditioning constant
  only). The rate counts simulator iteration-completion records and does not
  reuse the reconstructed `dec_tok / batch` approximation.

Hardware profiles: A100 (312e12 FLOP/s, 2.0e12 B/s), H100 (990e12,
3.35e12) — same cited constants as feature-test. No communication column.

`predict(design, coefficients, tp, hardware)` returns `design @ c`. It clamps
only when the caller supplies a measured per-GPU `power_limit_w`; the model
does not treat a datasheet board rating as the run's configured limit.

### 3. power-test/response_chain.py (module)

`apply_chain(pred, dt, hardware, delay_s)`: (1) delay by
round(delay_s/dt) bins (shift right, pad with the first value);
(2) H100 only: trailing moving average over round(1.0/dt) bins
(partial windows at the start use the available prefix). A100 gets the
delay only. Applied per run, never across run boundaries.

Because the chain is linear, fitting filters the DESIGN COLUMNS through
apply_chain and regresses against raw measured power.

### 4. power-test/fit_power_surface.py (CLI)

Per hardware, the baseline uses dense bins with role == train and finite
power. MoE training bins are counted and excluded. The A100 candidate also
uses the source Llama-70B prefill/decode staircase artifact produced by
`build_probe_power_calibration.py`: raw power remains at 250 ms, bursty engine
counters are conserved over each request-active level, and each level receives
equal total weight after per-GPU squared-error scaling. Cached-context,
mixed-grid, and transient probes are excluded because they do not identify one
instantaneous dynamic-power component. No target or holdout bundle enters the
artifact.

For each delay in the plan-fixed grid (0.0, 0.25, 0.5, 0.75) s: filter design
columns per run through apply_chain, RMS-scale columns, scipy NNLS, un-scale;
pick delay by train RMSE; report the full grid, not just the winner. The probe
candidate replaces the baseline only if it is no worse on every source dense
`test_indomain` median and strictly better on at least one of energy error,
ACF-MAE, ACF R2, and range NRMSE. Soft-DTW and target results are report-only
and cannot select the fit. Write
power-test/fitted_surface.json: per hardware {coefficients by name,
delay_s, delay_grid_rmse, optional cap_w_per_gpu, provenance: every constant
tagged cited | plan-fixed | fitted-here}.

Evaluation (frozen metrics): predictions per run through apply_chain,
scored with feature-test/evaluation_core.py trace_metrics against joined
measured power. NaN power bins are linearly interpolated per run before
scoring (fraction reported; ~0.7%) — the one documented deviation, needed
because trace_metrics' 1 s aggregation cannot hold gaps. Score ONLY roles
train and test_indomain (dev). holdout_twin / holdout_rate /
holdout_model / dtype_calibration are NOT scored in this stage — they are
plan item 3's matrix and are not burned during development. Report per
(hardware, role): energy_error_pct median/p90/worst, acf_mae, acf_r2,
soft_dtw_divergence, nrmse_range, signed mean_bias_pct by rate; plus the same
table restricted
to dense models (gpt-oss flagged MoE-blocked). Write
power-test/fit_report.json, including the baseline/candidate source-development
selection record, and print the tables. Failures are reported
as failures; nothing is dropped to make a table look better.

### 5. power-test/evaluate_arrival_only.py (CLI)

Loads the frozen surface without refitting and scores every `test_indomain`,
`holdout_rate`, `holdout_twin`, `holdout_model`, and `dtype_calibration` run.
It writes `arrival_only_report.json` and `arrival_only_per_run.csv`. Exact
source-ID matches to the v2 M4A and same-configuration B2 rows are contextual
references only: their timing contracts and trace horizons differ from the
arrival-only path.

The 2026-07-16 result passes dense energy transfer but not the full temporal
gate. Dense A100/H100 held-rate energy medians are 2.85%/1.68%; H100 405B is
3.49% median and 6.06% P90. Rate-4 70B cells reach ACF-MAE 0.30-0.37 and
range NRMSE 0.19-0.35, consistent with the timing layer's 7-17% request-time
errors in those cells. MoE remains blocked and is not used to judge the dense
surface.

### 6. power-test/thermal_ablation.py (diagnostic, not selected)

Adds one causal first-order heat state
`s[t] = exp(-dt/tau) * s[t-1] + (1-exp(-dt/tau)) * q[t]`, reset to zero
at every run boundary. `q` is preliminary predicted board power above the
idle and always-on fabric floor. The surface coefficients and non-negative
thermal multiplier are fit jointly on finite dense training bins; delay and
`tau` in (30, 60, 120, 240, 480, 960) seconds are selected by training RMSE.
Two declared drivers are reported: all dynamic load, and dynamic load gated
to TP8 as a chassis-level hypothesis. Holdout targets cannot affect the fit.

The all-load state lowers rate-4 70B TP8 median ACF-MAE from 0.314 to 0.114
on A100 and 0.314 to 0.260 on H100, but it chooses 30/60-second constants
and worsens the TP4 ACF controls from 0.0355 to 0.1388 and 0.0222 to 0.0305.
The TP8-only state chooses the 960-second grid boundary on both hardwares and
does not improve H100. This is evidence that the current training split does
not identify a transferable several-minute thermal law. The ablation is
report-only and does not change `fitted_surface.json`.

### 7. power-test/changepoint_ablation.py (retrospective diagnostic)

Refits the unchanged surface in memory on finite dense training bins, then
detects the best two-mean split in each measured-minus-predicted residual:

`argmin_k SSE(residual[:k]) + SSE(residual[k:])`,

with both segments at least 120 seconds. It reports split time, before/after
means, jump, and variance explained for every run and aggregates the 450 runs
by hardware/model/TP/rate. Detection is retrospective and never presented as
a deployable causal trigger.

The evaluated correction is
`P_step[t] = P_surface[t] + b + delta * 1[t >= t_star]`. In
leave-one-repeat/model folds, `t_star`, `b`, and `delta` are medians fitted
only from the other runs. Cross-hardware tests fit residuals relative to
baseline power. A separate original-split diagnostic uses only dense-70B TP8
rate-2 training runs, doubles their per-GPU jump under an explicit
linear-in-rate assumption, and scores the frozen rate-4 targets. It centers
that jump around zero so aggregate energy is not improved by an oracle offset.

All 12 rate-4 70B TP8 runs split at 304.2-306.5 seconds with jumps of
12.35-19.04 W/GPU. H100 leave-one-model-out improves energy/ACF-MAE/range
NRMSE from 3.41%/0.314/0.304 to 0.34%/0.057/0.187. A100 improves ACF-MAE
from 0.321 to 0.206 but regresses energy from 3.19% to 5.08% and NRMSE from
0.193 to 0.204. The original-training rate-2 extrapolation reaches H100
ACF-MAE 0.053 but leaves NRMSE at 0.289; A100 ACF-MAE is 0.254. Applying
the fitted step to the other 132 dense TP8 runs worsens median energy,
ACF-MAE, and NRMSE from 1.95%/0.012/0.080 to 5.53%/0.024/0.111.
Therefore the affected-cell step is identified, but a general trigger is not.
This diagnostic does not modify `fitted_surface.json`.

## Test discipline (YAGNI)

Smallest set that catches believable semantic errors; expected values
justified by hand-worked examples, not snapshots of current output:
- join: synthetic CSV/JSON with a known delta -> samples land in the
  right bins, gaps stay NaN, K != 0 raises.
- surface: one hand-computed design row; one-hinge concavity; fractional FP8
  scale; no cap without an explicit operating-limit binding.
- chain: delay shifts exactly n bins; H100 window is exactly a 4-bin
  trailing mean; runs never bleed into each other.
- fitting: changing MoE or development targets cannot change a dense fit;
  coefficient lookup follows design names rather than JSON key order.
- evaluation: exact-source reference matching preserves transfer versus
  same-configuration contracts.
- thermal diagnostic: exact first-order heating/cooling, seconds-based time
  constant, run reset, static-floor exclusion, TP8 node-level gating, and no
  MoE/holdout target leakage.
- changepoint diagnostic: exact hand-worked boundary, minimum segment length,
  additive invariance, held-out residual isolation, and correct target-power
  scaling for cross-hardware relative corrections.

No fixtures beyond inline arrays. No mocking. No parametrized sweeps.
