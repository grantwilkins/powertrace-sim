# PIPELINE_PLAN: request-to-power modeling — state, hypothesis, and route

Status: session handoff, 2026-07-16. Written after the timing layer was
built and validated and before the power stage was rebuilt on top of it.
Read together with `FEATURE_TEST_LEARNINGS.md` (power-side evidence and
no-go list), `timing-test/DESIGN.md` (timing contract), and `TODO.md`
(blocking measurements).

## 1. The hypothesis

One pipeline predicts node power from requests alone, with nothing trained
per deployment or per served model:

```text
(arrival time, prompt tokens, output tokens)          marked arrival process
        |
        v
discrete-event scheduler simulation                   engine policy, no fitted constants
        |  (continuous batching, chunked prefill,
        |   seat + KV admission)
        v
per-iteration work from architecture arithmetic       FLOPs/bytes per operator class,
        |                                             no fitted constants
        v
iteration time = summed per-operator rooflines        11 named scalars per GPU type
        |         + layer-scaled launch/sync
        |         + per-token sampling
        v
phase-resolved executed work on wall time (250 ms)
        |
        v
static power surface + first-order device response    fitted per GPU type on
        |                                             training cells only
        v
moving-average filter of the driver's power reading   documented meter behavior
        |
        v
node power trace -> energy, trace-shape metrics, Monte Carlo
```

Division of labor, which is the entire transfer argument:
- architecture arithmetic carries model-to-model transfer (two 8B models
  are one point in descriptor space; a 70B is interpolation);
- the small named-scalar sets carry hardware calibration, done once per
  GPU type (plus per-dtype class constants like the FP8 streaming scale);
- the simulator carries all load dependence (queueing, batching,
  interference) so nothing about load is memorized;
- the measurement filter reconciles predicted power with what the meter
  actually reports.

## 2. What is built and validated (timing-test/)

The timing layer is DONE and is the basis for the power stage:

- Per-request accuracy on frozen holdouts (results/timing_test_v1_fp8/):
  in-domain 3.0/5.4% median end-to-end error (H100/A100); held-out rate
  4.0: 5.6-6.6%; zero-shot architecture twins 2.9/5.6%; zero-shot held-out
  models: llama-3-405b 2.9-8.1% at rates <= 2 (after the one-scalar FP8
  streaming calibration, 0.781), gpt-oss-120b 9.4% median. Time to first
  token within 3-21 ms median everywhere. Beats the per-config log-fit
  baseline in 110/150 cells without per-model data; a Vidur-style random
  forest on identical features loses catastrophically zero-shot (34%/61%)
  while matching in-domain — the structure earns its keep exactly where
  transfer matters.
- 250 ms work placement validated (timing-test/validate_bin_placement.py):
  simulated vs measured decode-rate correlation 0.97-0.999 median at 1 s in
  every role, exact token conservation. This was the gate for using the
  simulator as the power stage's front end. (Caution: the measured ledger
  cache's time axis silently drops empty-power bins; comparisons must use
  a uniform clock or the keep_power_gaps builder path.)
- Known, bounded failures: no preemption mechanism (405B rate 4: -12.5%);
  gpt-oss-120b TP4 +8-17% (launch-bound MoE, see TODO); two fitted values
  flagged suspicious in timing-test/README.md (A100 110 us/token sampling
  cost; H100 bandwidth efficiency at its 1.0 bound).
- Hard-won contract lessons already encoded in code comments: engine
  configuration is part of the model (the legacy gpt-oss serving used
  async scheduling, the probes did not — 3.9 vs 6.9 ms at batch 1; never
  blend calibration data across engine modes); MoE routing is
  unidentifiable from latency or node power (fit_moe_routing.py is
  report-only, see TODO.md item 1 for the cheap offline measurement).

## 3. What the power side established (feature-test/, results/feature_test_v2/)

- A single response surface over aggregate work rates cannot pass the
  gates: communication is collinear with compute (r ~ 0.99), and the same
  ledger bytes/s draw different power across regimes (TP synchronization;
  MoE iteration granularity). Confirmed twice: v2 forensics and the
  measured-engine scoring, which also proved the old apparent 405B accuracy
  was reconstruction error cancelling model error.
- Cited-constant corrections that stand regardless of model choice: 405B
  FP8 weight bytes 487.23e9; FP8 FLOP fraction 0.7996; gpt-oss MXFP4 bytes
  confirmed; board-power caps (400/700 W) instead of fitted quantiles.
- The power reading itself: H100 reports a 1 s moving average, A100 is
  near-instant (identified from step events, matching arXiv:2312.02741),
  plus a ~0.3 s work-onset alignment lag that belongs in the timing layer
  (adversarial review, learnings section 14 — the identification script's
  EMA constants 0.75/0.8 are a known laundering artifact; use the pure
  window + explicit delay in the new stage).
- Metric discipline: use ACF-MAE for trace shape (ACF R2 denominators vary
  10x across rates); report signed bias by rate; report component fidelity
  AND end-to-end (cancellation hides in end-to-end-only scores).

## 4. Next steps, in order (dense first, by decision 2026-07-16)

1. **Simulated-ledger emitter** — DONE 2026-07-16
   (timing-test/simulated_ledger.py + tests). Converts the simulator's
   iteration trace into 250 ms channels with the measured ledger's exact
   names and semantics. Shared token/KV arithmetic comes from
   `bin_work_rates`/`effective_context`; weight-memory traffic comes directly
   from the timing model's per-iteration weight bytes, so one mixed
   prefill/decode iteration remains one weight sweep. Decode tokens and KV
   reads are completion events at iteration end (measured exact-ITL
   convention, so dec_tok integrates to sum(n_out-1)); coverage channels by
   interval overlap; running/waiting from the simulator's true admission
   times (new additive trace field n_chunks and per-request admitted_s).
   Built cache: feature-test/ledger_cache_sim_250ms.npz — all 450 manifest
   runs with role_idx and power=NaN. Simulator-only channels are `busy`,
   prefill/decode duty, exact iteration/s, scheduled-token/s, and
   tokens/iteration. Exact iteration coordinates count trace completion
   records; they do not inherit the reconstructed `dec_tok / batch` tail
   (which reached 0.2-0.9 million/s in boundary bins). Verified exact token
   and iteration conservation plus exact weight-byte conservation with
   hand-worked mixed-phase tests. The correction was locally validated
   without overwriting generated artifacts; the checked-in fitted surface
   and arrival-only reports must be regenerated before their metrics
   represent this code path.
2. **Power surface on iteration-granularity coordinates** — DONE 2026-07-16
   (`power-test/`). The concave per-hardware surface fits dense training bins
   only; known-blocked MoE bins cannot move its coefficients. The response
   chain selects A100 delay 0.0 s and H100 delay 0.25 s, with the fixed H100
   1 s reading window; no unmeasured first-order device constant is invented.
   Exact iteration/s and tokens/iteration receive zero NNLS weight on current
   dense support; decode duty is the identified phase term. Dense development
   energy: A100 median/P90 3.36/5.36%, H100 1.85/4.31%.
3. **End-to-end arrival-only evaluation** — DONE, DOES NOT FULLY PASS
   2026-07-16 (`power-test/arrival_only_report.json`). Requests -> watts was
   frozen and scored on 290 non-training timing-role runs, with exact-source
   contextual comparisons to conditional M4A (186 matches) and B2 (150).
   Dense energy transfers: A100/H100 held-rate medians 2.85/1.68%, H100 405B
   median/P90 3.49/6.06%, twins 3.51/2.00%. Temporal failures remain in
   rate-4 70B cells (ACF-MAE 0.30-0.37; range NRMSE 0.19-0.35) and the A100
   70B tail. A 2026-07-17 exact-work audit of H100 TP8 run 376 found zero
   best lag and 0.984-0.985 correlation in output-token rate and batch
   occupancy; the measured node-power baseline instead rises coherently by
   about 91 W late in the run. A training-only first-order heat ablation
   improves the TP8 ACF but selects 30-60 s constants and damages TP4
   controls; a TP8-only term hits the 960 s grid boundary and does not improve
   H100. A subsequent all-run changepoint audit finds the actual affected-cell
   structure: all 12 rate-4 70B TP8 repetitions jump at 304.2-306.5 s by
   12.35-19.04 W/GPU. On H100 the step transfers across 70B models
   (ACF-MAE 0.314 -> 0.057), and a rate-2 training-only extrapolation predicts
   the held-rate ACF improvement (0.314 -> 0.053). A100 offsets do not transfer
   cleanly, and applying the step to other dense TP8 cells worsens all three
   primary metrics. Do not ship the smooth or step fit or tune the static
   surface to these holdouts.
4. **NEXT: identify or bound the TP8 hardware-state failure** in TODO.md
   item 2 and identify a request-visible trigger or documented support
   boundary. Temperature/clock/P-state telemetry distinguishes a physical
   threshold from a benchmark/runtime phase. Then rerun the frozen
   arrival-only evaluator. Only after dense passes, run TODO.md item 1
   (MoE routing, one GPU-hour),
   substitute the measured routing law, re-run MoE cells.
5. Only then: the sealed campaigns (still unrun, still uninspected) for
   the paper's validation claim.

## 5. Discipline that must survive the session boundary

- Holdout roles live in timing-test/split_manifest_fp8.json; gpt-oss-120b
  probe bundles are quarantined until its zero-shot scoring freezes.
- Every constant carries a provenance class: cited hardware/architecture
  fact, source-fitted under frozen procedure, dedicated probe, or
  plan-fixed threshold. "It makes cell X pass" is not a class.
- Failed fits are reported, not shipped (see fit_moe_routing.py for the
  pattern); an identification whose parameter slams a bound or whose
  residual dwarfs measurement noise is a finding, not a calibration.
- B2 rows are bit-deterministic and reusable during development; the
  frozen evaluator reruns everything for results that get reported.
- The retrospective targets are burnt as validation evidence; only the
  sealed campaign discharges that.

## 6. File map

- Timing model: timing-test/ (DESIGN.md contract; README.md results and
  flagged caveats; fitted_efficiencies.json = the 11+1 scalars/hardware;
  iteration_time.py, scheduler_sim.py, fit_*.py, evaluate_timing.py,
  validate_bin_placement.py). Results: results/timing_test_v1_fp8/
  (v1 = pre-FP8-amendment snapshot).
- Power model: feature-test/ (evaluate_candidates.py frozen evaluator,
  physics kernel in model/classifiers/physics.py, meter identification,
  gates). Results: results/feature_test_v2/ read with learnings sec. 14.
- Evidence and history: FEATURE_TEST_LEARNINGS.md (sections 10-13 are the
  collaborator's measured-engine audit and design memo; section 14 the
  adversarial review).
- Blockers: TODO.md.

Verify before trusting anything: `uv run -m pytest -q` (376),
`uv run -m pytest -q timing-test/tests feature-test/tests` (23 + 64).
