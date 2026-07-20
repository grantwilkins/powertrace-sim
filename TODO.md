# TODO

## 1. MoE expert-routing measurement (BLOCKS the mixture-of-experts power axis)

Status: specified, not run — needs ~1 hour on any GPU (16 GB for gpt-oss-20b,
one 80 GB card for gpt-oss-120b). Until it runs, every mixture-of-experts
power claim carries the uniform-independent routing assumption, which is
provably unidentifiable from latency or node power (timing-test/README.md,
FEATURE_TEST_LEARNINGS.md section 6 item 2, timing-test/fit_moe_routing.py).

What to do:

1. Reconstruct the exact benchmark prompts: the serving campaigns sample
   ShareGPT (public) with a recorded seed, so the same dataset + seed +
   num_prompts regenerates the prompt set deterministically. No stored text
   is needed (the gpt-oss raw JSONs kept only lengths and timing).
2. One offline forward pass per model (public checkpoints
   openai/gpt-oss-20b and openai/gpt-oss-120b), generating continuations
   and logging, per layer and per token, the router's chosen expert IDs
   (~50-line hook on the MoE layer; speed and hardware are irrelevant —
   routing is a pure function of weights + text, independent of batch,
   engine, and GPU type).
3. From the logs, estimate the routing law the power model needs: the
   distribution of DISTINCT experts touched by a group of B co-scheduled
   tokens (skew across experts, overlap along a request), per layer.
   Replace `expected_weight_bytes_per_sweep`'s uniform-independent
   expectation (timing-test/iteration_time.py) and the ledger's identical
   formula (model/training_data/ledger_view.py:167) with the measured law.
4. Output-token routing comes from freshly generated continuations (the
   original outputs were not stored) — statistically equivalent for
   estimating the law.

Why it matters: expert weight traffic is bytes moved, and bytes cost energy
even when latency hides them (they sit below the roofline max, which is
exactly why timing cannot identify them but power mis-predicts). This is
the last measurement standing between the current pipeline and the
gpt-oss/A100 power cells; the same instrumentation resolves the
launch-bound MoE timing residual (+8-17% on gpt-oss-120b TP4).

Also collected in the same pass, for free: per-token routing entropy and
expert-load histograms usable as citations in the paper.

## 2. Dense arrival-only temporal fidelity (BLOCKS progression to MoE/sealed work)

Status: identified by the frozen 2026-07-16 arrival-only evaluation. Energy
transfer passes on dense held roles, but rate-4 70B cells on both hardwares
reach ACF-MAE 0.30-0.37 and range NRMSE 0.19-0.35; the A100 70B twin also has
a high temporal-error tail at rates 1-4.

The 2026-07-17 audit bounds timing as the primary H100 TP8 cause: run 376 has
zero best lag and 0.984-0.985 simulated/measured correlation for output-token
rate and batch occupancy, while all eight GPUs gain about 11-14 W late in the
run with almost unchanged utilization. TP8 second-half residual drift is
81-91 W on H100 and 114-133 W on A100; matched TP4 drift is about 0-6 W.

`power-test/thermal_ablation.py` tries the smallest causal heat state using
dense training targets only. A generic state improves TP8 ACF but chooses
30/60 s time constants and worsens TP4 controls. A TP8-only state chooses the
960 s grid boundary and does not improve H100. Neither is identified well
enough to ship.

`power-test/changepoint_ablation.py` establishes the residual structure from
all 450 runs. Every rate-4 70B TP8 repetition on both hardwares and both
models jumps at 304.2-306.5 s by 12.35-19.04 W/GPU. H100 leave-one-model-out
improves energy/ACF-MAE/NRMSE from 3.41%/0.314/0.304 to
0.34%/0.057/0.187. A rate-2 original-training extrapolation also predicts the
H100 held-rate ACF improvement, but A100 offsets do not transfer and an
ungated correction worsens the other 132 dense TP8 runs. The step is
identified; its general trigger is not.

Next diagnostic: repeat one TP8 rate-4 70B run and its TP4 control while
recording per-GPU temperature, SM/memory clocks, P-state, power limit, and
clock-event reasons (`nvidia-smi dmon -s pcu` plus the corresponding query
fields). Attribute the coherent drift to temperature/leakage, DVFS, or the
onboard meter before choosing a trigger. Also vary workload duration so a
fixed 300 s runtime phase can be separated from a temperature or cumulative
work threshold. Rerun
`power-test/evaluate_arrival_only.py` unchanged after an identified correction
or documented support exclusion.

## 3. Deferred (tracked elsewhere)

- Preemption mechanism in the scheduler simulation (405B at rate 4;
  counters exist in the new bundles) — timing-test/README.md.
- Sealed campaigns remain unrun by design; do not inspect
  (profiling/MODEL_READINESS_RUNBOOK.md).
- gpt-oss-120b probe bundles stay quarantined until its zero-shot scoring
  freezes (timing-test/build_probe_calibration.py).
