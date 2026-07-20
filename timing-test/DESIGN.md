# timing-test: first-principles request-timing model

Status: design and holdout contract frozen 2026-07-16, BEFORE any fitting.
The component-accounting revision below is retrospective development work from
2026-07-19; its Qwen and legacy results are not part of the original frozen
claim.
Goal: per hardware, predict request timing — time to first token, decode
duration, and total request latency — from request marks and architecture
descriptors, with efficiencies calibrated once per hardware, not per model.
"Accurate" is defined by the provisional targets in section 5 and by beating
the log-fit baseline (section 6) on the held-out cells.

## 1. Model structure (all standard terms)

Three layers, fitted and validated separately:

1. **Work per iteration** (no fitted constants): from the architecture
   registry (`model/training_data/arch.py`), compute per engine iteration
   with decode batch `B`, mean context `c`, or prefill chunk of `C` tokens:
   - transformer FLOPs: `2 * transformer_active_params * tokens`;
     input embeddings are row gathers, while output-head FLOPs are
     `2 * output_head_params * logits`;
   - decode attention:
     `4 * n_q_heads * head_dim * n_layers * B * c_eff`;
   - dense weight bytes: transformer sweep, output-head sweep when logits are
     produced, and only the selected input-embedding rows; MoE descriptors
     without component metadata retain their routing-aware legacy sweep;
   - each prefill request/chunk contributes its own causal attention work.
     Chunks are summed after computing their separate prior contexts, so
     concurrent prompts never attend across requests.
2. **Iteration time** (fitted per hardware; per-(hardware, GPU-count) for
   the synchronization term):
   ```text
   T_iter = T_launch(hw, tp)
          + max( FLOPs / (eff_F(hw) * peak_FLOPs * tp),
                 bytes / (eff_B(hw) * HBM_bandwidth * tp) )
   ```
   `peak_FLOPs` and `HBM_bandwidth` are datasheet constants; the FP8
   fraction scales peak as in the power model. `T_launch(hw, tp)` absorbs
   kernel-launch and collective-synchronization overhead per iteration;
   fitting it per GPU count is allowed (parallelism generality is a bonus,
   not a requirement). Efficiencies are shared across ALL models on a
   hardware — that sharing is the transfer claim.
3. **Scheduler simulation** (no fitted constants; engine policy replicated):
   continuous batching, first-come-first-served admission, chunked prefill,
   and decode-first composition. Bundle evaluation requires
   `max_num_seqs` and `max_num_batched_tokens` from the run manifest rather
   than assuming 2048 tokens. Cached prefixes reserve KV capacity and remain
   in attention context while only the uncached suffix is prefetched.
   KV-cache capacity comes from architecture and 0.9 * HBM. Replays recorded
   arrival timestamps. Emits per-request time
   to first token, per-token latencies, and end-to-end latency.

Known ~0.3 s work-onset alignment lag (adversarial review, learnings
section 14) is a timing-layer phenomenon; the simulator does not add it —
if it appears, it must show up as a fitted component of `T_launch` or as a
documented residual, never a silent shift of predictions toward targets.

## 2. Data

- Training/evaluation serving runs: the 450-run legacy set (per-request
  arrival, time to first token, per-token latencies; 25 configs x 6 rates
  x 3 repeats).
- Calibration probes (local bundles under `data/runs/`): decode staircase
  (batch 1..256), prefill staircase, context grids/holds for llama-3-70b
  (A100 TP4; H100 TP8 and TP4). Probe points are direct observations of
  iteration time versus work and get first priority in fitting.
- The gpt-oss iteration probes exist on the cluster but are not synced;
  the gpt-oss-120b holdout below is therefore genuinely zero-shot.

## 3. Frozen holdout contract (defined before fitting; no exceptions)

Per hardware, three nested claims, hardest first:

| holdout | A100 | H100 | claim tested |
|---|---|---|---|
| held-out model (never in any fit) | gpt-oss-120b (all rates) | llama-3-405b (all rates) | scale/architecture transfer from descriptors |
| held-out architecture twin | deepseek-r1-distill-70b (all rates) | deepseek-r1-distill-8b (all rates) | two same-architecture models are one point in descriptor space |
| held-out operating point | rate 4.0 req/s for every training model | rate 4.0 req/s for every training model | extrapolation past the training load range |

Training cells: remaining model x rate {0.125, 0.25, 0.5, 1, 2} cells,
repeats 0 and 1. In-domain test: repeat 2 of training cells. Prompt/output
SIZE cannot be held out of serving runs (sizes are mixed within a run), so
size generalization is reported as stratified error over the top decile of
prompt length and of output length on all test cells.

Leakage rules: holdout requests never enter efficiency fitting, overhead
fitting, scheduler policy tuning, or any normalization; the throughput
database (per-config medians) is treated as a LOG-FIT artifact and may be
used only by the baseline, never by the principled model.

## 4. Metrics (per request, aggregated per cell by median/P90; report both
absolute seconds and percent)

- time to first token: predicted minus measured (s);
- decode duration (sum of inter-token latencies) error (s, %);
- end-to-end latency error (s, %);
- per-phase RMSE (s) per cell; run-level total-time error (%);
- stratified by prompt-length and output-length decile.

## 5. Provisional targets ("pretty well"), per cell

- median |end-to-end error| <= 10% and <= 1.0 s at rates <= 2;
- median |time-to-first-token error| <= 0.3 s below saturation, <= 20%
  at the held-out rate 4.0;
- median |decode-duration error| <= 10%;
- no cell with systematic sign bias worse than 15% at P50.

Targets are provisional (set from serving-SLO practice, not from peeking at
holdouts); misses are reported, not renegotiated after scoring.

## 6. Baseline to beat

Per-config log fit: predicted time to first token = queue-free prompt
tokens / measured per-config median prefill rate; decode duration = output
tokens / measured per-config median decode rate at that concurrency bin
(`model/throughput_database.json`). This baseline gets per-config measured
medians (an advantage the principled model refuses); it cannot exist for
held-out models — where the principled model must stand alone. Beating or
matching it on training-model test cells while covering holdout cells it
cannot address is the success criterion.

## 7. Fit procedure (frozen)

1. Fit eff_F, eff_B per hardware and T_launch per (hardware, tp) by least
   squares on probe iteration-time points (first priority) plus per-run
   median inter-token latency at observed concurrency and queue-free
   time-to-first-token from TRAINING runs only.
2. No per-model constants anywhere. Architecture enters only through the
   work calculator.
3. Freeze, simulate all test/holdout cells open-loop (recorded arrivals
   in, timing out), score once, report.
