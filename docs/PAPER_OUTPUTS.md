# Paper Output Contract

This is the allowlist for figures and tables produced by the selected
PowerTrace-Sim pipeline. An output is a paper result only when it appears
below, its producer and provenance sidecar are retained, and its caption uses
the stated evidence label. Everything else under `power-test/`, `figures/`, or
`results/` is development evidence unless the allowlist is deliberately
updated.

Generated values and plots are never edited by hand. Regeneration runs the
producer, reviews its provenance file, and then copies only the allowlisted
rendered output into the manuscript.

## Main paper

| Claim | Producer | Allowlisted outputs | Required evidence |
| --- | --- | --- | --- |
| Request-phase timing | `uv run python power-test/plot_timing_parity.py` | `power-test/timing_prefill_parity.pdf`; `power-test/timing_decode_parity.pdf` | Frozen split and timing fit |
| End-to-end simulator fidelity | `uv run python power-test/coverage_model_fidelity_table.py` | `power-test/coverage_model_fidelity_table.tex` | The matching CSV and JSON provenance files |
| Representative held-out traces | `uv run python power-test/plot_best_coverage_holdout_traces.py` | Four `power-test/power_trace_*_coverage_1s.pdf` panels at 0.125, 1, 2, and 4 requests/s | `best_coverage_holdout_traces_1s.{csv,json}` and one fixed configuration across rates |
| Facility consequences | `uv run -m scripts.eval.run_azure_pipeline` | `figures/azure_figure_{1..5}_*.pdf`; `figures/azure_oversubscription_{capacity,lines}.pdf` | `azure_figure_manifest.json` and the corresponding `results/eval_paper/azure_*` CSV/JSON files |
| Facility sizing | `uv run -m scripts.eval.generate_azure_facility_sizing_table` | `results/eval_paper/azure_facility_sizing_table.tex` | Matching CSV and JSON files |
| Final external score | `uv run python power-test/score_sealed_campaign.py` | The aggregate score table selected after the external campaign | One unchanged model artifact and the complete per-run score report; no post-score fitting |

The paper may use a subset of the five Azure panels, but it must not substitute
an unlisted development plot merely because that plot looks clearer. Timing,
power, and facility results must all identify the exact selected-model artifact
used to generate them.

## Appendix transfer evidence

These panels are useful boundary evidence, not zero-shot validation. Captions
must call them **retrospective** and state every target-derived calibration.

| Evidence | Producer | Allowlisted outputs | Required label |
| --- | --- | --- | --- |
| Dense and MoE Qwen transfer | `uv run python power-test/plot_transfer_traces.py` | `power_trace_qwen3_14b_a100_tp1_dense_transfer_1s.pdf`; `power_trace_qwen3_30b_a3b_h100_tp2_moe_transfer_1s.pdf` | Retrospective idle/platform-calibrated transfer |
| BurstGPT arbitrary arrivals | `uv run python power-test/plot_burstgpt_transfer.py` | Three `power_trace_burstgpt_*_idle_1s.pdf` panels | Retrospective idle-calibrated arbitrary-arrival transfer |
| OpenHands agent workload | Current analysis recorded in `FEATURE_TEST_LEARNINGS.md` Sections 15--16 | `power-test/openhands_platform_calibrated_prediction_overlay.png` | Retrospective few-shot platform calibration; not a cache-effect claim |

Each family retains its exact plotted samples and JSON report. Qwen and
BurstGPT also retain their generated LaTeX captions. The OpenHands panel has no
standalone maintained renderer yet, so it is allowlisted as a frozen appendix
artifact; changing it requires first adding a deterministic producer.

## Development archive boundary

The following remain reproducibility evidence but are excluded from the paper
output surface:

- candidate-selection, ablation, residual, metric-audit, and rate-4 diagnostic
  plots;
- unsupported dense-as-MoE or cross-support counterfactuals;
- routing-law investigations and retrospective GPT-OSS timing diagnostics;
- historical GMM-BiGRU figures, which live in `archive/gmm_bigru_v1/`;
- the stale pre-audit artifacts identified by `results/eval_paper/README.md`.

Their data and code may remain in the research tree while active experiments
are being audited, but they are not maintained paper entry points. Once their
metric differences are documented, duplicate historical reruns should be
removed or moved to the relevant archive rather than added to this allowlist.

## Regeneration order

1. Rebuild and hash-bind the prepared data.
2. Refit the frozen selected equations and verify artifact equivalence.
3. Generate timing parity, coverage fidelity, and representative traces.
4. Run the facility pipeline from the same selected artifact.
5. Run the external score once, without fitting or threshold changes.
6. Regenerate the retrospective appendix families separately and preserve their
   labels.
7. Verify that every manuscript figure/table is present in this document and
   has its listed provenance sidecar.

The disaggregated Sherlock campaign is intentionally outside this v1 allowlist.
It becomes a paper claim only after role-aware prefill, decode, and KV-transfer
composition is implemented and evaluated.
