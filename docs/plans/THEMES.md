# e-Energy Themes for PowerTrace-Sim

This note turns the repo audit and a current e-Energy literature scan into a
paper direction for `eenergy-paper`. It is written as guidance for revising the
paper, not as a claim that every item is already supported by the code.

As of 2026-07-07, the strongest fit is the ACM e-Energy 2026 "Systems and
applied modeling" track, with "Learning" as a secondary lens. The paper should
read as an empirical model of real AI inference loads for grid and facility
planning. The machine learning model is a means to produce planner-grade load
profiles, not the main story.

## Core Position

AI inference facilities should not enter grid studies as rigid nameplate loads.
They are stochastic, workload-driven, accelerator-heavy loads whose peaks,
ramps, and flexibility depend on request arrivals, serving policy, hardware,
and aggregation across servers. PowerTrace-Sim is useful if it gives planners a
transparent way to generate those load shapes before the facility exists or
before telemetry can be shared.

The title should move toward:

```text
AI Inference Load Trace Generation for Grid and Facility Planning
```

or:

```text
Planner-Grade Power Traces for Grid-Interactive AI Inference Facilities
```

## Themes From Recent e-Energy Work

### 1. Data centers as grid assets, not passive demand

Recent work frames AI data centers as flexible resources that can react to grid
signals while preserving service quality. The strongest external comparison is
the Emerald Conductor field demonstration, which reports a software-only demand
response trial on a 256-GPU commercial cluster with a 25 percent cluster power
reduction over a three-hour peak event while maintaining QoS.

How to reflect this in the paper:

- Keep the facility trace as the main output, because grid operators consume
  load profiles, peaks, ramps, and event response envelopes.
- Add one result that reports "available reduction under a grid event" if the
  code can support it without new machinery. If not, present this as near-term
  use rather than a demonstrated claim.
- Use the occupancy roofline work only after the measured-state ledger is fixed
  or clearly label it as reconstruction-based.

### 2. Flexibility must be quantified under constraints

e-Energy reviewers will expect flexibility claims to specify what is flexible,
on what timescale, and at what cost. Recent carbon-aware and demand-response
papers emphasize temporal shifting, spatial shifting, QoS, robust guarantees,
and diminishing returns from longer deferral windows.

How to reflect this in the paper:

- For inference, separate hard real-time serving from latency-tolerant work.
- Report peak, average, load factor, max ramp, and oversubscription headroom as
  current metrics.
- Do not claim demand response value until there is an explicit event policy,
  workload deferral rule, or routing rule in code.
- Add sensitivity to arrival correlation before making strong oversubscription
  claims.

### 3. Carbon signals are contested

e-Energy 2024 included multiple papers on carbon accounting and carbon-aware
optimization, including work on location versus market carbon intensity and
average versus marginal carbon signals. That matters because a weak carbon
signal can undermine otherwise strong compute-shifting results.

How to reflect this in the paper:

- Avoid making carbon-reduction claims from generated power traces alone.
- If carbon appears, frame the contribution as producing the load trace that a
  separate grid or carbon model can consume.
- Prefer "grid-facing load characterization" over "carbon optimization" unless
  marginal or market signal assumptions are made explicit.

### 4. Open tools and reproducible pipelines matter

The e-Energy 2026 accepted-paper list includes open-source tools and simulation
frameworks for computational decarbonization and flexible data centers. That
raises the bar for this repo: reviewers will notice if the scripts are large,
stale, path-dependent, or if generated artifacts are mixed with source.

How to reflect this in the paper:

- Make `scripts/eval/run_azure_pipeline.py` the canonical paper command.
- Add a `results/eval_paper/README.md` mapping each paper figure/table to one
  command and its upstream artifacts.
- Ensure manifests use repo-relative paths and record command, git commit,
  seeds, config IDs, generation mode, timestamp mode, and fallback status.
- Move old or exploratory analyses out of the maintained evaluation path.

### 5. Real data and testbeds carry more weight than pure simulation

Recent e-Energy data center papers emphasize field trials, testbeds, measured
systems, and production traces. This is a strength of PowerTrace-Sim if the
paper clearly separates measured data, fitted models, generated traces, and
derived planning quantities.

How to reflect this in the paper:

- Lead with measured GPU power traces and the production Azure workload trace.
- Treat synthetic Poisson sweeps as validation coverage, not the central grid
  result.
- Keep MoE results, but present them as a limitation and a stress test because
  hidden expert routing weakens temporal fidelity.
- Make every paper table state whether it is measured, replayed, generated, or
  aggregated.

## Results Worth Showing From This Codebase

### Main paper figures and tables

- Server trace comparison: `figures/baselines_node_groundtruth_trace.pdf`
  validates that the generator captures dynamics missed by TDP, mean, and LUT
  baselines.
- Trace fidelity table: `results/eval_paper/trace_fidelity_table.tex` supports
  the claim that dense models preserve energy and autocorrelation well, while
  MoE remains harder.
- Facility diurnal profile: `figures/azure_figure_1_diurnal_profile.pdf` is the
  grid-facing artifact.
- Facility sizing table: `results/eval_paper/azure_facility_sizing_table.tex`
  converts traces into peak, average, ramp, and load-factor numbers.
- Hierarchy figure: `figures/azure_hierarchy_*.pdf` explains why aggregation
  smooths server-level variability.
- Oversubscription figure: `figures/azure_oversubscription_lines.pdf` is useful
  only if the text is careful about arrival correlation assumptions.

### Appendix or secondary figures

- Power CDFs for dense and MoE configurations.
- Appendix surrogate validity for measured versus generated active-request
  features.
- Feature sufficiency only if it directly defends the chosen two-feature model.

### Not ready for main claims

- Occupancy roofline results until outputs exist under `results/occupancy_roofline/`
  and the analysis states whether occupancy is measured from `engine.csv` or
  reconstructed from request timing.
- Agentic workload results until cache-on/cached-prefill behavior is measured
  rather than reconstructed.
- Ledger/two-price/saturating-fit results unless the paper deliberately adds a
  separate operator-model section.

## Recommended Paper Flow

1. Start with the grid problem: AI inference is a new large load class, and
   nameplate values are too crude for interconnection, distribution planning,
   and facility provisioning.
2. State the interface: given topology, server configuration, workload scenario,
   overhead, and PUE, emit load profiles at server, rack, row, and site levels.
3. Explain the measured decomposition: workload features drive operating-state
   transitions; configuration-specific state models drive power.
4. Validate server-level fidelity with measured traces.
5. Show facility-level consequences under a production Azure workload: peak,
   average, ramp, load factor, and oversubscription sensitivity.
6. Discuss grid use cases and limitations: demand response is a downstream use,
   not yet a solved control policy; carbon claims require a separate signal
   model; MoE hidden routing is a known hard case.

## Claims To Avoid Until Code Catches Up

- "We optimize carbon."
- "We provide demand response guarantees."
- "The roofline measures true GPU occupancy" unless `engine.csv` is the source.
- "Inference CLI and evaluation use the same generation semantics" until IID
  versus AR(1) behavior is unified or explicitly documented.
- "The Azure trace is fully reproducible" until raw input paths and generated
  manifests are repo-relative and command-complete.

## Source Scan

- ACM e-Energy 2026 CFP and tracks:
  https://energy.acm.org/conferences/eenergy/2026/pages/cfp.php
- ACM e-Energy 2026 accepted papers:
  https://energy.acm.org/conferences/eenergy/2026/pages/accepted-papers.php
- ACM e-Energy 2024 accepted papers:
  https://energy.acm.org/conferences/eenergy/2024/acceptedpapers.php
- ACM e-Energy 2025 program:
  https://energy.acm.org/conferences/eenergy/2025/program.php
- Turning AI Data Centers into Grid-Interactive Assets:
  https://arxiv.org/abs/2507.00909
- Carbon-Aware Computing for Data Centers with Probabilistic Performance
  Guarantees: https://arxiv.org/abs/2410.21510
- Carbon-Aware Computing in a Network of Data Centers:
  https://arxiv.org/abs/2405.18070
- To Defer or To Shift? The Role of AI Data Center Flexibility on Grid
  Interconnection: https://arxiv.org/abs/2604.05376
- Carbon-Aware Compute-Power Scheduling for AI Data Centers with Microgrid
  Prosumer Operations: https://arxiv.org/abs/2605.03751
