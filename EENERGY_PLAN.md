# EENERGY_PLAN: audited path to an e-Energy paper

Written 2026-07-09 and audited against the repository on 2026-07-09. This is
the execution plan; `THEMES.md` remains the framing document. Repository status
in this file describes the working tree inspected on that date, not a clean
release commit.

## Executive decision

The paper direction is worth pursuing, but the previous version of this plan
was not execution-ready. It assumed traffic behavior that the code does not
have, treated several research scripts as one deployable first-principles
model, mixed reconstruction with end-to-end prediction, and called a small set
of scenario simulations "bounds."

Do not start the large Monte Carlo grid until the workload, timing, model, and
statistical contracts below pass their gates.

The defensible core claim is:

> Given a documented offered-load scenario, topology, hardware/model
> configuration, and generator version, PowerTrace-Sim estimates the
> scenario-conditioned distribution of facility power and ramps at 1 s, 1 min,
> and 15 min resolution.

These outputs are simulated scenario envelopes. They are not confidence bounds
for all data centers, and the Azure trace does not identify cross-server
correlation because it contains aggregate requests rather than server IDs.

## Evidence labels

Every result and artifact must use one of these labels.

| Label | Meaning |
|---|---|
| measured | Directly derived from recorded requests, timing, or GPU power |
| reconstructed | Internal work/state inferred from measured request timing |
| simulated | Produced from arrivals and a fitted timing/power model |
| scaled | Extrapolated to a different fleet size or configuration |

"Zero-shot" is reserved for a target whose power, timing, throughput, cap,
noise, and validation data were not used for fitting or model selection.

## Verified repository state

| Area | What is true now | Consequence |
|---|---|---|
| Azure node streams | `split_azure_requests_to_nodes` assigns each request to exactly one random node. It uses no offsets or duplication and conserves request and token totals. | Preserve this as `partitioned_replay`; do not describe it as decorrelation by offset. |
| Azure driver | `run_azure_pipeline.py` starts from prebuilt node streams; it does not parse a day or construct streams. | Monte Carlo orchestration must own parse -> traffic -> power -> aggregation in run-isolated paths. |
| Azure days | Seven raw day rows are listed, but only one parsed day is present in the inspected tree. | Parse and validate all eligible days before promising a seven-day grid. |
| Facility scaling | Repartitioning one fixed day over more nodes lowers per-node load. It does not create a comparable larger facility workload. | Define the offered-load scaling invariant before comparing facility sizes. |
| Facility power | The default config is TP8, but the current pipeline can pass TP4 to sizing metrics when `--tp-gpus` is omitted. | Fix and test config -> GPU count -> node/site nameplate accounting before using existing facility numbers. |
| GRU path | Train/eval/infer code and tests exist. Evaluation supports IID/AR modes and oracle first-activation alignment; standalone inference is IID. | Main comparisons need common, non-oracle generation and initialization semantics. |
| Timing surrogate | The rollout builder uses independent constant prefill/decode rates and omits queueing, batching, preemption, and saturation feedback. The checked appendix explicitly omits queue analysis. | It is not yet a validated state engine for the first-principles ledger. |
| First-principles path | `final_model.py` exports an older NNLS feature set and lag. The cap, saturating-bandwidth basis, changed priors, and FP8 transform used in holdout scripts are not exported as one model. | Create one canonical implementation and versioned artifact before simulator integration. |
| First-principles holdouts | Existing 405B and gpt-oss results reconstruct work from target-run TTFT/decode timing. Both also informed model development. | Treat them as retrospective conditional-timing case studies, not sealed end-to-end zero-shot tests. |
| Bundle ledger | Request-timing reconstruction exists. `bins_from_engine_csv` is intentionally unimplemented. | Cache-on/agentic state and phase attribution remain reconstruction-based until engine parsing is validated. |
| Profiling | Tier-1 and validation configs are schema-tested, not universally live-tested. A100 Llama-70B explicitly defers its second-TP communication probe. | Track readiness per campaign; do not call the full matrix ready to run. |
| Agent gaps | `gap_params.json` contains literature priors with zero fitted samples, not an OpenHands fit. | Fit and version the artifact or report a prior sensitivity study. |
| Test discovery | `uv run -m pytest -x` is constrained by `pyproject.toml` to `model/tests`. Profiling and `feature-test` tests require explicit paths. | Code touching those areas must run both the repository command and the explicit research/profiling suites. |
| PyPSA | PyPSA is not a project dependency and no grid case, solver, or procurement objective is specified. | Cut the PyPSA figure from the active plan. |

## Non-negotiable contracts

1. Preserve current public commands and the fixed-seed `partitioned_replay`
   output until an explicit retirement decision. New semantics use a new module
   or an opt-in mode with manifests.
2. Version every stochastic layer separately: traffic, request marks, power
   noise, and model sampling each get a recorded seed.
3. Keep power-kernel validation separate from end-to-end simulation. Measured
   TTFT/decode timing may validate the power law, but may not enter an
   arrival-only claim.
4. Fit coefficients, lag, caps, normalization, throughput/timing parameters,
   and noise on training data only. Freeze development and sealed test roles
   before collecting confirmatory campaigns.
5. Compare generators at a common 1 s resolution unless the first-principles
   dynamics are refit at 250 ms. The existing 250 ms GRU fidelity table remains
   a separate result. State ACF horizons in seconds.
6. Report failed, skipped, clipped, capped, overloaded, and out-of-support
   cases. Do not silently clip features to training support.
7. A paper artifact is reproducible only when its command, inputs, hashes,
   seeds, code revision, model artifact version, topology, power domain, and
   expected runtime are recorded.

## W0 - Freeze provenance and correct the current reference

This is the first implementation stage.

1. Record the current one-day, 240-node `partitioned_replay` command and its
   stream/trace manifests before changing traffic code.
2. Parse all seven Azure days into run-isolated paths and record source file
   hash, row count, UTC span, token totals, and rejection counts.
3. Establish one explicit reference mapping from the aggregate Azure trace to
   `N_ref` servers. Use `N_ref=240` only as a documented scenario assumption,
   not as Azure metadata.
4. Fix the TP8/TP4 facility sizing mismatch. Derive nameplate from the resolved
   config, GPUs per server, per-GPU limit, non-GPU overhead, and PUE. Never
   hard-code 0.75/7.5/75 MW labels from one generated trace.
5. Create a campaign-role registry with `fit`, `calibration`, `development`,
   `workload_holdout`, and `sealed_external_validation`. Existing 405B and
   gpt-oss holdouts are development evidence. Gemma validation campaigns that
   have matching tier-2 data are not architecture holdouts.
6. Add a campaign-readiness matrix: schema-valid, container built, weights
   staged, hardware feasible, dry-run passed, live smoke-tested, complete.

Acceptance:

- Fixed-seed legacy streams match the recorded golden hashes and conservation
  totals.
- Facility nameplate has a hand-worked TP8 test.
- All seven day manifests are command-complete and repo-relative.
- Sealed campaign IDs are committed before their power data are inspected.

## W1 - Define traffic and facility semantics

Keep two distinct traffic modes.

### `partitioned_replay`

This is today's behavior: each observed request is assigned once to a node.
It preserves the observed aggregate day exactly and provides one reference
replay, but it cannot vary site-wide demand synchronization independently.

### `correlated_intensity`

This is a synthetic sensitivity experiment, not a reconstruction of Azure
server behavior. The preferred starting model is a positive latent intensity:

```text
log lambda[i,b] = log mu[day,b]
                + sigma * (sqrt(rho) * z_site[b]
                         + sqrt(1-rho) * z_node[i,b])
                - normalization
```

`mu` is the documented day envelope. `z_site` and `z_node` are block-bootstrap
residual series whose temporal ACF is preserved. Conditional counts are drawn
from a stated point process, and `(n_in, n_out)` marks are sampled jointly from
the day. The aggregate-only Azure trace can guide the envelope and residual
shape, but cannot identify `rho` or decompose site and node variation.

`rho` is latent log-intensity correlation. At `rho=1`, node intensities share
the same burst signal; independent count noise still means arrivals are not
identical. If exact common events are studied, implement and name a separate
common-shock model.

Facility-size contract:

- At size `N`, scale expected total offered requests and joint token volume by
  `N/N_ref` so per-server offered load is comparable.
- Keep topology explicit: servers per rack, racks per physical row, row power
  limit, and whether each value is IT or facility power.
- The existing six-rack `FacilityLayout` row is not the old oversubscription
  figure's virtual 23-rack capacity group.
- For agent sessions, all turns remain on one assigned server unless migration
  is an explicit scenario.

Acceptance:

- `partitioned_replay` remains golden-output equivalent for fixed inputs/seeds.
- Ensemble mean request and token rates match the size-scaled targets; do not
  require every random node to be within 1%.
- Detrended 1 s and 1 min achieved rate correlations increase with `rho`, with
  uncertainty over sampled node pairs and seeds.
- Marginal rate variance, residual ACF, and joint token-mark distributions are
  reported at every `rho`; the correlation sweep must not silently change them.
- Separate traffic, mark, and power seeds reproduce byte-identical manifests.
- A builder -> parser -> stream integration test covers both raw Azure columns
  and normalized `arrival_time,n_in,n_out` columns.

## W2 - Build a deployable first-principles path

### W2a - Complete and validate measurements

Run tier-1 probes only after the readiness matrix passes. H100 already declares
a TP pair. A100 communication identification needs a separately scheduled
owners-partition TP run; the current A100 config says it is deferred.

Merge old and new bundles through one deterministic command. Bundle discovery
must handle `data/runs/<campaign>/<run>/`, use per-run throughput, preserve
bundle roles, and record exactly which bundles entered each fit.

Identifiability requires more than posterior contraction. Require cold-start
fits, synthetic parameter recovery, design condition/posterior correlations,
plausible prior drift, cross-probe prediction, and held-out probe levels.

### W2b - Consolidate the power model

Extract pure, tested functions for:

1. request timing -> per-bin work ledger;
2. ledger + architecture -> mean node power;
3. physical cap and meter lag at a stated `dt`;
4. optional stochastic residual generation.

Export one versioned deployment artifact containing feature equations,
coefficients and priors, family-multiplier policy, architecture schema, `dt`,
lag, cap and its provenance, training bundle IDs, fit revision, and validation
summary. Do not deploy `feature-test/results/final_coefficients.json` as-is.

Before transfer claims, audit and test prefill attention work, SWA/hybrid layer
ratios, MoE expert touches, KV accounting, node-total TP communication, unused
GPU power, and FP8 scaling. Use dimensional and hand-worked tests for each.

### W2c - Separate timing modes

- `measured_timing`: uses recorded TTFT/decode timing to validate the power
  kernel. Label outputs reconstructed.
- `arrival_only`: consumes only arrivals, token counts, configuration, and
  source-approved artifacts. This is the end-to-end simulator path.

The current constant-throughput rollout is only a candidate. Validate TTFT,
decode duration, active count, prefill/decode work, backlog, and completion rate
by load regime. Add scheduler/queue behavior or restrict the operating envelope
if saturation fails.

### W2d - Add stochastic residuals last

Freeze the deterministic model first. Fit mean-zero residual innovations on
training runs only, after deciding whether residuals live before or after the
meter lag. Validate conditional mean/variance, ACF, energy neutrality, cap
interaction, and seed reproducibility. Start with the smallest supported model;
utilization-decile AR parameters are not a requirement.

Acceptance:

- Legacy ledger parity tests pass where equations are intentionally unchanged.
- Synthetic work ledgers conserve tokens and recover hand-worked dense, MoE,
  TP, long-context, and hybrid-attention values.
- Engine-derived and reconstructed work agree within predeclared tolerances on
  real bundles before cache-on/agentic phase attribution is used.
- Kernel and end-to-end metrics are reported separately on held-out runs.
- One inference manifest fully identifies the deployed physics artifact and
  timing mode.

## W3 - Run a fair generator comparison

Create one immutable train/development/test manifest shared by both paths.
Refit all learned preprocessing and stochastic parameters on training data.
The main comparison must not use measured-power alignment or measured initial
power for one model only.

Report three questions separately:

1. In-configuration trace fidelity on held-out runs.
2. Power-kernel fidelity conditional on measured timing.
3. End-to-end transfer with target timing/power artifacts excluded.

Aggregate at 1 s and report KS, ACF R2 at a fixed time horizon, NRMSE, energy
error, P95/P99 error, cap rate, support violations, and failures. Use paired
seeds and confidence intervals clustered by trace/configuration, not bins.
Predeclare practical non-inferiority margins after development runs and before
opening the sealed test set.

The decision is per use case, not one global winner. The GRU may remain the
in-domain generator while the first-principles model supports only validated
fleet transfer. Existing 405B and gpt-oss results may appear as retrospective
case studies, not confirmatory rows.

Acceptance:

- Same requests, horizon, resolution, initialization, and metric code for both
  end-to-end paths.
- Train/development/test bundle IDs and every fitted artifact are auditable.
- The model-selection rule, margins, seeds, and failure handling are frozen
  before sealed evaluation.

## W4 - Pilot before Monte Carlo

### W4a - Scale pilot

Start with one day, 240 servers, three seeds, and three `rho` values. Then test
2,400 servers. Profile wall time, peak RAM, requests processed, and output size.
Do not enable 24,000 servers until a compressed/cohort method is numerically
equivalent to explicit simulation at smaller sizes and has a stated resource
budget.

The current full-day per-node feature stack and one-file-per-node aggregation
do not make the old 2,100-run grid credible. Use chunked accumulation. Retain a
1 s site trace for every accepted run, representative row/rack traces, and
daily summaries; full node traces are debug artifacts for small runs only.

### W4b - Statistical design

Treat the seven Azure days as seven observed scenarios. Twenty seeds nested
inside each day do not create 140 independent days. Keep per-day results and
use day-clustered intervals. Select replicate counts using a predeclared Monte
Carlo standard-error or interval-width target.

Do not report a conditional P99 from 20 seeds. Use P50/P95 and maxima until the
replicate count supports a stable P99, and label any quantile as conditional on
the observed days and selected `rho`.

For each run define:

- peak at 1 s and maximum rolling 1 min/15 min mean;
- signed maximum up/down and P95 absolute ramp for 1 s/1 min/15 min;
- energy, mean, load factor, backlog/completion/SLO, cap rate, and support rate;
- rack/row headroom only for a declared physical topology and power domain.

### W4c - Baselines and ablations

- Nameplate and constant training mean are sizing references, not stochastic
  traces.
- `partitioned_replay` is the current single facility replay.
- Independent circular/block shifts preserve each node's temporal structure
  while removing alignment.
- IID time shuffle destroys temporal and cross-node structure; label it an
  IID-marginal ablation.
- Noise-off isolates the deterministic power model.
- Keep a normal-sum calculation only for pointwise power quantiles. Mean and
  variance alone cannot produce daily maxima or ramp distributions.
- Keep LUT results only where support/fallback status is reported; do not imply
  a full LUT Monte Carlo grid unless it is actually run.

Acceptance:

- Streaming and explicit small-fleet aggregation agree numerically for peaks,
  ramps, energy, and hierarchy totals.
- The run schema records day, `N`, topology, `rho` semantics, all seeds, model,
  timing mode, power domain, and every summary metric.
- Runtime and interval convergence justify the final grid; otherwise reduce
  sizes, `rho` points, or replicates transparently.

Candidate Result 1 artifacts after the gate:

| Item | Content |
|---|---|
| Fig P1 | Conditional site peak/rolling-peak distribution versus `N` and `rho` |
| Fig P2 | Conditional 1 s/1 min/15 min ramp summaries with baseline ablations |
| Table P3 | P50/P95 peak, ramp, energy, load factor, SLO/backlog, cap/support rates |
| Appendix | Row headroom sensitivity, pointwise normal-sum check, scale convergence |

## W5 - Workload scenarios

Agentic traffic cannot be reduced to the three-column Azure schema without
losing session affinity and cache semantics. Use an extended request schema
with `session_id`, `turn_index`, accumulated context, new input tokens, output
tokens, tool class, gap source, and cache regime. The generic parser may project
this to three columns only for cache-off tests that do not claim session state.

Prerequisites:

1. Pin SWE-smith dataset and tokenizer revisions and cache hashes.
2. Either fit OpenHands gaps and record per-class sample counts/fit revision, or
   retain literature priors and run an explicit prior sensitivity analysis.
3. Create and version the reasoning-length source artifact; it does not exist as
   a documented distribution today.
4. Live-smoke-test one agent replay and validate context-window truncation.

Primary workload comparison: reuse the same external arrival envelope and
equal request/turn count, then report total input tokens, output tokens, and
offered work so the extra work is visible. Add an equal-output-token sensitivity
instead of implying the scenarios provide identical service.

Keep session turns on one server. Study cache off first. Cache-on results and
prefill/decode attribution require validated engine-state parsing and measured
cache behavior. Agent traffic changes arrival gaps, token lengths, context,
placement, and potentially cache work; do not say it changes only arrivals.

Acceptance:

- Deterministic builder -> parser -> placement tests cover 24-hour span, session
  order/affinity, token totals, context limits, and source provenance.
- Workload tables report demand normalization and offered/served work.
- Out-of-support prompt/context fractions are reported, never silently clipped.

## W6 - Fleet and transfer scenarios

The current facility driver applies one config to every node. Add heterogeneous
node manifests, routing, capacity/backlog accounting, and unused-GPU/extra-
replica power before fleet mixtures.

Every sweep must state its invariant:

- A100 -> H100: fixed offered requests, plus a separate equal-served-work or
  equal-SLO provisioning comparison.
- 70B/405B mix: fixed routing policy and demand split, with model-specific
  capacity and support checks.
- Tensor parallelism: state whether server count, total GPU count, replicas, or
  SLO is held fixed and account for unused GPUs.
- External validation: distinguish workload, architecture, hardware, and TP
  holdouts. A campaign used to tune the model is no longer sealed validation.

Architecture-only transfer is not established if target throughput or measured
execution timing is required. Label those results conditional-timing transfer.
Run a true zero-shot row only after W2's source-only timing path passes.

Acceptance:

- Routing/request conservation, placement, capacity, overload, and power
  accounting have hand-worked heterogeneous-fleet tests.
- Each validation row names all target artifacts that were excluded from fit
  and model selection.

## W7 - Grid example

Cut PyPSA from the active plan. Reconsider only if a grid collaborator defines
a specific network, assets, costs, solver, objective, comparison, and analytic
sanity case. Do not add a dependency to produce a tautological max-load sizing
figure.

## W8 - Paper and artifact update

Paper edits happen after result gates, not in parallel with unstable semantics.

1. Replace existing offset/shared-intensity prose with the implemented traffic
   contract and mark `rho` as a synthetic sensitivity parameter.
2. Separate measured timing reconstruction from arrival-only simulation.
3. Replace "bounds" and broad zero-shot language with the evidence labels in
   this plan.
4. State resolution, power domain, demand normalization, topology, failure
   counts, support limits, and uncertainty unit next to each result.
5. Verify regulatory and queue claims against primary, jurisdiction-specific
   sources. Do not use "regulators now require" without a precise citation.
6. Add exact commands and upstream manifests to `results/eval_paper/README.md`.
7. Compile the manuscript with no undefined references/citations and regenerate
   every referenced number.

## Figure-to-command policy

Do not publish speculative flags as commands. A row becomes `ready` only after
the CLI exists, its integration test passes, and the artifact map contains the
full invocation.

| Paper item | Producer | Status |
|---|---|---|
| Existing server fidelity figure | `uv run -m scripts.eval.run_baselines_node_groundtruth` | Existing; preserve |
| Existing trace fidelity table | `uv run -m scripts.eval.generate_trace_fidelity_table ...` | Existing; preserve |
| Model comparison | `model.scripts.compare_generators` | Proposed; blocked on W2/W3 |
| Fig P1/P2 and Table P3 | one new Monte Carlo orchestrator plus renderer | Proposed; blocked on W1-W4 pilot |
| Existing hierarchy figure | `uv run -m scripts.eval.hierarchy_figure` | Existing; preserve |
| Workload comparison | one workload orchestrator plus renderer | Proposed; blocked on W5 |
| Fleet comparison | one fleet orchestrator plus renderer | Proposed; blocked on W6 |

## Schedule and cut order

The official e-Energy 2027 CFP was not available in the official site search on
2026-07-09. September 2026 and January 2027 are planning assumptions based on
the 2026 fall/winter cadence, not confirmed deadlines. Verify the 2027 CFP and
resubmission policy before choosing a cycle:
<https://energy.acm.org/conferences/eenergy/2026/pages/cfp.php>.

Sequence by evidence gate, not optimistic calendar overlap:

| Phase | Work | Exit condition |
|---|---|---|
| 0 | W0 provenance/correctness + W1 design | Reference replay frozen; scaling/correlation contract approved |
| 1 | W2 measurements, ledger audit, canonical artifact | Kernel and timing gates pass on held-out development data |
| 2 | W3 fair comparison | Model role frozen before sealed evaluation |
| 3 | W4 scale pilot, then justified final grid | Runtime and Monte Carlo convergence recorded |
| 4 | W5 workload scenarios; W6 only if transfer gate passes | Scenario claims stay inside validated support |
| 5 | W8 paper/artifact regeneration and internal review | Every claim maps to a command and manifest |

Cut in this order when time is constrained:

1. PyPSA remains cut.
2. Cut broad external-model transfer before weakening sealed validation.
3. Cut cache-on phase attribution if engine-state parsing is not validated.
4. Cut 24,000 servers before using an unvalidated compressed simulator.
5. Reduce `rho` points and tail quantiles before pooling dependent seeds.

The minimum viable paper is the existing server-fidelity evidence, a corrected
and reproducible 240-node replay, one clearly synthetic traffic-correlation
sensitivity at validated scale, and an honest limitations section.

## Cleanup relationship

Cleanup is governed by `cleaning-plan.md`; it is not a scientific workstream.
Source-control noise may be removed in separate commits, but do not archive an
active producer before its replacement lands.

In particular, `occupancy_roofline.py` is called by the current roofline
campaign and documented in `README.md`; moving it to an attic would break an
active workflow. Before untracking large Azure artifacts, retain compact source
and stream/trace manifests, exact commands, checksums, topology/seeds, and prove
regeneration from a fresh checkout. There is no
`results/azure_facility/manifest.json` to retain as previously claimed.

Required test gates for code changes in these areas:

```bash
uv run -m pytest -x
uv run -m pytest -x model/tests profiling feature-test
```
