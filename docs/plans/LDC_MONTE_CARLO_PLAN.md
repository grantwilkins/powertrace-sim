# LDC_MONTE_CARLO_PLAN: Reference-backed Monte Carlo LDCs for LLM inference fleets

Drafted 2026-07-23 against the audited repository state. This is the
load-duration-curve evaluation that follows the trace and power validation in
`EENERGY_PLAN.md`. It does not reopen that validation campaign. It asks whether
validated request traces can support credible, useful distributions of
site-level load-duration curves (LDCs).

The paper succeeds only if it demonstrates the complete chain

> held-out traffic fidelity -> held-out LDC skill -> materially different
> scenario-conditioned IT design demand.

The final 100--200 MW results are conditional projections. They are not
measurements, forecasts of a real site, transformer ratings, or interconnection
determinations.

## 1. Paper story and research questions

The paper has three sequential results.

1. **Trace and power validity.** PowerTrace matches held-out measured LLM
   execution traces and transfers to supported unprofiled workloads and
   deployments with only the changes permitted by `EENERGY_PLAN.md`.
2. **Traffic-to-LDC validity.** A site-level traffic generator first reconstructs
   the original seven-day Azure case, then predicts LDCs induced by unseen,
   nonoverlapping BurstGPT weeks. The comparison target is the unseen empirical
   request trace passed through the frozen, previously validated PowerTrace
   translator.
3. **Conditional planning use.** The validated generator samples site arrivals,
   routes them to nodes, and produces annual LDC distributions under explicit
   workload, load, routing, calendar, and fleet scenarios. These distributions
   determine the hourly IT active-power level exceeded for at most 22, 44, or
   88 hours in a conditional design year.

These become four falsifiable research questions.

- **RQ1:** Does the frozen trace-to-power model reproduce held-out execution and
  power behavior over the support used by the LDC study?
- **RQ2:** Does the traffic generator predict empirical-reference weekly LDCs
  better than strong replay, block-resampling, point-process, and external
  workload-generator baselines?
- **RQ3:** At an exactly simulated 20 MW-class fleet, how do demand level and
  chat/reasoning/agentic composition change the distribution of required hourly
  IT design demand?
- **RQ4:** Which conclusions survive projection to 100--200 MW and sensitivity
  to routing, cross-pool synchronization, seasonality, and PUE?

RQ1 is established by the first part of the paper and is a prerequisite for
RQ2--RQ4. The Monte Carlo evaluation must not substitute more self-consistency
tests for RQ2.

## 2. Claims, evidence labels, and boundaries

Every result carries one of four evidence labels.

| Label | Meaning |
|---|---|
| `measured request` | Request timestamps, marks, or sessions came from an observed trace. |
| `model-referenced` | Power came from a measured request trace passed through frozen PowerTrace and a declared router. |
| `exact simulated` | Every node in the declared fleet was simulated through PowerTrace. |
| `scaled projection` | Fleet power used a composition approximation gated against exact PowerTrace runs. |

The central RQ2 target is an **empirical-reference, model-referenced LDC**:
requests are observed, but power is produced by the already validated simulator.
This is legitimate for comparing traffic generators because every method uses
the same translator, census, and router. It is not a site-boundary power
measurement.

The annual product is a **conditional design distribution**. Within one fixed
structural scenario, Monte Carlo quantiles describe randomness in arrivals,
marks, sessions, routing, and fitted residual processes. Routing policy,
workload shares, synchronization structure, seasonality, PUE, and fleet census
are structural assumptions and are never pooled into one percentile band.

The paper does not claim:

- a calibrated annual forecast for Azure, BurstGPT, or a real datacenter;
- physical validation of a 100--200 MW inference facility;
- transformer, feeder, or interconnection capacity;
- curtailment capability, flexible-load headroom, or demand response;
- reactive-power, cooling, UPS, protection, contingency, or grid-coincidence
  behavior;
- that Q95 is a confidence interval or a 95% reliability guarantee.

## 3. Prior art and e-Energy positioning

The old separation between "workload without power" and "power without
workload" is too strong. The revised positioning acknowledges direct overlap.

- **Workload generation.** ServeGen models client and session behavior, model
  mix, token distributions, diurnal variation, and burstiness. BurstGPT
  publishes 110--121-day production chronologies with timestamps, request and
  response tokens, model, log type, and session IDs.
- **Serving and power.** Splitwise, POLCA, DynamoLLM, and TAPAS show that model,
  request length, batching, offered load, and serving policy materially affect
  power.
- **Whole-facility simulation.** The NLR study, *Measurement of Generative AI
  Workload Power Profiles for Whole-Facility Data Center Infrastructure
  Planning*, combines measured hardware profiles with DIPLOEE to generate
  minute-level annual facility traces. It is the closest system, not evidence
  of a gap between the literatures.
- **Grid-facing simulation.** FlexDC-Sim and related e-Energy work demonstrate
  that measured component behavior can support carefully bounded scaled
  simulation, while also showing that flexibility claims require a controller,
  service contract, rebound behavior, and QoS evaluation that are outside this
  paper.

The closest evaluation precedent is the e-Energy 2019 paper *Using Synthetic
Traces for Robust Energy System Sizing*. Its generators are judged by decisions
made from training data and outcomes on future real years, rather than by
visual plausibility. This paper follows that lesson at the layer for which
future observations exist: unseen empirical request weeks and their
model-referenced LDCs. The annual MW-scale outputs remain explicitly
conditional.

The defensible contribution is:

> Reference-backed probabilistic LDC generation for heterogeneous LLM
> inference workloads, demonstrating how traffic modeling and workload
> composition change scenario-conditioned IT active-power requirements.

LDC sorting is not itself novel. The novelty must come from request-semantic
traffic generation, an explicit router and fleet translator, held-out LDC
evaluation, and a downstream capacity-risk quantity.

## 4. Data hierarchy and frozen splits

### 4.1 Azure: original conditional reconstruction

The seven Azure days retain only:

- normalized Monday--Sunday time-of-day envelopes;
- short-range residual, burst, and autocorrelation shape;
- the original observed request trace used for a descriptive reconstruction
  target.

Azure no longer supplies current absolute request volume, current model mix, or
modern token distributions.

For the reconstruction task, every method receives the observed normalized
weekday envelope and total daily volume. The fitted joint mark law and residual
model may use the full seven-day case because this is a posterior-predictive
reconstruction, not a held-out forecast. The evaluation reports request, token,
traffic-shape, and derived-LDC closure without claiming weekly calibration.

### 4.2 BurstGPT: primary held-out weekly evidence

Use the full failure-free v2 chronology rather than the three existing
15-minute campaign windows.

- Concatenate `BurstGPT_without_fails_1.csv` and
  `BurstGPT_without_fails_2.csv` into the 121-day development chronology after
  verifying timestamp continuity and source metadata.
- Use the first 14 complete weeks for fitting and the next three complete weeks
  for model selection and frozen tolerance construction.
- Exclude the final two incomplete days from scoring.
- Use fifteen complete, nonoverlapping weeks from
  `BurstGPT_without_fails_3.csv` as the sealed test set.
- Exclude its final five incomplete days.
- Retain the files containing failed requests only for a sensitivity analysis;
  zero-response-token rows are not silently assigned executable output marks.

For each sealed week, all methods receive:

- weekday and weekend identities;
- the declared total weekly request volume;
- the same training and development histories.

They do not receive:

- the target hourly envelope;
- target residual or burst paths;
- target token-mark marginals;
- target sessions or per-client paths.

The week is the statistical unit. LDC ranks, hours, requests, and routing seeds
are not independent replicates.

### 4.3 Modern workload marks and sessions

Use source-specific distributions rather than inventing one "current industry
mix."

| Workload family | Source | Role |
|---|---|---|
| Interactive chat | ServeGen released chat/client distributions | Modern annual mark and session preset |
| Reasoning | ServeGen reasoning/model-specific distributions | Modern annual reasoning preset |
| Agentic | OpenHands traces and fitted inter-call/session structure | Constructed agentic stress scenario |
| Bursty production traffic | BurstGPT long chronology | Empirical arrival validation and burst stress |
| Weekday temporal shape | Seven Azure days | Normalized reference calendar shape only |

BurstGPT's GPT-3.5/GPT-4 marks are valid historical held-out observations but
are not described as the modern deployment mix. ServeGen output is an external
generated reference/baseline, not empirical truth. OpenHands superpositions are
constructed scenarios, not observations of a production site's annual agentic
traffic.

## 5. Facility, workload, and demand contracts

### 5.1 Facility boundary

The primary output is **IT active power**. The primary input is an explicit
node census:

```text
pool = {
    deployment_preset,
    node_count,
    workload_family,
    router
}
```

Paper anchors:

- approximately 240 nodes / 2 MW-class aggregate installed IT reference;
- approximately 2,400 nodes / 20 MW-class aggregate installed IT reference;
- 100 MW and 200 MW composition projections.

The artifact reports the exact node count and aggregate installed IT reference;
"2 MW" and "20 MW" are readable scenario names, not exact equipment ratings.
MW-to-census conversion is a convenience helper, not the authoritative input.

The exact 20 MW-class, 60%-demand scenario carries the main scientific
conclusion. The 2 MW case supplies audit detail. The 100 MW and 200 MW cases
communicate potential grid-scale consequence and always display their numerical
approximation bound.

PUE is excluded from the primary IT-power result. A constant PUE of 1.3 may be
applied as a separately labeled facility-boundary sensitivity; it is not mixed
into the primary ensemble.

### 5.2 Named deployment presets

Do not run a factorial workload x model x hardware grid. Bind each workload to
a named preset that passed the Stage 1 support and transfer gates.

- `interactive_chat_dense`: expected Qwen3-8B on A100.
- `reasoning_moe`: expected Qwen3-30B-A3B on H100.
- `agentic_cache_off`: OpenHands sessions on the validated cache-off
  deployment.

The final artifact IDs come from the frozen Stage 1 model bundle. If a candidate
preset does not pass, it cannot be promoted by the Monte Carlo evaluation.
Agentic stays an exploratory sensitivity unless its transfer gate passes.

### 5.3 Demand anchor and workload replacement

Measure the chat-only fleet's aggregate SLO-feasible service capacity using the
same Stage 1 SLO. Convert 30%, 60%, and 85% of that capacity into absolute
top-level arrival rates once, then hold those rates fixed across mix sweeps.

"30/60/85%" means offered top-level arrival rate relative to the chat-only
SLO-feasible anchor. It does not mean GPU utilization, load factor, completed
throughput, or capacity factor.

Run separate one-axis replacement sweeps:

- reasoning replaces chat at `{0, 25, 50, 75, 100}%` of top-level jobs;
- agentic replaces chat at `{0, 10, 25, 50}%` of top-level jobs.

A reasoning job replaces one chat request. An agentic job replaces one chat
request with one session start; dependent backend calls produced by the session
are retained and reported. They are not capped to make agentic work resemble a
single request.

All top-level jobs are admitted into one continuous chronology:

- no dropped or rejected work;
- no queue reset at day, week, or year boundaries;
- generate one unscored warm-up week before every scored weekly or annual
  interval, carry its queue state into the scored interval, and simulate one
  unscored drain week afterward to measure remaining work;
- offered, completed, unfinished, and out-of-support work are conserved and
  reported.

Every scenario cell reports completed request and token throughput, queue growth,
unfinished work, TTFT/TBT or the Stage 1 SLO, and support violations. An unstable
queue or SLO failure is visibly hatched and labeled `overload stress`; it is
excluded from normal-design conclusions. This prevents a heavy workload from
appearing electrically favorable merely because work remains queued.

The paper shows:

1. an equal-offered-top-level-rate comparison as the operational result; and
2. a post-hoc mean-power-normalized LDC comparison that isolates duration-curve
   shape from average power.

The normalized view is descriptive and is not used for capacity values in MW.

## 6. Site traffic generation and routing

Generate traffic at the site/workload-pool level before assigning requests to
nodes.

For pool `p` and time `t`, the proposed generator consists of:

1. a normalized weekday envelope;
2. a declared pool volume and workload share;
3. an overdispersed site-level residual process generated in contiguous weekly
   blocks;
4. joint request marks conditioned on workload family and coarse time stratum;
5. client/session unfolding where the source supports it.

The intensity is mean-normalized so residual scale changes temporal shape, not
total conditioned volume. Input and output tokens are sampled jointly.
Conversation/session identity is preserved where available.

Reference cross-pool residual processes are independent. A shared common-site
factor is a separate synchronization sensitivity. There is no per-node `rho`
parameter. Cross-node dependence emerges from the common site arrival stream,
the router, batching, queues, and serving behavior.

Routing policies:

- uniform random routing is the headline;
- least-loaded routing is a structural sensitivity.

The empirical-reference trace and every generated method use identical routing
rules and eight predeclared common random-number routing seeds per week. Scores
are averaged over those seeds within a week; the week remains the replication
unit. Each method produces 200 traffic ensembles per sealed week so finite
ensemble size is identical.

## 7. Baselines and comparison target

Hold the PowerTrace translator, fleet census, router, conditioning information,
training windows, workload support, and ensemble budget fixed while changing
only the traffic generator.

### 7.1 Deployable traffic baselines

1. **Fixed training replay.** Repeat a training week selected without target
   information and rescale only by the supplied weekly volume.
2. **Joint site-level block bootstrap.** Resample chronological count, session,
   and mark blocks together. This is the strongest nonparametric baseline and
   must not be weakened by independent mark sampling.
3. **Calendar-conditioned NHPP.** Fit weekday/time-of-day intensity from
   training data and retain joint conditional marks, but omit residual
   overdispersion and session dependence not implied by the NHPP.
4. **ServeGen-aligned generator.** Use the official client/session structure
   wherever the training source exposes the necessary fields, with no
   target-week fitting.
5. **Proposed generator.** Use the site residual, joint marks, sessions, and
   explicit router defined in Section 6.

The unseen empirical trace routed through PowerTrace is the oracle/reference,
not a deployable baseline. Flat mean and aggregate installed IT rating are
planning references, not peer scientific baselines. DIPLOEE is a closest-work
comparison at the power/facility layer; it is not inserted into RQ2 unless it
can consume exactly the same requests and conditioning information.

### 7.2 Required ablations

Change one property at a time:

- joint versus independently shuffled input/output marks;
- weekly residual blocks versus shuffled residuals;
- independent versus shared cross-pool residual;
- uniform versus least-loaded routing;
- exact versus cohort-composed fleet;
- IT power versus constant-PUE facility projection;
- zero-season versus declared seasonal modulation.

Every ablation reports both its LDC score and its change in the design-demand
quantity. The independent-mark case is an ablation, not the headline Poisson
baseline.

## 8. LDC definitions and held-out scoring

### 8.1 Weekly LDC target

For sealed week `j`, route its observed requests through the frozen translator
to obtain reference LDC `L*_j(u)`. Method `m` produces an equal-sized ensemble
`L_mjr(u)` on a fixed exceedance-rank grid.

Primary score:

- integrated marginal CRPS, equivalently integrated quantile loss, over the
  complete rank grid.

Secondary scores:

- tail-weighted integrated CRPS over the upper 10% of ranks;
- functional ensemble energy score over the discretized LDC;
- central predictive-band coverage and width;
- integrated absolute, Wasserstein-style distance between the predictive
  median and reference LDC;
- peak, energy, and capacity errors at weekly-supported ranks of 2, 8, and
  17 hours.

The whole-curve score prevents selecting favorable ranks after seeing results.
The separate tail score prevents central ranks from hiding peak errors. A wide
band does not pass merely because it covers the reference: coverage and
sharpness are reported together.

Compare methods using paired differences for the same sealed week and routing
seeds. Report every week, mean and median paired skill relative to joint block
bootstrap, and week-clustered uncertainty. Do not treat ranks or requests as
replicates. Azure remains one descriptive seven-day reconstruction and is not
included in weekly calibration counts.

### 8.2 Closure and superiority gates

Freeze tolerances on the development chronology before revealing
`BurstGPT_without_fails_3.csv` results.

- For each power-valued closure metric, use the development 95th-percentile
  reconstruction error as the equivalence margin.
- Cap the permissible margin at 2% of aggregate installed IT reference. If
  development variability requires a wider margin, the method is not precise
  enough to claim equivalence.
- Apply the analogous 2% relative cap to energy.
- Report absolute MW and MWh errors even when equivalence passes.

Equivalence and superiority answer different questions. The proposed generator
must both pass the closure gate and improve the prespecified full or tail score
over the joint block bootstrap to support a complexity claim. Failure to
establish equivalence is not reported as proof of nonequivalence.

## 9. Conditional annual design distributions

### 9.1 Calendar construction

The reference calendar contains the seven normalized Azure Monday--Sunday
envelopes. The headline uses the fixed 2025 non-leap calendar: 52 complete
weeks plus the extra Wednesday, preserving the observed weekday identities and
producing exactly 8,760 hourly observations.

Residual traffic is generated or sampled in contiguous seven-day blocks
calibrated on the training chronology. Do not draw 365 independent days:
sorting removes ordering from one LDC, but multi-day persistence still changes
the across-year tail distribution.

There is no fitted full-year season model because neither Azure nor BurstGPT
observes a full annual cycle. Use separate structural cases:

- zero-season reference;
- mean-one sinusoidal daily-volume modulation with 10% amplitude and summer
  peak;
- 10% amplitude and winter peak;
- 20% amplitude and summer peak;
- 20% amplitude and winter peak.

The sinusoid is normalized to preserve annual offered volume. Seasonal cases
are sensitivities and are never pooled with the zero-season ensemble.

### 9.2 Capacity-risk quantity

For annual hourly IT trace `x`, define

```text
C_k(x) = inf { c : count_t(x[t] > c) <= k }
```

for `k in {22, 44, 88}`. With descending hourly values, this is the `(k + 1)`th
order statistic using strict exceedance and no interpolation.

For annual replicate `r` within structural scenario `s`, report:

```text
C50_k(s) = Q0.50[C_k(x_r) | s]
C95_k(s) = Q0.95[C_k(x_r) | s]
```

Plain-language interpretation:

> Under this declared scenario, only 5% of simulated design years require more
> than `C95_22` MW of hourly IT demand to limit observations above that level
> to at most 22 hours.

This is a conditional ensemble quantile, not empirical future-site coverage.

From the same chronological replicates, retain four secondary metrics above a
selected threshold:

- excess MWh;
- maximum excess MW;
- number of excess events;
- longest consecutive excess event.

These preserve the distinction between isolated high hours and a sustained
event. They do not establish a curtailment policy, because the simulator does
not clip demand, defer work, or model rebound.

### 9.3 Monte Carlo convergence

Expensive traffic/node simulation and cheap year composition have separate
stopping rules.

- Build each exact or cohort week library in batches of 10, with at least 30
  replicates for central claims and at least 100 for any Q95/tail claim.
- Stop a library cell only after three consecutive batches keep the induced
  `C50_k`, `C95_k`, peak, and excess-energy summaries inside their declared
  tolerances under annual recomposition.
- Compose annual replicates in batches of 1,000, with a minimum of 2,000.
- Stop annual composition when the 95% Monte Carlo interval half-width is at
  most 0.25% of aggregate installed IT reference for every reported capacity
  and peak quantity, and at most 1% of the estimated mean for excess energy.
- Cap annual composition at 100,000 replicates and display unresolved Monte
  Carlo error rather than hiding it.

Report measurement/translator error, fitted-parameter uncertainty, finite
library sensitivity, Monte Carlo error, and structural scenario variation
separately. A half-library refit/recomposition is mandatory for every headline
cell.

## 10. Exact fleets and 100--200 MW composition

The expensive library record is a simulated facility week. It stores:

- scenario and source provenance;
- site and per-pool arrivals, completed work, queues, and support counters;
- site IT power at 1 second and hourly resolution;
- exact cohort-node traces needed to audit composition;
- shared residual paths, seed tree, and manifest hash;
- summary and convergence rows.

Native 250 ms data remains transient inside the streaming accumulator. The
hourly tier is the headline LDC input. One-second and five-minute power/ramp
outputs are secondary trace-fidelity diagnostics, not equal-weight grid
products.

Run every node exactly for the 2 MW- and 20 MW-class anchors. Beyond the exact
20 MW-class fleet, fit per-pool conditional cohort distributions and compose
larger censuses. The preferred approximation is conditional Gaussian
composition; if its gate fails, fall back to empirical cohort-block resampling.

Validate composition against exact 240- and 2,400-node runs using common seeds:

- energy error at most 0.5%;
- peak, `C50_k`, and `C95_k` error at most 0.5% of aggregate installed IT
  reference;
- predictive-band width error at most 10%;
- no systematic queue, throughput, or support-count discrepancy.

Do not loosen the gate. Agreement proves only that composition approximates
PowerTrace over the tested regimes. It is not independent evidence for the
traffic law or a physical 200 MW site.

## 11. Implementation architecture

Reuse the `EENERGY_PLAN.md` traffic, translator, accumulator, and orchestration
boundaries. Add the minimum new interfaces below; exact filenames may reuse an
existing coherent module rather than create a duplicate.

### 11.1 Scenario manifest

One versioned manifest contains:

```text
scenario_id
evidence_label
source_split
node_census[]
deployment_presets[]
top_level_load_anchor
workload_replacement
router
cross_pool_residual_mode
calendar
season_case
pue_case
seed_root
library_stopping_rule
annual_stopping_rule
```

It must distinguish fitted stochastic parameters from structural scenario
choices. A loader rejects manifests that attempt to pool routers, season cases,
PUE cases, synchronization modes, or evidence classes.

### 11.2 Producers and pure consumers

- **Trace adapters:** parse Azure, all BurstGPT v2 files, ServeGen pools, and
  OpenHands sessions into one marked-arrival/session schema with source hashes.
- **Traffic producer:** generates site-level marked arrivals and session calls;
  it does not know node power.
- **Router:** maps marked arrivals to a declared census and emits auditable
  placement records.
- **Power producer:** calls frozen `prepare_simulation` and the streaming
  PowerTrace iterator; it does not fit traffic parameters.
- **Facility accumulator:** consumes node streams without materializing the
  full fleet tensor and emits exact site sums, hourly means, and counters.
- **Week-library builder:** stores exact or cohort records and convergence
  metadata.
- **Year composer:** consumes hourly week records only and emits chronological
  years, LDCs, capacity quantities, and excess-event summaries.
- **Scorer:** compares predictive weekly ensembles with empirical-reference
  weeks; it never renders.
- **Renderer:** reads frozen artifacts and never simulates, fits, or changes
  scenario weights.

Every random choice derives from one named `SeedSequence` tree covering traffic
residuals, counts, marks, sessions, placement, power residuals, composition,
calendar blocks, and annual assembly.

## 12. Build order and falsifiable gates

| Phase | Work | Exit gate |
|---|---|---|
| 0 | Use the Stage 1 PowerTrace artifact IDs, SLO, support rules, and seven parsed Azure days. | Existing model tests pass; all input hashes and evidence labels resolve. |
| 1 | Add streaming accumulator, LDC/capacity summaries, and deterministic fake-power integration path. | Hand-worked multi-pool power, LDC, `C_k`, energy, and events are exact; existing Azure aggregation is reproduced. |
| 2 | Ingest full BurstGPT v2 chronology and freeze train/development/test weeks. | Counts, marks, sessions, timestamps, excluded failures, continuity, and split nonoverlap reconcile to source metadata. |
| 3 | Implement site traffic generator, workload pools, continuous queue, and routers. | Volume/mark/session conservation, determinism, support reporting, and no-boundary-reset tests pass. |
| 4 | Implement fixed replay, joint block bootstrap, NHPP, ServeGen-aligned baseline, common conditioning, and weekly scorer. | No target leakage; identical conditioning and ensemble budgets; shifted synthetic negative control fails. |
| 5 | Run Azure reconstruction. | Traffic and model-referenced LDC closure are reported with development-frozen tolerances and honest descriptive labeling. |
| 6 | Run the sealed fifteen-week BurstGPT comparison. | Proposed method passes closure and its paired full/tail score relative to joint block bootstrap is reported without rank-level pseudoreplication. |
| 7 | Build exact 2 MW- and 20 MW-class libraries for feasible 30/60/85% and mix cells. | Every cell is convergence-certified or labeled unresolved/overload; SLO and unfinished work accompany power. |
| 8 | Gate cohort composition and produce 100/200 MW projections. | Exact-composition gates pass; otherwise empirical cohort-block fallback passes or projections are cut. |
| 9 | Compose zero-season and declared seasonal years. | Calendar, annual volume, weekly dependence, energy, `C_k`, and stopping-rule tests pass. |
| 10 | Render paper artifacts from sealed run directories. | Every figure/table carries source, conditioning, census, power boundary, SLO feasibility, exact/scaled status, and scenario assumptions. |

## 13. Paper artifacts

The main result is intentionally narrow and readable.

### 13.1 Headline figure

Two panels:

1. **Does it work?** One sealed empirical-reference weekly LDC overlaid with
   predictive bands from fixed replay, joint block bootstrap, NHPP,
   ServeGen-aligned generation, and the proposed method. Include full/tail
   proper-score insets rather than visual coverage alone.
2. **Why does it matter?** Exact 20 MW-class, 60%-demand hourly IT design
   demand versus allowed annual exceedance hours, with Q50/Q95 curves for the
   proposed method and strongest baseline.

### 13.2 Baseline scoreboard

For each method report:

- full and tail integrated CRPS;
- functional energy score;
- coverage and band width;
- peak, energy, and weekly supported-rank error;
- difference in annual conditional `C50_k` and `C95_k`;
- runtime and required input data.

This table answers both "why is this more correct than another generator?" and
"does the difference change the result somebody would use?"

### 13.3 Scenario results

- Heatmap of 30/60/85% demand by reasoning replacement, with SLO-infeasible
  cells hatched.
- Separate agentic heatmap, promoted to the main paper only if its transfer
  gate passes.
- Mean-power-normalized LDC-shape comparison.
- Uniform versus least-loaded routing sensitivity.
- Independent versus common-site residual sensitivity.
- Zero-season versus +/-10% and +/-20% summer/winter seasonal cases.
- Exact/cohort validation and secondary 100/200 MW projection.
- Excess MWh, maximum excess MW, event count, and longest event for the
  selected headline threshold.

Every headline table includes the exact census, aggregate installed IT
reference, offered and completed work, achieved SLO, router, synchronization
case, evidence label, `C50_k`, `C95_k`, and difference from each intelligent
baseline in MW and percent.

## 14. Tests and reproducibility

New behavior requires the smallest semantic tests that would catch believable
research errors.

- Request, session, token, completed-work, queue, and energy conservation.
- Continuous queue state and warm-up/carry-in across day, week, and year
  boundaries.
- Unstable or SLO-failing cells cannot enter normal-design summaries.
- Joint mark and session dependence are preserved by the proposed and block
  baselines.
- Uniform and least-loaded routers conserve requests and are deterministic
  under a seed.
- Baselines receive identical conditioning and cannot read target-week fields.
- BurstGPT file continuity, failure filtering, source totals, and disjoint
  chronological splits.
- Hand-worked LDC order statistics, strict ties, `C_k`, Q50/Q95, integrated
  CRPS, energy score, and excess-event cases.
- Azure positive reconstruction and deliberately shifted negative control.
- Structural scenarios cannot be pooled into one ensemble.
- Weekly block composition preserves calendar counts and multi-day blocks.
- Seasonal multipliers have the declared amplitude and mean one.
- Annual energy equals chronological hourly energy before LDC sorting.
- Seed/provenance round trips reproduce the same artifacts.
- Streaming accumulation matches the existing Azure aggregator.
- Conditional Gaussian and empirical cohort composition are tested against
  analytic and exact small-fleet cases, including the `N >> K` variance failure.
- Monte Carlo stopping is tested against analytic distributions and reports an
  unresolved cap.

For every code phase:

```bash
uv run -m pytest -x
```

also run the smallest affected producer and verify its primary user-visible
artifact. Update the root README only when setup, inputs, commands, outputs, or
assumptions change.

## 15. Compute budget and cut order

Calibrate cost from the exact 2 MW path before producing the complete matrix.
Cache simulated weeks by traffic law and deployment preset; annual recomposition
is cheap and must never rerun node power.

Preserve, in order:

1. Azure reconstruction;
2. full BurstGPT sealed weekly comparison;
3. joint block-bootstrap and ServeGen-aligned baselines;
4. exact 20 MW-class, 60%-demand headline;
5. Q50/Q95 capacity and SLO feasibility;
6. deterministic provenance and convergence reporting.

Cut, in order, if compute or paper space is insufficient:

1. 200 MW, then 100 MW projections;
2. agentic headline promotion;
3. PUE and seasonal sensitivities;
4. one-second/five-minute secondary diagnostics;
5. non-headline 2 MW scenario plots.

Never cut the held-out weekly comparison, strong block baseline, exact
20 MW-class result, SLO/unfinished-work reporting, composition gate, or
conditional-claim language.

## 16. Main risks and required wording

1. **Model-referenced target.** The empirical requests are real; facility power
   is translated by PowerTrace. State this in every RQ2 result.
2. **Historical source semantics.** Azure and BurstGPT validate temporal and
   mark structure in their observed domains; they do not define a universal
   modern workload mix.
3. **Agentic construction.** OpenHands supplies session mechanics, not a
   naturally observed annual site arrival process.
4. **Saturation.** Heavy mixes at the fixed chat-derived rate may be infeasible.
   Hatching and excluding those cells is a result, not a reason to drop or hide
   work.
5. **Weekly versus annual tails.** Weekly LDCs validate supported ranks only.
   Annual 22/44/88-hour results have no observed annual ground truth.
6. **Seasonality.** The chosen data cannot identify a full annual cycle.
   Seasonal amplitude and phase remain declared sensitivities.
7. **Composition.** Exact agreement validates a numerical approximation to
   PowerTrace, not physical fleet-scale dependence.
8. **Chronology.** An LDC validates magnitudes, not ramps, event coincidence,
   or flexible-load response. Secondary event summaries do not create a
   curtailment claim.
9. **Grid terminology.** Use "scenario-conditioned hourly IT design demand" or
   "IT demand threshold." Avoid "grid capacity," "interconnection capacity,"
   "transformer sizing," "curtailment headroom," "95% reliability," and "site
   forecast."

## 17. References

- ServeGen: *Workload Characterization and Generation of Large Language Model
  Serving in Production*. NSDI 2026. <https://github.com/alibaba/ServeGen>
- BurstGPT: *A Real-World Workload Dataset to Optimize LLM Serving Systems*.
  KDD 2025. <https://github.com/HPMLL/BurstGPT>
- NLR/DIPLOEE: *Measurement of Generative AI Workload Power Profiles for
  Whole-Facility Data Center Infrastructure Planning*.
  <https://arxiv.org/abs/2604.07345>
- Splitwise: *Efficient Generative LLM Inference Using Phase Splitting*. NSDI
  2024. <https://www.usenix.org/conference/nsdi24/presentation/patel>
- DynamoLLM: *Designing LLM Inference Clusters for Performance and Energy
  Efficiency*. HPCA 2025. <https://arxiv.org/abs/2408.00741>
- POLCA: *Characterizing Power Management Opportunities for LLMs in the
  Cloud*. ASPLOS 2024. <https://doi.org/10.1145/3620666.3651329>
- *Using Synthetic Traces for Robust Energy System Sizing*. ACM e-Energy 2019.
  <https://doi.org/10.1145/3307772.3328306>
- *Data Center Participation in Demand Response with QoS Guarantees*. ACM
  e-Energy 2019.
  <https://www.bu.edu/peaclab/files/2019/06/YZhang_eEnergy2019_QoSG_published.pdf>
- FlexDC-Sim: *AI Data Center Flexibility*. ACM e-Energy 2026.
  <https://www.bu.edu/peaclab/files/2026/03/FlexDC_Sim_ACM_E_Energy26.pdf>
- *Flexible Connection to Accelerate Load Interconnection*. ACM e-Energy 2026.
  <https://doi.org/10.1145/3744255.3811733>
