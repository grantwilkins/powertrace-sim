# Research Review: Monte Carlo LDC Plan

## Executive recommendation

The proposed work is worth pursuing, but the paper should not claim to be the first
bottom-up workload-to-facility power generator. The 2026 NLR work on
[DIPLOEE](https://arxiv.org/pdf/2604.07345) already combines measured AI hardware
power profiles, probabilistic workload behavior, and full-year whole-facility
simulation. A stronger and more defensible contribution is:

> An uncertainty-aware, request-semantic inference load generator that separates
> measured uncertainty from planning assumptions, validates the effect of
> aggregation and routing, and quantifies the error those modeling choices cause
> in grid-planning decisions.

Four changes are important before implementation:

1. Generate requests at the site or service-pool level, then route them to nodes
   through an explicit policy. Per-node arrival streams with a shared latent factor
   skip the load balancer, even though routing can suppress, amplify, or temporally
   shift node correlation.
2. Do not present an independently composed 365-day trace as empirically validated
   from seven days of traffic. It can be a transparent scenario. Annual claims need
   longer data or a separate externally validated calendar model.
3. Replace the proposed curtailment-enabled headroom calculation. Subtracting a
   facility LDC from facility nameplate measures unused connection capacity, not
   grid headroom. The facility chronology must be added to a regional grid-load
   chronology and evaluated against a system threshold.
4. Use strong generative and power-model baselines, proper probabilistic scores,
   and a decision-error experiment. Nameplate, flat mean, and a deliberately
   weakened Poisson model are useful references or ablations, but not sufficient
   baselines for an academic comparison.

With those changes, this can make a compelling e-Energy paper. The distinctive
story is not merely that AI demand varies. It is that common workload-modeling
shortcuts create measurable errors in the grid quantities used to plan large
loads, and that PowerTrace-Sim provides a reproducible way to expose those errors.

## What the current plan proposes

The plan couples four layers:

1. A traffic generator with a diurnal envelope, block-bootstrapped log-intensity
   residuals, a shared site factor, and joint input/output token marks.
2. PowerTrace-Sim's request scheduler and node power model.
3. Exact small-fleet simulation plus a Gaussian cohort-composition approximation
   for 100--200 MW facilities.
4. Monte Carlo annual composition and load-duration curves (LDCs) at one-second,
   five-minute, and hourly resolutions.

That overall decomposition is sensible. The main issue is not the use of Monte
Carlo; it is which stochastic quantities are learned, which are assumed, whether
chronology is preserved for the decision being studied, and how success is
validated.

## Scientific audit

| Issue | Why it matters | Recommended change |
|---|---|---|
| Novelty overlaps DIPLOEE | NLR already publishes a bottom-up full-facility AI workload and power simulator with measured H100 profiles and full-year, one-minute runs. | Cite it as the closest system and distinguish request semantics, queueing, routing uncertainty, cross-node dependence, LDC calibration, and downstream decision error. |
| Node arrivals precede routing | Production requests normally arrive at a service pool and are then assigned to replicas. Least-loaded, sticky-session, round-robin, and random routing produce different queue and power correlations. | Generate site/pool arrivals first. Make router policy an explicit scenario, then feed routed requests to unchanged per-node schedulers. |
| Cross-node dependence is not identifiable | The Azure trace lacks server identifiers, placement, routing, and power. Its site timestamps cannot identify node-level correlation. | Treat factor loadings and router policy as planning scenarios, not fitted facts. Report realized correlation at the request-count, token-work, and power levels; the latent log-rate parameter is not itself power correlation. |
| One shared factor couples all pools | A single site factor silently imposes positive cross-pool dependence. Real services can share demand shocks or have different calendars. | Use explicit pool loadings or a small declared cross-pool correlation matrix, with independent, moderate, and common-mode cases. |
| Seven days cannot validate a year | Leave-one-day-out validation tests interpolation across those seven days, not month, season, holiday, growth, or annual extreme behavior. | Make day/week behavior the empirical result. Label annual LDCs as conditional scenarios unless longer data are added. Use the 213-day BurstGPT release for an external long-horizon check, while noting that it still does not cover a complete year. |
| Independent days destroy chronology | This is acceptable for estimating a marginal LDC, but not for ramps across boundaries, consecutive curtailment hours, rebound, storage, minimum event duration, or reserve deployment. | Maintain two products: a nonchronological marginal-LDC ensemble and a chronological sequence model for operational analyses. Never use the former for duration-dependent claims. |
| Circular block boundaries can be artificial | Wrapping residual blocks across an observed-day boundary may manufacture transitions that never occurred. | Use a stationary/block bootstrap that respects calendar boundaries, and report block-length sensitivity. |
| Gaussian fleet composition may miss tails | Queueing and saturation are nonlinear. A normal approximation can match means while missing maxima, skew, and duration above thresholds. | Validate it against exact simulation at several fleet sizes using energy, maxima, high quantiles, ramp tails, and threshold-duration statistics. Use empirical cohort blocks if Gaussian tails fail. |
| Variability scaling needs precise language | For independent nodes, absolute standard deviation grows as square root of fleet size; only relative variability shrinks as its inverse. Common-mode absolute variability can grow linearly. | Report both MW variability and percentage-of-mean variability, and separate independent and common-mode components. |
| Nameplate inversion is a scenario convention | GPU TDP, a fixed host allowance, and constant PUE do not uniquely determine contracted point-of-interconnection capacity. PUE and auxiliary power vary with load and environment. | Prefer explicit node census, IT capacity, auxiliary-load model, PUE scenario, and POI cap. Keep nameplate inversion as a clearly labeled convenience mode with sensitivity analysis. |
| “Transformer sizing” is too broad | Transformer and interconnection design also depend on thermal duration, redundancy, power factor/reactive demand, harmonics, protection, faults, and contingency criteria. | Call the output a probabilistic active-power planning envelope or contracted-demand sensitivity unless an electrical and thermal model is added. |
| The naive baseline bundles causes | Removing burstiness, synchronization, and mark dependence at once cannot show which simplification caused an error. | Change one factor at a time: time shuffle, independent marks, no residual burstiness, no site factor, and alternate routing. |
| The tail ensemble is small | With 100 replicates, an empirical 95th percentile is controlled by only about five upper-tail samples. | Stop based on uncertainty in the target quantile/decision, not a fixed replicate count. Publish bootstrap confidence intervals or quantile standard errors. |
| Held-out band coverage is insufficient | Checking whether one held-out curve lies inside a pointwise 90% band at 85% of correlated ranks is neither a proper score nor strong evidence of joint calibration. | Use calibration and sharpness together, with rank/PIT diagnostics, interval coverage and width, CRPS for scalar/rank-indexed targets, and an energy or variogram score for selected multivariate summaries. |
| Simulator agreement is not physical validation | Exact-versus-composed simulation validates the approximation to the simulator. It does not validate unmeasured site power. | Keep numerical and physical validation separate. Add external hardware/facility profiles or a controlled synchronized multi-node experiment. |

The calibration recommendation follows the standard principle that useful
probabilistic forecasts must be both calibrated and sharp, and should be compared
with proper scoring rules
([Gneiting, Balabdaoui, and Raftery](https://doi.org/10.1111/j.1467-9868.2007.00587.x);
[Gneiting and Raftery](https://doi.org/10.1198/016214506000001437)).

## Prior art and the lessons to take from it

### Workload generation

[ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf) is the
most relevant workload-generation baseline. It models clients, conversations,
client-specific arrival behavior, burstiness, and mark heterogeneity. Its central
lesson is that aggregate request rate is not enough: the number and behavior of
active clients and the association between rate and request size affect serving
performance and provisioning. The paper reports a large provisioning error from
its naive workload in one evaluated setting. PowerTrace should therefore preserve
session/client structure when it is observable and should not independently
resample arrival times and token marks.

[BurstGPT](https://arxiv.org/pdf/2401.17644) provides a much longer production
trace than the seven-day Azure sample and documents conversation, token-length,
failure, daily/weekly, and time-varying burstiness behavior. Its current release
covers 10.31 million records over 213 days. The lesson is that burst parameters are
not necessarily stationary across time or service type. It is useful as an
external validation corpus and as a burst-aware generator baseline, not as proof
that Azure traffic has the same distribution.

Classical network-traffic research reached the same broad warning much earlier:
aggregated traffic can remain bursty over long time scales, and Poisson assumptions
can fail. For this paper, ServeGen and BurstGPT provide more direct and modern
evidence than a long detour through that literature.

### Workload-to-power and whole-facility simulation

The closest prior system is the NLR study
[“Measurement of Generative AI Workload Power Profiles for Whole-Facility Data
Center Infrastructure Planning”](https://arxiv.org/pdf/2604.07345). It publishes
0.1-second H100 power measurements for training, fine-tuning, and inference and
uses DIPLOEE to generate one-minute, full-year facility traces from probabilistic
utilization and user-behavior inputs. Its associated
[open dataset](https://data.nlr.gov/submissions/312) is especially valuable for
external validation.

The right response is to make DIPLOEE a mandatory system baseline, not to draw an
artificial boundary around it. PowerTrace-Sim can still be more informative for
inference because it models token-bearing requests, queueing, and scheduling
rather than sampling a workload profile from utilization alone. The comparison
should hold the same requested workload and facility configuration constant and
ask whether request semantics materially improve power traces and grid decisions.

[DynamoLLM](https://arxiv.org/abs/2408.00741) shows that inference energy and
power depend on request length, offered load, and serving configuration, and uses
dynamic reconfiguration to improve efficiency.
[POLCA](https://doi.org/10.1145/3620666.3651329) and
[Splitwise](https://www.usenix.org/conference/nsdi24/presentation/patel) similarly
connect serving decisions to power and capacity. Their lesson is that a traffic
generator cannot be evaluated independently from its routing, batching, queueing,
and serving policy.

### Data-center flexibility and e-Energy

The e-Energy paper
[“Exploding AI Power Use: an Opportunity to Rethink Grid Planning and
Management”](https://doi.org/10.1145/3632775.3661959) argues that flexible
interconnection and relaxed availability guarantees can allow substantially more
data-center capacity to connect. A more recent e-Energy study,
[FlexDC-Sim](https://www.bu.edu/peaclab/files/2026/03/FlexDC_Sim_ACM_E_Energy26.pdf),
combines measured hardware power/performance behavior with demand-response and
regulation scenarios. These papers establish the grid-facing question, but they
also set a higher bar than an unconstrained LDC: a flexibility claim needs a
control action, a grid-service contract, tracking error, recovery behavior, and
an application-level QoS cost.

Consequently, this paper should make a clean choice:

- If it studies only uncontrolled demand, call the result planning headroom or
  coincidence reduction, not flexibility.
- If it claims curtailment or regulation, add an explicit policy and report
  response time, sustained MW, rebound/reconnection, deadline or latency effects,
  and energy not served to the compute workload.

Recent field demonstrations reinforce that distinction. For example,
[Emerald Conductor](https://arxiv.org/abs/2507.00909) demonstrates sustained
power reduction on a 256-GPU commercial cluster while monitoring service quality.
A simulator-only paper need not reproduce that scale, but it should use the same
contract-and-QoS vocabulary.

### Grid-planning practice

Power-system adequacy models use chronological load and Monte Carlo simulation,
but they report more than a duration curve. NERC's
[probabilistic adequacy report](https://www.nerc.com/comm/RSTC/PAWG/Probabilistic_Adequacy_and_Measures_Report.pdf)
emphasizes frequency, duration, and magnitude metrics such as LOLH and expected
unserved energy. Commercial planning tools similarly run many chronological
hourly scenarios while preserving correlations
([GE Vernova PlanOS/MARS](https://www.gevernova.com/consulting/planos/resource-adequacy)).
The lesson is to keep LDCs as a compact marginal summary and export chronology for
the decisions that depend on event duration and coincidence.

Current NERC guidance for emerging large loads requests much more than active
power traces: firm and flexible components, time-coupled constraints, behind-the-
meter resources, reactive behavior, protection, voltage/frequency response,
transfer and reconnection thresholds, and reconnection ramp rates
([NERC Risk Mitigation for Emerging Large Loads](https://www.nerc.com/globalassets/our-work/guidelines/reliability/RG_Risk-Mitigation-For-Emerging-Large-Loads.pdf)).
PowerTrace-Sim should explicitly present itself as a normal-operation active-power
input to those studies, not as a complete interconnection or dynamic load model.

Grid timescales should also drive the output schema. Hourly chronology is useful
for resource adequacy, while five- and fifteen-minute series align with common
market dispatch intervals; CAISO, for example, operates fifteen- and five-minute
real-time markets
([CAISO market overview](https://www.caiso.com/market-operations/products-services)).
One-second output is useful for ramps and control studies but is not a substitute
for an electrical transient model.

## What the methods should be compared against

The baseline methods do not need to be compared with an unknowable “true 200 MW
facility distribution.” They need to predict measurements that were not used to
fit them. The primary target should be a withheld, timestamp-aligned, measured AC
power profile from a real multi-node inference deployment under a known request
workload.

This creates an essential information boundary:

- Every method receives the same training traces, facility description, and
  request information.
- No method sees the held-out power measurements or parameters estimated from
  them.
- Every method emits either a point prediction or a fixed-size ensemble before the
  measured target is revealed.
- All methods are scored against the same measured profile and the grid decision
  calculated from that profile.

Without this experiment, the paper can establish simulator consistency and
component accuracy, but it cannot establish that its generated load-profile
distribution is more predictive than another generator.

### No single dataset is ground truth for every layer

The paper should use a validation ladder and state the evidence available for each
claim:

| Claim | Ground-truth target | Required experiment | Evidentiary status |
|---|---|---|---|
| Traffic model reproduces request behavior | Withheld real request timestamps, input/output tokens, and sessions | Fit on earlier days; generate the held-out day without seeing it | Supported by Azure and external serving traces, but only at their observed calendar horizon |
| Request-to-power model predicts physical load | Timestamp-aligned measured node or rack AC power under the exact held-out requests | Give each model the request sequence and configuration; predict the measured power trace | Supported by local measurements; can be externally tested |
| Aggregation predicts fleet behavior | Simultaneously measured aggregate AC power from 2/4/8 or more real nodes | Fit lower-scale components; predict an unseen node count and workload synchronization pattern | Requires a multi-node holdout rather than sums of independently replayed measurements |
| Stochastic generator predicts unseen load profiles | Multiple held-out measured power windows generated by real request workloads | Fit traffic and power layers on training windows; predict an ensemble for each unseen window | Requires paired request-and-power measurements across enough independent windows |
| Annual 100--200 MW distribution is correct | A year of real site-boundary power plus sufficient workload/operations metadata | Train elsewhere or on earlier history; predict the held-out site-year | Not currently supported; must remain a conditional projection |
| A planning decision is useful | Decision computed from withheld measured load plus historical grid demand | Select capacity/headroom with each model using training data; evaluate against measured load | Directly testable at the measured benchmark scale |

The fifth row is the irreducible limitation. Accurate node measurements and exact
simulation do not turn an unobserved annual site distribution into ground truth.
Unless an operator supplies site-boundary data, the paper should say that the
large-facility result is a scenario-conditioned projection whose components have
been validated at observable scales.

### Three benchmark tasks

One benchmark dataset can support three nested tasks.

#### Task A: conditional power reconstruction

Give every method the exact held-out request arrivals, token lengths, routing, and
hardware configuration. Ask it to predict the measured AC power trace. This
removes traffic uncertainty and isolates the request-to-power model.

For this task, pointwise MAE/RMSE, energy error, peak error, ramp error, and
threshold-duration error are meaningful because the request realization is fixed.
The paper's first visual anchor should be an hour or day of:

- the real measured AC load;
- PowerTrace's prediction;
- the strongest published or lookup-based baseline;
- a residual panel.

This is the literal “here is a load profile, and here is how closely we reproduce
it” result.

#### Task B: out-of-sample probabilistic load prediction

Hide both the request realization and power measurements for the target window.
Fit each generator only on preceding training windows and ask it for an ensemble
of complete load profiles. Then reveal the measured profile.

The generated traces should not be expected to reproduce the random path
point-for-point. Compare them using proper distributional scores, coverage and
width, and prespecified functionals such as energy, peak, high LDC ranks, ramps,
and duration above thresholds. Plot the measured profile against the predictive
band so the calibration result is visually concrete.

Use a rolling-origin protocol: train on windows \(1,\ldots,k\), predict window
\(k+1\), score it, and repeat. One held-out trace is a useful illustration; many
held-out windows are the statistical evidence.

#### Task C: scale-out extrapolation

Fit or calibrate on one- and two-node measurements, then predict simultaneously
measured four- and eight-node aggregate power without retuning. Repeat with
synchronized, staggered, and routed request streams. This is the direct test of
the aggregation and dependence mechanism.

Synthetic sums of node traces are not sufficient ground truth for this task
because they omit shared power infrastructure, actual routing, queue coupling,
and common-mode behavior.

### Practical sources of external measurements

The strongest feasible external target is NLR's
[public generative-AI power dataset](https://data.nlr.gov/submissions/312). It
contains 5/10 Hz measured power profiles, multiple AI workloads and node counts,
and workload metadata. Its simulated whole-facility examples are not physical
ground truth, but its raw Kestrel measurements can test conditional power
reconstruction and some scale-out behavior.

Two complementary sources can broaden configuration coverage:

- The [ML.ENERGY Benchmark dataset](https://ml.energy/data/) exposes H100/B200
  inference runs and per-request details including power timelines, inter-token
  latency, and output lengths. It is useful for external request-to-energy and
  request-to-power tests, though not for validating annual facility behavior.
- [TokenPowerBench](https://ojs.aaai.org/index.php/AAAI/article/download/40535/44496)
  defines a reproducible measurement harness that time-aligns GPU, CPU, DRAM, and
  full-node/PDU readings with inference stages and evaluates an eight-node H100
  cluster. Reproducing a subset of its configurations would create a credible
  independent benchmark even if its published results do not contain the
  long-duration stochastic workload needed for Task B.

These sources do not replace a local paired experiment. The minimum compelling
new dataset would contain several independent real workload windows, exact request
and routing logs, and synchronized rack-PDU or node AC power. Measuring 1/2/4/8
nodes is much more valuable for this paper than adding many more single-node
configuration sweeps.

### Turn the measured profile into the “so what”

The downstream experiment should also use a measured target. For each held-out
window:

1. Compute an oracle planning result \(D^\ast\) using the measured AC load and the
   chosen historical grid chronology.
2. Using training data only, compute \(D_m\) from each model's predicted
   distribution.
3. Report decision error \(D_m-D^\ast\), unsafe violations, and conservative MW
   left unused.

This answers both reviewer questions. “How do we know it is more correct?” is
answered by out-of-sample predictive skill against physical measurements. “How
does it help?” is answered by a smaller and safer planning-decision error. A model
that produces prettier distributions but does not improve either result has not
demonstrated value.

The site-scale claim should remain conditional:

> PowerTrace is more predictive than the evaluated baselines on held-out measured
> clusters, its aggregation error is bounded over the measured scale range, and
> its large-site ensembles transparently propagate the remaining workload,
> routing, and common-mode assumptions.

Do not replace the last clause with a claim that the 100--200 MW distribution is
physically validated unless site-boundary measurements are obtained.

### What e-Energy's publication history says

There is direct e-Energy precedent for synthetic-trace research, and it resolves
the ground-truth question in almost exactly the way proposed above.

The 2019 full paper
[“Using Synthetic Traces for Robust Energy System Sizing”](https://cs.stanford.edu/~fiodar/pubs/synthetic_trace_generation.pdf)
compares ARMA, Gaussian-mixture, GAN, and direct-resampling generators. It does
not argue that a generated trajectory is intrinsically realistic. Instead, it:

1. trains from one year of real hourly solar and household-load data;
2. sizes solar and storage using each generator;
3. deploys those sizing decisions against three held-out years of real data; and
4. compares capital cost and the number of QoS failures in those testing years.

That is the clearest template for this work. Replace “solar/storage sizing” with
“large-load interconnection or curtailment planning,” and replace the testing
years with withheld measured inference-load windows. The generator wins only if
it produces a safer or less conservative decision on future real measurements.

Other e-Energy papers reinforce a consistent evidence pattern:

| e-Energy precedent | Evidence pattern | Lesson for PowerTrace |
|---|---|---|
| [Data Center Participation in Demand Response with QoS Guarantees, 2019](https://www.bu.edu/peaclab/files/2019/06/YZhang_eEnergy2019_QoSG_published.pdf) | Real experiments on a 12-server cluster anchor large-scale simulations; the paper evaluates target tracking, cost, and job QoS. | A small physical cluster plus carefully bounded scale-out simulation is acceptable when the physical and simulated claims are separated. |
| [Using Synthetic Traces for Robust Energy System Sizing, 2019](https://doi.org/10.1145/3307772.3328306) | Generators are judged by decisions made from training data and failures/cost on future real years. | Trace similarity alone is not the primary proof; held-out operational regret is. |
| [Decision-Focused Retraining, 2024](https://publikationen.bibliothek.kit.edu/1000172455/153547971) | Forecast models are evaluated on both forecast quality and their value in a downstream feeder optimization over data from 199 buildings. | A slightly worse-looking forecast can be better if it produces a better grid decision; report both metrics. |
| [LACS, 2024](https://noman-bashir.github.io/assets/pdf/eEnergy-2024-LACS.pdf) | The learned method is compared with online and offline oracles, including baselines with perfect future information, and evaluated on carbon and deadline outcomes. | Include an empirical oracle even when it is not deployable; it shows the attainable decision bound. |
| [AI Data Center Flexibility/FlexDC-Sim, 2026](https://www.bu.edu/peaclab/files/2026/03/FlexDC_Sim_ACM_E_Energy26.pdf) | Real hardware power-performance profiles parameterize a 1,000-server simulator; results use explicit EDR/RSR contracts, tracking error, cost, and QoS. | This is enough to demonstrate a simulator-based opportunity in a short paper, but it does not validate an uncontrolled annual facility-load distribution. A full trace-generation paper should go further with held-out measured profiles. |
| [Flexible Connection to Accelerate Load Interconnection, 2026](https://doi.org/10.1145/3744255.3811733) | The best-paper study combines realistic load data, a standard IEEE feeder, an intervention-probability constraint, and the resulting hosting capacity under partial observability. | e-Energy values a precise interconnection decision and risk contract. It does not require observing the future large load if the methodological claim and input assumptions are explicit. |

The venue history therefore does **not** imply that the team must obtain a year of
metered 200 MW inference-facility data. It does imply that the paper needs one of
two evidentiary forms:

1. **Empirical systems paper:** paired, withheld multi-node request/power profiles,
   external measurements, scale-out checks, and decision regret on the withheld
   profiles. This is the best fit for the existing PowerTrace work.
2. **Formal planning-method paper:** a precise uncertainty set or stochastic model,
   a theorem or strong methodological result, and a standard grid case study with
   a declared risk contract. This would resemble the 2026 flexible-connection
   paper but would be a materially different project.

The weak middle position is a full paper that presents a plausible Monte Carlo
facility distribution, validates the power model only on the same small scenarios
used to build it, and reports LDCs without held-out decision consequences. Recent
e-Energy notes show that measured component profiles plus scaled simulation can
motivate an opportunity; the 2019 full-paper history shows what is needed to
establish that one stochastic generator is better than another.

For this project, the recommended acceptance bar is:

- one sealed measured profile figure for Task A;
- multiple rolling held-out measured windows for Task B;
- at least one measured multi-node scale-out target for Task C;
- an external NLR or ML.ENERGY result;
- and a Sun et al.-style table reporting decision cost, unsafe failures, and
  conservatism on held-out observations.

If paired long-duration power measurements cannot be collected, the fallback is
to use held-out real request traces as the future observations, translate them
with the already sealed PowerTrace model, and call the result **model-referenced
decision validation**, not physical ground truth. That may still demonstrate the
value of the workload generator, but the paper must state that the final load and
decision target inherit PowerTrace's measured-scale validation and are not new
facility measurements.

## The strongest paper demonstration

The paper should be organized around falsifiable research questions rather than
around the mechanics of generating an LDC.

### RQ1: Does the request-to-power model reproduce observed power behavior?

Use the existing held-out node traces and, where configurations can be aligned,
the NLR H100 data. Report:

- energy, mean, maximum, and high power quantiles;
- ramp-rate and threshold-duration distributions;
- autocorrelation or spectral behavior at the claimed output resolutions;
- latency, throughput, and queue behavior, so a power fit is not obtained by
  simulating the wrong computation;
- unsupported request-rate or request-shape regions explicitly.

An external dataset mismatch should be treated as a domain-shift result, not hidden
through calibration.

### RQ2: Is the stochastic traffic generator calibrated out of sample?

Train on a subset of days and generate an ensemble for each held-out day. Evaluate
request counts, token work, conversations if available, node power, and selected
LDC ranks. Report:

- interval coverage together with interval width;
- CRPS or another proper univariate score for daily energy, peak, ramps, and
  prespecified LDC exceedance ranks;
- rank/PIT diagnostics;
- an energy or variogram score for a small, predeclared vector of temporal
  features;
- skill relative to every generative baseline.

The seven Azure days can support day-level cross-validation. Use BurstGPT for a
separate external test of longer calendar patterns. Unless more data are obtained,
do not claim empirical calibration of month, season, or annual extremes.

### RQ3: Does aggregation preserve fleet-scale distributions and tails?

Run exact simulations for multiple sizes, not only one 240-node and one 2400-node
case. A logarithmic sequence such as 1, 8, 32, 128, and the largest tractable fleet
would expose the transition from node noise to common-mode behavior. At each size:

- compare exact simulation with cohort composition;
- vary router policy and the declared cross-pool/site dependence;
- report absolute MW and normalized variability;
- compare peak, high LDC ranks, ramps, and durations above planning thresholds;
- give uncertainty on approximation error.

A controlled 1/2/4/8-node hardware experiment with synchronized and staggered
requests would materially strengthen the correlation story if feasible.

### RQ4: Do modeling choices change a real grid-planning decision?

This should be the headline e-Energy experiment. For each stochastic facility
trace, align local time with one or more historical regional load chronologies and
compute a planning decision. The cleanest first target is curtailment-enabled
headroom.

The Duke
[“Rethinking Load Growth”](https://icc.illinois.gov/docket/P2025-0679/documents/371138/files/650737.pdf)
method tests a candidate new load against a regional seasonal peak threshold and
aggregates the energy above that threshold. A PowerTrace version should use:

\[
c_{r,t}(L)=\max\left(0,\;G_{y,t}+F_{r,t}(L)-T_{y,s(t)}\right)
\]

where \(G_{y,t}\) is historical grid demand, \(F_{r,t}(L)\) is replicate \(r\) of
the candidate facility at scale \(L\), and \(T_{y,s(t)}\) is the applicable system
threshold. For a time-varying candidate, define its curtailed-energy fraction as
\(\phi_r(L)=\sum_t c_{r,t}(L)/\sum_t F_{r,t}(L)\), then solve for the largest
\(L\) whose chosen risk measure over \(\phi_r(L)\) is below the declared target.
This generalizes Duke's constant-load calculation while retaining its system-load
coincidence.

This is fundamentally different from
`facility_nameplate - facility_LDC(e)`. The latter is a facility utilization
statistic and contains no information about coincidence with grid stress.

For each baseline, report:

- error in selected connectable MW versus held-out replay or the best available
  empirical oracle;
- underestimation risk and over-conservatism separately;
- expected curtailment energy, event count, event duration, and largest event;
- sensitivity to time zone, regional load year, router, common-mode dependence,
  PUE, and facility mix.

A second decision can be five-minute net-load ramp or reserve requirement. Avoid
adding more grid services unless each has a real operational contract.

### RQ5: Can a practitioner reproduce and interpret the result?

Release one end-to-end case with public inputs, fixed seeds, a machine-readable
manifest, and one documented command. A small formative review with grid
planners/operators would improve the paper: ask participants to configure a site,
identify a planning quantile, and locate model limitations. Report the concrete
problems found rather than treating a generic satisfaction score as validation.

## Intelligent baseline ladder

Baselines should be separated by layer so the experiment identifies where any
improvement comes from.

### Traffic and dependence baselines

1. **Held-out chronological replay.** This is the empirical oracle/ceiling for the
   observed day, not a deployable generator.
2. **Joint site-level block bootstrap.** Resample count and mark blocks together,
   preserving empirical short-range dependence. This is a strong nonparametric
   baseline and likely the hardest simple baseline to beat.
3. **Nonhomogeneous Poisson with joint conditional marks.** Fit the same
   time-of-day rate but preserve the empirical joint input/output mark
   distribution by time bucket. This isolates the value of residual burstiness
   without making the baseline needlessly bad.
4. **Burst-aware external generator.** Implement the relevant BurstGPT
   Gamma/time-varying model or use its trace-rescaling generator.
5. **Client/session-aware ServeGen baseline.** Use it where client or conversation
   structure is available. If the local trace cannot support those fields, run it
   as a separate public-data benchmark rather than inventing identities.
6. **Proposed site-factor plus explicit-router generator.**

Deep neural time-series generators or Hawkes processes should be added only if
there is enough training data and a clear hypothesis. Seven days do not justify a
large learned generator merely to make the baseline list look sophisticated.

### Power and facility baselines

1. **Measured-trace replay** under the exact observed request sequence.
2. **DIPLOEE-style request-rate/utilization to measured power-profile sampling**
   using the NLR data and the same facility assumptions.
3. **A published serving/power lookup approach such as POLCA or Splitwise** where
   its hardware and request regime can be reproduced fairly.
4. **Full PowerTrace request scheduler and power model.**

When comparing traffic generators, hold the scheduler and power model fixed. When
comparing power models, hold traffic and routing fixed. Nameplate and flat mean
remain useful planning references, but should not be presented as peer scientific
baselines.

### Required ablations

Run one change at a time:

- joint versus independent token marks;
- residual blocks versus shuffled residuals;
- site factor on versus off;
- different site-factor loadings;
- random, round-robin, sticky, and least-loaded routing where applicable;
- exact versus cohort-composed fleets;
- constant versus load-dependent PUE;
- chronological versus independently sampled day composition.

The outcome of interest is not only trace distance. Each ablation should also
report its change in the grid decision from RQ4.

## A product grid operators can use

The useful artifact is not a plot generator. It is a versioned scenario bundle
whose assumptions are legible to a planner.

### Two input modes

**Explicit engineering mode, preferred**

- POI active-power cap and facility time zone;
- node counts by hardware/service pool;
- measured or declared IT and auxiliary-power models;
- routing/scheduling policy;
- firm and controllable workload shares;
- on-site storage or generation, if modeled;
- calendar/workload scenario and seed.

**Early planning mode**

- requested connection MW;
- a named reference node/configuration;
- PUE/auxiliary scenarios;
- workload-mix and dependence scenarios.

The early mode must label inferred node count and capacity allocation as
assumptions. It should not manufacture false precision from a single nameplate
number.

### Output contract

Each run should emit:

- chronological UTC and local-time active-power series at hourly, 15-minute,
  5-minute, and optional 1-second resolution;
- `scenario_id`, replicate/weight, seed, pool, POI/facility boundary, and evidence
  label for every series;
- LDC quantiles with the ensemble quantile and exceedance probability named
  separately;
- peak, load factor, energy, ramp, event-count, event-duration, and coincidence
  tables;
- a manifest containing input hashes, code/data versions, assumptions, and
  out-of-support flags;
- CSV/Parquet exports plus adapters for common chronological planning-model
  schemas.

Use unambiguous notation such as
\(Q_{0.95}[LDC_r(e=0.01)]\), rather than “P99 exceedance,” which confuses the
within-year exceedance rank with the across-replicate uncertainty quantile.

### Load model card

Ship a one-page human-readable summary with:

- the intended planning use and prohibited uses;
- facility boundary and timezone;
- min/median/peak MW and load factor;
- five- and fifteen-minute ramp distributions;
- largest modeled reduction, reconnection, and rebound where controls exist;
- firm/flexible fractions and QoS assumptions;
- measured inputs versus inferred parameters versus sensitivity scenarios;
- validation scores and support limits;
- missing electrical behavior.

The missing-behavior section should explicitly state that the current model does
not provide reactive power/power factor, harmonic behavior, voltage/frequency
ride-through, protection, fault response, or validated reconnection dynamics.
Those require a site-specific dynamic model for an interconnection study.

### Planning versus operations

The Monte Carlo product is a planning scenario generator, not a live load
forecast. Operational adoption would additionally require telemetry-based
calibration, next-seven-day hourly forecasts, uncertainty updates, real-time
active/reactive power measurements, and explicit dispatch/reconnection
interfaces. Keeping those products separate will make the planning artifact
credible rather than overextended.

## Recommended minimum paper scope

To keep the paper tight:

1. Implement site/pool generation followed by explicit routing.
2. Validate node power on local held-out measurements and NLR data where feasible.
3. Compare the proposed generator with joint block bootstrap, conditional NHPP,
   BurstGPT/ServeGen, and DIPLOEE-style power sampling.
4. Validate exact versus composed fleets over several scales and dependence
   scenarios.
5. Run one strong chronological grid decision: curtailment-enabled connectable
   load using historical regional demand.
6. Release the operator scenario bundle and model card.

Annual synthetic LDCs, transformer claims, regulation, storage, forecasting, and
dynamic electrical behavior should not all be first-paper requirements. An annual
LDC can remain as a transparent scenario output, while the paper's validated
claims stay within the evidence.

## Suggested claim language

Avoid:

> We present the first Monte Carlo generator that composes stochastic AI
> workloads with validated node power into annual facility LDCs.

Prefer:

> We present an uncertainty-aware generator for inference data-center active
> power that preserves request-size dependence, makes site synchronization and
> routing assumptions explicit, and propagates their uncertainty to
> grid-planning decisions. Across held-out traces and facility scales, we quantify
> when simpler traffic and aggregation models misestimate power-duration and
> grid-coincidence risk.

That claim is narrower, testable, and meaningfully differentiated from the closest
prior systems.
