# Node Power Model: Measured Parameters and Validation

**Data.** Raw measured traces only: nvidia-smi power CSVs + vLLM per-request
JSONs (`data/sharegpt-benchmark-*/`), 19 node types (model × hardware × TP),
~80k five-second windows. Prefill rate `f` and decode rate `g` reconstructed
per window from request timestamps, TTFTs, and per-token ITLs.

---

## The operator-facing model (what dispatch consumes)

Six numbers per node type, each readable off a plot — no fit coefficients:

| value | one-sentence definition |
|---|---|
| `F`, `G` | the fastest prefill / decode rate the node ever sustained over 5 s [tok/s] |
| `ℓ = f/F + g/G` | how full the node is, as a fraction |
| `ρ*` | the highest ℓ at which inter-token latency is still within 25% of its best value |
| `P_idle` | median measured power of windows serving zero tokens [W] |
| `P_busy` | median measured power of windows running at or above ρ* [W] |
| `p̄ = P_busy / ρ*` | the pool-level price of load [W per node-unit] |

**Per-job score: ΔP_j = p̄ · ℓ_j.** Continuous and additive — dispatch is a sort.

Selected rows (full table: `operator_table.csv`; `~` = ρ* fallback 0.8, sweep
never pushed latency up):

| node type | P_idle | P_busy | ρ* | p̄ (W/node) |
|---|---|---|---|---|
| llama-3-405b H100 tp8 | 1007 | 4030 | 0.65 | 6240 |
| llama-3-70b A100 tp4 | 290 | 1289 | 0.55 | 2366 |
| llama-3-70b H100 tp8 | 955 | 3009 | ~0.80 | 3761 |
| llama-3-8b H100 tp1 | 120 | 466 | ~0.80 | 583 |
| gpt-oss-120b A100 tp4 | 267 | 947 | 0.87 | 1093 |

---

## Validation (why the linear pool model is licensed)

The node-level fits live here, not in the model:

1. **Node power saturates.** A saturating fit reaches R² 0.91–0.99 on all 19
   node types; a linear fit gets as low as 0.25 on 70B/405B dense models, where
   power is nearly a step (idle → ~peak by ℓ ≈ 0.1, then flat). This is the
   empirical premise for the autoscaler framing: pool power is linear in load
   only because the autoscaler converts load to node count.

2. **Two knees, in the right order.** Power flattens early (dense 70B+:
   ℓ ≈ 0.1–0.3); latency departs late (ℓ ≈ 0.55–1.0). ρ* sits just below the
   latency knee — measured, not assumed, wherever the sweep reached it.

3. **The bracket is quantified.** Fixed-node plateau slope vs. amortized price:
   ~3–5× apart for MoE, 17–44× for dense 70B, **58×** for the 405B. Ranking
   uses the additive amortized side (p̄·ℓ_j); the certified guarantee uses the
   conservative plateau side. On a fixed node power genuinely is a step and
   isn't additive — that regime is covered by the guarantee, never by the sort.

4. **Decode tokens cost ~7–10× more than prefill tokens** (c₁/c₂ ≈ 0.04–0.19,
   stable across all configs) — but collapsing to a single token price costs
   only ΔR² 0.02–0.07 on this workload (phases 0.81-correlated, decode dominates
   variance). The single-price ℓ is licensed for ShareGPT-like traffic.

---

## Caveats

- **F, G are workload-limited, not hardware-limited**: ~1600–1900 tok/s across
  all configs because every sweep used the same ShareGPT mix at ≤ 4 qps. They
  are lower bounds on capacity — harmless for ranking within a node type
  (normalization cancels), conservative for absolute watts, but cross-node-type
  comparisons of ℓ inherit the bias. A capacity-targeted sweep would pin them.
- ρ* = 0.8 by convention on H100 8B/70B (latency never rose at ≤ 4 qps); a
  higher-rate sweep is needed to measure those knees.
- `llama-3-8b-a100` excluded (older JSONs lack request timestamps).

**Node-level picture.** `figures/two_price_fit/operator_model_grid.png` shows,
per node type, the explainable ramp-plateau model — power climbs linearly from
P_idle to P_busy until the power knee, then stays flat — over the measured
windows. Only the knee is fitted; the levels keep their plain definitions.
R² 0.72–0.89 for 8B/MoE, 0.62–0.71 for dense 70B (0.44 for the 405B, whose
plateau keeps drifting upward — the residual gap vs. the 4-parameter saturating
fit, R² 0.91–0.99, is the price of the 3-number model). The knee column doubles
as the hardware-class signature: ℓ ≈ 0.03–0.15 for dense 70B+ (step-like),
0.3–0.6 for 8B/MoE (gradual).

**Artifacts.** Model table: `operator_table.csv` (`scripts/eval/operator_table.py`).
Validation: `saturating_summary.csv`, `summary.csv`, figures in
`figures/two_price_fit/sat_*.png` (Figure 2: `sat_llama-3-70b-a100_tp4.png`
step regime 42×, paired with `sat_gpt-oss-120b-a100_tp4.png` smooth regime 4×).
