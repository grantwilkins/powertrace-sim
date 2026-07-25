# Results

The maintained model writes only to the following result families:

- `clean_model/`: prepared-data manifest and regenerated selected release.
- `paper/`: current timing, replay, transfer, and facility paper artifacts,
  bound by `paper/manifest.json`.
- `eval_paper/`: compact current Azure facility CSV, JSON, and LaTeX support.
- `azure_facility/`: reproducible node and hierarchy intermediates.
- `stage0/`: current data-inventory summaries.
- `moe_routing/`: active routing-law profiling evidence for future extensions.
- `disaggregated/`: compact retrospective GPT-OSS prefill/decode transfer
  metrics, representative diagnostics, and the two-panel held-out phase power
  time-series figure. The plot and primary phase soft-DTW use the raw recorded
  approximately 250 ms samples without smoothing, interpolation, or warping.

Run the complete maintained regeneration path with:

```bash
uv run --extra train --extra paper -m scripts.paper.regenerate
```

Rejected feature scorecards, older fits, historical paper tables, and old
figures live under `archive/research_artifacts/`. The first-generation
GMM-BiGRU result tree is under `archive/gmm_bigru_v1/results/`.

Large `.npy` facility arrays and raw experimental outputs are intentionally
local-only; the accepted confirmatory disaggregated campaign under
`data/disagg/` is the explicit retained-evidence exception. The tracked
manifests hash the compact paper contract.
The disaggregated result family is intentionally outside that paper manifest
and records its raw local campaign hash in its own report.
