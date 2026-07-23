# PowerTrace-Sim Cleaning Plan

This plan is for preparing the repo and `eenergy-paper` for fast, relevant,
grid-facing evaluations. The goal is a smaller maintained surface, clearer data
contracts, and paper results that can be defended.

## Non-Negotiables

- Keep changes small. Delete or move only after a command proves the path is not
  in the maintained story.
- Prefer one clear path over old/new compatibility layers.
- Preserve existing public CLI flags unless a cleanup task explicitly retires a
  script.
- Every code cleanup PR must end with `uv run -m pytest -x`.
- Every result-changing PR must say which paper numbers or figures changed.
- Do not edit bulk `data/`, `results/`, or `figures/` artifacts unless that PR is
  explicitly about artifact provenance or regeneration.

## Target Maintained Surface

Keep these as the live paper and grid-evaluation paths:

- Model training and inference: `model/pipeline/`, `model/scripts/`,
  `model/classifiers/`, `model/training_data/`, `model/utils/`.
- Canonical facility evaluation: `scripts/eval/run_azure_pipeline.py` and its
  Azure submodules.
- Model validation: `scripts/eval/run_baselines_node.py`,
  `scripts/eval/run_baselines_node_groundtruth.py`,
  `scripts/eval/generate_trace_fidelity_table.py`,
  `scripts/eval/generate_power_cdf_comparison.py`.
- Profiling campaign path: `profiling/jobs/run_campaign.sh`,
  `profiling/jobs/campaign_config.py`, `profiling/probes/`,
  `profiling/BUNDLE_SCHEMA.md`, and campaign JSONs that are actually runnable.
- Emerging roofline path: `scripts/eval/occupancy_roofline.py`, but only after
  provenance and measured-state semantics are explicit.

Everything else must be either documented as legacy, moved under an archive, or
removed from the maintained evaluation README.

## Shaky Results Registry

Fix or label these before strengthening paper claims:

| Area | Risk | Required action |
| --- | --- | --- |
| Throughput DB | Defaults point at `model/config/throughput_database.json`, while the tracked file is `model/throughput_database.json`. | Pick one canonical path and update defaults, tests, and docs together. |
| Evaluation alignment | Evaluation estimates an offset from measured power before scoring. | Label as oracle alignment or replace with the inference path. |
| Eval/inference generation | Evaluation can use AR(1)/thresholded generation, while inference is IID-only. | Unify modes or write the divergence into output manifests and README. |
| Synthetic timestamps | Several CLIs allow timestamp fabrication. | Every output manifest must state whether recorded or synthetic timestamps were used. |
| Absolute paths | Generated manifests contain `/Users/grantwilkins/...`. | Store repo-relative paths plus repo root metadata. |
| Azure provenance | Splitter defaults to a local Downloads path. | Require `--input-csv` or document a repo-relative raw location. |
| MoE fidelity | MoE temporal fidelity is weak relative to dense models. | Keep as limitation/stress test, not a broad success claim. |
| Roofline | Outputs are not checked in, and occupancy may be reconstructed. | Do not make main-paper claims until measured-state ledger is complete. |
| Agentic/cache-on | `engine.csv` parser is not implemented in the ledger builder. | Implement measured-state bins before trusting cache-on/agentic numbers. |
| Splitwise baseline | Fallbacks can select family or scalar support. | Report fallback status in metrics tables and manifests. |
| Dry-run bundles | Dry runs write sample bundles into live roots. | Route dry runs to a dry-run root or require explicit sample writing. |
| Campaign checkpointing | `ls -dt` can pick stale bundles. | Have runners return the exact `run_dir` and checkpoint that path. |

## Phase 0: Inventory Before Cleanup

Run these before any cleanup PR:

```bash
git status --short
git ls-files '*__pycache__*' '*.pyc' '.DS_Store'
git ls-files 'results/**' 'figures/**' 'data/**' 'feature-test/results/**'
rg -n "model/config/throughput_database|/Users/grantwilkins|Downloads|allow-synthetic-request-timestamps|NotImplementedError|ls -dt" model scripts profiling feature-test README.md
```

Expected outcome:

- A list of tracked generated noise.
- A list of tracked paper artifacts that must be kept, regenerated, or removed.
- A list of path and fallback issues to handle in focused PRs.

## Phase 1: Source-Control Hygiene

Purpose: remove false code evidence without changing behavior.

Tasks:

1. Remove tracked `.DS_Store`, `__pycache__`, and `.pyc` files.
2. Confirm `.gitignore` prevents recurrence.
3. Decide whether tracked generated `figures/`, `results/`, and
   `feature-test/results/` are canonical paper artifacts or local outputs.
4. If canonical, keep them and add provenance. If not, untrack them in a
   dedicated artifact cleanup PR.

Do not delete raw data or paper figures in this phase. Only remove cache/noise
files with no scientific content.

Tests:

```bash
uv run -m pytest -x
```

## Phase 2: Canonical Data and Path Contracts

Purpose: make reruns portable and make failures informative.

Tasks:

1. Choose `model/throughput_database.json` as the canonical throughput DB unless
   there is a strong reason to create `model/config/`.
2. Replace repeated default literals with one helper or constant.
3. Update CLI defaults in model and eval scripts.
4. Update tests that currently create `model/config/throughput_database.json`.
5. Remove local absolute defaults, especially the Azure raw trace splitter.
6. Add result-manifest fields:
   `command`, `git_commit`, `config_id`, `seed`, `input_paths`,
   `generation_mode`, `alignment_mode`, `timestamp_source`,
   `fallback_status`.

YAGNI gate:

- Do not invent a new manifest framework. Extend the existing JSON outputs with
  a small shared writer only if at least two scripts need the exact same logic.

Tests:

```bash
uv run -m pytest -x model/tests/test_infer_gmm_bigru.py model/tests/test_eval_gmm_bigru.py model/tests/test_eval_baselines_scripts.py
uv run -m pytest -x
```

## Phase 3: Evaluation Semantics

Purpose: make paper numbers match the claimed interface.

Tasks:

1. Decide whether measured-power alignment is an oracle diagnostic or the
   production evaluation path.
2. If oracle: rename output fields and table text so reviewers see it clearly.
3. If non-oracle: remove measured-power offset estimation from the main metric
   path and share request-building semantics with inference.
4. Unify IID/AR(1)/thresholded generation between evaluation and inference, or
   expose an explicit `generation_mode` in both paths.
5. Make skipped/failed trace counts required columns in result summaries.

Tests:

```bash
uv run -m pytest -x model/tests/test_eval_gmm_bigru.py model/tests/test_infer_gmm_bigru.py model/tests/test_pipeline_roundtrip.py
uv run -m pytest -x
```

## Phase 4: Trim Evaluation Scripts

Purpose: keep only scripts that support the grid story.

Keep in maintained docs:

- `scripts/eval/run_azure_pipeline.py`
- `scripts/eval/azure_*`
- `scripts/eval/oversubscription_figure.py`
- `scripts/eval/hierarchy_figure.py`
- `scripts/eval/run_baselines_node.py`
- `scripts/eval/run_baselines_node_groundtruth.py`
- `scripts/eval/run_baselines_facility.py`
- `scripts/eval/generate_*table.py`
- `scripts/eval/generate_power_cdf_comparison.py`
- `scripts/eval/feature_sufficiency_figure.py`
- `scripts/eval/appendix_surrogate_validity.py`
- `scripts/eval/occupancy_roofline.py` after provenance cleanup

Move to legacy or archive docs unless explicitly used in the paper:

- `scripts/eval/two_price_fit.py`
- `scripts/eval/saturating_fit.py`
- `scripts/eval/operator_table.py`
- `scripts/eval/ledger_fit_lomo.py`
- `scripts/eval/rps_power_sweep.py`
- `scripts/run_training.sh`
- `scripts/run_ablation_study.sh`
- `scripts/collect_random_weights.sh`
- `feature-test/` outputs and one-off analysis scripts

YAGNI gate:

- Do not split a large file just because it is large. Split only when the new
  module has a stable purpose: data selection, metric aggregation, figure
  rendering, or manifest writing.

Tests:

```bash
uv run -m pytest -x model/tests/test_run_azure_pipeline.py model/tests/test_azure_generate_traces.py model/tests/test_generate_trace_fidelity_table.py
uv run -m pytest -x
```

## Phase 5: Profiling Bundle Discipline

Purpose: make new measurements easy to trust.

Tasks:

1. Make `data/runs/<campaign_id>/<run_id>/` the only live run-bundle format.
2. Treat `data/sharegpt-benchmark-*` as historical imports or add a converter.
3. Route dry-run sample bundles outside live run roots.
4. Make every runner emit the exact `run_dir`.
5. Replace glob checkpointing with checkpointing of the emitted `run_dir`.
6. Document Sherlock-specific assumptions in one place instead of scattering
   `$SCRATCH`, `$HOME`, cache paths, and Apptainer env through scripts.
7. Implement `bins_from_engine_csv` before using cache-on or agentic results in
   main claims.

Tests:

```bash
uv run -m pytest -x profiling feature-test
uv run -m pytest -x
```

## Phase 6: Paper Artifact Map

Purpose: make `eenergy-paper` easy to regenerate and audit.

Tasks:

1. Add `results/eval_paper/README.md`.
2. For each figure/table in the paper, list:
   output path, command, upstream artifacts, seed policy, and expected runtime.
3. Mark artifacts as one of:
   measured, fitted, generated, aggregated, or rendered.
4. Keep the main paper to grid-facing results:
   trace fidelity, facility profile, sizing table, hierarchy smoothing, and
   oversubscription sensitivity.
5. Move feature, surrogate, CDF, and MoE stress-test material to appendix unless
   it directly supports a main claim.

Tests:

```bash
uv run -m pytest -x
```

## Mission-Critical Tests To Preserve

- `model/tests/test_pipeline_roundtrip.py`
- `model/tests/test_eval_gmm_bigru.py`
- `model/tests/test_infer_gmm_bigru.py`
- `model/tests/test_train_gmm_bigru.py`
- `model/tests/test_inventory.py`
- `model/tests/test_stage0_inventory.py`
- `model/tests/test_manifest.py`
- `model/tests/test_metrics.py`
- `model/tests/test_gmm_bigru_utils.py`
- `model/tests/test_run_azure_pipeline.py`
- `model/tests/test_azure_generate_traces.py`
- `model/tests/test_azure_metrics.py`
- `model/tests/test_occupancy_roofline.py`
- `profiling/jobs/tests/test_campaign_config.py`
- `profiling/probes/tests/test_validate_runner.py`
- `profiling/probes/tests/test_agentic_replay.py`

## Definition of Clean

The repo is clean enough for rapid e-Energy iteration when:

- `uv run -m pytest -x` passes.
- `README.md`, `scripts/eval/README.md`, and `results/eval_paper/README.md`
  name the maintained commands.
- Generated manifests are repo-relative and include command, seed, git commit,
  generation mode, timestamp source, and fallback status.
- The paper does not depend on untracked scripts or untracked result paths.
- `git ls-files '*__pycache__*' '*.pyc' '.DS_Store'` returns nothing.
- Every main-paper claim maps to one maintained command and one checked result
  artifact or a documented regeneration step.
- Roofline and agentic claims are either backed by measured `engine.csv`
  occupancy/state parsing or explicitly labeled reconstruction-based.
