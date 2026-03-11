# `gpt_oss_20b_a100_tp1` Sandbox

This directory is a minimal, isolated training and evaluation sandbox for the
single config `gpt-oss-20b_A100_tp1`.

It is intentionally separate from the main repo training and evaluation
pipelines. The goal is to make it easy to change the model logic in one place,
run training, and immediately see the heldout metrics that matter.

## What Is In Here

This directory contains four files:

- `prepare.py`
- `train.py`
- `eval.py`
- `README.md`

It also creates a local `prepared/` directory after you run `prepare.py`.

## Design

The sandbox is deliberately small:

- `prepare.py` packages the exact source data for `gpt-oss-20b_A100_tp1`
- `train.py` trains the model
- `eval.py` evaluates the trained model on heldout ground-truth traces
- `train.py` calls `eval.py` only after training completes

Important constraints:

- No `run_id` directories
- No large result trees
- No dependency on `scripts/eval/generate_trace_fidelity_table.py`
- No `iid`, `ar1`, or `ar1_thresholded` branching
- Evaluation logic lives in `eval.py`, not in the training loop body

## Files

### `prepare.py`

Purpose:

- Reads the existing experimental manifest entry for `gpt-oss-20b_A100_tp1`
- Copies the source dataset, split, and normalization files into `prepared/`
- Filters the throughput database down to this one config
- Builds a fast heldout request cache for the test split

Default source inputs:

- `results/experimental_continuous_v1_gptoss_a100/manifest.json`
- `results/stage0_sharegpt_gptoss_a100/data_inventory.json`
- `results/stage0_sharegpt_gptoss_a100/throughput_database.json`

Outputs in `prepared/`:

- `dataset.npz`
- `split.json`
- `norm_params.json`
- `throughput_database.json`
- `heldout_requests.npz`
- `bundle_manifest.json`

Notes:

- `prepare.py` does not create a new split
- It uses the split already defined in the experimental manifest
- It needs the stage0 `data_inventory.json` because the heldout request arrays
  are recovered from the original request JSONs referenced there

### `train.py`

Purpose:

- Loads the prepared dataset bundle
- Builds train and validation features
- Computes `delta_A` normalization stats
- Runs BIC-based K selection unless `--k` is explicitly set
- Fits the 1D power GMM
- Builds state labels
- Trains the `GRUClassifier`
- Calls `eval.py` after training completes
- Prints a compact final summary to stdout

What it owns:

- Model architecture
- K-selection policy
- GMM fitting
- Label construction
- Training loop
- Early stopping

What it does not own:

- Heldout metric logic
- Heldout rollout evaluation logic

Default training settings:

- `feature_set = f2`
- `bic_candidates = 6,8,10,12,14,16,18,20`
- `hidden_dim = 64`
- `num_layers = 1`
- `epochs = 1000`
- `lr = 1e-3`
- `patience = 50`
- `scheduler_patience = 20`
- `scheduler_factor = 0.5`
- `seed = 42`

Console output includes:

- `config_id`
- `feature_set`
- `k`
- `k_selection_reason`
- `hidden_dim`
- `num_layers`
- `architecture`
- `training_time_sec`
- `best_val_loss`
- `final_train_loss`
- `final_val_loss`
- `heldout_acf_r2`
- `heldout_ks_stat`
- `heldout_delta_energy_pct`

### `eval.py`

Purpose:

- Loads the heldout bundle from `prepared/`
- Rebuilds heldout rollout features from cached request arrays
- Runs the trained model on the heldout traces
- Produces one deterministic predicted power trace per heldout trace
- Computes heldout metrics against ground truth

This file is imported by `train.py`. It does not currently expose a standalone
CLI. Its main interface is:

```python
evaluate_trained_model(...)
```

Returned metrics:

- `acf_r2`
- `ks_stat`
- `delta_energy_pct`

Aggregation rule:

- Metrics are computed per heldout trace
- Final values are medians across heldout traces

## Evaluation Path

The evaluation path is intentionally fixed and deterministic.

`eval.py` does not sample rollouts or compare multiple generation modes.
Instead, it does the following:

1. Builds rollout features from the heldout request schedule
2. Runs the trained classifier to get per-timestep logits
3. Applies `softmax(logits)`
4. Maps the state posterior directly to expected power using the fitted GMM
   means:

```text
predicted_power_t = softmax(logits_t) @ gmm_means
```

This keeps evaluation simple and stable for iterative training changes.

## Prepared Artifacts

### `prepared/dataset.npz`

Copied from the existing experimental manifest entry. Contains the local traces
for this config, including:

- power traces
- active request traces
- `t_arrive_log`
- `pair_key`
- `power_start_epoch_s`
- `dt`

### `prepared/split.json`

The original train/val/test split for this config.

### `prepared/norm_params.json`

The original normalization payload from the experimental manifest.

### `prepared/throughput_database.json`

A filtered throughput DB that contains only `gpt-oss-20b_A100_tp1`.

### `prepared/heldout_requests.npz`

Fast cache for heldout request schedules.

Keys:

- `trace_idx`
- `pair_key`
- `rate`
- `request_arrival_time_s`
- `input_tokens`
- `output_tokens`

## Typical Workflow

### 1. Prepare the local bundle

```bash
/Users/grantwilkins/anaconda3/envs/powertrace/bin/python \
  isolated/gpt_oss_20b_a100_tp1/prepare.py --force
```

### 2. Run a smoke train

```bash
/Users/grantwilkins/anaconda3/envs/powertrace/bin/python \
  isolated/gpt_oss_20b_a100_tp1/train.py --epochs 1 --device cpu
```

### 3. Run a normal train

```bash
/Users/grantwilkins/anaconda3/envs/powertrace/bin/python \
  isolated/gpt_oss_20b_a100_tp1/train.py
```

## Important Environment Note

The system `python3` on this machine is not usable for this sandbox because its
`numpy` / `scikit-learn` installation is ABI-incompatible with the repo's ML
stack.

The verified working interpreter for this sandbox is:

```text
/Users/grantwilkins/anaconda3/envs/powertrace/bin/python
```

If you use another interpreter, expect import failures unless it has compatible
`numpy`, `scikit-learn`, and `torch` builds.

## What This Sandbox Is Not

This sandbox is not:

- a replacement for the main repo training pipeline
- a paper-table generator
- a multi-mode surrogate evaluator
- a checkpoint management system

It is a focused place to modify training behavior for one config and immediately
inspect the heldout metrics that come out of that model.

## If You Want To Change Behavior

Use these files as the main control points:

- change model or training behavior in `train.py`
- change heldout scoring behavior in `eval.py`
- change packaged data sources in `prepare.py`

If you want evaluation to use a different prediction path, that change belongs
in `eval.py`, not `train.py`.
