# Tests

This directory contains unit tests and integration tests for the PowerTrace-Sim model package.

## Running Tests

```bash
# Run all tests
uv run -m pytest -x

# Run specific test file
uv run -m pytest -x model/tests/test_train_gmm_bigru.py

# Run with coverage
uv run -m pytest -x model/tests/ --cov=model --cov-report=html

# Run tests matching a pattern
uv run -m pytest -x model/tests/ -k "gmm"
```

## Test Groups

### Core Model and Pipeline

Representative files:

- `test_config.py`
- `test_decode_time.py`
- `test_feature_utils.py`
- `test_gmm_bigru_utils.py`
- `test_gru.py`
- `test_io.py`
- `test_metrics.py`
- `test_model_loading.py`
- `test_pipeline_roundtrip.py`
- `test_pipeline_utils.py`
- `test_data_loading_pipeline.py`

### Training Data and Manifest Flow

Representative files:

- `test_inventory.py`
- `test_manifest.py`
- `test_stage0_inventory.py`
- `test_timestamp_alignment.py`
- `test_split_azure_week_to_days.py`

### CLI Wrappers

- `test_train_gmm_bigru.py`
- `test_eval_gmm_bigru.py`
- `test_infer_gmm_bigru.py`
- `test_run_azure_pipeline.py`

### Reporting and Figure Scripts

- `test_collect_results.py`
- `test_eval_baselines_scripts.py`
- `test_generate_methods_figures.py`

### Azure Trace and Figure Suites

- `test_azure_aggregate.py`
- `test_azure_figures.py`
- `test_azure_generate_traces.py`
- `test_azure_metrics.py`
- `test_azure_to_node_streams.py`
- `test_azure_trace_utils.py`
- `test_feature_sufficiency_figure.py`
- `test_generate_azure_facility_sizing_table.py`
- `test_generate_baselines_node_table.py`
- `test_generate_trace_fidelity_table.py`
- `test_hierarchy_figure.py`
- `test_oversubscription_figure.py`
- `test_parse_azure_trace.py`

The tests are organized as standard `pytest` modules and exercise both package internals and the thin CLI wrappers under `model.scripts` and `scripts.eval`.
