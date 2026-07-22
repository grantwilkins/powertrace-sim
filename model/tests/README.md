# Tests

This directory contains unit tests and integration tests for the PowerTrace-Sim model package.

## Running Tests

```bash
# Run all tests
uv run -m pytest -x

# Run a specific maintained test file
uv run -m pytest -x model/tests/test_clean_lifecycle.py

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
- `test_io.py`
- `test_metrics.py`
- `test_clean_model.py`
- `test_clean_lifecycle.py`

### Training Data and Manifest Flow

Representative files:

- `test_inventory.py`
- `test_manifest.py`
- `test_stage0_inventory.py`
- `test_timestamp_alignment.py`
- `test_split_azure_week_to_days.py`

### CLI Wrappers

- `test_clean_lifecycle.py`
- `test_run_azure_pipeline.py`

### Reporting and Figure Scripts

- `test_collect_results.py`
- `test_collect_results.py`

### Azure Trace and Figure Suites

- `test_azure_aggregate.py`
- `test_azure_figures.py`
- `test_azure_generate_traces.py`
- `test_azure_metrics.py`
- `test_azure_to_node_streams.py`
- `test_azure_trace_utils.py`
- `test_generate_azure_facility_sizing_table.py`
- `test_hierarchy_figure.py`
- `test_oversubscription_figure.py`
- `test_parse_azure_trace.py`

The historical model has an independent suite under
`archive/gmm_bigru_v1/tests/`; its README documents the archive command.
