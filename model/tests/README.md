# Tests

This directory contains unit tests and integration tests for the PowerTrace-Sim model package.

## Running Tests

```bash
# Run all tests
pytest model/tests/ -v

# Run specific test file
pytest model/tests/test_train_gmm_bigru.py -v

# Run with coverage
pytest model/tests/ --cov=model --cov-report=html

# Run tests matching a pattern
pytest model/tests/ -k "gmm" -v
```

## Test Files

### GMM-BiGRU Pipeline Tests

| File | Description |
|------|-------------|
| `test_train_gmm_bigru.py` | Training pipeline tests |
| `test_eval_gmm_bigru.py` | Evaluation pipeline tests |
| `test_infer_gmm_bigru.py` | Inference pipeline tests |
| `test_gmm_bigru_utils.py` | GMM-BiGRU utility function tests |

### Simulator Tests

| File | Description |
|------|-------------|
| `test_servegen_simulator.py` | ServeGen integration tests |
| `validate_token_simulator.py` | Token simulation validation |
| `token_sim.py` | Token simulator unit tests |

### Evaluation Script Tests

| File | Description |
|------|-------------|
| `test_eval_baselines_scripts.py` | Baseline comparison script tests |
| `test_collect_results.py` | Results collection tests |

### Azure Trace Tests

| File | Description |
|------|-------------|
| `test_parse_azure_trace.py` | Azure trace parsing tests |
| `test_split_azure_week_to_days.py` | Trace splitting tests |
| `test_azure_trace_utils.py` | Trace utility tests |
| `test_azure_to_node_streams.py` | Node stream conversion tests |
| `test_azure_aggregate.py` | Aggregation tests |
| `test_azure_generate_traces.py` | Trace generation tests |
| `test_azure_metrics.py` | Azure metrics computation tests |
| `test_azure_figures.py` | Azure figure generation tests |

### Figure Generation Tests

| File | Description |
|------|-------------|
| `test_generate_methods_figures.py` | Methods figure generation tests |
| `test_hierarchy_figure.py` | Hierarchy visualization tests |
| `test_oversubscription_figure.py` | Oversubscription figure tests |
| `test_aggregation_variance.py` | Aggregation variance figure tests |
| `test_aggregation_resolution.py` | Resolution analysis figure tests |

### Other Tests

| File | Description |
|------|-------------|
| `ac.py` | Autocorrelation utility tests |

## Test Structure

Tests follow pytest conventions:

```python
# test_example.py
import pytest
import numpy as np

class TestExampleFeature:
    """Test suite for example feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = example_function(input_data)
        assert result is not None
        assert result.shape == expected_shape

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            example_function(invalid_input)

    @pytest.mark.parametrize("input,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
    ])
    def test_parametrized(self, input, expected):
        """Test multiple inputs."""
        assert square(input) == expected
```

## Fixtures

Common fixtures are defined in test files or `conftest.py`:

```python
@pytest.fixture
def sample_requests():
    """Generate sample request data."""
    return [
        ServeGenRequest(i, arrival_time=i*0.5, input_tokens=100, output_tokens=50)
        for i in range(10)
    ]

@pytest.fixture
def temp_checkpoint(tmp_path):
    """Create temporary checkpoint file."""
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}}, path)
    return path
```

## Validation Results

The `validation_results/` subdirectory contains:

- Ground truth data for regression testing
- Expected output files for comparison
- Baseline metrics for validation

## Environment Variables

Some tests respect these environment variables:

```bash
# Limit CPU threads for determinism
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Skip slow tests
export SKIP_SLOW_TESTS=1
```

## Writing New Tests

1. Create test file matching `test_*.py` pattern
2. Import modules from `model.*` namespace
3. Use fixtures for shared setup
4. Test both success and failure cases
5. Add docstrings explaining test purpose

Example template:

```python
"""Tests for new_feature module."""

import pytest
import numpy as np
import torch

from model.new_module import NewFeature


class TestNewFeature:
    """Test suite for NewFeature class."""

    @pytest.fixture
    def feature_instance(self):
        """Create feature instance for testing."""
        return NewFeature(param=10)

    def test_initialization(self, feature_instance):
        """Test proper initialization."""
        assert feature_instance.param == 10

    def test_process_valid_input(self, feature_instance):
        """Test processing with valid input."""
        result = feature_instance.process(np.ones(100))
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    def test_process_empty_input(self, feature_instance):
        """Test handling of empty input."""
        with pytest.raises(ValueError, match="empty"):
            feature_instance.process(np.array([]))
```
