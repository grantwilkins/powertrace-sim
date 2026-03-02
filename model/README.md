# Model Package

The `model/` package contains all machine learning models, simulators, and utilities for power trace generation. This is the core codebase for training, evaluating, and running inference with the GMM-BiGRU pipeline.

## Package Structure

```
model/
├── classifiers/          # ML model implementations
│   ├── gmm_bigru.py      # Main GMM-BiGRU pipeline (primary)
│   ├── gru.py            # GRU classifier architecture
│   ├── feature_utils.py  # Feature computation utilities
│   └── metrics.py        # Evaluation metrics (KS, ACF, etc.)
├── core/                 # Dataset and utility classes
│   ├── dataset.py        # PowerTraceDataset for training
│   └── utils.py          # Feature matrix construction
├── examples/             # Runnable demonstration scripts
├── scripts/              # Training, evaluation, inference CLI
│   ├── train_gmm_bigru.py
│   ├── eval_gmm_bigru.py
│   ├── infer_gmm_bigru.py
│   └── compare_gmm_bigru.py
├── simulators/           # Request arrival and power simulation
│   ├── arrival_simulator.py     # ServeGen integration, request processing
│   ├── server_power_simulator.py # Single-server power simulation
│   └── datacenter_simulator.py  # Multi-node datacenter simulation
├── tests/                # Unit and integration tests
├── training_data/        # Data preprocessing pipeline
│   ├── utils/            # Preparation and alignment scripts
│   └── losses/           # Training loss visualization
├── predictors/           # Power smoothing utilities
├── utils/                # Performance statistics extraction
├── config/               # Configuration files
│   └── performance_database.json  # TTFT/TPOT distributions
├── best_weights/         # Curated model checkpoints
├── gru_classifier_weights/  # Additional weight storage
└── new_weights/          # Experimental weights
```

## Core Pipeline: GMM-BiGRU

The GMM-BiGRU pipeline is the primary method for power trace generation:

1. **GMM Fitting**: Fit a Gaussian Mixture Model to power values, creating K power "states"
2. **Feature Extraction**: Compute workload features (active requests, token rates, delta features)
3. **BiGRU Training**: Train a bidirectional GRU to predict power state sequences from features
4. **Generation**: Sample from GMM components based on predicted state probabilities

### Key Components

#### `classifiers/gmm_bigru.py`

Main pipeline implementation:

- `fit_power_gmm()` - Fit GMM to power values
- `build_features_from_active()` - Construct feature matrices
- `build_state_labels()` - Generate GMM state labels for training
- `generate_gmm_bigru_trace()` - Generate power traces at inference time
- `generate_gmm_bigru_trace_ar1_thresholded()` - AR(1) smoothed generation

#### `classifiers/gru.py`

Neural network architecture:

- `GRUClassifier` - Bidirectional GRU with configurable layers/hidden size
- Supports variable sequence lengths and batch processing

#### `classifiers/feature_utils.py`

Feature computation:

- `compute_inference_features()` - Features for inference from request stream
- `compute_delta_active_requests()` - Rate of change in concurrent requests
- `normalize_delta_active_requests()` - Standardization with stored parameters

#### `classifiers/metrics.py`

Evaluation metrics:

- `compute_power_metrics()` - Full metric suite (KS, ACF R², NRMSE, percentiles, energy)

## Scripts

### Training

```bash
python -m model.scripts.train_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --out-dir results/my_experiment \
    --num-components 10 \
    --feature-set f2
```

### Evaluation

```bash
python -m model.scripts.eval_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --checkpoint-dir results/my_experiment/checkpoints \
    --norm-dir results/my_experiment/norm_params \
    --gmm-dir results/my_experiment/gmms \
    --out-dir results/my_experiment/eval_metrics
```

### Inference

```bash
python -m model.scripts.infer_gmm_bigru \
    --requests requests.csv \
    --checkpoint checkpoint.pt \
    --norm norm.json \
    --gmm gmm.json \
    --out-csv power_trace.csv
```

### Comparison

```bash
python -m model.scripts.compare_gmm_bigru \
    --results-dir results/my_experiment
```

## Simulators

### `simulators/arrival_simulator.py`

Integrates with ServeGen for realistic workload generation:

- `ServingConfig` - Model/hardware configuration
- `ServingSystemSimulator` - Request processing simulation
- `ServeGenPowerSimulator` - End-to-end power from workload
- `create_llama_config()`, `create_deepseek_config()` - Configuration factories

### `simulators/server_power_simulator.py`

Single-server power simulation:

- `ServerPowerSimulator` - Converts request stream to power trace
- Supports both fast (synthetic) and ServeGen workloads

### `simulators/datacenter_simulator.py`

Multi-node datacenter modeling:

- `DataCenterSimulator` - Rack/row/datacenter hierarchy
- `RowSpec` - Row configuration (A100/H100 rack counts)
- Parallel simulation with configurable worker count

## Configuration

### Feature Sets

- **f2**: 2 features - active_requests, delta_active_requests (normalized)
- **f3**: 3 features - adds cumulative token rate

### Hyperparameters

Default GMM-BiGRU settings:

- `num_components`: 10 (K=10 GMM components)
- `hidden_size`: 64 (GRU hidden dimension)
- `num_layers`: 2 (GRU layers)
- `bidirectional`: True
- `learning_rate`: 1e-3
- `epochs`: 100

## Testing

```bash
# Run all model tests
pytest model/tests/ -v

# Run specific test
pytest model/tests/test_train_gmm_bigru.py -v
```

## Dependencies

The model package requires:

- PyTorch >= 2.0
- NumPy >= 1.24
- scikit-learn >= 1.3 (for GMM)
- scipy >= 1.11 (for statistics)

Optional:
- matplotlib (for visualization scripts)
- ServeGen (for realistic workload generation)
