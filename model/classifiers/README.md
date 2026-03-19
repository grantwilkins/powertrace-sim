# Classifiers

This directory contains the model components used by the GMM-BiGRU pipeline.

## Files

- `gmm_bigru.py`: GMM fitting plus parameter serialization and state-label helpers.
- `features.py`: feature construction and normalization for training and inference.
- `trace_generation.py`: power-trace rollout and AR(1)-augmented generation utilities.
- `gru.py`: BiGRU classifier used for state-logit prediction.
- `metrics.py`: trace-quality metric functions.
- `model_loading.py`: checkpoint loading helpers shared by scripts.

## Key APIs

### `gmm_bigru.py`

- `fit_power_gmm(power_values, k=10, random_state=42, n_init=10, max_iter=300, reg_covar=1e-6)`
- `gmm_params_to_json_dict(gmm_params)`
- `load_gmm_params_json_dict(payload)`
- `build_state_labels(power_values, gmm_params)`
- `predict_sorted_gmm_labels_from_params(power_values, gmm_params)`

### `features.py`

- `compute_delta_active_requests(active_requests)`
- `compute_inference_features(requests, config, T=None, dt=0.25)`
- `extract_norm_params(norm_payload)`
- `build_features_from_active(active_requests, t_arrive_log, norm, feature_set="f2", max_length=None)`
- `build_rollout_features_from_requests(requests, throughput, norm, T=None, dt=0.25, feature_set="f2")`

### `trace_generation.py`

- `generate_gmm_bigru_trace(logits, gmm_params, seed=None, decode_mode="stochastic", median_filter_window=1, clamp_range=None)`
- `estimate_ar1_params(gmm_params, training_power_traces, training_labels_traces, K, min_run_length=5)`
- `generate_gmm_bigru_trace_ar1_thresholded(logits=..., gmm_params=..., phi=..., sigma_innov=..., sigma_marginal=..., p0=..., seed=None, decode_mode="stochastic", median_filter_window=1, phi_threshold=0.3, clamp_range=None)`

### `metrics.py`

- `ks_statistic(x, y)`
- `autocorrelation_r2(real, synthetic, max_lag=50)`
- `autocorrelation_r2_aggregate(real_traces, synthetic_traces, max_lag=50)`
- `compute_power_metrics(ground_truth_w, generated_w, dt, acf_max_lag=50)`
- `compute_aggregate_power_metrics(ground_truth_traces, generated_traces, dt, acf_max_lag=50)`

### `model_loading.py`

- `load_gru_classifier(checkpoint_path, k, input_dim, hidden_dim, num_layers, device)`

## Feature Set Policy

- Only `f2` is supported.
- `f3` paths were removed from the pipeline and will raise an error if selected.

## Example

```python
import numpy as np
from model.classifiers.features import build_rollout_features_from_requests
from model.classifiers.trace_generation import generate_gmm_bigru_trace

feature_pack = build_rollout_features_from_requests(
    requests=requests,
    throughput={"lambda_prefill": 120.0, "lambda_decode": 60.0},
    norm=norm_params,
    T=400,
    dt=0.25,
    feature_set="f2",
)

trace = generate_gmm_bigru_trace(
    logits=logits,
    gmm_params=gmm_params,
    seed=7,
    decode_mode="stochastic",
)
power_w = np.asarray(trace["power_w"], dtype=np.float64)
```
