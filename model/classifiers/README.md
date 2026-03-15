# Classifiers

This directory contains the core model code for the GMM-BiGRU pipeline.

## Files

- `gmm_bigru.py`: pipeline utilities for GMM fitting, feature building, label generation, and trace rollout.
- `gru.py`: BiGRU classifier module used for state-logit prediction.
- `feature_utils.py`: low-level feature helpers used by rollout/training code.
- `metrics.py`: trace-quality metric functions.
- `model_loading.py`: model/loading helpers shared by scripts.

## `gmm_bigru.py` API (Current)

### GMM + Serialization

- `fit_power_gmm(power_values, k=10, random_state=42, n_init=10, max_iter=300, reg_covar=1e-6)`
- `gmm_params_to_json_dict(gmm_params)`
- `load_gmm_params_json_dict(payload)`

### Labels + Feature Prep

- `extract_norm_params(norm_payload)`
- `build_state_labels(power_values, gmm_params)`
- `build_features_from_active(active_requests, t_arrive_log, norm, feature_set="f2", max_length=None)`
- `build_rollout_features_from_requests(requests, throughput, norm, T=None, dt=0.25, feature_set="f2")`
- `predict_sorted_gmm_labels_from_params(power_values, gmm_params)`

### Trace Generation

- `generate_gmm_bigru_trace(logits, gmm_params, seed=None, decode_mode="stochastic", median_filter_window=1, clamp_range=None)`
- `estimate_ar1_params(gmm_params, training_power_traces, training_labels_traces, K, min_run_length=5)`
- `generate_gmm_bigru_trace_ar1_thresholded(logits=..., gmm_params=..., phi=..., sigma_innov=..., sigma_marginal=..., p0=..., seed=None, decode_mode="stochastic", median_filter_window=1, phi_threshold=0.3, clamp_range=None)`

## Feature Set Policy

- Only `f2` is supported.
- `f3` paths were removed from the pipeline and will raise an error if selected.

## Example

```python
import numpy as np
from model.classifiers.gmm_bigru import (
    build_rollout_features_from_requests,
    generate_gmm_bigru_trace,
)

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
