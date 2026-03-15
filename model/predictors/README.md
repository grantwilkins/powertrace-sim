# Predictors

This package is intentionally minimal after the pipeline cleanup.

## Status

- Legacy predictor modules were removed.
- Power-trace sampling and AR(1) logic now live in `model.classifiers.gmm_bigru`.

## Active APIs

Use these functions from `model.classifiers.gmm_bigru`:

- `generate_gmm_bigru_trace(...)` for logits -> sampled power traces.
- `estimate_ar1_params(...)` for per-state AR(1) calibration from training traces.
- `generate_gmm_bigru_trace_ar1_thresholded(...)` for AR(1)-aware trace generation with threshold fallback.

## Migration Example

```python
from model.classifiers.gmm_bigru import (
    estimate_ar1_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
)
```

There are no standalone predictor modules in this directory anymore.
