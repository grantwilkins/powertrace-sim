#!/usr/bin/env python3
"""
Pipeline utilities for evaluation scripts.

This module re-exports functions from the primary GMM-BiGRU pipeline
to provide a consistent interface for evaluation scripts.
"""
from __future__ import annotations

# Re-export from model.classifiers.gmm_bigru
from model.classifiers.gmm_bigru import (
    build_rollout_features_from_requests,
    generate_gmm_bigru_trace,
    load_gmm_params_json_dict,
)

# Re-export from model.scripts.eval_gmm_bigru
from model.scripts.eval_gmm_bigru import (
    estimate_ar1_params,
    generate_gmm_bigru_trace_ar1_thresholded,
    predict_sorted_gmm_labels_from_params,
)

# Public API
__all__ = [
    "build_rollout_features_from_requests",
    "estimate_ar1_params",
    "generate_gmm_bigru_trace",
    "generate_gmm_bigru_trace_ar1_thresholded",
    "load_gmm_params_json_dict",
    "predict_sorted_gmm_labels_from_params",
]
