#!/usr/bin/env python3
"""
Pipeline utilities for evaluation scripts.

This module re-exports functions from the primary GMM-BiGRU pipeline
to provide a consistent interface for evaluation scripts.
"""
from __future__ import annotations

# Re-export from model.classifiers.gmm_bigru
from model.classifiers.gmm_bigru import (
    AR1_MIN_RUN_LENGTH,
    AR1_PHI_THRESHOLD,
    build_rollout_features_from_requests,
    estimate_ar1_params,
    extract_norm_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
    load_gmm_params_json_dict,
    predict_sorted_gmm_labels_from_params,
)
from model.classifiers.model_loading import load_gru_classifier
from model.pipeline.artifact_resolution import (
    resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths,
    resolve_throughput,
)

# Public API
__all__ = [
    "AR1_MIN_RUN_LENGTH",
    "AR1_PHI_THRESHOLD",
    "build_rollout_features_from_requests",
    "estimate_ar1_params",
    "extract_norm_params",
    "generate_gmm_bigru_trace",
    "generate_gmm_bigru_trace_ar1_thresholded",
    "load_gmm_params_json_dict",
    "load_gru_classifier",
    "predict_sorted_gmm_labels_from_params",
    "resolve_checkpoint_norm_gmm_paths",
    "resolve_experimental_paths",
    "resolve_throughput",
]
