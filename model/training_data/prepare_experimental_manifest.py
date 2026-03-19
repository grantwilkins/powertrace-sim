#!/usr/bin/env python3
"""Backward-compatible wrapper for the split manifest preparation modules."""
from __future__ import annotations

from model.training_data.alignment import (
    align_trace_to_grid as _align_trace_to_grid,
    compute_active_requests as _compute_active_requests,
    compute_t_arrive_log as _compute_t_arrive_log,
)
from model.training_data.manifest import (
    build_arg_parser,
    main,
    run_prepare_experimental_manifest,
)
from model.training_data.normalization import (
    compute_normalization_stats as _compute_normalization_stats,
    create_train_val_test_split as _create_train_val_test_split,
)
from model.training_data.power_parsing import (
    parse_power_csv as _parse_power_csv,
    parse_request_json as _parse_request_json,
)
from model.utils.io import power_timestamp_to_epoch as _power_timestamp_to_epoch


if __name__ == "__main__":
    main()
