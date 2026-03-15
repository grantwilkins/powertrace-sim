"""Tests for scripts/eval/run_azure_pipeline.py."""

from scripts.eval.azure_defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_METHODS_GENERATION,
    DEFAULT_OVERSUB_METHODS,
    DEFAULT_SPLITWISE_SOURCE_TP,
    DEFAULT_TRACE_KINDS,
)
from scripts.eval.run_azure_pipeline import build_parser


def test_run_azure_pipeline_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.config_id == DEFAULT_CONFIG_ID
    assert args.methods == ",".join(DEFAULT_METHODS_GENERATION)
    assert args.trace_kinds == ",".join(DEFAULT_TRACE_KINDS)
    assert args.oversub_methods == ",".join(DEFAULT_OVERSUB_METHODS)
    assert args.method_order == ",".join(DEFAULT_TRACE_KINDS)
    assert args.splitwise_source_tp == DEFAULT_SPLITWISE_SOURCE_TP
    assert args.show_redundant_baseline_cells is False
