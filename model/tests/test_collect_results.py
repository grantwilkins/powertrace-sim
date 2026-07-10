"""
Tests for scripts/eval/collect_results.py merge policy and aggregation.
"""

import csv
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from collect_results import (  # noqa: E402
    _parse_and_annotate_rows,
    aggregate_by_model,
    infer_arch_type,
    load_result_csv,
    parse_config_id,
    select_generation_rows,
)


class TestParseConfigId:
    def test_valid_config(self):
        parsed = parse_config_id("deepseek-r1-distill-70b_H100_tp4")
        assert parsed["model_family"] == "deepseek-r1-distill"
        assert parsed["model_size"] == "70"
        assert parsed["hardware"] == "H100"
        assert parsed["tp"] == "4"

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="Invalid config_id format"):
            parse_config_id("bad-format")


class TestInferArchType:
    def test_arch_rules(self):
        assert infer_arch_type("deepseek-r1-distill", 8) == "dense"
        assert infer_arch_type("gpt-oss", 20) == "moe"
        assert infer_arch_type("gpt-oss", 8) == "dense"
        assert infer_arch_type("llama-3", 70) == "dense"


class TestLoadResultCSV:
    def _make_temp_csv(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def test_load_valid_csv(self):
        path = self._make_temp_csv(
            [
                {
                    "config_id": "llama-3-8b_H100_tp1",
                    "ks_stat_median": "0.5",
                    "acf_r2_median": "0.8",
                    "nrmse_median": "0.3",
                    "delta_energy_pct_median": "5.0",
                    "generation_mode": "iid",
                }
            ]
        )
        try:
            df = load_result_csv(path, generation_mode="iid")
            assert len(df) == 1
            assert df.loc[0, "generation_mode"] == "iid"
        finally:
            os.unlink(path)

    def test_missing_column_raises(self):
        path = self._make_temp_csv(
            [{"config_id": "llama-3-8b_H100_tp1", "ks_stat_median": "0.5"}]
        )
        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                load_result_csv(path, generation_mode="iid")
        finally:
            os.unlink(path)

    def test_missing_generation_mode_is_rejected(self):
        path = self._make_temp_csv([{
            "config_id": "llama-3-8b_H100_tp1",
            "ks_stat_median": "0.5",
            "acf_r2_median": "0.8",
            "nrmse_median": "0.3",
            "delta_energy_pct_median": "5.0",
        }])
        try:
            with pytest.raises(ValueError, match="generation_mode"):
                load_result_csv(path, generation_mode="iid")
        finally:
            os.unlink(path)

    def test_recorded_generation_mode_cannot_be_overwritten(self):
        """Claim: a CSV's recorded mode is conserved; an IID request cannot relabel AR1.

        This catches both unconditional column assignment and validation after overwrite.
        """
        path = self._make_temp_csv(
            [{
                "config_id": "llama-3-8b_H100_tp1",
                "ks_stat_median": "0.5",
                "acf_r2_median": "0.8",
                "nrmse_median": "0.3",
                "delta_energy_pct_median": "5.0",
                "generation_mode": "ar1",
            }]
        )
        try:
            with pytest.raises(ValueError, match="generation_mode"):
                load_result_csv(path, generation_mode="iid")
        finally:
            os.unlink(path)


class TestSelectionPolicy:
    def test_all_architectures_keep_iid_rows(self):
        iid_df = pd.DataFrame(
            {
                "config_id": [
                    "llama-3-8b_H100_tp1",  # dense
                    "gpt-oss-20b_H100_tp1",
                    "gpt-oss-20b_H100_tp2",
                ],
                "ks_stat_median": [0.10, 0.20, 0.30],
                "acf_r2_median": [0.90, 0.80, 0.70],
                "nrmse_median": [0.11, 0.22, 0.33],
                "delta_energy_pct_median": [1.0, 2.0, 3.0],
                "generation_mode": ["iid", "iid", "iid"],
            }
        )

        selected = select_generation_rows(iid_df)

        assert len(selected) == 3
        assert set(selected["source_mode"]) == {"iid"}
        assert selected.loc[
            selected["config_id"] == "gpt-oss-20b_H100_tp2", "ks_stat_median"
        ].iloc[0] == pytest.approx(0.30)

    def test_iid_selection_rejects_non_iid_source_rows(self):
        """Claim: IID-only selection must reject, not relabel, an AR1 source row."""
        row = pd.DataFrame({
            "config_id": ["llama-3-8b_H100_tp1"],
            "ks_stat_median": [0.1],
            "acf_r2_median": [0.9],
            "nrmse_median": [0.1],
            "delta_energy_pct_median": [1.0],
            "generation_mode": ["ar1"],
        })
        with pytest.raises(ValueError, match="IID"):
            select_generation_rows(row)


class TestAggregation:
    def test_generation_mode_labels_and_counts(self):
        selected_df = pd.DataFrame(
            {
                "config_id": [
                    "llama-3-8b_H100_tp1",
                    "llama-3-8b_A100_tp1",
                    "gpt-oss-20b_H100_tp1",
                    "gpt-oss-20b_H100_tp2",
                ],
                "ks_stat_median": [0.10, 0.20, 0.30, 0.40],
                "acf_r2_median": [0.9, 0.8, 0.7, 0.6],
                "nrmse_median": [0.1, 0.2, 0.3, 0.4],
                "delta_energy_pct_median": [1.0, 2.0, 3.0, 4.0],
                "generation_mode": ["iid", "iid", "iid", "iid"],
                "model_family": [
                    "llama-3",
                    "llama-3",
                    "gpt-oss",
                    "gpt-oss",
                ],
                "model_size": [8, 8, 20, 20],
                "hardware": ["H100", "A100", "H100", "H100"],
                "tp": [1, 1, 1, 8],
                "arch_type": ["dense", "dense", "moe", "moe"],
                "source_mode": ["iid", "iid", "iid", "iid"],
            }
        )

        out = aggregate_by_model(selected_df)
        assert len(out) == 2

        dense = out.loc[out["arch_type"] == "dense"].iloc[0]
        moe = out.loc[out["arch_type"] == "moe"].iloc[0]

        assert dense["generation_mode"] == "iid"
        assert moe["generation_mode"] == "iid"
        assert int(dense["n_configs"]) == 2
        assert int(moe["n_configs"]) == 2

    def test_mixed_generation_modes_are_not_collapsed(self):
        """Claim: aggregation cannot report one mode for a mixed-mode model group."""
        rows = pd.DataFrame({
            "config_id": ["llama-3-8b_H100_tp1", "llama-3-8b_A100_tp1"],
            "ks_stat_median": [0.1, 0.2],
            "acf_r2_median": [0.9, 0.8],
            "nrmse_median": [0.1, 0.2],
            "delta_energy_pct_median": [1.0, 2.0],
            "generation_mode": ["iid", "ar1"],
        })
        with pytest.raises(ValueError, match="mixed generation modes"):
            aggregate_by_model(rows)


class TestParseAndAnnotateRows:
    def test_parse_and_annotate_rows_adds_arch_column(self):
        df = pd.DataFrame(
            {
                "config_id": ["gpt-oss-20b_H100_tp1"],
                "ks_stat_median": [0.2],
                "acf_r2_median": [0.8],
                "nrmse_median": [0.3],
                "delta_energy_pct_median": [1.5],
                "generation_mode": ["iid"],
            }
        )
        out = _parse_and_annotate_rows(df)
        assert len(out) == 1
        assert out.loc[0, "arch_type"] == "moe"
