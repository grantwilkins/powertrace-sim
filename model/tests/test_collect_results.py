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


class TestSelectionPolicy:
    def test_dense_iid_and_moe_ar1_with_fallback(self):
        iid_df = pd.DataFrame(
            {
                "config_id": [
                    "llama-3-8b_H100_tp1",  # dense
                    "gpt-oss-20b_H100_tp1",  # moe (fallback)
                    "gpt-oss-20b_H100_tp2",  # moe (replaced)
                ],
                "ks_stat_median": [0.10, 0.20, 0.30],
                "acf_r2_median": [0.90, 0.80, 0.70],
                "nrmse_median": [0.11, 0.22, 0.33],
                "delta_energy_pct_median": [1.0, 2.0, 3.0],
                "generation_mode": ["iid", "iid", "iid"],
            }
        )

        ar1_df = pd.DataFrame(
            {
                "config_id": [
                    "gpt-oss-20b_H100_tp2",  # should replace i.i.d.
                    "llama-3-8b_H100_tp1",  # dense row should be ignored
                ],
                "ks_stat_median": [0.05, 0.99],
                "acf_r2_median": [0.95, 0.01],
                "nrmse_median": [0.15, 0.99],
                "delta_energy_pct_median": [0.5, 99.0],
                "generation_mode": ["ar1", "ar1"],
            }
        )

        selected, fallback_moe = select_generation_rows(iid_df=iid_df, ar1_df=ar1_df)

        assert len(selected) == 3
        assert fallback_moe == ["gpt-oss-20b_H100_tp1"]

        dense_row = selected.loc[selected["config_id"] == "llama-3-8b_H100_tp1"].iloc[0]
        assert dense_row["source_mode"] == "iid"
        assert dense_row["ks_stat_median"] == pytest.approx(0.10)

        replaced_row = selected.loc[
            selected["config_id"] == "gpt-oss-20b_H100_tp2"
        ].iloc[0]
        assert replaced_row["source_mode"] == "ar1"
        assert replaced_row["ks_stat_median"] == pytest.approx(0.05)
        assert replaced_row["delta_energy_pct_median"] == pytest.approx(0.5)

    def test_all_moe_fallback_when_ar1_missing(self):
        iid_df = pd.DataFrame(
            {
                "config_id": ["gpt-oss-120b_H100_tp4"],
                "ks_stat_median": [0.2],
                "acf_r2_median": [0.8],
                "nrmse_median": [0.3],
                "delta_energy_pct_median": [1.5],
                "generation_mode": ["iid"],
            }
        )
        selected, fallback_moe = select_generation_rows(iid_df=iid_df, ar1_df=None)
        assert len(selected) == 1
        assert fallback_moe == ["gpt-oss-120b_H100_tp4"]
        assert selected.iloc[0]["source_mode"] == "iid"


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
                "generation_mode": ["iid", "iid", "iid", "ar1"],
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
                "source_mode": ["iid", "iid", "iid", "ar1"],
            }
        )

        out = aggregate_by_model(selected_df)
        assert len(out) == 2

        dense = out.loc[out["arch_type"] == "dense"].iloc[0]
        moe = out.loc[out["arch_type"] == "moe"].iloc[0]

        assert dense["generation_mode"] == "iid"
        assert moe["generation_mode"] == "ar1_with_iid_fallback"
        assert int(dense["n_configs"]) == 2
        assert int(moe["n_configs"]) == 2


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
