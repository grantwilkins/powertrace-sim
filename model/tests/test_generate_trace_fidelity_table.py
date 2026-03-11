import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.eval.generate_trace_fidelity_table import (
    filter_negative_acf_configs,
    load_request_timestamp_availability_map_merged,
    select_best_generation_mode,
)


class TestGenerateTraceFidelityTable(unittest.TestCase):
    def test_filter_negative_acf_configs_drops_entire_config(self):
        df = pd.DataFrame(
            [
                {"config_id": "cfg_good", "generation_mode": "iid", "acf_r2": 0.2},
                {"config_id": "cfg_good", "generation_mode": "ar1_thresh", "acf_r2": 0.1},
                {"config_id": "cfg_bad", "generation_mode": "iid", "acf_r2": -0.01},
                {"config_id": "cfg_bad", "generation_mode": "ar1_thresh", "acf_r2": 0.5},
            ]
        )
        out, stats = filter_negative_acf_configs(df)
        self.assertEqual(set(out["config_id"].tolist()), {"cfg_good"})
        self.assertEqual(int(stats["rows_excluded"]), 2)
        self.assertEqual(stats["excluded_configs"], ["cfg_bad"])

    def test_select_best_generation_mode_requires_max_coverage(self):
        df = pd.DataFrame(
            [
                {
                    "model": "deepseek-r1-distill",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "ar1",
                    "n_configs": 1,
                    "KS_mean": 0.10,
                },
                {
                    "model": "deepseek-r1-distill",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "iid",
                    "n_configs": 4,
                    "KS_mean": 0.30,
                },
                {
                    "model": "deepseek-r1-distill",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "ar1_thresh",
                    "n_configs": 4,
                    "KS_mean": 0.20,
                },
            ]
        )

        out = select_best_generation_mode(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["generation_mode"]), "ar1_thresh")

    def test_select_best_generation_mode_tie_prefers_expected_mode(self):
        df = pd.DataFrame(
            [
                {
                    "model": "llama-3",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "iid",
                    "n_configs": 3,
                    "KS_mean": 0.25,
                },
                {
                    "model": "llama-3",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "ar1_thresh",
                    "n_configs": 3,
                    "KS_mean": 0.25,
                },
            ]
        )

        out = select_best_generation_mode(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["generation_mode"]), "iid")

    def test_select_best_generation_mode_tie_prefers_moe_fallback_order(self):
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-oss",
                    "model_size": 120,
                    "arch_type": "moe",
                    "generation_mode": "iid",
                    "n_configs": 2,
                    "KS_mean": 0.30,
                },
                {
                    "model": "gpt-oss",
                    "model_size": 120,
                    "arch_type": "moe",
                    "generation_mode": "ar1_thresh",
                    "n_configs": 2,
                    "KS_mean": 0.30,
                },
            ]
        )

        out = select_best_generation_mode(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["generation_mode"]), "ar1_thresh")

    def test_load_request_timestamp_availability_map_merged_uses_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv1 = root / "data_inventory_1.json"
            inv2 = root / "data_inventory_2.json"

            inv1.write_text(
                json.dumps(
                    {
                        "paired_runs": [
                            {
                                "pair_key": "a",
                                "request_extraction_stats": {
                                    "missing_request_timestamps_array": True
                                },
                            },
                            {
                                "pair_key": "b",
                                "request_extraction_stats": {
                                    "missing_request_timestamps_array": False
                                },
                            },
                        ]
                    }
                )
            )
            inv2.write_text(
                json.dumps(
                    {
                        "paired_runs": [
                            {
                                "pair_key": "a",
                                "request_extraction_stats": {
                                    "missing_request_timestamps_array": False
                                },
                            }
                        ]
                    }
                )
            )

            merged = load_request_timestamp_availability_map_merged(
                [str(inv1), str(inv2)]
            )
            self.assertTrue(bool(merged["a"]))
            self.assertTrue(bool(merged["b"]))


if __name__ == "__main__":
    unittest.main()
