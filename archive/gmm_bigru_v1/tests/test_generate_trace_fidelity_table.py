import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.eval.generate_trace_fidelity_table import (
    aggregate_seed_to_config_total_energy,
    canonicalize_generation_mode,
    filter_negative_acf_configs,
    load_config_summary_csv,
    load_per_seed_csv,
    load_request_timestamp_availability_map_merged,
    select_expected_generation_mode,
    select_generation_mode,
    select_best_generation_mode,
)


class TestGenerateTraceFidelityTable(unittest.TestCase):
    def test_canonicalize_generation_mode_maps_thresholded_alias(self):
        self.assertEqual(canonicalize_generation_mode("ar1_thresholded"), "ar1_thresh")
        self.assertEqual(canonicalize_generation_mode("ar1_thresh"), "ar1_thresh")

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

    def test_select_expected_generation_mode_prefers_expected_mode(self):
        df = pd.DataFrame(
            [
                {
                    "model": "llama-3",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "iid",
                    "n_configs": 4,
                    "KS_mean": 0.30,
                },
                {
                    "model": "llama-3",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "ar1_thresh",
                    "n_configs": 4,
                    "KS_mean": 0.10,
                },
            ]
        )

        out = select_expected_generation_mode(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["generation_mode"]), "iid")

    def test_select_expected_generation_mode_requires_max_coverage(self):
        df = pd.DataFrame(
            [
                {
                    "model": "gpt-oss",
                    "model_size": 120,
                    "arch_type": "moe",
                    "generation_mode": "ar1",
                    "n_configs": 1,
                    "KS_mean": 0.10,
                },
                {
                    "model": "gpt-oss",
                    "model_size": 120,
                    "arch_type": "moe",
                    "generation_mode": "ar1_thresh",
                    "n_configs": 3,
                    "KS_mean": 0.40,
                },
                {
                    "model": "gpt-oss",
                    "model_size": 120,
                    "arch_type": "moe",
                    "generation_mode": "iid",
                    "n_configs": 3,
                    "KS_mean": 0.50,
                },
            ]
        )

        out = select_expected_generation_mode(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["generation_mode"]), "ar1_thresh")

    def test_select_generation_mode_rejects_unknown_strategy(self):
        df = pd.DataFrame(
            [
                {
                    "model": "llama-3",
                    "model_size": 8,
                    "arch_type": "dense",
                    "generation_mode": "iid",
                    "n_configs": 1,
                    "KS_mean": 0.25,
                }
            ]
        )

        with self.assertRaises(ValueError):
            select_generation_mode(df, selection_strategy="nope")

    def test_aggregate_seed_to_config_total_energy_sums_across_traces_first(self):
        df = pd.DataFrame(
            [
                {
                    "config_id": "cfg",
                    "generation_mode": "iid",
                    "seed": 42,
                    "energy_gt_j": 10.0,
                    "energy_pred_j": 0.0,
                    "delta_energy_pct": 100.0,
                },
                {
                    "config_id": "cfg",
                    "generation_mode": "iid",
                    "seed": 42,
                    "energy_gt_j": 90.0,
                    "energy_pred_j": 100.0,
                    "delta_energy_pct": 11.1111111111,
                },
            ]
        )

        out = aggregate_seed_to_config_total_energy(df)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["delta_energy_pct"]), 0.0, places=9)

    def test_load_config_summary_csv_prefers_full_heldout_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "config_summary.csv"
            run_manifest_path = Path(tmp) / "run_manifest.json"
            run_manifest_path.write_text(
                json.dumps({"defaults": {"generation_mode": "iid"}})
            )
            pd.DataFrame(
                [
                    {
                        "config_id": "cfg",
                        "status": "evaluated",
                        "num_eval_traces": 7,
                        "ks_stat_all_heldout": 0.11,
                        "acf_r2_all_heldout": 0.92,
                        "nrmse_all_heldout": 0.21,
                        "delta_energy_pct_all_heldout": 3.4,
                    }
                ]
            ).to_csv(csv_path, index=False)

            out = load_config_summary_csv(
                str(csv_path), generation_mode="iid", prefer_all_heldout=True
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(str(out.iloc[0]["generation_mode"]), "iid")
            self.assertAlmostEqual(float(out.iloc[0]["ks_stat"]), 0.11, places=9)
            self.assertAlmostEqual(float(out.iloc[0]["acf_r2"]), 0.92, places=9)
            self.assertAlmostEqual(float(out.iloc[0]["nrmse"]), 0.21, places=9)
            self.assertAlmostEqual(
                float(out.iloc[0]["delta_energy_pct"]), 3.4, places=9
            )

    def test_load_per_seed_csv_rejects_manifest_generation_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "per_seed_metrics.csv"
            run_manifest_path = Path(tmp) / "run_manifest.json"
            run_manifest_path.write_text(
                json.dumps({"defaults": {"generation_mode": "ar1_thresholded"}})
            )
            pd.DataFrame(
                [
                    {
                        "config_id": "cfg",
                        "trace_idx": 0,
                        "pair_key": "pair",
                        "seed": 42,
                        "status": "ok",
                        "ks_stat": 0.1,
                        "acf_r2": 0.9,
                        "nrmse": 0.2,
                        "delta_energy_pct": 1.0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "Generation mode mismatch"):
                load_per_seed_csv(str(csv_path), generation_mode="iid", require_pair_key=True)

    def test_load_config_summary_csv_rejects_manifest_generation_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "config_summary.csv"
            run_manifest_path = Path(tmp) / "run_manifest.json"
            run_manifest_path.write_text(
                json.dumps({"defaults": {"generation_mode": "ar1_thresholded"}})
            )
            pd.DataFrame(
                [
                    {
                        "config_id": "cfg",
                        "status": "evaluated",
                        "generation_mode": "ar1_thresholded",
                        "num_eval_traces": 7,
                        "ks_stat_all_heldout": 0.11,
                        "acf_r2_all_heldout": 0.92,
                        "nrmse_all_heldout": 0.21,
                        "delta_energy_pct_all_heldout": 3.4,
                    }
                ]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "Generation mode mismatch"):
                load_config_summary_csv(
                    str(csv_path), generation_mode="iid", prefer_all_heldout=True
                )

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
