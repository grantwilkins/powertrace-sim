import unittest
from pathlib import Path

from scripts.eval.generate_baselines_node_table import (
    _filter_negative_acf_configs,
    _select_configs,
)


class TestGenerateBaselinesNodeTable(unittest.TestCase):
    def test_filter_negative_acf_configs_excludes_entire_config(self):
        rows = [
            {
                "config_id": "cfg_a",
                "method": "ours",
                "status": "evaluated",
                "acf_r2": "0.5",
            },
            {
                "config_id": "cfg_b",
                "method": "splitwise_strict",
                "status": "evaluated",
                "acf_r2": "-0.1",
            },
            {
                "config_id": "cfg_b",
                "method": "ours",
                "status": "evaluated",
                "acf_r2": "0.7",
            },
        ]
        filtered, excluded = _filter_negative_acf_configs(
            rows=rows,
            selected_configs=["cfg_a", "cfg_b"],
        )
        self.assertEqual(filtered, ["cfg_a"])
        self.assertEqual(excluded, ["cfg_b"])

    def test_filter_negative_acf_configs_ignores_non_evaluated_rows(self):
        rows = [
            {
                "config_id": "cfg_a",
                "method": "ours",
                "status": "evaluated",
                "acf_r2": "0.5",
            },
            {
                "config_id": "cfg_b",
                "method": "splitwise_strict",
                "status": "failed",
                "acf_r2": "-0.9",
            },
        ]
        filtered, excluded = _filter_negative_acf_configs(
            rows=rows,
            selected_configs=["cfg_a", "cfg_b"],
        )
        self.assertEqual(filtered, ["cfg_a", "cfg_b"])
        self.assertEqual(excluded, [])

    def test_representative_selection_targets_llama70_a100_tp4_and_tp8(self):
        rows = [
            {"config_id": "llama-3-70b_A100_tp4", "status": "evaluated"},
            {"config_id": "llama-3-70b_A100_tp8", "status": "evaluated"},
            {"config_id": "llama-3-70b_H100_tp4", "status": "evaluated"},
            {"config_id": "deepseek-r1-distill-70b_A100_tp4", "status": "evaluated"},
        ]
        selected = _select_configs(
            rows=rows,
            config_ids=[],
            arch_filter="dense",
            representative_only=True,
        )
        self.assertEqual(
            selected,
            ["llama-3-70b_A100_tp4", "llama-3-70b_A100_tp8"],
        )

    def test_run_baselines_node_has_synthetic_timestamp_opt_out_flag(self):
        path = Path("scripts/eval/run_baselines_node.py")
        text = path.read_text()
        self.assertIn("--allow-synthetic-request-timestamps", text)
        self.assertIn("require_recorded_timestamps", text)

    def test_run_baselines_node_groundtruth_has_splitwise_mode_flag(self):
        path = Path("scripts/eval/run_baselines_node_groundtruth.py")
        text = path.read_text()
        self.assertIn("--splitwise-mode", text)
        self.assertIn("splitwise_strict", text)


if __name__ == "__main__":
    unittest.main()
