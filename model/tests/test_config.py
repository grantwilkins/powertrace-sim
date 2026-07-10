import unittest

import torch

from model.utils.config import (
    is_moe_config,
    parse_config_id,
    parse_config_ids,
    parse_csv_list,
    resolve_device,
    safe_float,
    tp_gpus_from_config_id,
)


class TestConfigUtils(unittest.TestCase):
    def test_resolve_device_auto(self):
        device = resolve_device("auto")
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"cpu", "cuda"})

    def test_resolve_device_explicit_cpu(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_resolve_device_torch_device_passthrough(self):
        device = torch.device("cpu")
        self.assertEqual(resolve_device(device), device)

    def test_parse_config_ids_empty(self):
        self.assertEqual(parse_config_ids([]), [])

    def test_parse_config_ids_dedupe(self):
        self.assertEqual(parse_config_ids(["a", "a", "b"]), ["a", "b"])

    def test_parse_config_ids_csv_split(self):
        self.assertEqual(parse_config_ids(["a,b,c"]), ["a", "b", "c"])

    def test_parse_config_id_valid(self):
        parsed = parse_config_id("llama-3-70b_A100_tp4")
        self.assertEqual(parsed["model_family"], "llama-3")
        self.assertEqual(parsed["model_size"], "70")
        self.assertEqual(parsed["hardware"], "A100")
        self.assertEqual(parsed["tp"], "4")

    def test_is_moe_config_gpt_oss(self):
        self.assertTrue(is_moe_config("gpt-oss-20b_A100_tp2"))

    def test_is_moe_config_llama(self):
        self.assertFalse(is_moe_config("llama-3-8b_A100_tp1"))

    def test_tp_gpus_from_config_id_parse_table(self):
        cases = {
            "llama-3-70b_A100_tp8": 8,
            "llama-3-70b_A100_tp4": 4,
            "gpt-oss-120b_H100_tp8": 8,
            "deepseek-r1-distill-llama-70b_H100_tp2": 2,
            "llama-3-8b_A100_tp1": 1,
            "  llama-3-70b_A100_tp8  ": 8,
        }
        for config_id, expected in cases.items():
            self.assertEqual(tp_gpus_from_config_id(config_id), expected)

    def test_tp_gpus_from_config_id_rejects_missing_or_invalid_suffix(self):
        for bad in ["llama-3-70b_A100", "llama-3-70b_A100_tp", "llama-3-70b_A100_tp0", ""]:
            with self.assertRaises(ValueError):
                tp_gpus_from_config_id(bad)

    def test_safe_float_valid(self):
        self.assertAlmostEqual(safe_float("3.14", "value"), 3.14, places=9)

    def test_safe_float_nan_raises(self):
        with self.assertRaises(ValueError):
            safe_float("nan", "value")

    def test_parse_csv_list(self):
        self.assertEqual(parse_csv_list("a,b,c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
