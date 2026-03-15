import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from model.utils.io import (
    ensure_dir,
    ensure_dir_for_file,
    load_json,
    power_timestamp_to_epoch,
    resolve_existing_path,
    safe_slug,
    write_csv,
    write_json,
)


class TestIOUtils(unittest.TestCase):
    def test_ensure_dir_creates_nested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "c"
            ensure_dir(path)
            self.assertTrue(path.exists())
            self.assertTrue(path.is_dir())

    def test_ensure_dir_existing_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing"
            path.mkdir(parents=True, exist_ok=True)
            ensure_dir(path)
            self.assertTrue(path.exists())

    def test_ensure_dir_for_file_creates_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "a" / "b" / "payload.json"
            ensure_dir_for_file(file_path)
            self.assertTrue((Path(tmp) / "a" / "b").exists())
            self.assertTrue((Path(tmp) / "a" / "b").is_dir())

    def test_ensure_dir_for_file_empty_parent_noop(self):
        ensure_dir_for_file("payload.json")

    def test_load_json_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            with open(path, "w") as f:
                json.dump({"a": 1}, f)
            self.assertEqual(load_json(path), {"a": 1})

    def test_load_json_non_dict_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            with open(path, "w") as f:
                json.dump([1, 2], f)
            with self.assertRaises(ValueError):
                load_json(path)

    def test_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "payload.json"
            payload = {"b": 2, "a": 1}
            write_json(path, payload)
            self.assertEqual(load_json(path), payload)

    def test_write_csv_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
            write_csv(path, rows, fields=["a", "b"])
            with open(path, "r", newline="") as f:
                loaded = list(csv.DictReader(f))
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["a"], "1")
            self.assertEqual(loaded[1]["b"], "y")

    def test_write_csv_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x" / "y" / "rows.csv"
            write_csv(path, [{"a": 1}], fields=["a"])
            self.assertTrue(path.exists())

    def test_safe_slug_special_chars(self):
        self.assertEqual(safe_slug("a/b:c d"), "a-b-c-d")

    def test_safe_slug_already_clean(self):
        self.assertEqual(safe_slug("abc_123"), "abc_123")

    def test_resolve_existing_path_absolute_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            path.write_text("x")
            out = resolve_existing_path(str(path), base_dir=tmp)
            self.assertEqual(out, str(path))

    def test_resolve_existing_path_absolute_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.txt"
            out = resolve_existing_path(str(path), base_dir=tmp)
            self.assertIsNone(out)

    def test_resolve_existing_path_relative_from_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            file_path = base / "sub" / "file.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("x")
            out = resolve_existing_path("sub/file.txt", base_dir=base)
            self.assertEqual(out, str(file_path))

    def test_power_timestamp_to_epoch_slash_format(self):
        out = power_timestamp_to_epoch("2025/01/01 00:00:00.000")
        expected = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(out, expected)

    def test_power_timestamp_to_epoch_dash_format(self):
        out = power_timestamp_to_epoch("2025-01-01 00:00:00")
        expected = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(out, expected)

    def test_power_timestamp_to_epoch_invalid(self):
        self.assertIsNone(power_timestamp_to_epoch("garbage"))


if __name__ == "__main__":
    unittest.main()
