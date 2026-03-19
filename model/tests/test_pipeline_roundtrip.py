import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.pipeline.evaluation import evaluate_from_artifacts
from model.pipeline.training import run_training_from_manifest
from model.utils.io import write_json as _write_json


class TestPipelineRoundTrip(unittest.TestCase):
    def test_train_then_eval_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = "toy-roundtrip_H100_tp1"

            exp_root = root / "results" / "experimental_continuous_v1"
            datasets_dir = exp_root / "datasets"
            splits_dir = exp_root / "splits"
            norms_dir = exp_root / "norm_params"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            splits_dir.mkdir(parents=True, exist_ok=True)
            norms_dir.mkdir(parents=True, exist_ok=True)

            pair_keys = np.asarray(["p0", "p1", "p2"], dtype=object)
            power = np.asarray(
                [
                    np.asarray([100.0, 102.0, 105.0, 104.0, 103.0, 101.0], dtype=np.float64),
                    np.asarray([98.0, 99.0, 102.0, 103.0, 101.0, 100.0], dtype=np.float64),
                    np.asarray([97.0, 100.0, 104.0, 106.0, 103.0, 101.0], dtype=np.float64),
                ],
                dtype=object,
            )
            active = np.asarray(
                [
                    np.asarray([0.0, 1.0, 2.0, 1.0, 1.0, 0.0], dtype=np.float64),
                    np.asarray([0.0, 1.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64),
                    np.asarray([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], dtype=np.float64),
                ],
                dtype=object,
            )
            dataset_path = datasets_dir / "toy-roundtrip_H100_tp1.npz"
            np.savez(
                dataset_path,
                config_id=np.asarray([cfg], dtype=object),
                dt=np.asarray([0.25], dtype=np.float64),
                pair_key=pair_keys,
                rate=np.asarray(["1", "1", "1"], dtype=object),
                power_start_epoch_s=np.asarray([1000.0, 1005.0, 1010.0], dtype=np.float64),
                power=power,
                active_requests=active,
            )

            split_path = splits_dir / "toy-roundtrip_H100_tp1.json"
            _write_json(
                split_path,
                {
                    "config_id": cfg,
                    "train_indices": [0],
                    "val_indices": [1],
                    "test_indices": [2],
                },
            )
            norm_path = norms_dir / "toy-roundtrip_H100_tp1.json"
            _write_json(
                norm_path,
                {
                    "config_id": cfg,
                    "dt": 0.25,
                    "power_mean": 101.5,
                    "power_std": 3.0,
                    "active_mean": 1.0,
                    "active_std": 1.0,
                    "t_arrive_log_mean": 0.0,
                    "t_arrive_log_std": 1.0,
                    "power_min": 90.0,
                    "power_max": 130.0,
                },
            )
            experimental_manifest_path = exp_root / "manifest.json"
            _write_json(
                experimental_manifest_path,
                {
                    "schema_version": "experimental-continuous-v1",
                    "configs": {
                        cfg: {
                            "written": True,
                            "dataset_npz": str(dataset_path),
                            "split_json": str(split_path),
                            "norm_params_json": str(norm_path),
                        }
                    },
                },
            )

            train_run = run_training_from_manifest(
                manifest_path=str(experimental_manifest_path),
                out_root=str(root / "results" / "continuous_v1_gmm_bigru"),
                config_ids=[cfg],
                k=2,
                feature_set="f2",
                hidden_dim=8,
                num_layers=1,
                epochs=2,
                lr=1e-3,
                patience=2,
                scheduler_patience=1,
                scheduler_factor=0.5,
                bic_candidates=[2],
                seed=5,
                device="cpu",
            )
            self.assertEqual(int(train_run["summary"]["num_trained"]), 1)

            run_manifest_path = Path(train_run["defaults"]["out_dir"]) / "run_manifest.json"
            throughput_path = root / "model" / "config" / "throughput_database.json"
            _write_json(
                throughput_path,
                {
                    "schema_version": "stage0-throughput-v1",
                    "configs": {
                        cfg: {
                            "prefill_rate_median_toks_per_s": 100.0,
                            "decode_rate_median_toks_per_s": 50.0,
                        }
                    },
                },
            )

            request_json = root / "data" / "requests_p2.json"
            _write_json(
                request_json,
                {
                    "input_lens": [24, 36],
                    "output_lens": [10, 8],
                    "request_timestamps": [1010.0, 1010.5],
                },
            )
            pair_manifest = root / "results" / "stage0" / "pair_manifest.csv"
            pair_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(pair_manifest, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["status", "pair_key", "json_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "matched",
                        "pair_key": "p2",
                        "json_path": str(request_json),
                    }
                )

            eval_out_dir = Path(train_run["defaults"]["out_dir"]) / "eval_metrics"
            eval_run = evaluate_from_artifacts(
                run_manifest=str(run_manifest_path),
                experimental_manifest=str(experimental_manifest_path),
                throughput_db=str(throughput_path),
                pair_manifest_csv=str(pair_manifest),
                out_dir=str(eval_out_dir),
                config_ids=[cfg],
                num_seeds=1,
                base_seed=9,
                device="cpu",
                decode_mode="stochastic",
                median_filter_window=1,
                plots=False,
            )
            self.assertEqual(int(eval_run["summary"]["num_evaluated_configs"]), 1)
            self.assertTrue((eval_out_dir / "config_summary.csv").exists())
            self.assertTrue((eval_out_dir / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
