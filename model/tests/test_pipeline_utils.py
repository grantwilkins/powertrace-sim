from pathlib import Path

import pytest

from scripts.eval.pipeline_utils import (
    resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths,
    resolve_throughput,
)


def test_resolve_experimental_paths_valid(tmp_path: Path):
    dataset = tmp_path / "dataset.npz"
    split = tmp_path / "split.json"
    dataset.write_bytes(b"")
    split.write_text("{}")
    manifest = {
        "configs": {
            "toy_H100_tp1": {
                "dataset_npz": str(dataset),
                "split_json": str(split),
            }
        }
    }

    dataset_path, split_path = resolve_experimental_paths(
        manifest, config_id="toy_H100_tp1", experimental_base=str(tmp_path)
    )
    assert dataset_path == str(dataset)
    assert split_path == str(split)


def test_resolve_experimental_paths_missing_config_raises(tmp_path: Path):
    manifest = {"configs": {}}
    with pytest.raises(ValueError, match="not found in experimental manifest"):
        resolve_experimental_paths(
            manifest, config_id="missing", experimental_base=str(tmp_path)
        )


def test_resolve_checkpoint_norm_gmm_paths_valid(tmp_path: Path):
    checkpoint = tmp_path / "ckpt.pt"
    norm = tmp_path / "norm.json"
    gmm = tmp_path / "gmm.json"
    checkpoint.write_bytes(b"")
    norm.write_text("{}")
    gmm.write_text("{}")
    entry = {
        "checkpoint_path": str(checkpoint),
        "norm_params_path": str(norm),
        "gmm_params_path": str(gmm),
    }

    ckpt_path, norm_path, gmm_path = resolve_checkpoint_norm_gmm_paths(
        entry, str(tmp_path)
    )
    assert ckpt_path == str(checkpoint)
    assert norm_path == str(norm)
    assert gmm_path == str(gmm)


def test_resolve_checkpoint_norm_gmm_paths_missing_checkpoint_raises(tmp_path: Path):
    norm = tmp_path / "norm.json"
    gmm = tmp_path / "gmm.json"
    norm.write_text("{}")
    gmm.write_text("{}")
    entry = {
        "checkpoint_path": str(tmp_path / "missing.pt"),
        "norm_params_path": str(norm),
        "gmm_params_path": str(gmm),
    }

    with pytest.raises(ValueError, match="Checkpoint path not found"):
        resolve_checkpoint_norm_gmm_paths(entry, str(tmp_path))


def test_resolve_throughput_valid():
    payload = {
        "configs": {
            "toy_H100_tp1": {
                "prefill_rate_median_toks_per_s": 100.0,
                "decode_rate_median_toks_per_s": 50.0,
            }
        }
    }

    result = resolve_throughput(payload, "toy_H100_tp1")
    assert result == {"lambda_prefill": 100.0, "lambda_decode": 50.0}


def test_resolve_throughput_invalid_prefill_raises():
    payload = {
        "configs": {
            "toy_H100_tp1": {
                "prefill_rate_median_toks_per_s": float("nan"),
                "decode_rate_median_toks_per_s": 50.0,
            }
        }
    }

    with pytest.raises(ValueError, match="Invalid prefill throughput"):
        resolve_throughput(payload, "toy_H100_tp1")


def test_resolve_throughput_missing_config_raises():
    payload = {"configs": {}}
    with pytest.raises(ValueError, match="not found in throughput DB"):
        resolve_throughput(payload, "missing")
