"""
Claim:
Legacy bundle manifests can recover missing operator-component descriptors
only when their recorded architecture matches a known compatibility entry.

Plausible wrong implementations:
- Trust a model-name alias even when the recorded parameter count differs.
- Mutate the manifest's architecture dictionary in place.
- Apply compatibility metadata to unrelated models.
"""

import pytest

from model.training_data.arch import arch_from_manifest


def test_qwen_legacy_manifest_gets_component_metadata_without_mutation():
    raw = {
        "n_active": 8190427136.0,
        "w_bytes": 16380854272.0,
        "d_model": 4096,
    }
    manifest = {"model": "Qwen/Qwen3-8B", "arch": raw}
    arch = arch_from_manifest(manifest)

    assert arch["input_embedding_params"] == 151936 * 4096
    assert arch["transformer_active_params"] > 0
    assert "input_embedding_params" not in raw


def test_qwen_compatibility_metadata_rejects_architecture_mismatch():
    manifest = {
        "model": "Qwen/Qwen3-8B",
        "arch": {"n_active": 7e9, "w_bytes": 14e9, "d_model": 4096},
    }
    with pytest.raises(ValueError, match="does not match"):
        arch_from_manifest(manifest)
