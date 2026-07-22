"""The paper manifest must bind every evidence family to immutable bytes.

A wrong implementation could list paths without hashing them, silently omit an
evidence family, or label target-calibrated transfer as frozen-release evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.paper import regenerate


def test_manifest_hashes_every_declared_output(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / regenerate.ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    outputs = ("results/paper/a.json", "figures/b.pdf")
    monkeypatch.setattr(regenerate, "OUTPUTS", outputs)
    for relative, content in zip(outputs, (b"one", b"two")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manifest = regenerate.write_manifest(tmp_path)
    payload = json.loads(manifest.read_text())

    assert [row["path"] for row in payload["outputs"]] == list(outputs)
    assert payload["outputs"][0]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert payload["evidence_boundary"]["appendix_transfer"].startswith("retrospective")


def test_manifest_rejects_partial_regeneration(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / regenerate.ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(regenerate, "OUTPUTS", ("missing.csv",))

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        regenerate.write_manifest(tmp_path)
