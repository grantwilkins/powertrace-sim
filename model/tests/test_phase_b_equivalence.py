"""
Claim:
Phase-B new mode permits only a baseline difference with identical normalized
rebuilt content, not another difference in the same filename.

Plausible wrong implementations:
- Whitelist a changed value because its filename was in the baseline record.
- Compare NPZ zip bytes, which change with archive metadata rather than arrays.
- Accept a legacy baseline that has no content fingerprint.
"""

import json

import numpy as np
import pytest

from scripts.gates import phase_b_equivalence as phase_b


def test_new_mode_rejects_changed_array_at_an_allowed_baseline_path(
    tmp_path, monkeypatch
):
    """Same file and same differing key are insufficient when the value changes."""
    monkeypatch.setattr(phase_b, "REPO_ROOT", tmp_path)
    reference = tmp_path / "reference.npz"
    baseline_rebuild = tmp_path / "baseline-rebuild.npz"
    unchanged_rebuild = tmp_path / "unchanged-rebuild.npz"
    changed_rebuild = tmp_path / "changed-rebuild.npz"
    np.savez(reference, power=np.array([0.0]))
    np.savez(baseline_rebuild, power=np.array([1.0]))
    np.savez(unchanged_rebuild, power=np.array([1.0]))
    np.savez(changed_rebuild, power=np.array([2.0]))

    baseline = phase_b._array_report(reference, baseline_rebuild)
    unchanged = phase_b._array_report(reference, unchanged_rebuild)
    changed = phase_b._array_report(reference, changed_rebuild)
    allowed = {
        baseline["file"]: (
            baseline["reference_content_sha256"],
            baseline["rebuilt_content_sha256"],
        )
    }

    assert phase_b._unexpected_diff_files([unchanged], allowed) == []
    assert phase_b._unexpected_diff_files([changed], allowed) == [baseline["file"]]


def test_new_mode_rejects_filename_only_baseline_record(tmp_path):
    """Old records cannot prove which content was an explained difference."""
    record = tmp_path / "phase_b_baseline_legacy.json"
    record.write_text(json.dumps({"diffs": [{"file": "results/norm.json"}]}))

    with pytest.raises(ValueError, match="lacks comparison-content fingerprints"):
        phase_b._baseline_diff_fingerprints(tmp_path)
