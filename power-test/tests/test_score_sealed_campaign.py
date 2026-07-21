import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "score_sealed_campaign.py"
SPEC = importlib.util.spec_from_file_location("score_sealed_campaign", MODULE_PATH)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def _bundle(tmp_path, *, role="sealed", evidence="measured_ledger",
            status="validated", run_id="run"):
    root = tmp_path / run_id
    root.mkdir()
    manifest = {
        "run_id": run_id,
        "validation_role": role,
        "evidence_profile": evidence,
        "instrumentation": {"profile": evidence, "status": status},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name in ("requests.json", "power.csv", "engine.csv"):
        (root / name).write_text("{}")
    (root / "engine.csv").write_text(
        "timestamp,prefix_cache_hits_total\n1000,0\n1001,0\n"
    )
    return root


def test_sealed_input_contract_accepts_only_validated_measured_bundles(tmp_path):
    root = _bundle(tmp_path)
    validated = score.validate_sealed_bundles([root])
    assert validated[0][1]["run_id"] == "run"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": "development"}, "not sealed"),
        ({"evidence": "core"}, "requires measured_ledger"),
        ({"status": "expected_dry_run"}, "not validated"),
    ],
)
def test_sealed_input_contract_fails_closed(tmp_path, kwargs, message):
    root = _bundle(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        score.validate_sealed_bundles([root])


def test_sealed_cache_off_bundle_rejects_observed_cache_hits(tmp_path):
    root = _bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["probe"] = {"type": "agentic", "prefix_cache": False}
    manifest_path.write_text(json.dumps(manifest))
    (root / "engine.csv").write_text(
        "timestamp,prefix_cache_hits_total\n1000,0\n1001,16\n"
    )
    with pytest.raises(ValueError, match="cache-off run observed 16"):
        score.validate_sealed_bundles([root])

def test_grade_requires_timing_energy_and_shape():
    run = {
        "timing": {"e2e_s_medabs_pct": 9.0},
        "power": {
            "energy_error_pct": 5.0, "duration_s": 900.0,
            "acf_mae": 0.04, "acf_r2": 0.91, "nrmse_range": 0.19,
        },
    }
    assert score.grade_run(run)["passed"]
    run["power"]["acf_r2"] = 0.89
    assert not score.grade_run(run)["passed"]


def test_agentic_cache_plan_requires_both_legs(tmp_path):
    root = _bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = "m"
    manifest["hardware"] = "A100"
    manifest["tp"] = 1
    manifest["probe"] = {
        "type": "agentic", "replay_plan_sha256": "plan",
        "prefix_cache": False, "pack_index": 0,
    }
    manifest_path.write_text(json.dumps(manifest))
    validated = score.validate_sealed_bundles([root])
    with pytest.raises(ValueError, match="complete off/on pairs"):
        score._cache_pairs(validated)


def test_campaign_matrix_requires_all_eleven_runs(tmp_path):
    validated = [score.validate_sealed_bundles([_bundle(tmp_path)])[0]]
    with pytest.raises(ValueError, match="exactly"):
        score.validate_campaign_matrix(validated)


def test_question_summary_reports_median_worst_and_failures():
    runs = []
    for index, error in enumerate((1.0, 2.0, 9.0)):
        runs.append(score.grade_run({
            "campaign": "sealed_burstgpt_qwen3-8b_a100",
            "run_id": f"run-{index}",
            "timing": {"e2e_s_medabs_pct": error},
            "power": {
                "energy_error_pct": 1.0, "duration_s": 900.0,
                "acf_mae": 0.01, "acf_r2": 0.95,
                "nrmse_range": 0.1,
            },
        }))
    runs[-1]["power"]["acf_r2"] = 0.89
    runs[-1] = score.grade_run(runs[-1])
    summary = score.summarize_questions(runs)["irregular_arrivals"]
    assert summary["metrics"]["timing_e2e_medabs_pct"] == {
        "median": 2.0, "worst": 9.0,
    }
    assert summary["metrics"]["acf_r2"]["worst"] == 0.89
    assert summary["failed_run_ids"] == ["run-2"]
