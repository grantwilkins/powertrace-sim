"""Open a sealed campaign once and score it with frozen source-only fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

_CLIENT = Path(__file__).resolve().parents[1] / "profiling" / "client"
sys.path.insert(0, str(_CLIENT))
from prefix_cache_contract import validate_prefix_cache_mode  # noqa: E402


GATES = {
    "timing_e2e_medabs_pct_max": 10.0,
    "energy_error_pct_max": 6.0,
    "acf_mae_max": 0.05,
    "acf_r2_min": 0.90,
    "nrmse_range_max": 0.20,
    "temporal_duration_s_min": 120.0,
}
REQUIRED_BUNDLE_FILES = (
    "manifest.json", "requests.json", "power.csv", "engine.csv",
)
CAMPAIGN_QUESTIONS = {
    "sealed_burstgpt_qwen3-8b_a100": ("irregular_arrivals", 3),
    "sealed_openhands_qwen3-8b_a100": ("agent_cache", 6),
    "sealed_qwen3-14b_a100": ("unseen_dense", 1),
    "sealed_qwen3-30b-a3b_h100": ("unseen_moe_hardware", 1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sealed_bundles(bundle_dirs) -> list[tuple[Path, dict]]:
    """Reject mixed-role, incomplete, or telemetry-invalid score inputs."""
    validated = []
    run_ids = set()
    for value in bundle_dirs:
        root = Path(value).resolve()
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(f"{root}: missing bundle files {missing}")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("validation_role") != "sealed":
            raise ValueError(f"{root}: bundle is not sealed")
        if manifest.get("evidence_profile") != "measured_ledger":
            raise ValueError(f"{root}: sealed scoring requires measured_ledger")
        instrumentation = manifest.get("instrumentation") or {}
        if (
            instrumentation.get("profile") != "measured_ledger"
            or instrumentation.get("status") != "validated"
        ):
            raise ValueError(f"{root}: measured instrumentation is not validated")
        probe = manifest.get("probe") or {}
        if probe.get("type") in {"agentic", "trace_replay"}:
            validate_prefix_cache_mode(
                root / "engine.csv", bool(probe.get("prefix_cache"))
            )
        run_id = manifest.get("run_id")
        if not run_id or run_id in run_ids:
            raise ValueError(f"{root}: missing or duplicate run_id")
        run_ids.add(run_id)
        validated.append((root, manifest))
    if not validated:
        raise ValueError("at least one sealed bundle is required")
    return validated


def grade_run(run: dict) -> dict:
    timing = run["timing"]
    power = run["power"]
    checks = {
        "timing_e2e": (
            timing["e2e_s_medabs_pct"]
            <= GATES["timing_e2e_medabs_pct_max"]
        ),
        "energy": power["energy_error_pct"] <= GATES["energy_error_pct_max"],
        "temporal_duration": (
            power["duration_s"] >= GATES["temporal_duration_s_min"]
        ),
        "acf_mae": power["acf_mae"] <= GATES["acf_mae_max"],
        "acf_r2": power["acf_r2"] >= GATES["acf_r2_min"],
        "nrmse_range": power["nrmse_range"] <= GATES["nrmse_range_max"],
    }
    return {**run, "gate_checks": checks, "passed": all(checks.values())}


def validate_campaign_matrix(validated) -> None:
    counts = defaultdict(int)
    for root, _ in validated:
        counts[root.parent.name] += 1
    expected = {
        campaign: count for campaign, (_, count) in CAMPAIGN_QUESTIONS.items()
    }
    if dict(counts) != expected:
        raise ValueError(
            f"sealed campaign matrix must contain exactly {expected}, got "
            f"{dict(counts)}"
        )


def _cache_pairs(validated):
    groups = defaultdict(dict)
    required = set()
    for root, manifest in validated:
        probe = manifest.get("probe") or {}
        plan_hash = probe.get("trace_plan_sha256") or probe.get("replay_plan_sha256")
        if plan_hash:
            key = (
                plan_hash, manifest.get("model"), manifest.get("hardware"),
                manifest.get("tp"), probe.get("pack_index", 0),
            )
            treatment = bool(probe.get("prefix_cache"))
            if treatment in groups[key]:
                raise ValueError(f"duplicate cache treatment for plan: {key}")
            groups[key][treatment] = root
            if probe.get("type") == "agentic":
                required.add(key)
    incomplete = [key for key in required if set(groups[key]) != {False, True}]
    if incomplete:
        raise ValueError(
            f"agentic cache plans require complete off/on pairs: {incomplete}"
        )
    return [pair for pair in groups.values() if set(pair) == {False, True}]


def _compare_pairs(validated) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling" / "probes"))
    from compare_trace_replays import compare_bundle_data

    results = []
    for pair in _cache_pairs(validated):
        off, on = pair[False], pair[True]
        results.append(compare_bundle_data(
            json.loads((off / "manifest.json").read_text()),
            json.loads((off / "requests.json").read_text()),
            json.loads((on / "manifest.json").read_text()),
            json.loads((on / "requests.json").read_text()),
        ))
    return results


def summarize_questions(runs: list[dict]) -> dict:
    metrics = {
        "timing_e2e_medabs_pct": ("timing", "e2e_s_medabs_pct", max),
        "energy_error_pct": ("power", "energy_error_pct", max),
        "acf_mae": ("power", "acf_mae", max),
        "acf_r2": ("power", "acf_r2", min),
        "nrmse_range": ("power", "nrmse_range", max),
    }
    grouped = defaultdict(list)
    for run in runs:
        question, _ = CAMPAIGN_QUESTIONS[run["campaign"]]
        grouped[question].append(run)
    return {
        question: {
            "runs": len(values),
            "passed": all(run["passed"] for run in values),
            "failed_run_ids": [
                run["run_id"] for run in values if not run["passed"]
            ],
            "metrics": {
                name: {
                    "median": median(run[section][field] for run in values),
                    "worst": worst(run[section][field] for run in values),
                }
                for name, (section, field, worst) in metrics.items()
            },
        }
        for question, values in sorted(grouped.items())
    }


def score_campaign(
    bundle_dirs, timing_fit_path, power_fit_path, out_path, *, dt=0.25
) -> dict:
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"sealed report already exists: {out}")
    validated = validate_sealed_bundles(bundle_dirs)
    validate_campaign_matrix(validated)
    timing_path, power_path = Path(timing_fit_path), Path(power_fit_path)
    timing_fit = json.loads(timing_path.read_text())
    power_fit = json.loads(power_path.read_text())

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "timing-test"))
    from evaluate_expansion import evaluate_bundle

    runs = [
        grade_run(evaluate_bundle(root, timing_fit, power_fit, dt=dt))
        for root, _ in validated
    ]
    cache_pairs = _compare_pairs(validated)
    report = {
        "schema_version": "sealed-campaign-score-v1",
        "validation_role": "sealed",
        "frozen_inputs": {
            "timing_fit_sha256": sha256_file(timing_path),
            "power_fit_sha256": sha256_file(power_path),
            "scorer_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "timing-test" / "evaluate_expansion.py"
            ),
            "bundle_manifest_sha256": {
                manifest["run_id"]: sha256_file(root / "manifest.json")
                for root, manifest in validated
            },
        },
        "gates": GATES,
        "runs": runs,
        "questions": summarize_questions(runs),
        "failures": [
            run["run_id"] for run in runs if not run["passed"]
        ],
        "cache_pair_identity": cache_pairs,
        "passed": all(run["passed"] for run in runs),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", action="append", required=True)
    parser.add_argument("--timing-fit", required=True)
    parser.add_argument("--power-fit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dt", type=float, default=0.25)
    args = parser.parse_args(argv)
    report = score_campaign(
        args.bundle_dir, args.timing_fit, args.power_fit, args.out, dt=args.dt
    )
    print(
        f"scored {len(report['runs'])} sealed bundles; "
        f"campaign pass={report['passed']}"
    )


if __name__ == "__main__":
    main()
