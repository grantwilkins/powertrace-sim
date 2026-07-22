"""Write the alternate 117/333 coverage split for the timing dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.training_data.coverage_split import ROLES, assign_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="timing-test/timing_dataset.npz")
    parser.add_argument("--out", default="timing-test/coverage_split_manifest.json")
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    roles = assign_runs(
        data["run_model"], data["run_hardware"], data["run_tp"], data["run_rate"]
    )
    role_runs = {
        role: sorted(rid for rid, value in roles.items() if value == role)
        for role in ROLES
    }
    manifest = {
        "schema_version": "timing-coverage-split-v1",
        "design": "13 training setups x 3 rates; whole cells remain together",
        "holdout_model": {
            "A100": "deepseek-r1-distill-70b",
            "H100": "deepseek-r1-distill-8b",
        },
        "holdout_twin": {},
        "roles": {str(rid): role for rid, role in sorted(roles.items())},
        "role_runs": role_runs,
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print({role: len(runs) for role, runs in role_runs.items()})


if __name__ == "__main__":
    main()
