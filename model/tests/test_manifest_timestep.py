"""
Claim:
Raw trace cadences for one GRU configuration must agree closely enough to be
projected onto one exact stored timestep.

Plausible wrong implementations:
- Accept a materially different cadence using a permissive tolerance.
- Store an unprojected trace with a cadence different from the config grid.
- Compare only an average cadence that hides one incompatible trace.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import model.training_data.manifest as manifest_module


def _record(pair_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_layout="sharegpt",
        provenance={
            "power_csv_path": f"{pair_key}.csv",
            "json_path": f"{pair_key}.json",
            "sha256": {},
            "request_rows": {},
            "request_projection": {},
            "request_projection_indices": [],
        },
    )


def _trace(dt: float) -> dict:
    values = np.asarray([1.0, 2.0, 3.0])
    return {"dt": dt, "power": values, "active_requests": values, "t_arrive_log": values}


def test_manifest_rejects_four_percent_cross_trace_timestep_difference(tmp_path, monkeypatch):
    pair_manifest = tmp_path / "pairs.csv"
    rows = []
    for i in range(3):
        power_path = tmp_path / f"power-{i}.csv"
        request_path = tmp_path / f"requests-{i}.json"
        power_path.touch()
        request_path.touch()
        rows.append(
            "matched,unit,H100,1,1.0,run-"
            f"{i},{power_path},{request_path}"
        )
    pair_manifest.write_text(
        "status,model_name,hardware,tensor_parallelism,rate,pair_key,power_csv_path,json_path\n"
        + "\n".join(rows)
        + "\n"
    )

    records = [_record(f"run-{i}") for i in range(3)]
    traces = {id(record): _trace(dt) for record, dt in zip(records, (0.25, 0.26, 0.25))}
    monkeypatch.setattr(manifest_module, "load_legacy_run", lambda *_args, **_kwargs: records.pop(0))
    monkeypatch.setattr(manifest_module, "gru_view_from_record", lambda record: traces[id(record)])

    with pytest.raises(ValueError, match="mixes incompatible sampling intervals"):
        manifest_module.run_prepare_experimental_manifest(
            pair_manifest_csv=str(pair_manifest), out_dir=str(tmp_path / "out")
        )
