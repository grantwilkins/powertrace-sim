"""
Claim:
The timing dataset builder reproduces the frozen DESIGN.md section 3 split
(one role per run, holdouts before rate before repeat), the shared repeat
assignment rule, lossless ragged ITL packing, and honest per-run exclusion
counting for inexact-ITL requests.

Plausible wrong implementations:
- Assign repeats by manifest order instead of sorted source identity.
- Send a holdout model's rate-4.0 runs to holdout_rate instead of the
  model holdout (rate check before model check).
- Hold out the twin model on both hardwares instead of one per hardware.
- Off-by-one ITL offsets that drop or duplicate the last interval.
- Count exclusions against the exact subset instead of the source rows.
- Shift arrival times when excluded requests include the earliest arrival.
"""

import numpy as np
import pytest

import build_timing_dataset as btd
from model.training_data.run_record import RunRecord


def _rows(specs):
    return [
        {"run_id": i, "config_id": cfg, "source_id": src, "rate": rate}
        for i, (cfg, rate, src) in enumerate(specs)
    ]


def test_repeat_assignment_sorts_by_source_identity_within_config_rate():
    rows = _rows([
        ("m_A100_tp4", 0.5, "m_A100_tp4|date=20250520"),  # run 0: 2nd sorted
        ("m_A100_tp4", 0.5, "m_A100_tp4|date=20250518"),  # run 1: 1st sorted
        ("m_A100_tp4", 0.5, "m_A100_tp4|date=20250522"),  # run 2: 3rd sorted
        ("m_A100_tp4", 1.0, "m_A100_tp4|date=20250519"),  # other rate: own group
        ("m_H100_tp4", 0.5, "m_H100_tp4|date=20250501"),  # other config: own group
    ])
    assert btd.assign_repeats(rows) == {0: 1, 1: 0, 2: 2, 3: 0, 4: 0}


def test_roles_cover_and_partition_the_synthetic_run_table():
    models = {
        "A100": ["gpt-oss-120b", "deepseek-r1-distill-70b", "llama-3-8b", "gpt-oss-20b"],
        "H100": ["llama-3-405b", "deepseek-r1-distill-8b", "llama-3-70b"],
    }
    rates = list(btd.TRAIN_RATES) + [btd.HOLDOUT_RATE]
    roles_by_run, run_ids, table = {}, [], []
    run_id = 0
    for hw, hw_models in models.items():
        for model in hw_models:
            for rate in rates:
                for repeat in (0, 1, 2):
                    role = btd.assign_role(model, hw, rate, repeat)
                    assert role in btd.ROLES  # every run gets exactly one role
                    roles_by_run[run_id] = role
                    run_ids.append(run_id)
                    table.append((hw, model, rate, repeat, role))
                    run_id += 1
    btd.check_split(roles_by_run, run_ids)  # coverage + disjointness

    for hw, model, rate, repeat, role in table:
        if model == btd.HOLDOUT_MODEL[hw]:
            assert role == "holdout_model"  # all rates, all repeats
        elif model == btd.HOLDOUT_TWIN[hw]:
            assert role == "holdout_twin"  # all rates, all repeats
        elif rate == btd.HOLDOUT_RATE:
            assert role == "holdout_rate"  # all repeats of remaining models
        else:
            assert role == ("train" if repeat in (0, 1) else "test_indomain")

    # Twin models are held out on exactly one hardware each.
    assert btd.assign_role("deepseek-r1-distill-70b", "H100", 0.5, 0) == "train"
    assert btd.assign_role("deepseek-r1-distill-8b", "A100", 0.5, 2) == "test_indomain"
    with pytest.raises(ValueError):
        btd.assign_role("llama-3-8b", "A100", 3.0, 0)  # rate outside contract
    with pytest.raises(ValueError):
        btd.assign_role("llama-3-8b", "A100", 0.5, 3)  # repeat outside contract
    with pytest.raises(ValueError):
        btd.check_split(roles_by_run, run_ids + [run_id])  # uncovered run


def test_itl_offsets_round_trip_ragged_lists():
    itls = [[0.02, 0.03, 0.05], [], [0.1], [0.04, 0.04]]
    values, offsets = btd.pack_itls(itls)
    assert offsets.tolist() == [0, 3, 3, 4, 6]
    assert values.size == offsets[-1]
    for i, expected in enumerate(itls):
        np.testing.assert_array_equal(values[offsets[i]:offsets[i + 1]], expected)
    empty_values, empty_offsets = btd.pack_itls([])
    assert empty_values.size == 0 and empty_offsets.tolist() == [0]


def _synthetic_record(output_lens, itls, timestamps):
    n = len(output_lens)
    return RunRecord(
        config_id="llama-3-8b_A100_tp1", model="llama-3-8b", hardware="A100",
        tp=1, gpus_per_node=8, source_layout="synthetic", clock_basis="epoch",
        power_timestamps=np.array([0.0, 1.0, 2.0]), power_per_gpu=None,
        util_per_gpu=None, mem_per_gpu=None,
        node_power=np.array([100.0, 110.0, 105.0]), device_ids=(),
        device_table={}, input_lens=np.full(n, 10.0),
        output_lens=np.asarray(output_lens, dtype=np.float64),
        ttfts=np.full(n, 0.2), itls=np.asarray(itls, dtype=object),
        decode_times=np.full(n, 0.5),
        request_timestamps=np.asarray(timestamps, dtype=np.float64),
        has_timestamps=True, timestamp_source="recorded", request_table={},
        engine_table={}, arch={}, provenance={},
    )


def test_exclusion_counting_and_run_relative_arrivals():
    # Requests 0 and 3 are exact (len(itls) == output_tokens - 1); 1 has a
    # truncated list, 2 has an extra interval. The earliest arrival (request
    # 1) is excluded, yet it still anchors t=0 for the retained rows.
    record = _synthetic_record(
        output_lens=[3.0, 4.0, 2.0, 1.0],
        itls=[[0.1, 0.1], [0.1], [0.1, 0.1], []],
        timestamps=[1000.0, 999.0, 1001.0, 1003.5],
    )
    requests = btd.extract_requests(record)
    assert requests["n_source"] == 4
    assert requests["n_exact"] == 2
    assert requests["n_source"] - requests["n_exact"] == 2
    np.testing.assert_allclose(requests["arrival_time_s"], [1.0, 4.5])
    assert requests["output_tokens"].tolist() == [3, 1]
    assert [v.tolist() for v in requests["itls"]] == [[0.1, 0.1], []]
