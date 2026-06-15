"""Unit tests for the pure probe-schedule core (CAMPAIGN.md §3 / §5-B)."""

from schedule import (
    build_context_holds,
    build_decode_staircase,
    build_idle_hold,
    build_mixed_grid,
    build_prefill_staircase,
    build_transients,
)


def test_decode_staircase_levels():
    s = build_decode_staircase(256, hold_s=45.0)
    conc = [l.concurrency for l in s.levels]
    assert conc == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert conc[-1] == 256  # ceiling included exactly
    r = s.levels[0].request
    assert r.input_len == 8 and r.output_len == 2048 and r.ignore_eos
    assert all(l.hold_seconds == 45.0 for l in s.levels)


def test_decode_staircase_non_power_of_two_ceiling():
    s = build_decode_staircase(192)
    conc = [l.concurrency for l in s.levels]
    assert conc[-1] == 192               # exact ceiling
    assert conc[:-1] == [1, 2, 4, 8, 16, 32, 64, 128]


def test_prefill_staircase():
    s = build_prefill_staircase()
    assert [l.request.input_len for l in s.levels] == [256, 1024, 4096, 16384, 65536]
    assert all(l.concurrency == 1 for l in s.levels)
    assert all(l.request.output_len == 1 for l in s.levels)
    assert s.server_overrides["enable_chunked_prefill"] is False


def test_context_holds():
    contexts = (2048, 8192, 32768, 131072)
    s = build_context_holds(contexts=contexts, batch=8)
    assert [l.request.prefix_len for l in s.levels] == list(contexts)
    assert all(l.concurrency == 8 for l in s.levels)
    assert s.server_overrides["max_model_len"] >= max(contexts)
    assert s.server_overrides["env"]["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"


def test_transients_alternate_idle_and_load():
    s = build_transients(concurrency=64, repeats=3)
    conc = [l.concurrency for l in s.levels]
    assert conc == [0, 64, 0, 64, 0, 64]  # sharp idle<->load steps


def test_idle_hold():
    s = build_idle_hold(hold_s=60.0)
    assert len(s.levels) == 1
    assert s.levels[0].concurrency == 0
    assert s.levels[0].hold_seconds == 60.0


def test_mixed_grid_deterministic_and_bounded():
    a = build_mixed_grid(n_points=16, seed=0)
    b = build_mixed_grid(n_points=16, seed=0)
    assert [l.label for l in a.levels] == [l.label for l in b.levels]
    assert len(a.levels) == 16
    for l in a.levels:
        assert 1 <= l.concurrency <= 128
        assert 256 <= l.request.input_len <= 16384


def test_mixed_grid_seed_changes_points():
    a = build_mixed_grid(n_points=16, seed=0)
    c = build_mixed_grid(n_points=16, seed=1)
    assert [l.concurrency for l in a.levels] != [l.concurrency for l in c.levels]


def test_total_seconds():
    s = build_decode_staircase(8, hold_s=45.0)  # levels 1,2,4,8 -> 4 levels
    assert s.total_seconds == 45.0 * 4


def test_num_prompts_sized_per_level():
    from schedule import estimate_num_prompts
    # idle / zero concurrency -> no prompts
    assert estimate_num_prompts(0, 8, 2048, 45.0) == 0
    # decode level scales with concurrency
    s = build_decode_staircase(16, hold_s=45.0)
    for lvl in s.levels:
        assert lvl.num_prompts >= lvl.concurrency  # at least one wave
    # short fast prefills need many prompts to fill the hold
    pf = build_prefill_staircase(input_lens=(256,), hold_s=45.0)
    assert pf.levels[0].num_prompts > 10
    # idle levels in transients carry no prompts
    tr = build_transients(concurrency=8, repeats=2)
    idle = [l for l in tr.levels if l.concurrency == 0]
    assert all(l.num_prompts == 0 for l in idle)
