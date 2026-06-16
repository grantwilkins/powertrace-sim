"""Pure tests for the dataset mixer (record schema + equal-balance + determinism).

The HF streaming in ``stream_pairs`` is the live layer, exercised when a mix is
built; here we pin the parts that decide the output file's correctness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

import mix_dataset  # noqa: E402


def test_to_sharegpt_record_schema():
    r = mix_dataset.to_sharegpt_record("hello", "world", "src/a")
    assert r["conversations"][0]["value"] == "hello"
    assert r["conversations"][1]["value"] == "world"
    assert len(r["conversations"]) >= 2  # ShareGPT loader requires >= 2 turns
    assert r["source"] == "src/a"


def _recs(prefix, k):
    return [mix_dataset.to_sharegpt_record(f"{prefix}-p{i}", f"{prefix}-c{i}", prefix)
            for i in range(k)]


def test_equal_random_mix_balances_to_smallest_source():
    by_source = {"A": _recs("A", 10), "B": _recs("B", 4)}
    mixed = mix_dataset.equal_random_mix(by_source, seed=0)
    counts = {}
    for r in mixed:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    assert counts == {"A": 4, "B": 4}        # equal, truncated to the smaller source
    assert len(mixed) == 8


def test_equal_random_mix_is_deterministic_and_shuffled():
    by_source = {"A": _recs("A", 6), "B": _recs("B", 6)}
    m1 = mix_dataset.equal_random_mix(by_source, seed=7)
    m2 = mix_dataset.equal_random_mix(by_source, seed=7)
    m3 = mix_dataset.equal_random_mix(by_source, seed=8)
    order = lambda m: [r["conversations"][0]["value"] for r in m]
    assert order(m1) == order(m2)            # same seed -> identical
    assert order(m1) != order(m3)            # different seed -> different order
    # not grouped by source (actually interleaved)
    sources = [r["source"] for r in m1]
    assert sources != sorted(sources)


def test_wildchat_and_opencode_extractors():
    wc = {"conversation": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}]}
    assert mix_dataset._wildchat_pair(wc) == ("q1", "a1")
    oc = {"input": "write a function", "output": "def f(): ..."}
    assert mix_dataset._opencode_pair(oc) == ("write a function", "def f(): ...")
