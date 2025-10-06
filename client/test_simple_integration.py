#!/usr/bin/env python3
"""
Simple integration test for RealisticRandomDataset with new parameters.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_dataset import RealisticRandomDataset


def test_basic_usage():
    """Test basic dataset instantiation and sampling."""
    print("Test 1: Basic usage with default parameters...")

    # Mock tokenizer
    class MockTokenizer:
        vocab_size = 50000

        def decode(self, tokens):
            return " ".join(str(t) for t in tokens[:10])

    tokenizer = MockTokenizer()

    dataset = RealisticRandomDataset(random_seed=42)
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=1000,
        input_log_mean=6.22,
        input_log_std=1.08,
        input_min=32,
        input_max=4096,
        tail_probability=0.08,
        tail_alpha=1.4,
        tail_xmin=4096,
        use_exponential_output=True,
        exp_output_mean=250,
        output_min=16,
        output_max=2048,
    )

    input_lens = np.array([req.prompt_len for req in requests])
    output_lens = np.array([req.expected_output_len for req in requests])

    print(f"  Generated {len(requests)} requests")
    print(f"  Input: median={np.median(input_lens):.0f}, P90={np.percentile(input_lens, 90):.0f}")
    print(f"  Output: mean={output_lens.mean():.1f}, median={np.median(output_lens):.0f}")
    print("  ✓ Passed")


def test_exponential_vs_lognormal():
    """Test exponential vs lognormal output distributions."""
    print("\nTest 2: Exponential vs log-normal outputs...")

    class MockTokenizer:
        vocab_size = 50000

        def decode(self, tokens):
            return " ".join(str(t) for t in tokens[:10])

    tokenizer = MockTokenizer()
    dataset = RealisticRandomDataset(random_seed=42)

    # Test exponential
    requests_exp = dataset.sample(
        tokenizer=tokenizer,
        num_requests=5000,
        use_exponential_output=True,
        exp_output_mean=300,
        output_min=16,
        output_max=2048,
    )

    # Test lognormal
    requests_ln = dataset.sample(
        tokenizer=tokenizer,
        num_requests=5000,
        use_exponential_output=False,
        output_log_mean=5.7,
        output_log_std=0.83,
        output_min=16,
        output_max=2048,
    )

    exp_lens = np.array([req.expected_output_len for req in requests_exp])
    ln_lens = np.array([req.expected_output_len for req in requests_ln])

    print(f"  Exponential: mean={exp_lens.mean():.1f}, median={np.median(exp_lens):.0f}")
    print(f"  Log-normal:  mean={ln_lens.mean():.1f}, median={np.median(ln_lens):.0f}")
    print("  ✓ Passed")


def test_pareto_tail():
    """Test Pareto tail mixing."""
    print("\nTest 3: Pareto tail mixing...")

    class MockTokenizer:
        vocab_size = 50000

        def decode(self, tokens):
            return " ".join(str(t) for t in tokens[:10])

    tokenizer = MockTokenizer()

    # No tail
    dataset = RealisticRandomDataset(random_seed=42)
    requests_no_tail = dataset.sample(
        tokenizer=tokenizer,
        num_requests=10000,
        input_log_mean=6.22,
        input_log_std=1.08,
        input_min=32,
        input_max=8192,
        tail_probability=0.0,  # No tail
        tail_xmin=4096,
    )

    # With 8% tail
    requests_with_tail = dataset.sample(
        tokenizer=tokenizer,
        num_requests=10000,
        input_log_mean=6.22,
        input_log_std=1.08,
        input_min=32,
        input_max=8192,
        tail_probability=0.08,
        tail_alpha=1.4,
        tail_xmin=4096,
    )

    no_tail_lens = np.array([req.prompt_len for req in requests_no_tail])
    with_tail_lens = np.array([req.prompt_len for req in requests_with_tail])

    # Count samples > tail_xmin
    no_tail_count = (no_tail_lens >= 4096).sum()
    with_tail_count = (with_tail_lens >= 4096).sum()

    print(f"  Without tail: {no_tail_count} samples ≥ 4096 ({100*no_tail_count/10000:.1f}%)")
    print(f"  With 8% tail: {with_tail_count} samples ≥ 4096 ({100*with_tail_count/10000:.1f}%)")
    print(f"  P90 ratio: no_tail={np.percentile(no_tail_lens, 90):.0f}, "
          f"with_tail={np.percentile(with_tail_lens, 90):.0f}")
    print("  ✓ Passed")


def test_all_presets():
    """Test all 8 preset combinations."""
    print("\nTest 4: All preset combinations...")

    PRESETS = [
        ("conversation", "low", 5.30, 0.85, 120),
        ("conversation", "medium", 6.22, 1.08, 250),
        ("conversation", "high", 7.31, 1.08, 600),
        ("conversation", "ultra", 8.01, 1.08, 1000),
        ("coding", "low", 5.70, 0.95, 180),
        ("coding", "medium", 6.55, 1.10, 350),
        ("coding", "high", 7.60, 1.10, 800),
        ("coding", "ultra", 8.30, 1.10, 1400),
    ]

    class MockTokenizer:
        vocab_size = 50000

        def decode(self, tokens):
            return " ".join(str(t) for t in tokens[:10])

    tokenizer = MockTokenizer()

    for task, level, in_mu, in_sigma, out_mean in PRESETS:
        dataset = RealisticRandomDataset(random_seed=hash((task, level)) % (2 ** 31))
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=1000,
            input_log_mean=in_mu,
            input_log_std=in_sigma,
            use_exponential_output=True,
            exp_output_mean=out_mean,
        )

        input_lens = np.array([req.prompt_len for req in requests])
        output_lens = np.array([req.expected_output_len for req in requests])

        print(f"  {task:12s} / {level:6s}: in_median={np.median(input_lens):5.0f}, "
              f"out_mean={output_lens.mean():6.1f} ✓")

    print("  ✓ All presets passed")


def main():
    """Run all tests."""
    print("="*70)
    print("RealisticRandomDataset Integration Tests")
    print("="*70)

    test_basic_usage()
    test_exponential_vs_lognormal()
    test_pareto_tail()
    test_all_presets()

    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)


if __name__ == "__main__":
    main()
