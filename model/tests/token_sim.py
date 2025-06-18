import matplotlib.pyplot as plt
import numpy as np
from core.dataset import PowerTraceDataset
from simulators.arrival_simulator import ModelConfig, TokenSimulator


def test_token_simulator():
    dataset = PowerTraceDataset(
        "/Users/grantwilkins/powertrace-sim/model/training_data/vllm-benchmark_deepseek-r1-8b_a100.npz"
    )
    sim = TokenSimulator.from_npz(
        "/Users/grantwilkins/powertrace-sim/model/training_data/vllm-benchmark_deepseek-r1-8b_a100.npz"
    )
    results = sim.run_simulation(config=ModelConfig(8, 2, "A100"))
    plt.figure(figsize=(5, 3))
    plt.plot(results["timestamps"], results["prefill_tokens"])
    plt.plot(results["timestamps"], results["decode_tokens"])
    plt.ylabel("Tokens")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    tp2_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == 2]
    orig_prefill_tokens = []
    orig_decode_tokens = []
    for idx in tp2_indices:
        orig_prefill_tokens.append(dataset.traces[idx]["prefill_tokens"])
        orig_decode_tokens.append(dataset.traces[idx]["decode_tokens"])

    # make a line cdf of the prefill tokens against that of the original
    orig_prefill_tokens = np.concatenate(orig_prefill_tokens)
    orig_decode_tokens = np.concatenate(orig_decode_tokens)
    sim_prefill_tokens = results["prefill_tokens"]
    sim_decode_tokens = results["decode_tokens"]
    sim_prefill_tokens = np.array(sim_prefill_tokens)
    sim_decode_tokens = np.array(sim_decode_tokens)
    orig_prefill_tokens = np.array(orig_prefill_tokens)
    orig_decode_tokens = np.array(orig_decode_tokens)
    sim_prefill_tokens = np.sort(sim_prefill_tokens)
    sim_decode_tokens = np.sort(sim_decode_tokens)
    orig_prefill_tokens = np.sort(orig_prefill_tokens)
    orig_decode_tokens = np.sort(orig_decode_tokens)
    sim_prefill_cdf = np.arange(1, len(sim_prefill_tokens) + 1) / len(
        sim_prefill_tokens
    )
    sim_decode_cdf = np.arange(1, len(sim_decode_tokens) + 1) / len(sim_decode_tokens)
    orig_prefill_cdf = np.arange(1, len(orig_prefill_tokens) + 1) / len(
        orig_prefill_tokens
    )
    orig_decode_cdf = np.arange(1, len(orig_decode_tokens) + 1) / len(
        orig_decode_tokens
    )
    plt.figure(figsize=(5, 3))
    plt.plot(
        sim_prefill_tokens,
        sim_prefill_cdf,
        label="Simulated Prefill Tokens",
        color="blue",
    )
    plt.plot(
        orig_prefill_tokens,
        orig_prefill_cdf,
        label="Original Prefill Tokens",
        color="orange",
    )
    plt.xlabel("Prefill Tokens")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(
        sim_decode_tokens,
        sim_decode_cdf,
        label="Simulated Decode Tokens",
        color="blue",
    )
    plt.plot(
        orig_decode_tokens,
        orig_decode_cdf,
        label="Original Decode Tokens",
        color="orange",
    )
    plt.xlabel("Decode Tokens")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_token_simulator()
