import numpy as np
import torch
from core.dataset import PowerTraceDataset

# Load the dataset
dataset = PowerTraceDataset("./training_data/vllm-benchmark_llama-3-70b_a100.npz")

# Check TP=8 specifically
tp = 8
tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

print(f"Number of traces for TP={tp}: {len(tp_indices)}")

# Check the first trace for TP=8
if tp_indices:
    idx = tp_indices[0]
    x_sample, y_sample, z_sample = dataset[idx]

    print(
        f"Sample trace shape: x={x_sample.shape}, y={y_sample.shape}, z={z_sample.shape}"
    )
    print(f"Unique z values: {torch.unique(z_sample)}")
    print(f"Number of unique z values: {len(torch.unique(z_sample))}")
    print(f"Min z value: {torch.min(z_sample)}")
    print(f"Max z value: {torch.max(z_sample)}")

    # Check what K was used in clustering
    print(f"Clustering model n_components: {dataset.state_labels[tp].n_components}")

    # Check if there's a mismatch
    unique_count = len(torch.unique(z_sample))
    expected_count = dataset.state_labels[tp].n_components

    print(f"Expected classes (K): {expected_count}")
    print(f"Actual unique labels: {unique_count}")

    if unique_count != expected_count:
        print("MISMATCH DETECTED!")
        print("This is likely causing the 'Target out of bounds' error")
else:
    print(f"No traces found for TP={tp}")
