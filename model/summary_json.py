import glob
import json
import os
import sys

import numpy as np
from core.dataset import PowerTraceDataset

if __name__ == "__main__":
    data_files = glob.glob("training_data/*.npz")
    for data_file in data_files:
        model = data_file.split("_")[-2]
        hw = data_file.split("_")[-1].split(".")[0]
        print(f"Processing {data_file} for model {model} on hardware {hw}")
        dataset = PowerTraceDataset(data_file, use_gmm=True)
        summary = {}
        for tp in list(set(dataset.tp_all)):
            summary[tp] = {}
            tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]
            min_power = float("inf")
            max_power = float("-inf")
            for idx in tp_indices:
                trace_power = dataset.traces[idx]["y"].flatten()
                min_power = min(min_power, np.min(trace_power))
                max_power = max(max_power, np.max(trace_power))
            summary[tp]["min_power"] = float(min_power)
            summary[tp]["max_power"] = float(max_power)
            summary[tp]["mu_values"] = list(dataset.mu[tp])
            summary[tp]["sigma_values"] = list(dataset.sigma[tp])
            summary[tp]["Dx"] = dataset.traces[0]["x"].shape[1]
            summary[tp]["K"] = dataset.state_labels[tp].n_components
            state_stats = {"median": [], "mad": [], "iqr": []}
            for k in range(dataset.state_labels[tp].n_components):
                powers = []
                for idx in tp_indices:
                    tr = dataset.traces[idx]
                    mask = tr["z"] == k
                    if mask.sum() > 0:
                        powers.extend(tr["y"][mask])

                if powers:
                    state_stats["median"].append(float(np.median(powers)))
                    state_stats["mad"].append(
                        float(np.median(np.abs(powers - np.median(powers))))
                    )
                    state_stats["iqr"].append(
                        float(np.percentile(powers, 75) - np.percentile(powers, 25))
                    )
            summary[tp]["state_stats"] = state_stats
        summary_file = f"model_summary_{model}_{hw}.json"
        print(f"Saving summary to {summary_file}")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {summary_file}")


def load_summary_from_json(summary_file: str):
    """Load the summary from a JSON file."""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file {summary_file} does not exist.")

    with open(summary_file, "r") as f:
        summary = json.load(f)

    return summary
