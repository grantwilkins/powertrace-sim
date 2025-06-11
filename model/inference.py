import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from core.dataset import PowerTraceDataset
from core.utils import load_classifier
from predictors.smooth_sampler import SmoothingSampler
from simulators.arrival_simulator import ModelConfig, TokenSimulator
from summary_json import load_summary_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer model to create power traces")
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama-3-8b",
            "llama-3-70b",
            "deepseek-r1-distill-8b",
            "deepseek-r1-distill-70b",
        ],
        help="LLM name",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism value to infer on",
    )
    parser.add_argument("--hardware_accelerator", type=str, default="a100")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./gru_classifier_weights/",
        help="Path to the classifier weights folder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on on (cuda, cuda:0, cpu, etc.).",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000,
        help="Time horizon for the simulation in seconds",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.1,
        help="Arrival rate for the token simulator (tokens per second)",
    )

    args = parser.parse_args()
    dataset = PowerTraceDataset(args.data_file, use_gmm=True)
    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = load_summary_from_json(
        f"./summary_data/model_summary_{args.model}_{args.hardware_accelerator}.json"
    )
    print(summary)

    tp = str(args.tp)
    classifier = load_classifier(
        args.weights_path + args.model + f"_{args.hardware_accelerator}_tp{tp}.pt",
        device=device,
        Dx=summary[tp]["Dx"],
        K=summary[tp]["K"],
    )
    smoother = SmoothingSampler(state_stats=summary[tp]["state_stats"])

    # # ENSURE WE CAN GET MEANINGFUL HISTs
    # sim = TokenSimulator.from_npz(args.data_file)
    # sim_results = sim.run_simulation(
    #     config=ModelConfig(
    #         model_size=int(args.model.split("-")[-1].replace("b", "")),
    #         tensor_parallelism=int(tp),
    #         hardware=args.hardware_accelerator,
    #     ),
    #     time_horizon=args.T,
    #     arrival_rate=args.rate,
    # )
    # schedule = sim.prepare_for_inference(sim_results)
    # get a schedule from the data set
    tp1_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == int(tp)]
    idx = np.random.choice(tp1_indices)
    # idx = 1
    # print arrival rate of this idx
    print(summary[tp]["mu_values"])
    time_vals, power, states = smoother.sample_power(
        classifier,
        # np.array(summary[tp]["mu_values"]),
        # np.array(summary[tp]["sigma_values"]),
        dataset.mu[int(tp)],
        dataset.sigma[int(tp)],
        schedule_x=dataset.traces[idx]["x"],
        dt=0.25,
        smoothing_window=5,
    )
    power = np.clip(power, summary[tp]["min_power"], summary[tp]["max_power"])
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(time_vals, power, label="Simulated Power Trace")
    plt.plot(
        time_vals,
        dataset.traces[idx]["y"].flatten(),
        label="Original Power Trace",
        linestyle="--",
    )
    plt.xlabel("Time (s)")
    plt.show()

    example_timestamp = time.time()
    time_vals += example_timestamp
    # ADD SKU
    if args.hardware_accelerator == "a100":
        sku = "Standard_ND96amsr_A100_v4"
    elif args.hardware_accelerator == "h100":
        sku = "Standard_ND96isr_H100_v5"
    else:
        sku = "Unknown_SKU"

    # Create DataFrame for easier CSV handling with mixed data types
    # df = pd.DataFrame({"Timestamp": time_vals, "GPU Power (W)": power, "VM SKU": sku})

    # # Save to CSV
    # output_file = (
    #     f"./power_traces/{args.model}_{args.hardware_accelerator}_tp{tp}_id{idx}.csv"
    # )
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # df.to_csv(output_file, index=False)
