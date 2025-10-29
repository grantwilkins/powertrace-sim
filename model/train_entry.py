import argparse
import os

import numpy as np
import torch
from classifiers.train import train_classifiers
from core.dataset import PowerTraceDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IOHMM model on power traces")
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to the NPZ data file"
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
        required=True,
        help="Tensor parallelism value to train. Use -1 to train all TP values in the dataset.",
    )
    parser.add_argument("--hardware_accelerator", type=str, required=True)
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./gru_classifier_weights/",
        help="Path to the classifier weights file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cuda:0, cpu, etc.). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        help="Number of epochs to train for",
    )
    args = parser.parse_args()

    dataset = PowerTraceDataset(args.data_file)
    print(dataset)

    os.makedirs("./training_data/losses", exist_ok=True)
    os.makedirs(args.weights_path, exist_ok=True)
    if args.tp == -1:
        # Train all TP values in the dataset
        unique_tps = list(set(dataset.tp_all))
        for tp in unique_tps:
            print(f"Training classifier for TP={tp}")
            classifier, losses = train_classifiers(
                dataset,
                tp=tp,
                device=torch.device(args.device) if args.device else None,
                num_epochs=args.num_epochs,
                lr=args.lr,
            )
            classifier.to("cpu")
            np.save(
                f"./training_data/losses/training_losses_{args.model}_{args.hardware_accelerator}_tp{tp}.npy",
                np.array(losses),
            )
            torch.save(
                classifier.state_dict(),
                f"{args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{tp}.pt",
            )
    else:
        # Train specific TP value
        classifier, losses = train_classifiers(
            dataset,
            tp=args.tp,
            device=torch.device(args.device) if args.device else None,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )
        # Ensure directories exist
        classifier.to("cpu")
        np.save(
            f"./training_data/losses/training_losses_{args.model}_{args.hardware_accelerator}_tp{args.tp}.npy",
            np.array(losses),
        )
        torch.save(
            classifier.state_dict(),
            f"{args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{args.tp}.pt",
        )
