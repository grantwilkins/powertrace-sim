# pipeline.py ------------------------------------------------------------------
#  One‑shot script that ties together the entire framework:
#      raw consolidated NPZ  →  bundles  →  token‑GPs  →  power‑GPs  →  hyper‑GP
#   + produces diagnostic plots (training losses, hold‑out errors).
# -----------------------------------------------------------------------------
#  Usage (from repo root, after `pip install -e .`):
#     python pipeline.py \
#         --npz   processed_data/power_trace_data.npz \
#         --work  experiments/debug            \
#         --device cuda                        \
#         --exclude deepseek
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse, json, random, math
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

import data_io as dataio
import token_generator as token_gen
import power_gp
import hyper_gp
import simulator
from token_generator import load_model as load_token_model
from hyper_gp import HyperGP
from power_gp import PowerSVGP, load_model as load_power_model


plt.rcParams.update({"font.size": 11})

OUT_DIR = Path("debug")  # or Path(args.work_dir) …
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ckpt_path(bundle: dict) -> Path:
    """Unique checkpoint name for one (rate,tp,ms) triple."""
    r = bundle["rate"]  # e.g. 0.25
    tp = bundle["tp"]  # e.g. 8
    ms = str(bundle["ms"])  # e.g. "7B"
    tag = f"pr{r:.2f}_tp{tp}_ms{ms}.pt".replace(" ", "")
    return OUT_DIR / tag


def plot_losses(loss_lists: Dict[str, List[float]], out: Path):
    plt.figure(figsize=(6, 4))
    for lbl, ls in loss_lists.items():
        plt.plot(ls, label=lbl)
    plt.xlabel("iteration")
    plt.ylabel("ELBO loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_error_hist(errors: List[float], out: Path, title: str):
    plt.figure(figsize=(5, 4))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Relative error (%)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="consolidated npz file")
    ap.add_argument("--work", required=True, help="output work directory")
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="case‑insensitive patterns to drop"
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    work = Path(args.work)
    (work / "plots").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1)  NPZ  →  bundles
    # ------------------------------------------------------------------
    bundles = dataio.npz_to_bundles(args.npz)
    if args.exclude:
        bundles = [
            b
            for b in bundles
            if not any(
                p.lower() in (b["power"].__repr__() + str(b)).lower()
                for p in args.exclude
            )
        ]
    dataio.save_bundles(bundles, work / "bundles")

    for field in ("prefill", "decode"):
        X, y = token_gen.build_training_set(bundles, field)
        mdl_dict = token_gen.train_token_gp(X, y, device=args.device)
        token_gen.save_model(mdl_dict, work / "token" / f"{field}.pt")

    power_loss_curves: Dict[str, List[float]] = {}
    rel_errors = []
    hyper_rows: List[Tuple[np.ndarray, np.ndarray]] = []

    for b in bundles:
        # leave‑one sample (last) out as test

        ckpt = _ckpt_path(b)
        if ckpt.exists():
            continue

        train_power = b["power"][:-1]
        test_power = b["power"][-1]

        tmp_bundle = dict(b)
        tmp_bundle["power"] = train_power
        model_dict = power_gp.train_power_gp(
            tmp_bundle, device=args.device, verbose=False
        )
        power_gp.save_model(
            model_dict, work / "power" / f"rate{b['rate']:.4g}_tp{b['tp']}_{b['ms']}.pt"
        )

        # collect for hyper‑GP
        hyper_rows.append(hyper_gp.collect_hyper_row(model_dict))

        # quick error on hold‑out
        svgp = power_gp.load_model(
            work / "power" / f"rate{b['rate']:.4g}_tp{b['tp']}_{b['ms']}.pt"
        )
        t = test_power.shape[0]
        time = torch.linspace(0, 1, t).unsqueeze(-1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = svgp.likelihood(svgp(time)).mean.squeeze().cpu().numpy() * b["tp"]
        err = np.mean(np.abs(pred - test_power) / np.maximum(test_power, 1e-3)) * 100
        rel_errors.append(err)

    plot_error_hist(
        rel_errors, work / "plots" / "power_rel_err_hist.png", "Power GP hold‑out MAPE"
    )

    # ------------------------------------------------------------------
    # 4)  hyper‑GP
    # ------------------------------------------------------------------
    hgp = hyper_gp.HyperGP()
    hgp.fit(hyper_rows)
    hgp.save(work / "hyper_gp.pt")

    # ------------------------------------------------------------------
    # 5)  quick simulation sanity
    # ------------------------------------------------------------------
    sim = simulator.PowerSimulator(
        bundle_dir=work / "bundles",
        model_dir=work / "power",
        token_dir=work / "token",
        hyper_path=work / "hyper_gp.pt",
        dt=0.25,
    )

    # choose a random configuration from training grid and a 120‑s horizon
    key = random.choice(list(sim.bundles.keys()))
    rate, tp, ms = key
    power_pred, _ = sim.run(
        duration_s=120, poisson_rate=rate, tensor_parallelism=tp, model_size=ms
    )
    np.save(work / "sample_power.npy", power_pred)
    print(
        f"saved example power prediction → {work/'sample_power.npy'}  (shape {power_pred.shape})"
    )


if __name__ == "__main__":
    main()
