#!/usr/bin/env python3
# demo_prefill_tokens.py
# -----------------------------------------------------------
# Visualise the pre-fill token GP by sampling a fresh sequence
# -----------------------------------------------------------

import argparse, math, numpy as np, matplotlib.pyplot as plt, torch
import token_generator as tok_gen  # your module


def load_bundle(path: str):
    """Return the first sample in the bundle as a 1-D numpy array."""
    data = np.load(path, allow_pickle=True)
    # adapt if your NPZ structure differs
    tokens = data["prefill"][0]  # (T,) first sample
    dt = data.get("dt", 0.25)  # 250 ms default
    return tokens, float(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="prefill GP checkpoint")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--ms", required=True, help="model size, e.g. 7 or 70B")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=1, help="# samples")
    ap.add_argument("--bundle", help="npz file to overlay a real trace")
    ap.add_argument(
        "--num_inducing", type=int, default=1000, help="Number of inducing points"
    )
    args = ap.parse_args()

    # 1) load GP ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp = tok_gen.load_model(args.ckpt, device=device, num_inducing=args.num_inducing)

    # 2) sample -------------------------------------------------------------
    seq = tok_gen.sample_tokens(
        gp,
        duration_s=args.seconds,
        dt=args.dt,
        rate=args.rate,
        tp=args.tp,
        ms=args.ms,
        n_samples=args.n,
        device=device,
    )[
        0
    ]  # take first sample

    t = np.linspace(0, args.seconds, len(seq), endpoint=False)

    # 3) optional ground-truth overlay -------------------------------------
    if args.bundle:
        true_tokens, true_dt = load_bundle(args.bundle)
        t_true = np.arange(len(true_tokens)) * true_dt
        plt.plot(t_true, true_tokens, "g-", lw=2, label="real prefill")

    # plot GP sample
    plt.step(t, seq, where="post", lw=1.5, label="GP sample")
    plt.xlabel("time [s]")
    plt.ylabel("prefill tokens / 250 ms")
    plt.title(f"λ={args.rate} /s, TP={args.tp}, MS={args.ms}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
