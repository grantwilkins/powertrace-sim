from typing import Dict, Optional

import numpy as np
import torch

from ..classifiers.gru import GRUClassifier


def histogram_requests(
    bin_ts: np.ndarray, req_ts: np.ndarray, in_tok: np.ndarray, out_tok: np.ndarray
):
    dt = np.median(np.diff(bin_ts)) if len(bin_ts) > 1 else 0.25
    if len(bin_ts) > 1:
        if not np.all(np.diff(bin_ts) > 0):
            bin_ts = np.unique(bin_ts)
            if len(bin_ts) <= 1:
                bin_ts = (
                    np.array([0, dt])
                    if len(bin_ts) == 0
                    else np.array([bin_ts[0], bin_ts[0] + dt])
                )
    else:
        bin_ts = (
            np.array([0, dt])
            if len(bin_ts) == 0
            else np.array([bin_ts[0], bin_ts[0] + dt])
        )
    edges = np.append(bin_ts, bin_ts[-1] + dt)
    new_req_cnt, _ = np.histogram(req_ts, edges)
    new_in_tok, _ = np.histogram(req_ts, edges, weights=in_tok)
    new_out_tok, _ = np.histogram(req_ts, edges, weights=out_tok)

    return new_req_cnt.astype("float32"), new_in_tok, new_out_tok


def make_schedule_matrix(
    trace_dict: Dict[str, np.ndarray],
    arrival_rate: float = None,
    add_diff_features: bool = True,
):
    """
    trace_dict contains 1-D numpy arrays *already cut to true length*.
    arrival_rate: Poisson arrival rate for the trace (optional, broadcasts to all timesteps).
    add_diff_features: Whether to add first-difference features for temporal signals.
    Returns x_t  (T × Dx)  where columns are z-scored.
    """

    cnt, tok_in, tok_out = histogram_requests(
        bin_ts=trace_dict["timestamps"],
        req_ts=trace_dict["request_ts"],
        in_tok=trace_dict["input_tokens"],
        out_tok=trace_dict["output_tokens"],
    )

    T = len(cnt)

    # Store key temporal signals for diff features
    active_req = trace_dict["active_requests"]
    prefill_tok = trace_dict["prefill_tokens"]
    decode_tok = trace_dict["decode_tokens"]

    # Base features (6 features)
    features = [
        cnt,
        tok_in,
        tok_out,
        active_req,
        prefill_tok,
        decode_tok,
    ]

    # Add arrival rate as 7th feature if provided
    if arrival_rate is not None:
        # Broadcast log2(arrival_rate) to all timesteps
        arrival_rate_feature = np.full(T, np.log2(arrival_rate), dtype="float32")
        features.append(arrival_rate_feature)

    # Add first-difference features to help detect transitions
    if add_diff_features:
        # Compute diffs with prepend (first timestep has diff=0)
        diff_active_req = np.diff(active_req, prepend=active_req[0])
        diff_prefill_tok = np.diff(prefill_tok, prepend=prefill_tok[0])
        diff_decode_tok = np.diff(decode_tok, prepend=decode_tok[0])

        features.extend([diff_active_req, diff_prefill_tok, diff_decode_tok])

    x = np.stack(features, axis=1).astype("float32")

    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


def fit_temperature_scaling(
    logits: np.ndarray, labels: np.ndarray, T_range: np.ndarray = None
) -> float:
    """
    Fit temperature scaling parameter to minimize NLL on validation set.

    Args:
        logits: Unnormalized logits (N, K)
        labels: True labels (N,)
        T_range: Temperatures to search (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    Returns:
        Best temperature T
    """
    if T_range is None:
        T_range = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])

    best_T = 1.0
    best_nll = float("inf")

    for T in T_range:
        # Apply temperature scaling
        scaled_logits = logits / T
        # Compute log probabilities
        log_probs = scaled_logits - np.log(
            np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        )
        # Compute NLL
        nll = -np.mean(log_probs[np.arange(len(labels)), labels])

        if nll < best_nll:
            best_nll = nll
            best_T = T

    return best_T


def apply_temperature_scaling(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Unnormalized logits (..., K)
        T: Temperature parameter

    Returns:
        Calibrated probabilities (..., K)
    """
    scaled_logits = logits / T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def viterbi_decode(
    logits: np.ndarray, transition_penalty: float = 1.0, self_loop_bonus: float = 0.5
) -> np.ndarray:
    """
    Viterbi decoding to smooth state sequences with transition penalties.

    Args:
        logits: Log probabilities (T, K) or (B, T, K)
        transition_penalty: Cost for switching states (higher = smoother)
        self_loop_bonus: Bonus for staying in same state

    Returns:
        Decoded state sequence (T,) or (B, T)
    """
    single_trace = logits.ndim == 2
    if single_trace:
        logits = logits[None, ...]  # (1, T, K)

    B, T, K = logits.shape
    decoded = np.zeros((B, T), dtype=np.int64)

    for b in range(B):
        # Build uniform transition matrix with self-loop preference
        trans_mat = -np.ones((K, K)) * transition_penalty
        np.fill_diagonal(trans_mat, self_loop_bonus)

        # Viterbi forward pass
        dp = np.zeros((T, K))
        path = np.zeros((T, K), dtype=np.int64)

        # Initialize with first frame
        dp[0] = logits[b, 0]

        # Forward pass
        for t in range(1, T):
            for k in range(K):
                # Score for transitioning to state k at time t
                scores = dp[t - 1] + trans_mat[:, k] + logits[b, t, k]
                best_prev = np.argmax(scores)
                dp[t, k] = scores[best_prev]
                path[t, k] = best_prev

        # Backward pass (traceback)
        decoded[b, -1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            decoded[b, t] = path[t + 1, decoded[b, t + 1]]

    return decoded[0] if single_trace else decoded


def load_classifier(
    path, device: Optional[torch.device] = None, Dx: int = 7, K: int = 6
):
    """
    Load a classifier from a file.
    """
    import os

    # Resolve path relative to project root
    if not os.path.isabs(path) and not os.path.exists(path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        resolved = os.path.join(project_root, path)
        if os.path.exists(resolved):
            path = resolved

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading classifier from {path} on device: {device}")
    classifier = GRUClassifier(H=64, Dx=Dx, K=K, num_layers=2).to(device)
    classifier.load_state_dict(torch.load(path, map_location=device))
    return classifier
