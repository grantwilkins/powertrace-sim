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


def make_schedule_matrix(trace_dict: Dict[str, np.ndarray]):
    """
    trace_dict contains 1-D numpy arrays *already cut to true length*.
    Returns x_t  (T × Dx)  where columns are z-scored.
    """

    cnt, tok_in, tok_out = histogram_requests(
        bin_ts=trace_dict["timestamps"],
        req_ts=trace_dict["request_ts"],
        in_tok=trace_dict["input_tokens"],
        out_tok=trace_dict["output_tokens"],
    )

    x = np.stack(
        [
            cnt,
            tok_in,
            tok_out,
            trace_dict["active_requests"],
            trace_dict["prefill_tokens"],
            trace_dict["decode_tokens"],
        ],
        axis=1,
    ).astype("float32")

    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


def load_classifier(
    path, device: Optional[torch.device] = None, Dx: int = 6, K: int = 6
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
    classifier = GRUClassifier(Dx=Dx, K=K).to(device)
    classifier.load_state_dict(torch.load(path, map_location=device))
    return classifier
