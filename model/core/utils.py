import numpy as np


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


def make_schedule_matrix(trace_dict):
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


class SmoothingSampler:
    def __init__(self, dataset, tp=None):
        self.state_stats = {}
        self.tp = tp

        for k in range(6):
            powers = []
            for tr in dataset.traces:
                if tp is not None and tr.get("tp", None) != tp:
                    continue

                mask = tr["z"] == k
                if mask.sum() > 0:
                    powers.extend(tr["y"][mask])

            if powers:
                self.state_stats[k] = {
                    "median": np.median(powers),
                    "mad": np.median(np.abs(powers - np.median(powers))),
                    "iqr": np.percentile(powers, 75) - np.percentile(powers, 25),
                    "tp": tp,
                }

    def smooth_sample(self, time, power, states, smoothing_window=5, tp=None):
        tp = tp if tp is not None else self.tp
        smoothed = power.copy()
        for i in range(len(power)):
            start = max(0, i - smoothing_window // 2)
            end = min(len(power), i + smoothing_window // 2 + 1)
            current_state = states[i]
            same_state_mask = states[start:end] == current_state

            if same_state_mask.sum() > 0:
                window_values = power[start:end][same_state_mask]
                smoothed[i] = np.median(window_values)

        return time, smoothed, states
