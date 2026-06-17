"""Fit the per-tool-class gap model on real OpenHands timing (offline, Slurm).

OpenHands native event streams carry per-event timestamps and a ``source`` field
(agent / environment), so the wall-clock gap between an agent's tool-call event and
the following environment observation is the *observed* tool-execution time. We fit

    log(gap_s) = mu_t + b1_t * log(observation_tokens + 1)   per tool class t

by OLS, with ``sigma_t`` = residual std, and fall back to literature priors for
classes with too few samples. Writes ``gap_params.json`` (the artifact the live
replayer loads).

Heavy (download + tokenize) -> run in a Slurm job with HF caches on $SCRATCH:

    HF_HOME=$SCRATCH/hf python -m openhands_gap_fit --model Qwen/Qwen3-8B -o gap_params.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

import tool_classes
from gap_sampler import DEFAULT_PARAMS

DATASET = "OpenHands/openhands-evaluation-outputs"
N_MIN = 30  # below this, a class keeps its literature prior rather than a noisy fit

# log-scale priors (median seconds), from the research synthesis (AgentServeSim /
# Claude-Code workload tables): local ops tight, bash wide, remote/subagent tails.
PRIORS = {
    "local_io": {"mu": -2.5, "sigma": 0.4, "b1": 0.05},
    "bash":     {"mu":  0.18, "sigma": 1.0, "b1": 0.18},
    "remote":   {"mu":  1.4,  "sigma": 1.3, "b1": 0.22},
    "subagent": {"mu":  4.1,  "sigma": 1.0, "b1": 0.10},
}


def _epoch(ts) -> float:
    """Parse an event timestamp (ISO-8601 string or epoch float) to seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def observed_gaps(events, tokenizer) -> list[tuple[str, int, float]]:
    """Walk one trajectory's events -> (tool_class, observation_tokens, gap_s).

    An ``agent`` event naming a tool opens a gap; the next ``environment`` event
    closes it (its content is the observation). Mixed LLM time is excluded because
    the gap is measured from the *action* event, not the LLM request.
    """
    out, pending = [], None
    for ev in events:
        source, name = ev.get("source"), ev.get("tool_name") or ev.get("action")
        if source == "agent" and name:
            pending = (tool_classes.classify(name), _epoch(ev["timestamp"]))
        elif source == "environment" and pending is not None:
            cls, t0 = pending
            gap = _epoch(ev["timestamp"]) - t0
            obs = len(tokenizer(ev.get("content") or "")["input_ids"])
            if gap > 0:
                out.append((cls, obs, gap))
            pending = None
    return out


def fit_class(samples: list[tuple[int, float]]) -> dict:
    """OLS fit of log(gap) ~ mu + b1*log(obs+1) for one class."""
    obs = np.array([o for o, _ in samples], dtype=float)
    y = np.log([g for _, g in samples])
    design = np.column_stack([np.ones_like(obs), np.log1p(obs)])
    (mu, b1), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ (mu, b1)
    sigma = float(resid.std(ddof=min(2, max(len(y) - 1, 1))))
    return {"mu": float(mu), "b1": float(b1), "sigma": sigma, "n": len(y)}


def fit_params(samples: list[tuple[str, int, float]], *, fit_commit=None) -> dict:
    """Group samples by class, fit where dense, else keep the literature prior."""
    classes = {}
    for cls in tool_classes.CLASSES:
        rows = [(o, g) for c, o, g in samples if c == cls]
        if len(rows) >= N_MIN:
            classes[cls] = {**fit_class(rows), "source": "fit"}
        else:
            classes[cls] = {**PRIORS[cls], "n": len(rows), "source": "literature_prior"}
    return {
        "schema_version": 1, "source": DATASET, "fit_commit": fit_commit,
        "n_total_samples": len(samples), "classes": classes,
    }


def _collect(model: str, max_trajectories: int | None) -> list[tuple[str, int, float]]:
    import itertools

    import transformers
    from datasets import load_dataset

    tok = transformers.AutoTokenizer.from_pretrained(model)
    ds = load_dataset(DATASET, split="train", streaming=True)
    if max_trajectories:
        ds = itertools.islice(ds, max_trajectories)
    samples = []
    for row in ds:
        events = row.get("events") or row.get("history") or []
        samples.extend(observed_gaps(events, tok))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="tokenizer for observation sizing")
    ap.add_argument("-o", "--out", default=str(DEFAULT_PARAMS))
    ap.add_argument("--max-trajectories", type=int, default=None)
    ap.add_argument("--emit", action="store_true", help="print params, don't write")
    args = ap.parse_args()

    params = fit_params(_collect(args.model, args.max_trajectories))
    text = json.dumps(params, indent=2)
    if args.emit:
        print(text)
    else:
        open(args.out, "w").write(text + "\n")
        print(f"wrote {args.out}: "
              + ", ".join(f"{c}(n={p['n']},{p['source']})"
                          for c, p in params["classes"].items()))


if __name__ == "__main__":
    main()
