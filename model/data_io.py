# dataio.py
# -----------------------------------------------------------------------------
#  Dataset utilities for the Power‑Trace GP project.
# -----------------------------------------------------------------------------
#  • Converts the consolidated NPZ (output of your earlier CSV‑processing stage)
#    into *parameter‑keyed* bundles so that each (rate, TP, model‑size) setting
#    can be trained independently.
#  • Serialises every bundle as a compact pickle (≈ few MB) that contains
#        {"power":  (S, T),
#         "prefill":(S, T) or None,
#         "decode": (S, T) or None,
#         "rate":   float,
#         "tp":     int,
#         "ms":     str/int}
#  • Provides loader helpers for downstream training / simulation code.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import pickle
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

__all__ = [
    "TraceBundle",
    "npz_to_bundles",
    "save_bundles",
    "load_bundle",
    "load_all_bundles",
]


class TraceBundle(Dict[str, np.ndarray]):
    """Typed alias for a parameter‑specific data bundle."""

    @property
    def key(self) -> Tuple[float, int, str]:
        """Short key (rate, tp, ms) used for dict lookup."""
        return float(self["rate"]), int(self["tp"]), str(self["ms"])

    @property
    def power(self) -> np.ndarray:  # (S, T)
        return self["power"]

    @property
    def prefill(self) -> Optional[np.ndarray]:
        return self.get("prefill", None)

    @property
    def decode(self) -> Optional[np.ndarray]:
        return self.get("decode", None)


def npz_to_bundles(npz_path: str) -> List[TraceBundle]:
    """Load the consolidated `.npz` file and split it by (rate,tp,ms).

    Parameters
    ----------
    npz_path : str
        Path to *processed_data/power_trace_data.npz* (or alike).

    Returns
    -------
    list[TraceBundle]
        One bundle per unique parameter triple.  Each bundle contains **all**
        (up to five) samples collected in the experiment.
    """

    npz = np.load(npz_path, allow_pickle=True)
    power = npz["power_traces"]  # (N, T)
    rate_arr = npz["poisson_rate"]  # (N,)
    tp_arr = npz["tensor_parallelism"]  # (N,)
    ms_arr = npz["model_sizes"]  # (N,)

    prefill_arr = npz.get("prefill_tokens", None)
    decode_arr = npz.get("decode_tokens", None)

    # sanity check ------------------------------------------------------------
    n, t = power.shape
    for arr, nm in ((rate_arr, "rate"), (tp_arr, "tp"), (ms_arr, "ms")):
        if len(arr) != n:
            raise ValueError(f"{nm} array length mismatch: {len(arr)} vs {n}")
    if prefill_arr is not None and prefill_arr.shape != (n, t):
        raise ValueError("prefill_tokens shape mismatch with power_traces")
    if decode_arr is not None and decode_arr.shape != (n, t):
        raise ValueError("decode_tokens shape mismatch with power_traces")

    # group indices by parameter triple --------------------------------------
    bundles: List[TraceBundle] = []
    for rate, tp, ms in sorted({tuple(k) for k in zip(rate_arr, tp_arr, ms_arr)}):
        mask = (rate_arr == rate) & (tp_arr == tp) & (ms_arr == ms)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        bundle: TraceBundle = {
            "rate": float(rate),
            "tp": int(tp),
            "ms": str(ms),
            "power": power[idx],
        }
        if prefill_arr is not None:
            bundle["prefill"] = prefill_arr[idx]
        if decode_arr is not None:
            bundle["decode"] = decode_arr[idx]

        bundles.append(bundle)

    return bundles


# -----------------------------------------------------------------------------
#  serialisation helpers
# -----------------------------------------------------------------------------

_DEF_FMT = "rate{rate:.4g}_tp{tp}_{ms}.pkl"


def _bundle_fname(bundle: TraceBundle) -> str:
    return _DEF_FMT.format(rate=bundle["rate"], tp=bundle["tp"], ms=bundle["ms"])


def save_bundles(bundles: List[TraceBundle], out_dir: str | os.PathLike) -> None:
    """Write every bundle as `*.pkl` so training can stream them lazily."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in bundles:
        fname = out_dir / _bundle_fname(b)
        with fname.open("wb") as fh:
            pickle.dump(dict(b), fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[dataio] saved {fname.relative_to(out_dir)} …  (power shape {b['power'].shape})"
        )


def load_bundle(
    root: str | os.PathLike, rate: float, tp: int, ms: str | int
) -> TraceBundle:
    """Load a *single* bundle by its parameter triple."""
    fname = Path(root) / _DEF_FMT.format(rate=rate, tp=tp, ms=ms)
    if not fname.exists():
        raise FileNotFoundError(fname)
    with fname.open("rb") as fh:
        bundle = pickle.load(fh)
    return bundle


def load_all_bundles(
    root: str | os.PathLike,
) -> Dict[Tuple[float, int, str], TraceBundle]:
    """Load **all** `*.pkl` bundles inside *root* into memory."""
    root = Path(root)
    bundles: Dict[Tuple[float, int, str], TraceBundle] = {}
    for pkl in root.glob("*.pkl"):
        with pkl.open("rb") as fh:
            b: TraceBundle = pickle.load(fh)
        bundles[b.key] = b
    return bundles


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert consolidated NPZ into per‑setting bundles"
    )
    ap.add_argument("npz", help="path to power_trace_data.npz")
    ap.add_argument(
        "out", help="directory to write bundles", nargs="?", default="bundles"
    )
    args = ap.parse_args()

    bnds = npz_to_bundles(args.npz)
    save_bundles(bnds, args.out)

    print(f"[dataio] {len(bnds)} bundles written to {args.out}")
