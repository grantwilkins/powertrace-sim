# core/state_discovery.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# -----------------------------
# Low-level fitting utilities
# -----------------------------


def _fit_dp_gmm(y: np.ndarray, Kmax=12, seed=0, reg=1e-6, max_iter=1000, n_init=2):
    """Dirichlet-process (variational) GMM; y shape (N,1)."""
    bg = BayesianGaussianMixture(
        n_components=Kmax,
        weight_concentration_prior_type="dirichlet_process",
        covariance_type="diag",
        reg_covar=reg,
        random_state=seed,
        max_iter=max_iter,
        n_init=n_init,
    ).fit(y)
    return bg  # has .weights_, .means_, .covariances_


def _prune_components(model, y: np.ndarray, weight_eps=0.01, min_frac=0.005):
    """Keep components with sufficient mixture mass AND occupancy."""
    resp = model.predict_proba(y)
    occ = resp.sum(axis=0) / len(y)
    keep = (model.weights_ > weight_eps) & (occ > min_frac)
    return keep


def _bhattacharyya_1d(m1: float, s1: float, m2: float, s2: float) -> float:
    """Bhattacharyya distance for two 1D Gaussians (stds s1, s2)."""
    s = 0.5 * (s1**2 + s2**2)
    term1 = 0.25 * ((m1 - m2) ** 2) / (s + 1e-12)
    term2 = 0.5 * np.log((s + 1e-12) / (s1 * s2 + 1e-12))
    return float(term1 + term2)


def _merge_close_1d(
    means: np.ndarray, stds: np.ndarray, weights: np.ndarray, thresh=0.05
):
    """Greedy merge of near-duplicate components (1D)."""
    means = means.astype(float).copy()
    stds = stds.astype(float).copy()
    weights = weights.astype(float).copy()
    merged = True
    while merged and len(means) > 1:
        merged = False
        K = len(means)
        best = None
        for i in range(K):
            for j in range(i + 1, K):
                d = _bhattacharyya_1d(means[i], stds[i], means[j], stds[j])
                if d < thresh:
                    best = (i, j)
                    break
            if best:
                break
        if best:
            i, j = best
            w = np.array([weights[i], weights[j]])
            w /= w.sum()
            m = w[0] * means[i] + w[1] * means[j]
            v = w[0] * (stds[i] ** 2) + w[1] * (stds[j] ** 2)
            keep_idx = [k for k in range(K) if k not in (i, j)]
            means = np.concatenate([means[keep_idx], [m]])
            stds = np.concatenate([stds[keep_idx], [np.sqrt(v + 1e-12)]])
            weights = np.concatenate([weights[keep_idx], [weights[i] + weights[j]]])
            merged = True
    return means, stds, weights


# -----------------------------
# Public API (fit/load/cache)
# -----------------------------


def fit_or_load_states_1d_power(
    y: np.ndarray,
    key: str,
    cache_dir: str,
    mode: str = "auto",  # "auto" | "fixed"
    K_fixed: int = 6,
    Kmax: int = 12,
    seed: int = 0,
    weight_eps: float = 0.01,
    min_frac: float = 0.005,
    merge_thresh: float = 0.05,
    version: str = "v1",
) -> Dict[str, np.ndarray]:
    """
    Fit or load cached 1D Gaussian state model from power samples y (N,1).
    Returns dict(mu(K,), sigma(K,), pi(K,), K, method, version).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{key}_states_{version}.npz"
    if path.exists():
        obj = np.load(path, allow_pickle=True)
        return {k: obj[k] for k in obj.files}

    if mode == "fixed":
        gm = GaussianMixture(
            n_components=K_fixed,
            covariance_type="diag",
            random_state=seed,
            reg_covar=1e-6,
            n_init=10,
        ).fit(y)
        mu = gm.means_.ravel()
        sigma = np.sqrt(gm.covariances_.ravel())
        pi = gm.weights_.ravel()
        method = f"fixed(K={K_fixed})"
    else:
        bg = _fit_dp_gmm(y, Kmax=Kmax, seed=seed)
        keep = _prune_components(bg, y, weight_eps=weight_eps, min_frac=min_frac)
        mu = bg.means_.ravel()[keep]
        sigma = np.sqrt(bg.covariances_.ravel()[keep])
        pi = bg.weights_.ravel()[keep]
        mu, sigma, pi = _merge_close_1d(mu, sigma, pi, thresh=merge_thresh)
        order = np.argsort(mu)
        mu, sigma, pi = mu[order], sigma[order], pi[order]
        method = "dp-gmm+merge"

    out = dict(mu=mu, sigma=sigma, pi=pi, K=len(mu), method=method, version=version)
    np.savez(path, **out)
    return out


# -----------------------------
# Manifest and helpers
# -----------------------------


@dataclass
class ManifestRow:
    file: str
    model: str
    hardware: str
    tp: int
    mode: str
    K: int
    mu: list
    sigma: list
    method: str
    version: str
    n_samples: int
    subsampled: int
    data_hash: str
    cache_path: str


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_filename(fname: str) -> Tuple[str, str]:
    # Expect "random_{model}_{hardware}.npz"
    base = Path(fname).name
    assert base.startswith("random_") and base.endswith(".npz"), (
        f"Unexpected name: {base}"
    )
    stem = base[len("random_") : -len(".npz")]
    model, hardware = stem.rsplit("_", 1)
    return model, hardware


# -----------------------------
# CLI main
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Discover/caches power states per (model,hardware,TP)."
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory with random_{model}_{hardware}.npz files",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="random_*.npz",
        help="Glob pattern inside input_dir",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default="states_cache",
        help="Where to write *_states_v1.npz",
    )
    ap.add_argument("--mode", type=str, choices=["auto", "fixed"], default="auto")
    ap.add_argument("--K_fixed", type=int, default=6)
    ap.add_argument("--Kmax", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_eps", type=float, default=0.01)
    ap.add_argument("--min_frac", type=float, default=0.005)
    ap.add_argument("--merge_thresh", type=float, default=0.05)
    ap.add_argument(
        "--subsample",
        type=int,
        default=100_000,
        help="Max timesteps to fit per (file,TP). Use -1 for all.",
    )
    ap.add_argument(
        "--manifest_csv",
        type=str,
        default="state_manifest.csv",
        help="Summary CSV written next to cache_dir",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(
            f"[state_discovery] No files matching {args.pattern} in {input_dir.resolve()}"
        )
        return

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[ManifestRow] = []

    for fpath in files:
        model, hardware = _parse_filename(fpath.name)
        d = np.load(fpath, allow_pickle=True)
        tp_all = d["tensor_parallelism"]
        power_traces = d["power_traces"]
        timestamps = d["timestamps"]

        # all TPs present in this file
        tps = sorted(set(int(t) for t in tp_all))
        file_hash = _sha256_of_file(fpath)

        if args.verbose:
            print(f"\n[File] {fpath.name}  model={model}  hw={hardware}  TPs={tps}")

        for tp in tps:
            # concatenate power for this TP across experiments
            segs = []
            for i in range(len(tp_all)):
                if int(tp_all[i]) != tp:
                    continue
                mask = timestamps[i] > 0
                if not np.any(mask):
                    continue
                p = power_traces[i][mask].astype(np.float64).reshape(-1, 1)
                segs.append(p)
            if not segs:
                if args.verbose:
                    print(f"  [TP={tp}] no segments")
                continue

            y = np.concatenate(segs, axis=0)  # (N,1)
            n_samples = int(y.shape[0])
            if args.subsample > 0 and n_samples > args.subsample:
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(n_samples, size=args.subsample, replace=False)
                y_fit = y[idx]
                subsampled = int(len(idx))
            else:
                y_fit = y
                subsampled = n_samples

            key = f"{hardware}_{model}_tp{tp}"
            info = fit_or_load_states_1d_power(
                y_fit,
                key=key,
                cache_dir=str(cache_dir),
                mode=args.mode,
                K_fixed=args.K_fixed,
                Kmax=args.Kmax,
                seed=args.seed,
                weight_eps=args.weight_eps,
                min_frac=args.min_frac,
                merge_thresh=args.merge_thresh,
                version="v1",
            )

            # record manifest row
            cache_path = cache_dir / f"{key}_states_v1.npz"
            row = ManifestRow(
                file=fpath.name,
                model=model,
                hardware=hardware,
                tp=int(tp),
                mode=args.mode,
                K=int(info["K"]),
                mu=[float(x) for x in info["mu"]],
                sigma=[float(x) for x in info["sigma"]],
                method=str(info["method"]),
                version=str(info["version"]),
                n_samples=n_samples,
                subsampled=subsampled,
                data_hash=file_hash,
                cache_path=str(cache_path),
            )
            manifest_rows.append(row)

            if args.verbose:
                print(
                    f"  [TP={tp}] K={row.K:>2}  method={row.method:12s}  "
                    f"y_fit={row.subsampled:,}/{row.n_samples:,}  → {cache_path.name}"
                )

    # write manifest CSV & JSON
    import pandas as pd

    man_df = pd.DataFrame([asdict(r) for r in manifest_rows])
    out_csv = cache_dir / args.manifest_csv
    man_df.to_csv(out_csv, index=False)

    out_json = cache_dir / (Path(args.manifest_csv).with_suffix(".json").name)
    with out_json.open("w") as f:
        json.dump([asdict(r) for r in manifest_rows], f, indent=2)

    print(f"\n[state_discovery] Wrote manifest: {out_csv} (and {out_json})")
    print(
        f"[state_discovery] Cached {sum(r.K for r in manifest_rows)} total states across {len(manifest_rows)} TP entries."
    )


if __name__ == "__main__":
    main()
