# token_gen.py
# -----------------------------------------------------------------------------
#  Conditional token‑count generators (Prefill / Decode) using a single SVGP.
#  The model learns p(tokens | time, rate, TP, model‑size) across *all* bundles.
# -----------------------------------------------------------------------------
#  Inputs  X = [t_norm, rate, tp, model_size_B]
#  Output  y = tokens (real, we round ≥0 when sampling)
# -----------------------------------------------------------------------------
#  Exposed helpers:
#      • build_training_set(bundles) → X, y  (numpy)
#      • train_token_gp(X, y, …)     → model_dict
#      • save_model / load_model
#      • sample_tokens(model, duration, dt, rate,tp,ms, n_samples)
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import gpytorch

__all__ = [
    "build_training_set",
    "train_token_gp",
    "save_model",
    "load_model",
    "sample_tokens",
]


class _TokenSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing: torch.Tensor, input_dim: int = 4):
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
        q_strat = gpytorch.variational.VariationalStrategy(
            self, inducing, q_dist, learn_inducing_locations=True
        )
        super().__init__(q_strat)

        self.mean_module = gpytorch.means.ConstantMean()
        # moderate – assume tokens evolve smoothly in inputs
        base = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)
        )
        self.covar_module = base + gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.WhiteNoiseKernel()
        )

        # Student‑T again: heavy‑tails for burstiness
        self.likelihood = gpytorch.likelihoods.StudentTLikelihood(df=4.0)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# -----------------------------------------------------------------------------

_try_kmeans = True
try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    _try_kmeans = False


def _inducing_pts(X: torch.Tensor, m: int) -> torch.Tensor:
    if _try_kmeans and len(X) >= m:
        km = MiniBatchKMeans(n_clusters=m, n_init=1, random_state=0).fit(X.cpu())
        return torch.tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)
    idx = torch.randperm(len(X))[:m]
    return X[idx]


def _ms_to_float(ms) -> float:
    """Convert model‑size token ('7B', '70B', etc.) to float(B)."""
    if isinstance(ms, (int, float)):
        return float(ms)
    s = str(ms).lower().replace("b", "")
    return float(s)


def build_training_set(
    bundles: List[Dict[str, Any]], token_field: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine bundles → global (X, y) arrays for a token type.

    token_field ∈ {"prefill", "decode"}
    """
    xs, ys = [], []
    for b in bundles:
        if token_field not in b:
            continue  # skip if unavailable
        tokens = b[token_field].astype(np.float32)  # (S,T)
        rate, tp, ms = float(b["rate"]), float(b["tp"]), _ms_to_float(b["ms"])

        s, t = tokens.shape
        # features ---------------------------------------------------------
        time_norm = np.linspace(0.0, 1.0, t, dtype=np.float32)
        # broadcast across S samples
        feat = np.stack(
            (
                np.broadcast_to(time_norm, (s, t)),  # (S,T)
                np.full((s, t), rate, dtype=np.float32),
                np.full((s, t), tp, dtype=np.float32),
                np.full((s, t), ms, dtype=np.float32),
            ),
            axis=-1,
        )  # (S,T,4)
        xs.append(feat.reshape(-1, 4))
        ys.append(tokens.reshape(-1))

    if not xs:
        raise ValueError(f"No bundles contained token field '{token_field}'")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def train_token_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    inducing: int = 1000,
    iters: int = 300,
    batch: int = 4096,
    lr: float = 2e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train TokenSVGP on (X,y) arrays."""
    device = torch.device(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    # inducing
    m = min(inducing, max(50, len(X_t) // 20))
    Z = _inducing_pts(X_t, m)

    model = _TokenSVGP(Z).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=len(y_t))

    model.train()
    model.likelihood.train()

    for ep in range(iters):
        perm = torch.randperm(len(X_t))[:batch]
        xb, yb = X_t[perm], y_t[perm]
        opt.zero_grad()
        loss = -mll(model(xb), yb)
        loss.backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == iters - 1):
            print(f"[token_gen] iter {ep:4d}/{iters}  loss={loss.item():.3f}")

    model.eval()
    model.likelihood.eval()
    model = model.cpu()
    return {"state_dict": model.state_dict()}


def save_model(mdl: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mdl, path)
    print(f"[token_gen] saved GP → {path}")


def load_model(path: str | Path, device: str | torch.device = "cpu") -> _TokenSVGP:
    saved = torch.load(path, map_location=device)
    Z = torch.zeros((10, 4), dtype=torch.float32, device=device)
    mdl = _TokenSVGP(Z).to(device)
    mdl.load_state_dict(saved["state_dict"])
    mdl.eval()
    mdl.likelihood.eval()
    return mdl


def _make_features(time_steps: int, rate: float, tp: int, ms: float) -> np.ndarray:
    time = np.linspace(0, 1, time_steps, dtype=np.float32)
    return np.stack(
        (
            time,
            np.full_like(time, rate),
            np.full_like(time, tp),
            np.full_like(time, ms),
        ),
        axis=-1,
    )


def sample_tokens(
    model: _TokenSVGP,
    duration_s: float,
    dt: float,
    rate: float,
    tp: int,
    ms: float | str,
    n_samples: int = 1,
    device: str | torch.device = "cpu",
) -> np.ndarray:  # (n_samples, T)
    """Sample token sequences from trained GP."""
    ms_f = _ms_to_float(ms)
    T = int(math.ceil(duration_s / dt))
    feats = _make_features(T, rate, tp, ms_f)
    X = torch.tensor(feats, dtype=torch.float32, device=device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = model.likelihood(model(X))
        if n_samples == 1:
            out = dist.sample().unsqueeze(0)
        else:
            out = dist.sample(torch.Size([n_samples]))
    seq = out.cpu().numpy()
    seq = np.maximum(seq, 0.0)  # no negative tokens
    return np.round(seq).astype(int)
