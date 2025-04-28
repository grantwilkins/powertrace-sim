# power_gp.py
# -----------------------------------------------------------------------------
#  Sparse‑variational GP for spiky power traces *per* (rate, TP, model‑size).
# -----------------------------------------------------------------------------
#  A PowerSVGP is trained on the 5 samples belonging to one parameter triple.
#  Kernel: (Matern ν=½  ×  Spectral‑Mixture)  ⊕  White
#           → handles sharp jumps while capturing quasi‑periodicity.
#  Likelihood: Student‑T (heavy‑tailed) to avoid over‑penalising outliers.
# -----------------------------------------------------------------------------
#  Exposed helpers:
#      • train_power_gp(bundle, …)   →  trained model, metadata dict
#      • save_model(model_dict, path)
#      • load_model(path, device)
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch
import gpytorch

__all__ = [
    "PowerSVGP",
    "train_power_gp",
    "save_model",
    "load_model",
]


class PowerSVGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP tailored for noisy power‑draw sequences."""

    def __init__(self, inducing_points: torch.Tensor, sm_components: int = 4):
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        q_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            q_dist,
            learn_inducing_locations=True,
        )
        super().__init__(q_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        time_matern = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(0,))
        )
        time_sm = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=sm_components, active_dims=(0,)
            )
        )
        white = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WhiteNoiseKernel())

        self.covar_module = time_matern * time_sm + white

        # heavy‑tailed likelihood
        self.likelihood = gpytorch.likelihoods.StudentTLikelihood(
            df=3.0,  # fixed degrees‑of‑freedom
            scale_prior=gpytorch.priors.GammaPrior(2.0, 0.2),
        )

    def forward(self, x: torch.Tensor):  # x.shape == (N,1)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


_KMEANS_FALLBACK = False
try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:  # keep import‑time cost low
    _KMEANS_FALLBACK = True


def _select_inducing(X: torch.Tensor, m: int) -> torch.Tensor:
    """k‑means centroids or random subset."""
    if _KMEANS_FALLBACK:
        idx = torch.randperm(len(X))[:m]
        return X[idx]
    else:
        km = MiniBatchKMeans(n_clusters=m, n_init=1, random_state=0).fit(X.cpu())
        return torch.tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)


def train_power_gp(
    bundle: Dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    num_inducing: int = 500,
    iters: int = 400,
    lr: float = 3e-3,
    sm_components: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a PowerSVGP on a *single* TraceBundle.

    Returns a dict that can be serialised directly via :func:`save_model`.
    """
    device = torch.device(device)

    power = bundle["power"].astype(np.float32)  # (S,T)
    tp = float(bundle["tp"])

    power = power / tp

    S, T = power.shape
    time = np.linspace(0.0, 1.0, T, dtype=np.float32)
    X = np.broadcast_to(time, (S, T)).reshape(-1, 1)  # (S*T,1)
    y = power.reshape(-1)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    M = min(num_inducing, max(50, len(X_t) // 10))
    Z = _select_inducing(X_t, M)

    # model
    model = PowerSVGP(Z, sm_components=sm_components).to(device)

    # optimiser / loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=len(y_t))

    model.train()
    model.likelihood.train()

    # mini‑batch size heuristic
    batch = min(2048, len(X_t))
    perm = torch.randperm(len(X_t))

    for epoch in range(iters):
        # simple SGD over a shuffled pass -------------------------------
        if epoch % math.ceil(len(X_t) / batch) == 0:  # reshuffle each epoch
            perm = torch.randperm(len(X_t))
        start = (epoch * batch) % len(X_t)
        idx = perm[start : start + batch]
        xb, yb = X_t[idx], y_t[idx]

        opt.zero_grad()
        out = model(xb)
        loss = -mll(out, yb)
        loss.backward()
        opt.step()

        if verbose and (epoch % 50 == 0 or epoch == iters - 1):
            print(f"[power_gp] epoch {epoch:4d}/{iters} → loss={loss.item():.3f}")

    # switch to eval mode & detach to CPU
    model.eval()
    model.likelihood.eval()
    model = model.cpu()

    return {
        "model_state": model.state_dict(),
        "likelihood_state": model.likelihood.state_dict(),
        "tp": tp,
        "rate": float(bundle["rate"]),
        "ms": str(bundle["ms"]),
        "normalisation": {"tp_div": tp},
        "sm_components": sm_components,
    }


def save_model(model_dict: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_dict, path)
    print(f"[power_gp] saved model → {path}")


def load_model(path: str | Path, device: torch.device | str = "cpu") -> PowerSVGP:
    mdl = torch.load(path, map_location=device)
    sm_components = mdl.get("sm_components", 4)
    Z = torch.zeros((10, 1), dtype=torch.float32, device=device)
    model = PowerSVGP(Z, sm_components=sm_components).to(device)
    model.load_state_dict(mdl["model_state"])
    model.likelihood.load_state_dict(mdl["likelihood_state"])

    model.eval()
    model.likelihood.eval()
    return model
