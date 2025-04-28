# hyper_gp.py
# -----------------------------------------------------------------------------
#  Meta-learner: Gaussian-process prior over the *hyper-parameters* θ of every
#  per-setting PowerSVGP.  This lets us **marginalise** kernel parameters when
#  predicting power at unseen (rate, TP, model-size) configurations.
# -----------------------------------------------------------------------------
#  Workflow
#  --------
#  1.  During power-GP training you call ::
#          hyp_rows = collect_hyper_rows(bundle_key, model_dict)
#      which extracts a 7-D vector θ (log-space) and stores (rate,tp,ms, θ).
#  2.  Feed the list of rows into `HyperGP.fit(rows)` → trains *D* independent
#      ExactGP regressors (one per θ-dimension).
#  3.  `sample_theta(rate,tp,ms, n)` draws Monte-Carlo samples θ* from the
#      posterior predictive of each 1-D GP.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import gpytorch

__all__ = [
    "collect_hyper_row",
    "HyperGP",
]

_HYP_NAMES = [
    "log_mat_ls",  # log Matern length-scale
    "log_mat_os",  # log Matern output-scale
    "log_sm_ls",  # log mean SM length-scale
    "log_sm_os",  # log SM output-scale (mean across mixtures)
    "log_white_os",  # log white-noise output-scale
    "log_lik_scale",  # log StudentT scale (noise)
]


def _ms_to_float(ms) -> float:
    if isinstance(ms, (int, float)):
        return float(ms)
    return float(str(ms).lower().replace("b", ""))


def _flatten_theta(model_state: Dict[str, torch.Tensor], sm_comp: int) -> np.ndarray:
    """Extract the 6-D hyper-parameter vector from the *state_dict* only."""
    from power_gp import PowerSVGP  # local import to avoid cycle

    Z = torch.zeros((10, 1))
    dummy = PowerSVGP(Z, sm_components=sm_comp)
    dummy.load_state_dict(model_state, strict=False)

    with torch.no_grad():
        matern_sk: gpytorch.kernels.ScaleKernel = dummy.covar_module.kernels[0]
        sm_sk: gpytorch.kernels.ScaleKernel = dummy.covar_module.kernels[1]
        white_sk: gpytorch.kernels.ScaleKernel = dummy.covar_module.kernels[2]

        vec = np.array(
            [
                math.log(float(matern_sk.base_kernel.lengthscale.squeeze())),
                math.log(float(matern_sk.outputscale)),
                math.log(float(sm_sk.base_kernel.mixture_scales.mean())),
                math.log(float(sm_sk.outputscale)),
                math.log(float(white_sk.outputscale)),
                math.log(float(dummy.likelihood.scale)),
            ],
            dtype=np.float32,
        )
    return vec


def collect_hyper_row(model_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, theta) where
    x     = [rate, tp, ms_float]
    theta = 6-D log-hyper vector.
    """
    x = np.array(
        [
            float(model_dict["rate"]),
            float(model_dict["tp"]),
            _ms_to_float(model_dict["ms"]),
        ],
        dtype=np.float32,
    )

    theta = _flatten_theta(
        model_dict["model_state"], model_dict.get("sm_components", 4)
    )
    return x, theta


class _ThetaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class HyperGP:
    """Meta-learner that models each log-hyper-parameter as a GP over (rate,tp,ms)."""

    def __init__(self):
        self.models: List[_ThetaGP] = []
        self.likelihoods: List[gpytorch.likelihoods.GaussianLikelihood] = []
        self.trained = False
        self.X_mean = None
        self.X_std = None

    def fit(
        self,
        rows: List[Tuple[np.ndarray, np.ndarray]],
        iters: int = 300,
        lr: float = 0.1,
    ):
        """`rows` is a list of (x, theta) tuples collected across all bundles."""
        X = np.stack([r[0] for r in rows])  # (N,3)
        Y = np.stack([r[1] for r in rows])  # (N,6)

        # normalise inputs for numerical stability
        self.X_mean = X.mean(0, keepdims=True)
        self.X_std = X.std(0, keepdims=True)
        self.X_std[self.X_std == 0] = 1.0
        Xn = (X - self.X_mean) / self.X_std

        X_t = torch.tensor(Xn, dtype=torch.float32)

        self.models = []
        self.likelihoods = []
        for d in range(Y.shape[1]):
            y_t = torch.tensor(Y[:, d], dtype=torch.float32)
            ll = gpytorch.likelihoods.GaussianLikelihood()
            m = _ThetaGP(X_t, y_t, ll)
            self.models.append(m)
            self.likelihoods.append(ll)

        # joint optimiser ------------------------------------------------
        opt = torch.optim.Adam(
            [p for m in self.models + self.likelihoods for p in m.parameters()], lr=lr
        )

        for ep in range(iters):
            opt.zero_grad()
            loss = 0.0
            for m, ll, d in zip(self.models, self.likelihoods, range(Y.shape[1])):
                output = m(X_t)
                ll_out = -gpytorch.mlls.ExactMarginalLogLikelihood(ll, m)(
                    output, torch.tensor(Y[:, d])
                )
                loss += ll_out
            loss.backward()
            opt.step()
            if ep % 50 == 0 or ep == iters - 1:
                print(f"[hyper_gp] iter {ep:3d}/{iters}  loss={loss.item():.2f}")

        for m in self.models:  # switch to eval mode
            m.eval()
        self.trained = True

    def _x_norm(self, rate, tp, ms):
        x = np.array([rate, tp, _ms_to_float(ms)], dtype=np.float32)
        return (x - self.X_mean) / self.X_std

    def sample_theta(
        self, rate: float, tp: int, ms: str | float, n: int = 5
    ) -> np.ndarray:
        """Return `(n,6)` matrix of θ samples (log-space)."""
        if not self.trained:
            raise RuntimeError("HyperGP.fit() must be called first")
        x = torch.tensor(self._x_norm(rate, tp, ms), dtype=torch.float32).unsqueeze(0)

        samples = []
        with torch.no_grad():
            for m, ll in zip(self.models, self.likelihoods):
                pred = ll(m(x))  # posterior
                s = pred.rsample(torch.Size([n]))  # (n,1)
                samples.append(s.squeeze(-1))
        theta = torch.stack(samples, dim=-1).cpu().numpy()  # (n,6)
        return theta

    def mean_theta(self, rate: float, tp: int, ms: str | float) -> np.ndarray:
        return self.sample_theta(rate, tp, ms, n=1).squeeze(0)

    def save(self, path: str | Path):
        pkg = {
            "models": [m.state_dict() for m in self.models],
            "likelihoods": [ll.state_dict() for ll in self.likelihoods],
            "X_mean": self.X_mean,
            "X_std": self.X_std,
        }
        torch.save(pkg, Path(path))
        print(f"[hyper_gp] saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "HyperGP":
        saved = torch.load(path, map_location=device)
        self = cls()
        self.X_mean = saved["X_mean"]
        self.X_std = saved["X_std"]

        self.models = []
        self.likelihoods = []
        for sd_m, sd_ll in zip(saved["models"], saved["likelihoods"]):
            # dummy inputs to build shapes
            dummy_x = torch.zeros((1, 3))
            dummy_y = torch.zeros((1,))
            ll = gpytorch.likelihoods.GaussianLikelihood()
            m = _ThetaGP(dummy_x, dummy_y, ll)
            m.load_state_dict(sd_m)
            ll.load_state_dict(sd_ll)
            m.eval()
            ll.eval()
            self.models.append(m)
            self.likelihoods.append(ll)
        self.trained = True
        return self
