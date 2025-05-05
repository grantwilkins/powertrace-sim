# simulator.py
# -----------------------------------------------------------------------------
#  End‑to‑end simulator:
#    • generates token sequences (prefill / decode) conditioned on (rate, TP, MS)
#    • samples GP hyper‑parameters via HyperGP
#    • instantiates a *runtime* PowerSVGP with those θ values and predicts a
#      power‑trace for the requested duration.
# -----------------------------------------------------------------------------
#  API
#  ===
#      sim = PowerSimulator(
#               bundle_dir="bundles",      # *.pkl files
#               model_dir="ckpt/power",    # *.pt power GP states
#               token_dir="ckpt/token",    # two .pt files (prefill.pt,decode.pt)
#               hyper_path="ckpt/hyper.pt",# meta‑GP
#               dt=0.25)
#
#      power, tokens = sim.run(duration_s=120,
#                              poisson_rate=0.5,
#                              tensor_parallelism=8,
#                              model_size="7B",
#                              n_samples=3)
# -----------------------------------------------------------------------------
#  Author: Grant Wilkins — 2025-04-27

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import gpytorch

from data_io import load_all_bundles
from power_gp import PowerSVGP, load_model as load_power_model
from token_generator import load_model as load_token_model, sample_tokens
from hyper_gp import HyperGP

__all__ = ["PowerSimulator"]

_DT_DEFAULT = 0.25  # seconds


# -----------------------------------------------------------------------------
#  helper — instantiate SVGP with new θ
# -----------------------------------------------------------------------------


def _ms_to_float(ms):
    if isinstance(ms, (int, float)):
        return float(ms)
    return float(str(ms).lower().replace("b", ""))


def _instantiate_power_svgp(base_state: Dict[str, Any], theta: np.ndarray) -> PowerSVGP:
    """Create a PowerSVGP and overwrite its kernel hyper-parameters with θ."""
    sm_comp = base_state.get("sm_components", 4)
    # induce points from base_state
    Z_shape = base_state["model_state"]["variational_strategy.inducing_points"].shape
    Z = torch.zeros(Z_shape)
    model = PowerSVGP(Z, sm_components=sm_comp)
    model.load_state_dict(base_state["model_state"])
    model.likelihood.load_state_dict(base_state["likelihood_state"])

    # unpack θ ---------------------------------------------------------------
    (log_mat_ls, log_mat_os, log_sm_ls, log_sm_os, log_white_os, log_lik_scale) = theta

    matern_sk: gpytorch.kernels.ScaleKernel = model.covar_module.kernels[0]
    sm_sk: gpytorch.kernels.ScaleKernel = model.covar_module.kernels[1]
    white_sk: gpytorch.kernels.ScaleKernel = model.covar_module.kernels[2]

    matern_sk.base_kernel.lengthscale = math.exp(log_mat_ls)
    matern_sk.outputscale = math.exp(log_mat_os)

    sm_sk.base_kernel.mixture_scales.data.fill_(math.exp(log_sm_ls))
    sm_sk.outputscale = math.exp(log_sm_os)

    white_sk.outputscale = math.exp(log_white_os)
    model.likelihood.scale = math.exp(log_lik_scale)

    model.eval()
    model.likelihood.eval()
    return model


# -----------------------------------------------------------------------------
#  main orchestrator
# -----------------------------------------------------------------------------


class PowerSimulator:
    def __init__(
        self,
        *,
        bundle_dir: str | Path,
        model_dir: str | Path,
        token_dir: str | Path,
        hyper_path: str | Path,
        dt: float = _DT_DEFAULT,
        device: str | torch.device = "cpu",
    ):
        self.dt = dt
        self.device = torch.device(device)

        # load bundles for quick meta‑data lookup (nearest neighbour)
        self.bundles = load_all_bundles(bundle_dir)
        self.bundle_keys = np.array(
            [
                (float(r), float(t), float(_ms_to_float(m)))
                for (r, t, m) in self.bundles.keys()
            ],
            dtype=float,
        )  # (N,3)

        # power GP states ----------------------------------------------------
        self.power_models: Dict[Tuple[float, int, str], Dict[str, Any]] = {}
        for pt in Path(model_dir).glob("*.pt"):
            state = torch.load(pt, map_location="cpu")
            key = (float(state["rate"]), int(state["tp"]), str(state["ms"]))
            self.power_models[key] = state

        # token GPs ----------------------------------------------------------
        self.token_gp_prefill = load_token_model(Path(token_dir) / "prefill.pt")
        self.token_gp_decode = load_token_model(Path(token_dir) / "decode.pt")

        # hyper GP -----------------------------------------------------------
        self.hyper_gp = HyperGP.load(hyper_path)

        print(f"[simulator] loaded {len(self.power_models)} power‑GP states")

    # ------------------------------------------------------------------
    #  nearest power model helper
    # ------------------------------------------------------------------
    def _nearest_power_state(self, rate: float, tp: int, ms) -> Dict[str, Any]:
        q = np.array([rate, tp, _ms_to_float(ms)], dtype=float)
        dists = np.linalg.norm(self.bundle_keys - q, axis=1)
        idx = int(np.argmin(dists))
        key = tuple(self.bundles.keys())[idx]
        return self.power_models[key]

    # ------------------------------------------------------------------
    #  run simulation
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        duration_s: float,
        poisson_rate: float,
        tensor_parallelism: int,
        model_size: str | float,
        n_samples: int = 1,
        return_tokens: bool = True,
        sample_power: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Simulate power (and optionally token) sequences.

        Returns
        -------
        power : ndarray shape (n_samples, T)
        tokens: dict with "prefill", "decode"  (each same shape)  or None
        """
        steps = int(math.ceil(duration_s / self.dt))

        tokens = None
        if return_tokens:
            pre = sample_tokens(
                self.token_gp_prefill,
                duration_s,
                self.dt,
                poisson_rate,
                tensor_parallelism,
                model_size,
                n_samples=n_samples,
            )
            dec = sample_tokens(
                self.token_gp_decode,
                duration_s,
                self.dt,
                poisson_rate,
                tensor_parallelism,
                model_size,
                n_samples=n_samples,
            )
            tokens = {"prefill": pre, "decode": dec}

        if not sample_power:
            return np.zeros((n_samples, steps)), tokens

        theta_samples = self.hyper_gp.sample_theta(
            poisson_rate, tensor_parallelism, model_size, n=n_samples
        )
        base_state = self._nearest_power_state(
            poisson_rate, tensor_parallelism, model_size
        )

        # prepare time tensor
        time_norm = torch.linspace(0, 1, steps, dtype=torch.float32)
        time_t = time_norm.unsqueeze(-1)  # (T,1)

        power_out = np.zeros((n_samples, steps), dtype=np.float32)
        for i in range(n_samples):
            model = _instantiate_power_svgp(base_state, theta_samples[i])
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                dist = model.likelihood(model(time_t))
                samp = dist.sample().cpu().numpy()
            power_out[i] = samp * tensor_parallelism

        return power_out, tokens
