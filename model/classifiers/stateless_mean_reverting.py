import math
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-12


def _expand_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=ref.dtype, device=ref.device)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_f = _expand_mask(mask, values)
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return (values * mask_f).sum() / denom


def compute_delta_active_requests(active_requests: np.ndarray) -> np.ndarray:
    arr = np.asarray(active_requests, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = np.zeros_like(arr, dtype=np.float64)
    if arr.size > 1:
        delta[1:] = arr[1:] - arr[:-1]
    return delta


def normalize_delta_active_requests(delta_active: np.ndarray, mean: float, std: float) -> np.ndarray:
    arr = np.asarray(delta_active, dtype=np.float64).reshape(-1)
    denom = max(float(std), 1e-6)
    return ((arr - float(mean)) / denom).astype(np.float32)


class StatelessMeanReverting(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        hd = int(hidden_dim)
        if hd <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")

        self.mu_net = nn.Sequential(
            nn.Linear(1, hd),
            nn.ReLU(),
            nn.Linear(hd, 1),
        )
        self.alpha_net = nn.Sequential(
            nn.Linear(1, hd),
            nn.ReLU(),
            nn.Linear(hd, 1),
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(1, hd),
            nn.ReLU(),
            nn.Linear(hd, 1),
        )

    def forward(self, A_t: torch.Tensor, delta_A_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if A_t.ndim == 2:
            A_t = A_t.unsqueeze(0)
        if delta_A_t.ndim == 2:
            delta_A_t = delta_A_t.unsqueeze(0)
        if A_t.ndim != 3 or A_t.shape[-1] != 1:
            raise ValueError(f"A_t must have shape (B, T, 1); got {tuple(A_t.shape)}")
        if delta_A_t.ndim != 3 or delta_A_t.shape[-1] != 1:
            raise ValueError(f"delta_A_t must have shape (B, T, 1); got {tuple(delta_A_t.shape)}")

        mu = self.mu_net(A_t)
        alpha = torch.sigmoid(self.alpha_net(delta_A_t))
        log_sigma = torch.clamp(self.sigma_net(A_t), min=-6.0, max=2.0)
        return mu, alpha, log_sigma


def stateless_mean_reverting_nll(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    log_sigma: torch.Tensor,
    p_prev: torch.Tensor,
    p_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_mu: float = 0.1,
    return_parts: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    lambda_mu = float(lambda_mu)
    if lambda_mu < 0.0:
        raise ValueError(f"lambda_mu must be >= 0; got {lambda_mu}")

    if p_prev.ndim == 2:
        p_prev = p_prev.unsqueeze(-1)
    if p_target.ndim == 2:
        p_target = p_target.unsqueeze(-1)

    sigma = torch.exp(log_sigma)
    pred_mean = ((1.0 - alpha) * p_prev) + (alpha * mu)
    nll = 0.5 * math.log(2.0 * math.pi) + log_sigma + 0.5 * ((p_target - pred_mean) / sigma) ** 2
    nll_loss = _masked_mean(nll, mask)

    mu_err2 = (mu - p_target) ** 2
    mu_loss = _masked_mean(mu_err2, mask)

    total_loss = nll_loss + (lambda_mu * mu_loss)
    if return_parts:
        return {
            "total_loss": total_loss,
            "nll_loss": nll_loss,
            "mu_loss": mu_loss,
            "pred_mean": pred_mean,
        }
    return total_loss


@torch.no_grad()
def generate_stateless_trace(
    model: nn.Module,
    A_norm: np.ndarray,
    delta_A_norm: np.ndarray,
    P_0: float,
    config_norm: Dict[str, float],
    seed: Optional[int] = None,
) -> np.ndarray:
    A = np.asarray(A_norm, dtype=np.float32).reshape(-1)
    dA = np.asarray(delta_A_norm, dtype=np.float32).reshape(-1)
    if A.size != dA.size:
        raise ValueError(f"A_norm and delta_A_norm must have equal length; got {A.size} vs {dA.size}")

    power_mean = float(config_norm["power_mean"])
    power_std = float(config_norm["power_std"])
    power_min = float(config_norm["power_min"])
    power_max = float(config_norm["power_max"])
    if (not np.isfinite(power_std)) or power_std <= 0.0:
        raise ValueError(f"power_std must be positive; got {power_std}")

    use_clamp = np.isfinite(power_min) and np.isfinite(power_max) and (power_max > power_min)
    margin = 0.05 * (power_max - power_min) if use_clamp else 0.0

    rng = np.random.default_rng(seed)
    model.eval()
    device = next(model.parameters()).device

    T = int(A.shape[0])
    if T <= 0:
        return np.zeros((0,), dtype=np.float64)

    p_prev_raw = float(P_0)
    out = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        p_prev_norm = (p_prev_raw - power_mean) / power_std
        A_t = torch.tensor([[[float(A[t])]]], device=device, dtype=torch.float32)
        dA_t = torch.tensor([[[float(dA[t])]]], device=device, dtype=torch.float32)
        mu, alpha, log_sigma = model(A_t, dA_t)

        mu_norm = float(mu[0, 0, 0].item())
        alpha_val = float(alpha[0, 0, 0].item())
        sigma_norm = float(torch.exp(log_sigma[0, 0, 0]).item())
        pred_mean_norm = ((1.0 - alpha_val) * p_prev_norm) + (alpha_val * mu_norm)
        p_t_norm = float(rng.normal(pred_mean_norm, sigma_norm))

        p_t_raw = (p_t_norm * power_std) + power_mean
        if use_clamp:
            p_t_raw = float(np.clip(p_t_raw, power_min - margin, power_max + margin))

        out[t] = p_t_raw
        p_prev_raw = p_t_raw

    return out
