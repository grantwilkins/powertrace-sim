import math
from typing import Dict, Optional, Tuple

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


def build_exogenous_features(
    T: int,
    dt: float,
    t0: float,
    request_timestamps: np.ndarray,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    prefill_rate: float,
    decode_rate: float,
    drop_backlog: bool = False,
) -> dict:
    """
    Build A_t, T_arrive_t, and T_backlog_t on an index-based timeline.
    """
    if T <= 0:
        z = np.zeros((0,), dtype=np.float64)
        return {"active_requests": z, "t_arrive": z, "t_backlog": z}

    req_ts = np.asarray(request_timestamps, dtype=np.float64).reshape(-1)
    in_tok = np.asarray(input_tokens, dtype=np.float64).reshape(-1)
    out_tok = np.asarray(output_tokens, dtype=np.float64).reshape(-1)
    n = int(min(len(req_ts), len(in_tok), len(out_tok)))
    req_ts = req_ts[:n]
    in_tok = np.clip(in_tok[:n], a_min=0.0, a_max=None)
    out_tok = np.clip(out_tok[:n], a_min=0.0, a_max=None)

    valid = np.isfinite(req_ts) & np.isfinite(in_tok) & np.isfinite(out_tok)
    req_rel = req_ts[valid] - float(t0)
    in_tok = in_tok[valid]
    out_tok = out_tok[valid]

    t_arrive = np.zeros((T,), dtype=np.float64)
    active_diff = np.zeros((T + 1,), dtype=np.float64)
    backlog_const_diff = np.zeros((T + 1,), dtype=np.float64)
    backlog_slope_diff = np.zeros((T + 1,), dtype=np.float64)

    if req_rel.size > 0:
        bins = np.floor(req_rel / dt).astype(np.int64)
        good = (bins >= 0) & (bins < T)
        if np.any(good):
            np.add.at(t_arrive, bins[good], in_tok[good])

    prefill_rate = max(float(prefill_rate), EPS)
    decode_rate = max(float(decode_rate), EPS)

    for arrival, nin, nout in zip(req_rel, in_tok, out_tok):
        prefill_end = float(arrival + (nin / prefill_rate))
        completion = float(prefill_end + (nout / decode_rate))

        start_idx = int(np.ceil(arrival / dt))
        end_idx = int(np.ceil(completion / dt))
        if end_idx <= 0 or start_idx >= T:
            continue

        l = max(0, start_idx)
        r = min(T, end_idx)
        if r <= l:
            continue
        active_diff[l] += 1.0
        active_diff[r] -= 1.0

        if drop_backlog:
            continue

        prefill_end_idx = int(np.ceil(prefill_end / dt))
        prefill_r = min(r, max(l, prefill_end_idx))
        if prefill_r > l:
            backlog_const_diff[l] += nout
            backlog_const_diff[prefill_r] -= nout

        decode_l = max(l, prefill_end_idx)
        decode_r = r
        if decode_r > decode_l:
            c = nout + decode_rate * prefill_end
            m = -decode_rate * dt
            backlog_const_diff[decode_l] += c
            backlog_const_diff[decode_r] -= c
            backlog_slope_diff[decode_l] += m
            backlog_slope_diff[decode_r] -= m

    active = np.clip(np.cumsum(active_diff[:-1]), a_min=0.0, a_max=None)
    if drop_backlog:
        backlog = np.zeros((T,), dtype=np.float64)
    else:
        c_arr = np.cumsum(backlog_const_diff[:-1])
        m_arr = np.cumsum(backlog_slope_diff[:-1])
        idx = np.arange(T, dtype=np.float64)
        backlog = np.clip(c_arr + (m_arr * idx), a_min=0.0, a_max=None)

    return {
        "active_requests": active.astype(np.float64),
        "t_arrive": t_arrive.astype(np.float64),
        "t_backlog": backlog.astype(np.float64),
    }


class ContinuousAutoregressiveGRU(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 1, output_mode: str = "gaussian"):
        super().__init__()
        if output_mode not in {"gaussian", "mdn"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        self.output_mode = output_mode
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        if output_mode == "gaussian":
            self.output_head = nn.Linear(hidden_dim, 2)
            self.M = 1
        else:
            self.M = 3
            self.output_head = nn.Linear(hidden_dim, self.M * 3)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, h_n = self.gru(x, h_0)
        return self.output_head(h), h_n


class ResidualAutoregressiveGRU(ContinuousAutoregressiveGRU):
    """
    Same architecture as ContinuousAutoregressiveGRU, but semantically used to model
    residual targets ΔP_t.
    """


class NoisyTeacherForcing:
    """
    Noise schedule used to corrupt P_{t-1} during training.
    """

    def __init__(
        self,
        noise_std_start: float = 0.0,
        noise_std_end: float = 0.1,
        warmup_epochs: int = 100,
        ramp_epochs: int = 200,
    ):
        self.noise_std_start = float(noise_std_start)
        self.noise_std_end = float(noise_std_end)
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.ramp_epochs = int(max(1, ramp_epochs))

    def get_noise_std(self, epoch_idx: int) -> float:
        epoch_idx = int(max(0, epoch_idx))
        if epoch_idx < self.warmup_epochs:
            return self.noise_std_start
        progress = min(1.0, float(epoch_idx - self.warmup_epochs) / float(self.ramp_epochs))
        return self.noise_std_start + progress * (self.noise_std_end - self.noise_std_start)

    def apply(self, p_prev_norm: torch.Tensor, epoch_idx: int, training: bool) -> torch.Tensor:
        std = self.get_noise_std(epoch_idx)
        if (not training) or std <= 0.0:
            return p_prev_norm
        return p_prev_norm + (torch.randn_like(p_prev_norm) * std)


class MeanRevertingGRU(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 1, n_mix: int = 1):
        """
        Args:
            input_dim: features per timestep (default: P_{t-1}, A_t, T_arrive_t)
            hidden_dim: GRU hidden size
            num_layers: GRU depth
            n_mix: number of mixture components (1 = single Gaussian)
        """
        super().__init__()
        self.n_mix = int(n_mix)
        if self.n_mix < 1:
            raise ValueError(f"n_mix must be >= 1; got {n_mix}")

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        if self.n_mix == 1:
            # (mu, raw_alpha, log_sigma)
            self.output_head = nn.Linear(hidden_dim, 3)
        else:
            # 1 (alpha) + M (pi logits) + M (mu) + M (log_sigma)
            self.output_head = nn.Linear(hidden_dim, 1 + (3 * self.n_mix))

    def forward(
        self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        h, h_n = self.gru(x, h_0)
        raw = self.output_head(h)

        if self.n_mix == 1:
            mu = raw[:, :, 0:1]
            alpha = torch.sigmoid(raw[:, :, 1:2])
            log_sigma = torch.clamp(raw[:, :, 2:3], min=-6.0, max=2.0)
            return {"mu": mu, "alpha": alpha, "log_sigma": log_sigma}, h_n

        M = int(self.n_mix)
        alpha = torch.sigmoid(raw[:, :, 0:1])
        logit_pi = raw[:, :, 1 : 1 + M]
        mu = raw[:, :, 1 + M : 1 + (2 * M)]
        log_sigma = torch.clamp(raw[:, :, 1 + (2 * M) : 1 + (3 * M)], min=-6.0, max=2.0)
        return {"mu": mu, "alpha": alpha, "log_sigma": log_sigma, "logit_pi": logit_pi}, h_n


class PowerNoiseInjector:
    """Corrupt P_{t-1} input during training to simulate rollout-time error."""

    def __init__(self, warmup_epochs: int = 100, ramp_epochs: int = 200, max_noise_std: float = 0.1):
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.ramp_epochs = int(max(1, ramp_epochs))
        self.max_noise_std = float(max(0.0, max_noise_std))

    def get_std(self, epoch: int) -> float:
        epoch = int(max(0, epoch))
        if epoch < self.warmup_epochs:
            return 0.0
        progress = min(1.0, float(epoch - self.warmup_epochs) / float(self.ramp_epochs))
        return float(self.max_noise_std * progress)

    def __call__(self, p_prev: torch.Tensor, epoch: int) -> torch.Tensor:
        std = self.get_std(epoch)
        if std <= 0.0:
            return p_prev
        return p_prev + (torch.randn_like(p_prev) * std)


def mean_reverting_nll(
    params: Dict[str, torch.Tensor],
    p_prev: torch.Tensor,
    p_target: torch.Tensor,
    n_mix: int = 1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NLL for P_t = (1 - alpha_t) * P_{t-1} + alpha_t * mu_t + sigma_t * epsilon_t.
    """
    if p_prev.ndim == 2:
        p_prev = p_prev.unsqueeze(-1)
    if p_target.ndim == 2:
        p_target = p_target.unsqueeze(-1)

    alpha = params["alpha"]
    log_sigma = params["log_sigma"]
    sigma = torch.exp(log_sigma)

    if int(n_mix) == 1:
        mu = params["mu"]
        pred_mean = ((1.0 - alpha) * p_prev) + (alpha * mu)
        nll = 0.5 * math.log(2.0 * math.pi) + log_sigma + 0.5 * ((p_target - pred_mean) / sigma) ** 2
        return _masked_mean(nll, mask)

    mu = params["mu"]
    logit_pi = params["logit_pi"]
    pred_mean = ((1.0 - alpha) * p_prev) + (alpha * mu)
    log_pi = torch.log_softmax(logit_pi, dim=-1)
    log_normal = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * ((p_target - pred_mean) / sigma) ** 2
    log_prob = torch.logsumexp(log_pi + log_normal, dim=-1, keepdim=True)
    return _masked_mean(-log_prob, mask)


def gaussian_nll_loss(params: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if targets.ndim == 2:
        targets = targets.unsqueeze(-1)
    mu = params[:, :, 0:1]
    log_sigma = torch.clamp(params[:, :, 1:2], min=-6.0, max=2.0)
    sigma = torch.exp(log_sigma)
    nll = 0.5 * math.log(2.0 * math.pi) + log_sigma + 0.5 * ((targets - mu) / sigma) ** 2
    return _masked_mean(nll, mask)


def mdn_nll_loss(params: torch.Tensor, targets: torch.Tensor, M: int = 3, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if targets.ndim == 2:
        targets = targets.unsqueeze(-1)
    logit_pi = params[:, :, :M]
    mu = params[:, :, M : 2 * M]
    log_sigma = torch.clamp(params[:, :, 2 * M : 3 * M], min=-6.0, max=2.0)
    sigma = torch.exp(log_sigma)

    log_pi = torch.log_softmax(logit_pi, dim=-1)
    y = targets.squeeze(-1).unsqueeze(-1)
    log_normal = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * ((y - mu) / sigma) ** 2
    log_prob = torch.logsumexp(log_pi + log_normal, dim=-1, keepdim=True)
    nll = -log_prob
    return _masked_mean(nll, mask)


@torch.no_grad()
def generate_trace(
    model: nn.Module,
    features: torch.Tensor,
    P_0: float,
    norm_params: dict,
    config_power_range: Optional[Tuple[float, float]] = None,
    output_mode: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a power trace autoregressively.

    Residual rollout:
      ΔP_t ~ model(...)
      P_t = P_{t-1} + ΔP_t

    Args:
        model: trained residual GRU
        features: (seq_len, 3) normalized exogenous features aligned to Δ steps (usually T-1)
        P_0: initial power in raw watts
        norm_params: must include power_mean/power_std and delta_mean/delta_std
        config_power_range: optional (p_min, p_max) to clamp generated trace
        output_mode: gaussian or mdn
        rng: optional numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()
    model.eval()
    device = next(model.parameters()).device

    power_mean = float(norm_params["power_mean"])
    power_std = float(norm_params["power_std"])
    delta_mean = float(norm_params.get("delta_mean", 0.0))
    delta_std = float(norm_params.get("delta_std", 1.0))
    p_prev = float(P_0)

    seq_len_delta = int(features.shape[0])
    h = None
    out = np.zeros((seq_len_delta + 1,), dtype=np.float64)
    out[0] = p_prev

    p_clip_min = None
    p_clip_max = None
    if config_power_range is not None:
        p_min, p_max = config_power_range
        margin = 0.05 * (float(p_max) - float(p_min))
        p_clip_min = float(p_min) - margin
        p_clip_max = float(p_max) + margin

    features = features.to(device=device, dtype=torch.float32)
    for t in range(seq_len_delta):
        p_prev_norm = (p_prev - power_mean) / (power_std + 1e-12)
        x_t = torch.cat(
            [
                torch.tensor([[[p_prev_norm]]], device=device, dtype=torch.float32),
                features[t : t + 1].unsqueeze(0),
            ],
            dim=-1,
        )
        params, h = model(x_t, h)

        if output_mode == "gaussian":
            mu_delta_norm = float(params[0, 0, 0].item())
            log_sigma = float(np.clip(params[0, 0, 1].item(), -6.0, 2.0))
            delta_norm = float(rng.normal(mu_delta_norm, math.exp(log_sigma)))
        elif output_mode == "mdn":
            M = int(getattr(model, "M", 3))
            p = params[0, 0]
            logits = p[:M].detach().cpu().numpy()
            mus = p[M : 2 * M].detach().cpu().numpy()
            log_sigmas = np.clip(p[2 * M : 3 * M].detach().cpu().numpy(), -6.0, 2.0)
            pi = np.exp(logits - np.max(logits))
            pi = pi / np.sum(pi)
            k = int(rng.choice(np.arange(M), p=pi))
            delta_norm = float(rng.normal(mus[k], np.exp(log_sigmas[k])))
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        delta_raw = (delta_norm * delta_std) + delta_mean
        p_t = p_prev + delta_raw
        if p_clip_min is not None:
            p_t = float(np.clip(p_t, p_clip_min, p_clip_max))
        out[t + 1] = p_t
        p_prev = p_t

    return out


def generate_trace_v2(
    model: nn.Module,
    features: torch.Tensor,
    P_0: float,
    norm_params: dict,
    config_power_range: Optional[Tuple[float, float]] = None,
    output_mode: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    return generate_trace(
        model=model,
        features=features,
        P_0=P_0,
        norm_params=norm_params,
        config_power_range=config_power_range,
        output_mode=output_mode,
        rng=rng,
    )
