"""MAP estimation with physical priors + robust likelihood, with uncertainty.

Coefficients ``c = exp(theta)`` (positive by construction) under log-normal
datasheet priors. The likelihood is Student-t (nu=4) with a state-dependent scale
``sigma_i = a + b * P_pred`` so saturated high-power bins and transition spikes do
not dominate the fit.

The optimizer is a hand-rolled damped Gauss-Newton / IRLS in ``theta`` (analytic
Jacobian ``dpred/dtheta = X * c``, t-likelihood IRLS weights). This needs only
numpy — it runs in the minimal numpy-only venv on the cluster, sidestepping the
scipy build problem on the python module. Optional scipy is not required.

Outputs, beyond point estimates:
  * Laplace posterior sd per coefficient (Gauss-Newton Hessian at the mode) and a
    DATA-IDENTIFIED / partial / PRIOR-DOMINATED label per coefficient.
  * Predictive intervals: var(P) = J cov J^T (coefficient uncertainty)
    + sigma^2 (intrinsic noise floor).
  * Per-factor variance attribution: each coefficient's share of predictive var.
"""

from __future__ import annotations

import numpy as np

from powermodel.priors import FEATS, PRIORS

NU = 4.0
SIGMA_FLOOR = 15.0


def _resolve_priors(priors=None):
    """Return (feats, priors_dict). Default to the merged-model schema in
    priors.py; pass an OrderedDict/dict {feat: (mean, sd_log)} to fit an
    alternative parameterization (e.g. the power-per-regime model) with the SAME
    machinery. Preserves current behavior when ``priors`` is None."""
    if priors is None:
        return FEATS, PRIORS
    return tuple(priors.keys()), priors


def _prior_vectors(priors=None):
    feats, pr = _resolve_priors(priors)
    mu = np.array([np.log(pr[f][0]) for f in feats])
    sd = np.array([pr[f][1] for f in feats])
    return mu, sd


def predict(X, theta):
    return X @ np.exp(theta)


def het_sigma(y, pred):
    """sigma_i = a + b*pred from robust regression of |resid| on pred."""
    r = np.abs(y - pred)
    A = np.column_stack([np.ones_like(pred), pred])
    w = 1.0 / np.maximum(pred, 100.0)
    coef, *_ = np.linalg.lstsq(A * w[:, None], r * w * 1.2533, rcond=None)
    return np.maximum(SIGMA_FLOOR, A @ coef)


def _objective(theta, X, y, sigma, mu, sd):
    c = np.exp(theta)
    r = y - X @ c
    nll = float(np.sum(0.5 * (NU + 1) * np.log1p(r**2 / (NU * sigma**2))))
    nll += float(np.sum((theta - mu) ** 2 / (2 * sd**2)))
    return nll


def map_fit(X, y, sigma, theta0=None, iters=60, tol=1e-8, priors=None):
    """Damped Gauss-Newton MAP fit in theta-space. Returns theta."""
    mu, sd = _prior_vectors(priors)
    theta = mu.copy() if theta0 is None else theta0.copy()
    prior_prec = np.diag(1.0 / sd**2)
    f_prev = _objective(theta, X, y, sigma, mu, sd)
    for _ in range(iters):
        c = np.exp(theta)
        pred = X @ c
        r = y - pred
        J = X * c[None, :]                       # dpred/dtheta
        w = (NU + 1) / (NU * sigma**2 + r**2)    # IRLS t-weights
        H = (J * w[:, None]).T @ J + prior_prec
        g = J.T @ (w * r) - (theta - mu) / sd**2  # gradient of -nll wrt theta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # backtracking line search
        t = 1.0
        for _ in range(30):
            cand = theta + t * step
            f = _objective(cand, X, y, sigma, mu, sd)
            if f < f_prev:
                break
            t *= 0.5
        if t * float(np.max(np.abs(step))) < tol:
            theta = theta + t * step
            break
        if f < f_prev:
            theta, f_prev = theta + t * step, f
        else:
            break
    return theta


def fit_two_stage(X, y, theta0=None, priors=None):
    """Fit, re-estimate heteroscedastic sigma, refit (2 passes)."""
    sigma = np.full(y.size, 100.0)
    theta = map_fit(X, y, sigma, theta0=theta0, priors=priors)
    for _ in range(2):
        sigma = het_sigma(y, predict(X, theta))
        theta = map_fit(X, y, sigma, theta0=theta, priors=priors)
    return theta, sigma


def laplace_cov(X, y, theta, sigma, priors=None):
    """Posterior covariance of theta via Gauss-Newton Laplace approximation."""
    mu, sd = _prior_vectors(priors)
    c = np.exp(theta)
    r = y - X @ c
    J = X * c[None, :]
    w = (NU + 1) / (NU * sigma**2 + r**2)
    H = (J * w[:, None]).T @ J + np.diag(1.0 / sd**2)
    return np.linalg.inv(H)


def identifiability(theta, cov, priors=None):
    """Per-coefficient posterior sd, shrink ratio, drift, and status label."""
    feats, pr = _resolve_priors(priors)
    mu, sd = _prior_vectors(priors)
    post_sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    rows = []
    for j, f in enumerate(feats):
        shrink = post_sd[j] / sd[j]
        drift = (theta[j] - mu[j]) / sd[j]
        status = ("DATA-IDENTIFIED" if shrink < 0.3
                  else "partial" if shrink < 0.7 else "PRIOR-DOMINATED")
        rows.append(dict(feature=f, value=float(np.exp(theta[j])),
                         prior_mean=pr[f][0], post_sd_log=float(post_sd[j]),
                         prior_sd_log=float(sd[j]), shrink=float(shrink),
                         drift_sigma=float(drift), status=status))
    return rows


def predictive(X, theta, cov, sigma):
    """Mean prediction, total predictive sd, and per-factor variance shares.

    var(P_i) = sum_jk J_ij cov_jk J_ik   (coefficient uncertainty)
             + sigma_i^2                  (intrinsic per-bin noise)
    Per-factor share approximates each coefficient's diagonal contribution.
    """
    c = np.exp(theta)
    pred = X @ c
    J = X * c[None, :]
    coef_var = np.einsum("ij,jk,ik->i", J, cov, J)
    total_sd = np.sqrt(np.maximum(coef_var + sigma**2, 0.0))
    # per-factor (diagonal) variance contribution, averaged over bins
    diag = np.diag(cov)
    factor_var = (J**2) * diag[None, :]            # (n_bins, n_feat)
    factor_share = factor_var.mean(axis=0)
    return dict(pred=pred, total_sd=total_sd, coef_sd=np.sqrt(coef_var),
                factor_var_mean=factor_share, sigma=sigma)
