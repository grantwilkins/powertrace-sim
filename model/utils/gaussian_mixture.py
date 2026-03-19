from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


class NumpyGaussianMixture1D:
    """
    Minimal 1D Gaussian mixture implementation used when sklearn is unavailable.
    Supports the subset of the sklearn API used in this repo.
    """

    def __init__(
        self,
        *,
        n_components: int,
        covariance_type: str = "full",
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        reg_covar: float = 1e-6,
    ) -> None:
        if str(covariance_type).strip().lower() != "full":
            raise ValueError("Only covariance_type='full' is supported.")
        self.n_components = int(max(1, n_components))
        self.covariance_type = "full"
        self.random_state = int(random_state)
        self.n_init = int(max(1, n_init))
        self.max_iter = int(max(10, max_iter))
        self.reg_covar = float(max(reg_covar, 1e-12))

        self.weights_ = np.full(
            (self.n_components,), 1.0 / float(self.n_components), dtype=np.float64
        )
        self.means_ = np.zeros((self.n_components, 1), dtype=np.float64)
        self.covariances_ = np.ones((self.n_components, 1, 1), dtype=np.float64)

    @staticmethod
    def _as_1d_finite(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("No finite points.")
        return arr

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
        a_max = np.max(a, axis=axis, keepdims=True)
        out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)

    @staticmethod
    def _log_normal_pdf(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        x_col = x.reshape(-1, 1)
        mean_row = np.asarray(means, dtype=np.float64).reshape(1, -1)
        var_row = np.clip(np.asarray(variances, dtype=np.float64).reshape(1, -1), 1e-12, None)
        return -0.5 * (
            np.log(2.0 * np.pi * var_row) + ((x_col - mean_row) ** 2) / var_row
        )

    def _log_likelihood(self, x: np.ndarray) -> float:
        log_prob = self._log_normal_pdf(
            x,
            self.means_.reshape(-1),
            self.covariances_.reshape(-1),
        )
        log_mix = log_prob + np.log(np.clip(self.weights_, 1e-12, None)).reshape(1, -1)
        return float(np.sum(self._logsumexp(log_mix, axis=1)))

    def _fit_once(self, x: np.ndarray, rng: np.random.Generator) -> Tuple[float, Any]:
        n = int(x.size)
        k = int(self.n_components)
        if n < k:
            raise ValueError(f"Need at least k samples to fit GMM; got n={n}, k={k}")

        quantiles = np.linspace(0.0, 1.0, k + 2, dtype=np.float64)[1:-1]
        means = np.quantile(x, quantiles).astype(np.float64)
        if k > 1:
            span = max(float(np.std(x)), 1e-6)
            means = means + rng.normal(0.0, 0.01 * span, size=k)
        variances = np.full((k,), max(float(np.var(x)), self.reg_covar), dtype=np.float64)
        weights = np.full((k,), 1.0 / float(k), dtype=np.float64)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_prob = self._log_normal_pdf(x, means, variances)
            log_resp = log_prob + np.log(np.clip(weights, 1e-12, None)).reshape(1, -1)
            log_den = self._logsumexp(log_resp, axis=1).reshape(-1, 1)
            resp = np.exp(log_resp - log_den)

            nk = np.sum(resp, axis=0) + 1e-12
            weights = nk / float(n)
            means = np.sum(resp * x.reshape(-1, 1), axis=0) / nk
            diffs = x.reshape(-1, 1) - means.reshape(1, -1)
            variances = np.sum(resp * (diffs * diffs), axis=0) / nk
            variances = np.clip(variances, self.reg_covar, None)

            ll = float(np.sum(log_den))
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < 1e-6:
                prev_ll = ll
                break
            prev_ll = ll

        return prev_ll, (weights, means, variances)

    def fit(self, x: np.ndarray) -> "NumpyGaussianMixture1D":
        arr = self._as_1d_finite(x)
        if arr.size < self.n_components:
            raise ValueError(
                f"Need at least k samples to fit GMM; got n={arr.size}, k={self.n_components}"
            )

        best_ll = -np.inf
        best_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.random_state + i)
            ll, state = self._fit_once(arr, rng)
            if ll > best_ll or best_state is None:
                best_ll = ll
                best_state = state
        assert best_state is not None
        weights, means, variances = best_state
        self.weights_ = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.means_ = np.asarray(means, dtype=np.float64).reshape(-1, 1)
        self.covariances_ = np.asarray(variances, dtype=np.float64).reshape(-1, 1, 1)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = self._as_1d_finite(x)
        log_prob = self._log_normal_pdf(
            arr,
            self.means_.reshape(-1),
            self.covariances_.reshape(-1),
        )
        log_mix = log_prob + np.log(np.clip(self.weights_, 1e-12, None)).reshape(1, -1)
        return np.argmax(log_mix, axis=1).astype(np.int64)

    def aic(self, x: np.ndarray) -> float:
        arr = self._as_1d_finite(x)
        ll = self._log_likelihood(arr)
        p = int(3 * self.n_components - 1)
        return float(2.0 * p - 2.0 * ll)

    def bic(self, x: np.ndarray) -> float:
        arr = self._as_1d_finite(x)
        ll = self._log_likelihood(arr)
        p = int(3 * self.n_components - 1)
        return float(-2.0 * ll + p * np.log(float(arr.size)))


def load_sklearn_gaussian_mixture_cls() -> Optional[Any]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None
    return GaussianMixture


def make_gaussian_mixture(
    *,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    reg_covar: float = 1e-6,
) -> Any:
    kwargs = dict(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        random_state=int(random_state),
        n_init=int(max(1, n_init)),
        max_iter=int(max(10, max_iter)),
        reg_covar=float(max(reg_covar, 1e-12)),
    )
    gaussian_mixture_cls = load_sklearn_gaussian_mixture_cls()
    if gaussian_mixture_cls is not None:
        return gaussian_mixture_cls(**kwargs)
    return NumpyGaussianMixture1D(**kwargs)
