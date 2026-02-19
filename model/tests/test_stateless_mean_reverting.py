import os
import unittest

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.classifiers.stateless_mean_reverting import (
    StatelessMeanReverting,
    compute_delta_active_requests,
    generate_stateless_trace,
    normalize_delta_active_requests,
    stateless_mean_reverting_nll,
)


class TestStatelessMeanReverting(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch

    def test_forward_shapes_and_ranges(self):
        model = StatelessMeanReverting(hidden_dim=16)
        A = self.torch.randn(3, 10, 1)
        dA = self.torch.randn(3, 10, 1)
        mu, alpha, log_sigma = model(A, dA)

        self.assertEqual(tuple(mu.shape), (3, 10, 1))
        self.assertEqual(tuple(alpha.shape), (3, 10, 1))
        self.assertEqual(tuple(log_sigma.shape), (3, 10, 1))
        self.assertTrue(self.torch.all(alpha > 0.0))
        self.assertTrue(self.torch.all(alpha < 1.0))
        self.assertTrue(self.torch.all(log_sigma >= -6.0))
        self.assertTrue(self.torch.all(log_sigma <= 2.0))

    def test_nll_lambda_zero_matches_manual(self):
        B, T = 2, 8
        model = StatelessMeanReverting(hidden_dim=8)
        A = self.torch.randn(B, T, 1)
        dA = self.torch.randn(B, T, 1)
        p_prev = self.torch.randn(B, T, 1)
        p_target = self.torch.randn(B, T, 1)

        mu, alpha, log_sigma = model(A, dA)
        loss = stateless_mean_reverting_nll(
            mu=mu,
            alpha=alpha,
            log_sigma=log_sigma,
            p_prev=p_prev,
            p_target=p_target,
            lambda_mu=0.0,
        )

        sigma = self.torch.exp(log_sigma)
        pred_mean = ((1.0 - alpha) * p_prev) + (alpha * mu)
        manual = (
            0.5 * np.log(2.0 * np.pi) + log_sigma + 0.5 * ((p_target - pred_mean) / sigma) ** 2
        ).mean()
        self.assertTrue(self.torch.allclose(loss, manual, atol=1e-6, rtol=1e-6))

    def test_aux_mu_loss_increases_grad(self):
        p_prev = self.torch.zeros(1, 5, 1)
        p_target = self.torch.full((1, 5, 1), 2.0)
        mu = self.torch.zeros(1, 5, 1, requires_grad=True)
        alpha = self.torch.full((1, 5, 1), 0.01)
        log_sigma = self.torch.zeros(1, 5, 1)

        loss_nll = stateless_mean_reverting_nll(
            mu=mu,
            alpha=alpha,
            log_sigma=log_sigma,
            p_prev=p_prev,
            p_target=p_target,
            lambda_mu=0.0,
        )
        loss_nll.backward()
        grad_nll = float(mu.grad.detach().abs().mean().item())

        mu.grad.zero_()
        loss_total = stateless_mean_reverting_nll(
            mu=mu,
            alpha=alpha,
            log_sigma=log_sigma,
            p_prev=p_prev,
            p_target=p_target,
            lambda_mu=0.1,
        )
        loss_total.backward()
        grad_total = float(mu.grad.detach().abs().mean().item())

        self.assertGreater(grad_nll, 0.0)
        self.assertGreater(grad_total, grad_nll * 5.0)

    def test_generate_seeded_and_clamped(self):
        model = StatelessMeanReverting(hidden_dim=8)
        A_norm = np.zeros((20,), dtype=np.float32)
        dA_norm = np.zeros((20,), dtype=np.float32)
        norm = {
            "power_mean": 200.0,
            "power_std": 20.0,
            "power_min": 180.0,
            "power_max": 260.0,
        }
        out_a = generate_stateless_trace(model, A_norm, dA_norm, P_0=210.0, config_norm=norm, seed=123)
        out_b = generate_stateless_trace(model, A_norm, dA_norm, P_0=210.0, config_norm=norm, seed=123)

        self.assertEqual(len(out_a), 20)
        self.assertTrue(np.all(np.isfinite(out_a)))
        self.assertTrue(np.allclose(out_a, out_b))
        lo = 180.0 - (0.05 * (260.0 - 180.0))
        hi = 260.0 + (0.05 * (260.0 - 180.0))
        self.assertTrue(np.all(out_a >= lo - 1e-6))
        self.assertTrue(np.all(out_a <= hi + 1e-6))

    def test_delta_helpers(self):
        active = np.array([2.0, 2.0, 5.0, 3.0], dtype=np.float64)
        delta = compute_delta_active_requests(active)
        np.testing.assert_allclose(delta, np.array([0.0, 0.0, 3.0, -2.0], dtype=np.float64), atol=1e-8)

        d_norm = normalize_delta_active_requests(delta, mean=0.5, std=2.0)
        np.testing.assert_allclose(d_norm, (delta - 0.5) / 2.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
