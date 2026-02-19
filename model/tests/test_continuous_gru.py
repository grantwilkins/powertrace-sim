import os
import unittest

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.classifiers.continuous_gru import (
    MeanRevertingGRU,
    NoisyTeacherForcing,
    PowerNoiseInjector,
    ResidualAutoregressiveGRU,
    build_exogenous_features,
    compute_inference_features,
    gaussian_nll_loss,
    generate_trace,
    generate_mean_reverting_trace,
    mean_reverting_nll,
    mdn_nll_loss,
)


class TestContinuousGRU(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch

    def test_forward_shapes_gaussian(self):
        model = ResidualAutoregressiveGRU(input_dim=4, hidden_dim=32, num_layers=1, output_mode="gaussian")
        x = self.torch.randn(3, 17, 4)
        params, h = model(x)
        self.assertEqual(tuple(params.shape), (3, 17, 2))
        self.assertEqual(tuple(h.shape), (1, 3, 32))

    def test_forward_shapes_mdn(self):
        model = ResidualAutoregressiveGRU(input_dim=4, hidden_dim=16, num_layers=1, output_mode="mdn")
        x = self.torch.randn(2, 11, 4)
        params, h = model(x)
        self.assertEqual(tuple(params.shape), (2, 11, 9))  # M=3 => 3*3
        self.assertEqual(tuple(h.shape), (1, 2, 16))

    def test_mean_reverting_forward_shapes_gaussian(self):
        model = MeanRevertingGRU(input_dim=3, hidden_dim=32, num_layers=1, n_mix=1)
        x = self.torch.randn(2, 10, 3)
        params, h = model(x)
        self.assertEqual(tuple(params["mu"].shape), (2, 10, 1))
        self.assertEqual(tuple(params["alpha"].shape), (2, 10, 1))
        self.assertEqual(tuple(params["log_sigma"].shape), (2, 10, 1))
        self.assertEqual(tuple(h.shape), (1, 2, 32))
        self.assertTrue(self.torch.all(params["alpha"] > 0.0))
        self.assertTrue(self.torch.all(params["alpha"] < 1.0))
        self.assertTrue(self.torch.all(params["log_sigma"] >= -6.0))
        self.assertTrue(self.torch.all(params["log_sigma"] <= 2.0))

    def test_mean_reverting_forward_shapes_mdn(self):
        model = MeanRevertingGRU(input_dim=3, hidden_dim=16, num_layers=1, n_mix=3)
        x = self.torch.randn(3, 8, 3)
        params, h = model(x)
        self.assertEqual(tuple(params["mu"].shape), (3, 8, 3))
        self.assertEqual(tuple(params["alpha"].shape), (3, 8, 1))
        self.assertEqual(tuple(params["log_sigma"].shape), (3, 8, 3))
        self.assertEqual(tuple(params["logit_pi"].shape), (3, 8, 3))
        self.assertEqual(tuple(h.shape), (1, 3, 16))
        self.assertTrue(self.torch.all(params["alpha"] > 0.0))
        self.assertTrue(self.torch.all(params["alpha"] < 1.0))
        self.assertTrue(self.torch.all(params["log_sigma"] >= -6.0))
        self.assertTrue(self.torch.all(params["log_sigma"] <= 2.0))

    def test_losses_finite_with_mask(self):
        B, T = 2, 9
        y = self.torch.randn(B, T, 1)
        mask = self.torch.zeros(B, T, dtype=self.torch.bool)
        mask[0, :9] = True
        mask[1, :5] = True

        gaussian_params = self.torch.randn(B, T, 2)
        loss_g = gaussian_nll_loss(gaussian_params, y, mask=mask)
        self.assertTrue(self.torch.isfinite(loss_g))

        mdn_params = self.torch.randn(B, T, 9)
        loss_m = mdn_nll_loss(mdn_params, y, M=3, mask=mask)
        self.assertTrue(self.torch.isfinite(loss_m))

    def test_mean_reverting_losses_finite_with_mask(self):
        B, T = 2, 9
        p_prev = self.torch.randn(B, T, 1)
        p_target = self.torch.randn(B, T, 1)
        mask = self.torch.zeros(B, T, dtype=self.torch.bool)
        mask[0, :9] = True
        mask[1, :4] = True

        model_g = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=1)
        params_g, _ = model_g(self.torch.randn(B, T, 3))
        loss_g = mean_reverting_nll(params_g, p_prev, p_target, n_mix=1, mask=mask)
        self.assertTrue(self.torch.isfinite(loss_g))
        self.assertEqual(loss_g.ndim, 0)

        model_m = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=3)
        params_m, _ = model_m(self.torch.randn(B, T, 3))
        loss_m = mean_reverting_nll(params_m, p_prev, p_target, n_mix=3, mask=mask)
        self.assertTrue(self.torch.isfinite(loss_m))
        self.assertEqual(loss_m.ndim, 0)

    def test_mean_reverting_lambda_mu_zero_matches_original_nll(self):
        B, T = 2, 7
        p_prev = self.torch.randn(B, T, 1)
        p_target = self.torch.randn(B, T, 1)

        model_g = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=1)
        params_g, _ = model_g(self.torch.randn(B, T, 3))
        loss_g = mean_reverting_nll(params_g, p_prev, p_target, n_mix=1, lambda_mu=0.0)

        alpha_g = params_g["alpha"]
        mu_g = params_g["mu"]
        log_sigma_g = params_g["log_sigma"]
        sigma_g = self.torch.exp(log_sigma_g)
        pred_mean_g = ((1.0 - alpha_g) * p_prev) + (alpha_g * mu_g)
        expected_g = (
            0.5 * np.log(2.0 * np.pi) + log_sigma_g + 0.5 * ((p_target - pred_mean_g) / sigma_g) ** 2
        ).mean()
        self.assertTrue(self.torch.allclose(loss_g, expected_g, atol=1e-6, rtol=1e-6))

        model_m = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=3)
        params_m, _ = model_m(self.torch.randn(B, T, 3))
        loss_m = mean_reverting_nll(params_m, p_prev, p_target, n_mix=3, lambda_mu=0.0)

        alpha_m = params_m["alpha"]
        mu_m = params_m["mu"]
        log_sigma_m = params_m["log_sigma"]
        sigma_m = self.torch.exp(log_sigma_m)
        logit_pi = params_m["logit_pi"]
        pred_mean_m = ((1.0 - alpha_m) * p_prev) + (alpha_m * mu_m)
        log_pi_m = self.torch.log_softmax(logit_pi, dim=-1)
        log_normal_m = -0.5 * np.log(2.0 * np.pi) - log_sigma_m - 0.5 * ((p_target - pred_mean_m) / sigma_m) ** 2
        expected_m = -(self.torch.logsumexp(log_pi_m + log_normal_m, dim=-1, keepdim=True)).mean()
        self.assertTrue(self.torch.allclose(loss_m, expected_m, atol=1e-6, rtol=1e-6))

    def test_mean_reverting_aux_mu_loss_increases_total(self):
        p_prev = self.torch.zeros(1, 4, 1)
        p_target = self.torch.zeros(1, 4, 1)
        params = {
            "alpha": self.torch.full((1, 4, 1), 0.05),
            "mu": self.torch.full((1, 4, 1), 5.0),
            "log_sigma": self.torch.zeros(1, 4, 1),
        }
        base = mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=0.0)
        parts = mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=0.1, return_parts=True)
        self.assertGreater(float(parts["total_loss"].item()), float(base.item()))
        self.assertAlmostEqual(
            float(parts["total_loss"].item()),
            float((parts["nll_loss"] + (0.1 * parts["mu_loss"])).item()),
            places=7,
        )

    def test_mean_reverting_aux_mu_loss_respects_mask(self):
        p_prev = self.torch.zeros(1, 3, 1)
        p_target = self.torch.tensor([[[0.0], [10.0], [0.0]]], dtype=self.torch.float32)
        params = {
            "alpha": self.torch.full((1, 3, 1), 0.1),
            "mu": self.torch.tensor([[[0.0], [1000.0], [0.0]]], dtype=self.torch.float32),
            "log_sigma": self.torch.zeros(1, 3, 1),
        }
        mask = self.torch.tensor([[True, False, True]], dtype=self.torch.bool)

        masked = mean_reverting_nll(params, p_prev, p_target, n_mix=1, mask=mask, lambda_mu=0.1, return_parts=True)
        unmasked = mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=0.1, return_parts=True)

        self.assertLess(float(masked["mu_loss"].item()), 1e-9)
        self.assertGreater(float(unmasked["mu_loss"].item()), 1e5)

    def test_mean_reverting_aux_mu_increases_grad_at_low_alpha(self):
        p_prev = self.torch.zeros(1, 5, 1)
        p_target = self.torch.full((1, 5, 1), 2.0)
        mu = self.torch.zeros(1, 5, 1, requires_grad=True)
        params = {
            "alpha": self.torch.full((1, 5, 1), 0.01),
            "mu": mu,
            "log_sigma": self.torch.zeros(1, 5, 1),
        }

        loss_nll = mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=0.0)
        loss_nll.backward()
        grad_nll = float(mu.grad.detach().abs().mean().item())

        mu.grad.zero_()
        loss_total = mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=0.1)
        loss_total.backward()
        grad_total = float(mu.grad.detach().abs().mean().item())

        self.assertGreater(grad_nll, 0.0)
        self.assertGreater(grad_total, grad_nll * 5.0)

    def test_mean_reverting_negative_lambda_mu_rejected(self):
        p_prev = self.torch.zeros(1, 2, 1)
        p_target = self.torch.zeros(1, 2, 1)
        params = {
            "alpha": self.torch.full((1, 2, 1), 0.1),
            "mu": self.torch.zeros(1, 2, 1),
            "log_sigma": self.torch.zeros(1, 2, 1),
        }
        with self.assertRaisesRegex(ValueError, "lambda_mu"):
            mean_reverting_nll(params, p_prev, p_target, n_mix=1, lambda_mu=-1.0)

    def test_generate_trace_length_and_finite(self):
        model = ResidualAutoregressiveGRU(input_dim=4, hidden_dim=8, num_layers=1, output_mode="gaussian")
        features = self.torch.zeros(15, 3)
        norm = {"power_mean": 200.0, "power_std": 20.0, "delta_mean": 0.0, "delta_std": 5.0}
        out = generate_trace(model, features, P_0=210.0, norm_params=norm, output_mode="gaussian")
        self.assertEqual(len(out), 16)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_generate_trace_clamped_to_power_range(self):
        model = ResidualAutoregressiveGRU(input_dim=4, hidden_dim=8, num_layers=1, output_mode="gaussian")
        features = self.torch.zeros(20, 3)
        norm = {"power_mean": 200.0, "power_std": 20.0, "delta_mean": 0.0, "delta_std": 10.0}
        out = generate_trace(
            model,
            features,
            P_0=210.0,
            norm_params=norm,
            output_mode="gaussian",
            config_power_range=(180.0, 260.0),
            rng=np.random.default_rng(0),
        )
        lo = 180.0 - (0.05 * (260.0 - 180.0))
        hi = 260.0 + (0.05 * (260.0 - 180.0))
        self.assertTrue(np.all(out >= lo - 1e-6))
        self.assertTrue(np.all(out <= hi + 1e-6))

    def test_noisy_teacher_forcing_schedule(self):
        sched = NoisyTeacherForcing(noise_std_start=0.0, noise_std_end=0.1, warmup_epochs=100, ramp_epochs=200)
        self.assertAlmostEqual(sched.get_noise_std(0), 0.0, places=8)
        self.assertAlmostEqual(sched.get_noise_std(99), 0.0, places=8)
        self.assertAlmostEqual(sched.get_noise_std(100), 0.0, places=8)
        self.assertAlmostEqual(sched.get_noise_std(300), 0.1, places=6)
        self.assertAlmostEqual(sched.get_noise_std(500), 0.1, places=6)

    def test_power_noise_injector_schedule_and_noop(self):
        sched = PowerNoiseInjector(warmup_epochs=5, ramp_epochs=10, max_noise_std=0.2)
        self.assertAlmostEqual(sched.get_std(0), 0.0, places=8)
        self.assertAlmostEqual(sched.get_std(4), 0.0, places=8)
        self.assertAlmostEqual(sched.get_std(5), 0.0, places=8)
        self.assertAlmostEqual(sched.get_std(10), 0.1, places=6)
        self.assertAlmostEqual(sched.get_std(25), 0.2, places=6)

        x = self.torch.ones(2, 3, 1)
        zero_sched = PowerNoiseInjector(warmup_epochs=0, ramp_epochs=10, max_noise_std=0.0)
        y = zero_sched(x, epoch=30)
        self.assertTrue(self.torch.allclose(x, y))

    def test_compute_inference_features_basic_counts_and_shape(self):
        requests = [
            {"arrival_time": 0.1, "input_tokens": 10.0, "output_tokens": 4.0},
            {"arrival_time": 0.4, "input_tokens": 20.0, "output_tokens": 0.0},
        ]
        cfg = {
            "lambda_prefill": 10.0,
            "lambda_decode": 4.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        feats = compute_inference_features(requests=requests, config=cfg, T=4, dt=1.0)
        self.assertEqual(tuple(feats.shape), (4, 2))
        expected_A = np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float32)
        expected_t_arr = np.array([30.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(feats[:, 0], expected_A, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(feats[:, 1], np.log1p(expected_t_arr), rtol=1e-6, atol=1e-6)

    def test_compute_inference_features_auto_T(self):
        requests = [{"arrival_time": 0.0, "input_tokens": 10.0, "output_tokens": 10.0}]
        cfg = {
            "lambda_prefill": 10.0,
            "lambda_decode": 10.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        feats = compute_inference_features(requests=requests, config=cfg, T=None, dt=0.25)
        self.assertEqual(tuple(feats.shape), (9, 2))

    def test_generate_mean_reverting_trace_gaussian_seeded_and_clamped(self):
        model = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=1)
        feats = np.zeros((15, 2), dtype=np.float32)
        norm = {
            "power_mean": 200.0,
            "power_std": 20.0,
            "power_min": 180.0,
            "power_max": 260.0,
        }
        out_a = generate_mean_reverting_trace(model, feats, P_0=210.0, config_norm=norm, n_mix=1, seed=123)
        out_b = generate_mean_reverting_trace(model, feats, P_0=210.0, config_norm=norm, n_mix=1, seed=123)
        self.assertEqual(len(out_a), 15)
        self.assertTrue(np.all(np.isfinite(out_a)))
        self.assertTrue(np.allclose(out_a, out_b))
        lo = 180.0 - (0.05 * (260.0 - 180.0))
        hi = 260.0 + (0.05 * (260.0 - 180.0))
        self.assertTrue(np.all(out_a >= lo - 1e-6))
        self.assertTrue(np.all(out_a <= hi + 1e-6))

    def test_generate_mean_reverting_trace_mdn_seeded_and_clamped(self):
        model = MeanRevertingGRU(input_dim=3, hidden_dim=8, num_layers=1, n_mix=3)
        feats = np.zeros((12, 2), dtype=np.float32)
        norm = {
            "power_mean": 300.0,
            "power_std": 25.0,
            "power_min": 220.0,
            "power_max": 400.0,
        }
        out = generate_mean_reverting_trace(model, feats, P_0=250.0, config_norm=norm, n_mix=3, seed=7)
        self.assertEqual(len(out), 12)
        self.assertTrue(np.all(np.isfinite(out)))
        lo = 220.0 - (0.05 * (400.0 - 220.0))
        hi = 400.0 + (0.05 * (400.0 - 220.0))
        self.assertTrue(np.all(out >= lo - 1e-6))
        self.assertTrue(np.all(out <= hi + 1e-6))

    def test_continuous_feature_builder_shape_and_nonnegative(self):
        T = 20
        dt = 0.25
        t0 = 1000.0
        req_ts = np.array([1000.1, 1000.7, 1001.8], dtype=np.float64)
        in_tok = np.array([100.0, 200.0, 50.0], dtype=np.float64)
        out_tok = np.array([40.0, 80.0, 20.0], dtype=np.float64)

        feats = build_exogenous_features(
            T=T,
            dt=dt,
            t0=t0,
            request_timestamps=req_ts,
            input_tokens=in_tok,
            output_tokens=out_tok,
            prefill_rate=1000.0,
            decode_rate=200.0,
            drop_backlog=False,
        )
        self.assertEqual(feats["active_requests"].shape[0], T)
        self.assertEqual(feats["t_arrive"].shape[0], T)
        self.assertEqual(feats["t_backlog"].shape[0], T)
        self.assertTrue(np.all(feats["active_requests"] >= 0.0))
        self.assertTrue(np.all(feats["t_arrive"] >= 0.0))
        self.assertTrue(np.all(feats["t_backlog"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
