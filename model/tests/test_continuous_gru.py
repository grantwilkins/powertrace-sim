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
    gaussian_nll_loss,
    generate_trace,
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
