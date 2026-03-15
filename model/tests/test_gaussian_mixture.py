import unittest

import numpy as np

from model.utils.gaussian_mixture import NumpyGaussianMixture1D, make_gaussian_mixture


class TestGaussianMixtureUtils(unittest.TestCase):
    def _bimodal_data(self) -> np.ndarray:
        rng = np.random.default_rng(123)
        left = rng.normal(-2.0, 0.4, size=300)
        right = rng.normal(3.0, 0.5, size=300)
        return np.concatenate([left, right]).astype(np.float64)

    def test_numpy_gmm_fit_predict(self):
        x = self._bimodal_data()
        gmm = NumpyGaussianMixture1D(n_components=2, random_state=7, n_init=5, max_iter=200)
        gmm.fit(x)
        labels = gmm.predict(x)
        self.assertEqual(labels.shape, (x.shape[0],))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 2))

    def test_numpy_gmm_weights_sum_to_one(self):
        x = self._bimodal_data()
        gmm = NumpyGaussianMixture1D(n_components=2, random_state=9, n_init=3, max_iter=150)
        gmm.fit(x)
        self.assertAlmostEqual(float(np.sum(gmm.weights_)), 1.0, places=8)

    def test_numpy_gmm_aic_bic_finite(self):
        x = self._bimodal_data()
        gmm = NumpyGaussianMixture1D(n_components=2, random_state=11, n_init=3, max_iter=150)
        gmm.fit(x)
        self.assertTrue(np.isfinite(gmm.aic(x)))
        self.assertTrue(np.isfinite(gmm.bic(x)))

    def test_make_gaussian_mixture_returns_callable(self):
        gmm = make_gaussian_mixture(n_components=2, random_state=3, n_init=2, max_iter=50)
        self.assertTrue(callable(getattr(gmm, "fit", None)))
        self.assertTrue(callable(getattr(gmm, "predict", None)))


if __name__ == "__main__":
    unittest.main()
