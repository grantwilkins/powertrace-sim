"""Unit tests for the gap sampler and the OLS fit."""

import numpy as np

import openhands_gap_fit as fit
import tool_classes as tc
from gap_sampler import GapSampler

PARAMS = {"classes": {
    "local_io": {"mu": -2.5, "sigma": 0.4, "b1": 0.05},
    "bash":     {"mu":  0.2, "sigma": 1.0, "b1": 0.30},
}}


def _mean_gap(sampler, cls, obs, n=20000):
    rng = np.random.default_rng(0)
    return np.mean([sampler.sample(cls, obs, rng) for _ in range(n)])


def test_deterministic_given_rng():
    s = GapSampler(PARAMS)
    a = [s.sample("bash", 100, np.random.default_rng(7)) for _ in range(3)]
    b = [s.sample("bash", 100, np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_larger_observation_means_longer_gap():
    s = GapSampler(PARAMS)
    assert _mean_gap(s, "bash", 1000) > _mean_gap(s, "bash", 1)  # b1 > 0


def test_unknown_class_falls_back_to_bash():
    s = GapSampler(PARAMS)
    rng = np.random.default_rng(0)
    assert s.sample("mystery", 50, rng) > 0  # no KeyError; uses bash params


def test_fit_recovers_lognormal_params():
    rng = np.random.default_rng(0)
    mu, b1, sigma = 0.5, 0.2, 0.6
    samples = []
    for _ in range(5000):
        obs = int(rng.integers(1, 500))
        gap = float(rng.lognormal(mu + b1 * np.log1p(obs), sigma))
        samples.append((obs, gap))
    p = fit.fit_class(samples)
    assert abs(p["mu"] - mu) < 0.1
    assert abs(p["b1"] - b1) < 0.05
    assert abs(p["sigma"] - sigma) < 0.05


def test_sparse_class_keeps_literature_prior():
    samples = [("bash", 10, 1.0)] * 3  # below N_MIN
    params = fit.fit_params(samples)
    assert params["classes"]["bash"]["source"] == "literature_prior"
    assert params["classes"]["local_io"]["source"] == "literature_prior"
    assert set(params["classes"]) == set(tc.CLASSES)
