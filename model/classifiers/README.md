# Legacy physics compatibility

This package now contains only `physics.py`, the compact physics-artifact
compatibility layer still exercised by retained research evaluations. The
maintained selected model lives in `model/timing/`, `model/power/`, and
`model/training/`.

The GMM fitting, BiGRU classifier, feature construction, checkpoint loading,
and stochastic trace generation modules were moved to
`archive/gmm_bigru_v1/model/classifiers/` with their tests.
