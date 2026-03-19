# Utils

Shared helper modules used across the training, evaluation, and inference code.

## Modules

- `config.py`: configuration constants and file-path helpers.
- `decode_time.py`: derive decode-time values from benchmark `itls` data.
- `gaussian_mixture.py`: wrapper around `sklearn.mixture.GaussianMixture`.
- `io.py`: JSON/CSV and filesystem helpers used by the CLI scripts.
- `runtime.py`: process/threading setup used by `model.scripts.*` entrypoints.

## Notes

- There is no longer an `extract_performance_stats.py` module in `model/utils/`.
- Performance and benchmark aggregation logic now lives in `model/training_data/`.
