# Maintained command-line interface

The default model exposes four commands:

| Module | Purpose |
|---|---|
| `model.scripts.prepare_data` | Validate and hash-bind canonical fitting inputs |
| `model.scripts.train` | Refit the frozen timing and power equations |
| `model.scripts.evaluate` | Score aligned measured and predicted power traces |
| `model.scripts.infer` | Predict request timing and deterministic node GPU power |

See the repository `README.md` for runnable examples. Historical GMM-BiGRU
commands live exclusively under `archive/gmm_bigru_v1/`.
