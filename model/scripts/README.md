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
The superseded arrivals-only physics wrapper lives under `scripts/legacy/`.

`model.scripts.infer` streams native 250 ms ledger and power bins directly to
`power.csv`; only request timing records and the engine iteration trace are
retained. Use the Python `simulate()` API when materialized ledger and power
arrays are required for analysis.
