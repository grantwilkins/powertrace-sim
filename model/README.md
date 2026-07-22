# Selected model package

`model/` contains the maintained architecture-aware timing and deterministic
power pipeline.

```text
request_schedule.py -> timing/ -> power/ -> simulation.py
                            ^          ^
                            |          |
                         training/ + artifacts/powertrace_v1.json
```

The public lifecycle is `prepare_data`, `train`, `evaluate`, and `infer` under
`model.scripts`. Shared architecture and ledger primitives remain in
`model.training_data` while their final package locations are consolidated.

The previous GMM-BiGRU implementation and its commands are runnable only from
`archive/gmm_bigru_v1/`.

Run the maintained test suite with:

```bash
uv run -m pytest -x
```
