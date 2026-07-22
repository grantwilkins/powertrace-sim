# GMM-BiGRU v1 historical artifact

This directory preserves the first PowerTrace-Sim model as a runnable,
self-contained Python source snapshot. It is historical evidence, not the
repository default.

The model fits a one-dimensional GMM to measured node power, labels every
training bin by mixture component, and trains a bidirectional GRU to predict
the component sequence from normalized active-request count and its first
difference. Inference samples power from the predicted component distribution.

Run the archived code from this directory so it is the Python import root:

```bash
cd archive/gmm_bigru_v1
uv run --project ../.. --extra archive-bigru python -m model.scripts.train_gmm_bigru --help
uv run --project ../.. --extra archive-bigru python -m model.scripts.infer_gmm_bigru --help
uv run --project ../.. --extra archive-bigru python -m scripts.eval.run_baselines_node --help
uv run --project ../.. --extra archive-bigru python -m pytest -x tests
```

The complete tracked artifact trees now live under
`results/continuous_v1_gmm_bigru/` and `results/experimental_continuous_v1/`
inside this archive. They preserve:

- prepared per-configuration datasets and normalization parameters;
- K10 and auto-K checkpoints and GMM parameters;
- IID, AR(1), and thresholded-AR parameters;
- training curves, run manifests, and one canonical metric summary per
  scientifically distinct variant.
- the earlier per-configuration `results/training/` checkpoints and curves;
- the tracked `results/rps_sweep*/` request-rate sweep outputs.

Scientifically distinct metric reruns remain: `eval_metrics_fullheldout` adds
all-heldout columns, `eval_metrics_rerun_energy` corrects energy accounting,
and the IID, AR(1), and thresholded-AR trees use different generation modes.
The two thresholded-AR reruns produced identical metric CSVs after normalizing
their output-directory strings, so only the later `eval_metrics_rerun2` copy is
retained. Regenerable evaluation overlays and AR-parameter plots were removed
after the intact archive was checkpointed; training curves remain preserved.
No code in this snapshot imports the new selected-model timing or power
implementation.

The archive also contains the model-specific node/facility comparisons,
feature-sufficiency and power-CDF figures, trace-fidelity table, request-rate
sweep, historical result collector, and their tests. The copied Splitwise helper is part of the frozen
comparison environment; the maintained facility pipeline uses its own root
copy.

`feature_test/gmm_bigru_baseline.py` preserves the B2 comparator from the later
feature-selection ladder. Its target-blind deterministic test is included in
the archive suite.
