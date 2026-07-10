# Legacy Scripts

Archived analysis scripts and shell wrappers whose results were cut from the
e-Energy paper. They were moved here from `scripts/eval/` and `scripts/` during
cleaning-plan.md Phase 4 and are no longer part of the maintained evaluation
path.

- `two_price_fit.py` - Two-price operator model fit over benchmark windows
- `saturating_fit.py` - Saturating-curve fits on the two-price windows
- `operator_table.py` - Descriptive operator table from the two-price windows
- `ledger_fit_lomo.py` - Leave-one-model-out ledger fit
- `rps_power_sweep.py` - Request-rate vs power sweep
- `run_training.sh` - Old training wrapper
- `run_ablation_study.sh` - Old ablation study wrapper
- `collect_random_weights.sh` - Old random-weights collection wrapper

These scripts are kept as-is for reference and are not tested or maintained.
Full provenance (original paths and history) is preserved in git history via
`git log --follow`.

`collect_random_weights.sh` resolves its input and output paths relative to the
repository root, so it can be launched from any working directory.
