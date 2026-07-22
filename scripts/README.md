# Scripts

This directory contains maintained evaluation entrypoints. The current
command-line surface lives in `scripts/eval/`; training entrypoints live under
`model/scripts/`. GMM-BiGRU evaluation scripts live in
`archive/gmm_bigru_v1/scripts/`.

## Layout

```text
scripts/
├── eval/                 # Selected-model and Azure evaluations
└── legacy/               # Non-BiGRU scripts cut from the paper path
```

## Common Entry Points

```bash
uv run -m scripts.eval.run_azure_pipeline
uv run -m scripts.eval.generate_azure_facility_sizing_table
uv run -m scripts.eval.appendix_surrogate_validity --dry-run
```

## See Also

- [scripts/eval/README.md](eval/README.md) - Current evaluation CLI reference
- [model/scripts/README.md](../model/scripts/README.md) - Training and inference entrypoints

## Testing

```bash
uv run -m pytest -x
```
