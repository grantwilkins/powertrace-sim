# Scripts

This directory contains maintained evaluation entrypoints. The current
command-line surface lives in `scripts/eval/`; training entrypoints live under
`model/scripts/`. GMM-BiGRU evaluation scripts live in
`archive/gmm_bigru_v1/scripts/`.

## Layout

```text
scripts/
├── eval/                 # Selected-model and Azure evaluation support
├── paper/                # One maintained local paper-regeneration entry point
└── legacy/               # Non-BiGRU scripts cut from the paper path
```

## Common Entry Points

```bash
uv run -m scripts.eval.run_azure_pipeline
uv run -m scripts.paper.regenerate
```

This refits one selected artifact, regenerates the replay, timing, transfer,
and facility families, then writes a hash-bound manifest. The complete
paper-output allowlist and evidence labels are in
[`docs/PAPER_OUTPUTS.md`](../docs/PAPER_OUTPUTS.md). Other analysis modules are
support code or development diagnostics, not additional paper entry points.

## See Also

- [scripts/eval/README.md](eval/README.md) - Current evaluation CLI reference
- [model/scripts/README.md](../model/scripts/README.md) - Training and inference entrypoints

## Testing

```bash
uv run -m pytest -x
```
