# Maintained Facility Support Artifacts

This directory contains the compact CSV, JSON, and LaTeX support files used by
the maintained Azure facility paper pipeline. Regenerate them with:

```bash
uv run --extra train --extra paper -m scripts.paper.regenerate
```

The current set consists of `azure_facility_*`,
`azure_oversubscription_capacity.*`, and the excluded-but-currently reproducible
`azure_hierarchy_figure.*` diagnostic. Maintained rendered figures live under
`results/paper/facility/`; large node and hierarchy arrays under
`results/azure_facility/` are local intermediates.

Retired appendix, GMM-BiGRU baseline, feature-sufficiency, and trace-fidelity
outputs are preserved under `archive/research_artifacts/eval_paper/`. They are
not valid inputs to the maintained paper manifest.
