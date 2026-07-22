# Evaluation Scripts

This directory contains the maintained selected-model and Azure facility
evaluation surface. The GMM-BiGRU node baselines, retrospective CDF and feature
figures, trace-fidelity table, and their support modules now live under
`archive/gmm_bigru_v1/scripts/eval/`.

## Azure Facility Pipeline

Run the end-to-end workflow with:

```bash
uv run -m scripts.eval.run_azure_pipeline
```

The pipeline parses the Azure request trace, assigns requests to nodes,
generates selected-model and Splitwise comparator traces, aggregates node power
to rack/row/site levels, computes metrics, and emits the facility figures and
sizing table.

The individual stages remain available as modules:

- `split_azure_week_to_days.py` and `parse_azure_trace.py` normalize requests.
- `azure_to_node_streams.py` assigns requests to the configured facility.
- `azure_generate_traces.py` generates `ours` and `splitwise_strict` traces.
- `azure_aggregate.py` aggregates node traces to rack, row, and site levels.
- `azure_metrics.py` computes facility and load-duration metrics.
- `oversubscription_figure.py`, `azure_figures.py`, and
  `generate_azure_facility_sizing_table.py` render paper outputs.
- `hierarchy_figure.py` renders server/rack/row/site trace comparisons.

The maintained Splitwise comparator is isolated in `splitwise.py`; it does not
import Torch or the historical learned model.
Node-trace generation accepts only `ours` and `splitwise_strict`; the rejected
physics prototype is no longer a facility method, and trace generation no
longer loads the archived training manifest or throughput database.

## Other Maintained Evaluations

`occupancy_roofline.py` retains the profiling-bundle occupancy analysis used to
interpret the selected power coordinates. It is support analysis, not an
allowlisted paper renderer.

## Historical Evaluation

Run GMM-BiGRU commands from the archive so Python resolves the archived `model`
and `scripts` packages:

```bash
cd archive/gmm_bigru_v1
uv run --project ../.. --extra archive-bigru python \
    -m scripts.eval.run_baselines_node --help
```

See `archive/gmm_bigru_v1/README.md` for the archived command and test surface.
The stale surrogate-validity renderer now lives under `scripts/legacy/`.

## Testing

```bash
uv run -m pytest -x
```
