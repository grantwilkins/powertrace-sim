"""vLLM ``/metrics`` scraper sidecar (Tier-0 instrumentation, CAMPAIGN.md §5-A).

Polls the engine's Prometheus endpoint at 4 Hz and writes the stock scheduler,
token, iteration, and cache metrics to ``engine.csv``. These measurements replace
only ledger fields with an exact mapping; phase-specific batch and context work
still come from request timing because stock vLLM does not expose them.

``parse_prometheus_metrics`` mirrors the parser in ``client_async`` but is defined
here directly: ``client_async`` pulls heavy optional deps (aiohttp/openai) at
import time, so importing it just for a 12-line pure-string parser would make this
otherwise-light module fail in test environments. The pure ``metrics_row`` mapping
is what the unit tests exercise; ``poll_loop`` is the thin live layer.
"""

from __future__ import annotations

import csv
import time
import urllib.request


def parse_prometheus_metrics(text: str) -> dict[str, list[float]]:
    """Parse exposition format into all series grouped by metric name."""
    metrics: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value = line.split(" ", 1)
            value = float(value)
        except ValueError:
            continue
        metrics.setdefault(left.split("{", 1)[0], []).append(value)
    return metrics


# (column, aliases, reduction). Sharded counters/gauges sum; cache occupancy is
# averaged because every worker reports a fraction on the same [0, 1] scale.
ENGINE_COLUMNS = (
    ("timestamp", ("__t__",), "first"),
    ("num_requests_running", ("vllm:num_requests_running",), "sum"),
    ("num_requests_waiting", ("vllm:num_requests_waiting",), "sum"),
    ("gpu_cache_usage_perc", ("vllm:gpu_cache_usage_perc",), "mean"),
    ("prompt_tokens_total", ("vllm:prompt_tokens_total",), "sum"),
    ("generation_tokens_total", ("vllm:generation_tokens_total",), "sum"),
    ("iteration_tokens_total_sum", ("vllm:iteration_tokens_total_sum",), "sum"),
    ("iteration_tokens_total_count", ("vllm:iteration_tokens_total_count",), "sum"),
    ("request_prefill_time_seconds_sum", ("vllm:request_prefill_time_seconds_sum",), "sum"),
    ("request_decode_time_seconds_sum", ("vllm:request_decode_time_seconds_sum",), "sum"),
    ("num_preemptions_total", ("vllm:num_preemptions_total",), "sum"),
    ("prefix_cache_queries_total", ("vllm:prefix_cache_queries_total",), "sum"),
    ("prefix_cache_hits_total", ("vllm:prefix_cache_hits_total",), "sum"),
)

ENGINE_HEADER = [name for name, _, _ in ENGINE_COLUMNS]


def _series(parsed: dict[str, list[float]], aliases: tuple[str, ...]):
    for alias in aliases:
        if alias in parsed:
            return parsed[alias]
    return None


def metrics_row(parsed: dict, t: float) -> list[float]:
    """Map a parsed-metrics dict to the fixed ``engine.csv`` column order.

    Missing counters map to ``nan`` (so downstream differencing can detect gaps);
    the injected wall time ``t`` fills the timestamp column.
    """
    row: list[float] = []
    for name, aliases, reduction in ENGINE_COLUMNS:
        if aliases == ("__t__",):
            row.append(float(t))
        else:
            values = _series(parsed, aliases)
            if values is None:
                row.append(float("nan"))
            elif reduction == "sum":
                row.append(float(sum(values)))
            elif reduction == "mean":
                row.append(float(sum(values) / len(values)))
            else:
                row.append(float(values[0]))
    return row


def observed_metrics_row(parsed: dict, t: float) -> list[float] | None:
    """Return a row only for a successful scrape.

    Preflight proves required counters exist. A transient HTTP failure should not
    write an all-NaN sample that poisons strict evidence validation.
    """
    return metrics_row(parsed, t) if parsed else None


def available_columns(parsed: dict[str, list[float]]) -> set[str]:
    return {
        name for name, aliases, _ in ENGINE_COLUMNS
        if aliases == ("__t__",) or _series(parsed, aliases) is not None
    }


def preflight_metrics(base_url: str, required_columns, *, timeout_s=5.0) -> None:
    """Fail before traffic if the server cannot expose required evidence."""
    with urllib.request.urlopen(metrics_url(base_url), timeout=timeout_s) as response:
        parsed = parse_prometheus_metrics(response.read().decode())
    missing = sorted(set(required_columns) - available_columns(parsed))
    if missing:
        raise ValueError(f"/metrics lacks required evidence columns: {missing}")


def metrics_url(base_url: str) -> str:
    """Derive the ``/metrics`` URL from an OpenAI-style ``base_url``."""
    root = base_url.rsplit("/v1", 1)[0].rstrip("/")
    return f"{root}/metrics"


async def poll_loop(base_url: str, out_csv: str, period_s: float, stop_event) -> None:
    """Thin live layer (not unit-tested): scrape ``/metrics`` until ``stop_event``."""
    import aiohttp  # local import: optional dependency

    url = metrics_url(base_url)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ENGINE_HEADER)
        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                t = time.time()
                try:
                    async with session.get(url, timeout=5) as resp:
                        parsed = (
                            parse_prometheus_metrics(await resp.text())
                            if resp.status == 200 else {}
                        )
                except Exception:
                    parsed = {}
                row = observed_metrics_row(parsed, t)
                if row is not None:
                    writer.writerow(row)
                    f.flush()
                await _sleep(period_s)


async def _sleep(period_s: float) -> None:
    import asyncio

    await asyncio.sleep(period_s)


def read_engine_csv(path: str) -> dict:
    """Load an ``engine.csv`` into column arrays (used by the ledger builder)."""
    import numpy as np

    cols: dict[str, list] = {h: [] for h in ENGINE_HEADER}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for h in ENGINE_HEADER:
                try:
                    cols[h].append(float(r[h]))
                except (TypeError, ValueError):
                    cols[h].append(float("nan"))
    return {h: np.asarray(v, dtype=float) for h, v in cols.items()}
