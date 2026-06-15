"""vLLM ``/metrics`` scraper sidecar (Tier-0 instrumentation, CAMPAIGN.md §5-A).

Polls the engine's Prometheus endpoint at 4 Hz and writes ``engine.csv`` with the
true per-bin engine state, turning every power sample into a *labeled*
``(state -> power)`` row and removing the error-prone ttft/itl reconstruction.

``parse_prometheus_metrics`` mirrors the parser in ``client_async`` but is defined
here directly: ``client_async`` pulls heavy optional deps (aiohttp/openai) at
import time, so importing it just for a 12-line pure-string parser would make this
otherwise-light module fail in test environments. The pure ``metrics_row`` mapping
is what the unit tests exercise; ``poll_loop`` is the thin live layer.
"""

from __future__ import annotations

import csv
import time


def parse_prometheus_metrics(text: str) -> dict:
    """Parse Prometheus exposition format -> {metric_name: value} (last wins)."""
    metrics: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value = line.split(" ", 1)
            value = float(value)
        except ValueError:
            continue
        metrics[left.split("{", 1)[0]] = value
    return metrics


# (engine.csv column, vLLM metric name) — the §2 data contract for engine.csv.
ENGINE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("timestamp", "__t__"),  # wall time injected by the logger
    ("num_requests_running", "vllm:num_requests_running"),
    ("num_requests_waiting", "vllm:num_requests_waiting"),
    ("gpu_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    ("prompt_tokens_total", "vllm:prompt_tokens_total"),
    ("generation_tokens_total", "vllm:generation_tokens_total"),
    ("iteration_tokens_total_sum", "vllm:iteration_tokens_total_sum"),
    ("iteration_tokens_total_count", "vllm:iteration_tokens_total_count"),
    ("request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_sum"),
    ("request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_sum"),
)

ENGINE_HEADER = [name for name, _ in ENGINE_COLUMNS]


def metrics_row(parsed: dict, t: float) -> list[float]:
    """Map a parsed-metrics dict to the fixed ``engine.csv`` column order.

    Missing counters map to ``nan`` (so downstream differencing can detect gaps);
    the injected wall time ``t`` fills the timestamp column.
    """
    row: list[float] = []
    for name, metric in ENGINE_COLUMNS:
        if metric == "__t__":
            row.append(float(t))
        else:
            v = parsed.get(metric)
            row.append(float(v) if v is not None else float("nan"))
    return row


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
                writer.writerow(metrics_row(parsed, t))
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
