import argparse
import asyncio
import os
import random
import time
from datetime import datetime

import aiohttp
import datasets
import numpy as np
import pandas as pd
import transformers
from openai import AsyncOpenAI

"""
Fully‑asynchronous version of the benchmarking client that preserves (almost) *all*
metric collection and accounting fields from the original synchronous script while
eliminating every synchronous/blocking call.

Key points
-----------
* **AsyncOpenAI** is used for completions – no `run_in_executor` threads.
* **aiohttp** is used for `/info` and `/metrics` endpoints.
* Per‑request timeout (`OPENAI_TIMEOUT`) and hard wall‑clock cutoff
  (`--time-window`) are enforced.
* Per‑request CSV append is kept (matching the old naming convention) so that
  partial data is persisted even if the program exits early.
* All original per‑request fields are computed:
  - Batch Size, Effective Batch Size
  - Prefill / Decode token deltas
  - Prefill / Decode time deltas and throughputs
* End‑of‑run aggregation & human‑readable report mirror the original output.
"""

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

OPENAI_TIMEOUT = 45  # seconds per request
CSV_LOCK = asyncio.Lock()  # serialize CSV writes across tasks


def parse_prometheus_metrics(text: str) -> dict:
    """Parse Prometheus exposition format -> {metric_name: value}."""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value = line.split(" ", 1)
            value = float(value)
        except ValueError:
            continue
        name = left.split("{", 1)[0]
        metrics[name] = value
    return metrics


async def fetch_metrics(session: aiohttp.ClientSession, url: str) -> dict:
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                return parse_prometheus_metrics(await resp.text())
    except Exception:
        pass
    return {}


async def log_vllm_info(session: aiohttp.ClientSession, base_url: str):
    root = base_url.rsplit("/v1", 1)[0]
    try:
        async with session.get(f"{root}/info", timeout=5) as resp:
            if resp.status == 200:
                print("vLLM /info:\n", await resp.json())
    except Exception as e:
        print("[warn] /info fetch failed:", e)

    metrics = await fetch_metrics(session, f"{root}/metrics")
    print(f"Sample /metrics keys: {list(metrics)[:10] if metrics else 'NONE'}")


# ---------------------------------------------------------------------------
# Per‑request coroutine
# ---------------------------------------------------------------------------


async def send_message(
    *,
    client: AsyncOpenAI,
    session: aiohttp.ClientSession,
    model_name: str,
    tokenizer,
    input_options,
    poisson_arrival_rate: float,
    tensor_parallel_size: int,
    reasoning: bool,
    base_url: str,
    date: str,
):
    """Fire one request, compute metric deltas, append to per‑request CSV."""
    root = base_url.rsplit("/v1", 1)[0]
    metrics_url = f"{root}/metrics"

    # ---------------------- choose prompt ----------------------
    row = random.choice(input_options)
    instruction_text = row["instruction"]
    data_source = row["data_source"]

    # ---------------- fetch pre‑metrics ------------------------
    pre = await fetch_metrics(session, metrics_url)

    # ------------------- build messages -----------------------
    messages = []
    if not reasoning:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": instruction_text})

    # ------------------ call OpenAI ---------------------------
    start_t = time.time()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                user=f"req_{int(start_t*1000)}_{random.randint(1000,9999)}",
            ),
            timeout=OPENAI_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception):
        return None  # skip on error/timeout
    latency = time.time() - start_t

    # ---------------- fetch post‑metrics ----------------------
    post = await fetch_metrics(session, metrics_url)

    # ---------------- token accounting -----------------------
    content = resp.choices[0].message.content
    in_tokens = len(tokenizer.encode(instruction_text))
    out_tokens = len(tokenizer.encode(content))

    def delta(name: str):
        return post.get(name, 0.0) - pre.get(name, 0.0)

    # Effective batch: ratio of sum / count deltas if both advance.
    eff_bs = (
        post.get("vllm:iteration_tokens_total_sum", 0)
        - pre.get("vllm:iteration_tokens_total_sum", 0)
    ) / max(
        1,
        post.get("vllm:iteration_tokens_total_count", 0)
        - pre.get("vllm:iteration_tokens_total_count", 0),
    )

    prefill_time = delta("vllm:request_prefill_time_seconds_sum")
    decode_time = delta("vllm:request_decode_time_seconds_sum")

    record = {
        "Request Time": datetime.fromtimestamp(start_t).isoformat(),
        "Model": model_name,
        "Data Source": data_source,
        "Poisson Arrival Rate": poisson_arrival_rate,
        "Tensor Parallel Size": tensor_parallel_size,
        "Input Tokens": in_tokens,
        "Output Tokens": out_tokens,
        "E2E Latency": latency,
        "Batch Size": post.get("vllm:num_requests_running", np.nan),
        "Effective Batch Size": eff_bs,
        "Prefill Tokens": delta("vllm:prompt_tokens_total"),
        "Decode Tokens": delta("vllm:generation_tokens_total"),
        "Prefill Time": prefill_time,
        "Decode Time": decode_time,
        "Prefill Throughput": (
            (delta("vllm:prompt_tokens_total") / prefill_time)
            if prefill_time > 0
            else 0
        ),
        "Decode Throughput": (
            (delta("vllm:generation_tokens_total") / decode_time)
            if decode_time > 0
            else 0
        ),
    }

    # -------------- append row to rolling CSV -----------------
    per_csv = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}_d{date}.csv"
    async with CSV_LOCK:
        file_exists = os.path.exists(per_csv)
        pd.DataFrame([record]).to_csv(
            per_csv, mode="a", header=not file_exists, index=False
        )

    return record


# ---------------------------------------------------------------------------
# Scheduler (Poisson arrivals + hard cutoff)
# ---------------------------------------------------------------------------


async def schedule_messages(
    *,
    client: AsyncOpenAI,
    session: aiohttp.ClientSession,
    model_name: str,
    tokenizer,
    input_options,
    poisson_arrival_rate: float,
    tensor_parallel_size: int,
    reasoning: bool,
    base_url: str,
    window: float,
    date: str,
):
    # ------ generate arrival schedule -------
    times, t = [], 0.0
    while True:
        t += np.random.exponential(1.0 / poisson_arrival_rate)
        if t >= window:
            break
        times.append(t)

    start = time.time()
    root_kwargs = dict(
        client=client,
        session=session,
        model_name=model_name,
        tokenizer=tokenizer,
        input_options=input_options,
        poisson_arrival_rate=poisson_arrival_rate,
        tensor_parallel_size=tensor_parallel_size,
        reasoning=reasoning,
        base_url=base_url,
        date=date,
    )

    async def one(arrive):
        await asyncio.sleep(max(0, start + arrive - time.time()))
        return await send_message(**root_kwargs)

    tasks = [asyncio.create_task(one(a)) for a in times]

    done, pending = await asyncio.wait(tasks, timeout=window)
    for p in pending:
        p.cancel()

    return [t.result() for t in done if t.exception() is None and t.result()]


# ---------------------------------------------------------------------------
# Aggregation & report (same fields as original)
# ---------------------------------------------------------------------------


def aggregate_and_report(records: list, args):
    if not records:
        print("No successful requests.")
        return

    df = pd.DataFrame(records)
    final_csv = f"results_{args.model_name.split('/')[-1]}_{args.poisson_arrival_rate}_{args.tensor_parallel_size}_d{args.date}_final.csv"
    df.to_csv(final_csv, index=False)
    print(f"Saved aggregated CSV: {final_csv} with {len(df)} rows")

    # ---- human‑readable stats ----
    print("\nPerformance summary (⁎ same as original):")
    print(
        f"Avg E2E latency: {df['E2E Latency'].mean():.2f}s | P90: {df['E2E Latency'].quantile(0.9):.2f}s"
    )
    print(
        f"Average batch size: {df['Batch Size'].mean():.2f}  Max: {df['Batch Size'].max()}"
    )
    print(f"Average prefill throughput: {df['Prefill Throughput'].mean():.2f} tok/s")
    print(f"Average decode throughput: {df['Decode Throughput'].mean():.2f} tok/s")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--poisson-arrival-rate", type=float, default=1.0)
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--time-window", type=float, default=600.0)
    args = p.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    dataset = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")["train"]

    client = AsyncOpenAI(
        api_key=args.api_key, base_url=args.base_url, timeout=OPENAI_TIMEOUT
    )

    async with aiohttp.ClientSession() as session:
        await log_vllm_info(session, args.base_url)
        recs = await schedule_messages(
            client=client,
            session=session,
            model_name=args.model_name,
            tokenizer=tokenizer,
            input_options=dataset,
            poisson_arrival_rate=args.poisson_arrival_rate,
            tensor_parallel_size=args.tensor_parallel_size,
            reasoning=args.reasoning,
            base_url=args.base_url,
            window=args.time_window,
            date=args.date,
        )

    aggregate_and_report(recs, args)


if __name__ == "__main__":
    asyncio.run(main())
