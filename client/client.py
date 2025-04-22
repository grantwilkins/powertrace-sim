import os
import time
import random
import argparse
import asyncio
import numpy as np
import pandas as pd
import datasets
import transformers
import json
import requests
from openai import OpenAI


def parse_prometheus_metrics(metrics_text):
    """Parse Prometheus format metrics into a dictionary."""
    parsed_metrics = {}
    for line in metrics_text.split("\n"):
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue

        try:
            # Extract metric name and value
            # Format is typically: metric_name{labels...} value
            parts = line.split(" ")
            if len(parts) >= 2:
                metric_full = parts[0]
                value = float(parts[1])

                # Extract the metric name from metric{labels}
                if "{" in metric_full:
                    metric_name, labels_str = metric_full.split("{", 1)
                    labels_str = labels_str.rstrip("}")
                    # Store with labels as a compound key or in a nested structure
                    parsed_metrics[metric_name] = value
                else:
                    parsed_metrics[metric_full] = value
        except Exception as e:
            print(f"Error parsing metric line: {line}, Error: {e}")

    return parsed_metrics


async def send_message(
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    date: str,
    reasoning: bool = False,
    tensor_parallel_size: int = 1,
    base_url: str = "http://localhost:8000/v1",
) -> dict:
    """
    Sends a message to the OpenAI API and records performance metrics.
    Also retrieves vLLM-specific metrics via the /metrics endpoint.
    """
    chosen_input = random.choice(input_options)
    instruction_text = chosen_input["instruction"]
    data_source = chosen_input["data_source"]

    messages = [
        (
            {"role": "system", "content": "You are a helpful assistant."}
            if not reasoning
            else {}
        ),
        {"role": "user", "content": instruction_text},
    ]

    start_time = time.time()

    # Get metrics before request to establish baseline
    metrics_url = f"{base_url.split('/v1')[0]}/metrics"
    pre_metrics = {}
    try:
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            # Store raw text and parse with the Prometheus parser
            pre_metrics_raw = metrics_response.text
            pre_metrics = parse_prometheus_metrics(pre_metrics_raw)
    except Exception as e:
        print(f"Failed to get pre-request metrics: {e}")

    # Create unique request ID to track this specific request
    request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    # Send the actual completion request
    loop = asyncio.get_event_loop()
    response_raw = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model_name,
            messages=messages,
            user=request_id,  # Use user field to track the request
        ),
    )
    e2e_time = time.time() - start_time

    # Get metrics after request
    post_metrics = {}
    try:
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            # Store raw text instead of parsing as JSON
            post_metrics_raw = metrics_response.text
            # Parse Prometheus format metrics
            post_metrics = parse_prometheus_metrics(post_metrics_raw)
    except Exception as e:
        print(f"Failed to get post-request metrics: {e}")

    prefill_tokens = 0
    decode_tokens = 0
    batch_size = 0
    prefill_throughput = 0
    decode_throughput = 0

    # Calculate differences in metrics before and after request
    if pre_metrics and post_metrics:
        try:
            # Extract batch size from appropriate metric
            # Note: You might need to find the right metric for batch size
            if "vllm:num_requests_running" in post_metrics:
                batch_size = post_metrics["vllm:num_requests_running"]

            # Extract prefill and decode stats
            if (
                "vllm:prompt_tokens_total" in pre_metrics
                and "vllm:prompt_tokens_total" in post_metrics
            ):
                prefill_tokens = (
                    post_metrics["vllm:prompt_tokens_total"]
                    - pre_metrics["vllm:prompt_tokens_total"]
                )

            if (
                "vllm:generation_tokens_total" in pre_metrics
                and "vllm:generation_tokens_total" in post_metrics
            ):
                decode_tokens = (
                    post_metrics["vllm:generation_tokens_total"]
                    - pre_metrics["vllm:generation_tokens_total"]
                )

            # Calculate throughput (tokens per second)
            if (
                "vllm:request_prefill_time_seconds" in pre_metrics
                and "vllm:request_prefill_time_seconds" in post_metrics
            ):
                prefill_time = (
                    post_metrics["vllm:request_prefill_time_seconds"]
                    - pre_metrics["vllm:request_prefill_time_seconds"]
                )
                if prefill_time > 0:
                    prefill_throughput = prefill_tokens / prefill_time

            if (
                "vllm:request_decode_time_seconds" in pre_metrics
                and "vllm:request_decode_time_seconds" in post_metrics
            ):
                decode_time = (
                    post_metrics["vllm:request_decode_time_seconds"]
                    - pre_metrics["vllm:request_decode_time_seconds"]
                )
                if decode_time > 0:
                    decode_throughput = decode_tokens / decode_time
        except Exception as e:
            print(f"Error processing metrics: {e}")

    response = response_raw.choices[0]
    content = response.message.content

    # Calculate input and output tokens
    input_tokens = len(tokenizers.encode(instruction_text))
    output_tokens = len(tokenizers.encode(content))

    record = {
        "Request Time": start_time,
        "Model": model_name,
        "Data Source": data_source,
        "Poisson Arrival Rate": poisson_arrival_rate,
        "Tensor Parallel Size": tensor_parallel_size,
        "Input Tokens": input_tokens,
        "Output Tokens": output_tokens,
        "E2E Latency": e2e_time,
        "Batch Size": batch_size,
        "Prefill Tokens": prefill_tokens,
        "Decode Tokens": decode_tokens,
        "Prefill Throughput": prefill_throughput,  # tokens per second
        "Decode Throughput": decode_throughput,  # tokens per second
    }

    outfile = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}_d{date}.csv"
    file_exists = os.path.exists(outfile)
    df_temp = pd.DataFrame([record])
    df_temp.to_csv(outfile, mode="a", header=not file_exists, index=False)

    return record


async def schedule_one(
    task_id: int,
    arrival_time: float,  # arrival time (relative to start_time)
    start_time: float,  # absolute time at which scheduling began
    end_time: float,  # start_time + T
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    reasoning: bool,
    tensor_parallel_size: int,
    base_url: str,
    date: str,
):
    """
    Sleeps until its scheduled arrival_time, then sends a request if still within the time window.
    """
    # Sleep until arrival_time (relative to start_time)
    now = time.time()
    sleep_duration = (start_time + arrival_time) - now
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)

    # Check if we've passed the time window
    if time.time() > end_time:
        print(f"[Task {task_id}] Skipped; arrival time is past the time window.")
        return None

    # Send the message
    print(f"[Task {task_id}] Sending request at {time.time()-start_time:.2f}s.")
    return await send_message(
        client=client,
        model_name=model_name,
        input_options=input_options,
        poisson_arrival_rate=poisson_arrival_rate,
        tokenizers=tokenizers,
        date=date,
        reasoning=reasoning,
        tensor_parallel_size=tensor_parallel_size,
        base_url=base_url,
    )


async def schedule_messages(
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    date: str,
    reasoning: bool = False,
    T: float = 600.0,
    tensor_parallel_size: int = 1,
    base_url: str = "http://localhost:8000/v1",
) -> list:
    """
    Generates arrival times via an exponential distribution (Poisson process),
    but only up to time T. Then schedules each query at its arrival time.

    After T seconds from the start, we best-effort cancel any tasks that
    have not started yet and gather the results of completed tasks.
    """
    # 1) Generate arrival times (exponential interarrivals) until we reach T
    arrival_times = []
    current_time = 0.0
    while True:
        interarrival = np.random.exponential(1.0 / poisson_arrival_rate)
        current_time += interarrival
        if current_time >= T:
            break
        arrival_times.append(current_time)

    print(f"Expected average number of tasks: {poisson_arrival_rate * T:.1f}")
    print(f"Actual number of tasks generated: {len(arrival_times)}")

    # 2) Schedule each arrival as a separate task
    start_time = time.time()
    end_time = start_time + T

    tasks = []
    for i, arrival_time in enumerate(arrival_times):
        task = asyncio.create_task(
            schedule_one(
                task_id=i,
                arrival_time=arrival_time,
                start_time=start_time,
                end_time=end_time,
                client=client,
                model_name=model_name,
                input_options=input_options,
                poisson_arrival_rate=poisson_arrival_rate,
                tokenizers=tokenizers,
                reasoning=reasoning,
                tensor_parallel_size=tensor_parallel_size,
                base_url=base_url,
                date=date,
            )
        )
        tasks.append(task)

    # 3) Allow up to T seconds for these tasks to start and (best-effort) complete
    print(f"Scheduling done. Waiting for up to {T:.1f}s from now...")

    # Sleep until T seconds have passed since scheduling started
    time_left = end_time - time.time()
    if time_left > 0:
        await asyncio.sleep(time_left)

    print(f"Time window of {T} seconds has elapsed. Attempting to end...")

    # Cancel any tasks that haven't started or are not done
    cancelled_count = 0
    for t in tasks:
        if not t.done():
            t.cancel()
            cancelled_count += 1

    print(f"Cancelled {cancelled_count} tasks that were not completed by the deadline.")

    # 4) Gather the results from tasks that finished
    results = []
    for t in tasks:
        # If it was cancelled or raised an exception, ignore or log it
        try:
            result = await t
            if result:  # Only append non-None results
                results.append(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Task raised an exception: {e}")

    total_time = time.time() - start_time
    print(f"Completed {len(results)} tasks in {total_time:.2f} seconds.")

    # Add batch statistics
    if results:
        df = pd.DataFrame(results)
        avg_batch_size = df["Batch Size"].mean()
        max_batch_size = df["Batch Size"].max()
        avg_prefill_throughput = df["Prefill Throughput"].mean()
        avg_decode_throughput = df["Decode Throughput"].mean()

        print(f"Average batch size: {avg_batch_size:.2f}")
        print(f"Maximum batch size: {max_batch_size}")
        print(f"Average prefill throughput: {avg_prefill_throughput:.2f} tokens/s")
        print(f"Average decode throughput: {avg_decode_throughput:.2f} tokens/s")

    return results


async def get_vllm_server_info(base_url: str) -> dict:
    """
    Get information about the vLLM server configuration.
    """
    info = {}
    try:
        # Try to get vLLM server info
        response = requests.get(f"{base_url.split('/v1')[0]}/info")
        if response.status_code == 200:
            info = response.json()
            print("vLLM Server Info:")
            print(json.dumps(info, indent=2))
        else:
            print(f"Failed to get vLLM server info: {response.status_code}")

        # Also try to get metrics to understand exactly what's available
        metrics_url = f"{base_url.split('/v1')[0]}/metrics"
        pre_metrics = {}
        try:
            metrics_response = requests.get(metrics_url)
            if metrics_response.status_code == 200:
                # Store raw text instead of parsing as JSON
                pre_metrics_raw = metrics_response.text
                # Parse Prometheus format metrics
                pre_metrics = parse_prometheus_metrics(pre_metrics_raw)
                print(pre_metrics)
        except Exception as e:
            print(f"Failed to get pre-request metrics: {e}")
    except Exception as e:
        print(f"Error retrieving vLLM server info: {e}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument("--poisson-arrival-rate", type=float, default=1.0)
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--time-window", type=float, default=600.0)
    args = parser.parse_args()

    model_name = args.model_name
    openai_api_key = args.api_key
    openai_api_base = args.base_url
    poisson_arrival_rate = args.poisson_arrival_rate
    reasoning = args.reasoning
    date = args.date
    tensor_parallel_size = args.tensor_parallel_size
    T = args.time_window

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    tokenizers = transformers.AutoTokenizer.from_pretrained(model_name)

    # Get vLLM server info before starting the test
    asyncio.run(get_vllm_server_info(openai_api_base))

    # Load a dataset to sample from (adjust as needed)
    input_options = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")[
        "train"
    ]

    # Run the scheduling loop
    records = asyncio.run(
        schedule_messages(
            client=client,
            model_name=model_name,
            input_options=input_options,
            poisson_arrival_rate=poisson_arrival_rate,
            tokenizers=tokenizers,
            date=date,
            reasoning=reasoning,
            T=T,
            tensor_parallel_size=tensor_parallel_size,
            base_url=openai_api_base,
        )
    )

    # Save final results to disk
    df = pd.DataFrame(records)
    final_csv = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}_d{date}_final.csv"
    df.to_csv(final_csv, index=False)
    print(f"Final results saved to {final_csv}")

    # Generate additional statistics
    if not df.empty:
        print("\nPerformance Statistics:")
        print(f"Total requests processed: {len(df)}")
        print(f"Average E2E latency: {df['E2E Latency'].mean():.2f}s")
        print(f"Average batch size: {df['Batch Size'].mean():.2f}")
        print(f"Max batch size: {df['Batch Size'].max()}")
        print(
            f"Average prefill throughput: {df['Prefill Throughput'].mean():.2f} tokens/s"
        )
        print(
            f"Average decode throughput: {df['Decode Throughput'].mean():.2f} tokens/s"
        )

        # Add throughput calculations at batch level
        input_tokens_sum = df["Input Tokens"].sum()
        output_tokens_sum = df["Output Tokens"].sum()
        prefill_time_total = (
            df["Prefill Time"].sum()
            if "Prefill Time" in df.columns and df["Prefill Time"].sum() > 0
            else (
                input_tokens_sum / df["Prefill Throughput"].mean()
                if df["Prefill Throughput"].mean() > 0
                else 0
            )
        )
        decode_time_total = (
            df["Decode Time"].sum()
            if "Decode Time" in df.columns and df["Decode Time"].sum() > 0
            else (
                output_tokens_sum / df["Decode Throughput"].mean()
                if df["Decode Throughput"].mean() > 0
                else 0
            )
        )

        print(f"\nAggregate Throughput Metrics:")
        print(f"Total input tokens processed: {input_tokens_sum}")
        print(f"Total output tokens processed: {output_tokens_sum}")
        print(f"Total prefill time (measured): {prefill_time_total:.2f}s")
        print(f"Total decode time (measured): {decode_time_total:.2f}s")

        # Calculate aggregate throughput
        aggregate_prefill_throughput = (
            input_tokens_sum / prefill_time_total if prefill_time_total > 0 else 0
        )
        aggregate_decode_throughput = (
            output_tokens_sum / decode_time_total if decode_time_total > 0 else 0
        )

        if aggregate_prefill_throughput > 0:
            print(
                f"Aggregate prefill throughput: {aggregate_prefill_throughput:.2f} tokens/s"
            )
        else:
            print("Aggregate prefill throughput: N/A")

        if aggregate_decode_throughput > 0:
            print(
                f"Aggregate decode throughput: {aggregate_decode_throughput:.2f} tokens/s"
            )
        else:
            print("Aggregate decode throughput: N/A")

        # Generate a more detailed performance report
        report_file = f"perf_report_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}.txt"
        with open(report_file, "w") as f:
            f.write(f"Performance Report for {model_name}\n")
            f.write(f"================================\n\n")
            f.write(f"Test Parameters:\n")
            f.write(f"- Poisson arrival rate: {poisson_arrival_rate}\n")
            f.write(f"- Tensor parallel size: {tensor_parallel_size}\n")
            f.write(f"- Time window: {T}s\n\n")

            f.write(f"Test Summary:\n")
            f.write(f"- Total requests: {len(df)}\n")
            f.write(f"- Total input tokens: {input_tokens_sum}\n")
            f.write(f"- Total output tokens: {output_tokens_sum}\n")
            f.write(
                f"- Avg tokens per request: {(input_tokens_sum + output_tokens_sum) / len(df):.2f}\n\n"
            )

            f.write(f"Latency Metrics:\n")
            f.write(f"- Average E2E latency: {df['E2E Latency'].mean():.2f}s\n")
            f.write(f"- P50 latency: {df['E2E Latency'].quantile(0.5):.2f}s\n")
            f.write(f"- P90 latency: {df['E2E Latency'].quantile(0.9):.2f}s\n")
            f.write(f"- P99 latency: {df['E2E Latency'].quantile(0.99):.2f}s\n\n")

            f.write(f"Throughput Metrics:\n")
            f.write(
                f"- Average prefill throughput: {df['Prefill Throughput'].mean():.2f} tokens/s\n"
            )
            f.write(
                f"- Average decode throughput: {df['Decode Throughput'].mean():.2f} tokens/s\n"
            )
            f.write(
                f"- Max prefill throughput: {df['Prefill Throughput'].max():.2f} tokens/s\n"
            )
            f.write(
                f"- Max decode throughput: {df['Decode Throughput'].max():.2f} tokens/s\n"
            )

            # Add aggregate throughput metrics
            if aggregate_prefill_throughput > 0:
                f.write(
                    f"- Aggregate prefill throughput: {aggregate_prefill_throughput:.2f} tokens/s\n"
                )
            else:
                f.write("- Aggregate prefill throughput: N/A\n")

            if aggregate_decode_throughput > 0:
                f.write(
                    f"- Aggregate decode throughput: {aggregate_decode_throughput:.2f} tokens/s\n\n"
                )
            else:
                f.write("- Aggregate decode throughput: N/A\n\n")

            f.write(f"Batch Metrics:\n")
            f.write(f"- Average batch size: {df['Batch Size'].mean():.2f}\n")
            f.write(f"- Max batch size: {df['Batch Size'].max()}\n")
            f.write(f"- Batch size P50: {df['Batch Size'].quantile(0.5):.2f}\n")
            f.write(f"- Batch size P90: {df['Batch Size'].quantile(0.9):.2f}\n\n")

            # Add section about throughput by batch size if we have enough data
            if len(df) > 10 and df["Batch Size"].max() > 1:
                f.write(f"Throughput Analysis by Batch Size:\n")
                batch_groups = df.groupby(
                    pd.cut(
                        df["Batch Size"],
                        bins=[0, 1, 2, 4, 8, 16, 32, 64, 128, float("inf")],
                    )
                )
                for batch_range, group in batch_groups:
                    if len(group) > 0:
                        f.write(
                            f"- Batch size {batch_range}: Prefill={group['Prefill Throughput'].mean():.2f} tokens/s, "
                        )
                        f.write(
                            f"Decode={group['Decode Throughput'].mean():.2f} tokens/s, "
                        )
                        f.write(f"Requests={len(group)}\n")
                f.write("\n")

            f.write(f"Token Distribution:\n")
            f.write(f"- Average input tokens: {df['Input Tokens'].mean():.2f}\n")
            f.write(f"- Average output tokens: {df['Output Tokens'].mean():.2f}\n")
            f.write(f"- Max input tokens: {df['Input Tokens'].max()}\n")
            f.write(f"- Max output tokens: {df['Output Tokens'].max()}\n")

        print(f"Performance report saved to {report_file}")
