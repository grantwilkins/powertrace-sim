import os
import time
import random
import argparse
import asyncio
import numpy as np
import pandas as pd
import datasets
import transformers
from openai import OpenAI


async def send_message(
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    reasoning: bool = False,
    tensor_parallel_size: int = 1,
) -> dict:
    """Sends a message to the OpenAI API and records performance metrics."""
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
    loop = asyncio.get_event_loop()
    response_raw = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(model=model_name, messages=messages),
    )
    e2e_time = time.time() - start_time

    response = response_raw.choices[0]
    content = response.message.content

    record = {
        "Request Time": start_time,
        "Model": model_name,
        "Data Source": data_source,
        "Poisson Arrival Rate": poisson_arrival_rate,
        "Tensor Parallel Size": tensor_parallel_size,
        "Input Tokens": len(tokenizers.encode(instruction_text)),
        "Output Tokens": len(tokenizers.encode(content)),
        "E2E Latency": e2e_time,
    }

    outfile = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}.csv"
    file_exists = os.path.exists(outfile)
    df_temp = pd.DataFrame([record])
    df_temp.to_csv(outfile, mode="a", header=not file_exists, index=False)

    return record


async def schedule_task_at_time(
    target_time: float,
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    reasoning: bool = False,
    tensor_parallel_size: int = 1,
) -> dict:
    """Schedule a task to run at a specific time."""
    # Calculate how long to wait until target_time
    now = time.time()
    wait_time = max(0, target_time - now)

    if wait_time > 0:
        await asyncio.sleep(wait_time)

    # Send the message after waiting
    return await send_message(
        client=client,
        model_name=model_name,
        input_options=input_options,
        poisson_arrival_rate=poisson_arrival_rate,
        tokenizers=tokenizers,
        reasoning=reasoning,
        tensor_parallel_size=tensor_parallel_size,
    )


async def schedule_messages(
    client: OpenAI,
    model_name: str,
    input_options: datasets.arrow_dataset.Dataset,
    poisson_arrival_rate: float,
    tokenizers,
    reasoning: bool = False,
    T: float = 600.0,
    tensor_parallel_size: int = 1,
) -> list:
    """
    Schedule requests (tasks) over a time window T using a Poisson process.
    Total number of tasks is determined by the poisson_arrival_rate * T.
    """
    expected_num_tasks = int(poisson_arrival_rate * T)
    interarrival_times = np.random.exponential(
        scale=1.0 / poisson_arrival_rate, size=expected_num_tasks
    )
    arrival_times = np.cumsum(interarrival_times)
    arrival_times = arrival_times[arrival_times <= T]
    print(f"Expected number of tasks: {expected_num_tasks}")
    print(f"Actual number of tasks: {len(arrival_times)}")
    print("Arrival times:", arrival_times)
    print(
        f"Scheduling {len(arrival_times)} tasks over {T} seconds (rate: {poisson_arrival_rate}/sec)"
    )

    start_time = time.time()
    tasks = []

    # Create all tasks without waiting
    for arrival_time in arrival_times:
        # Schedule the task to run after the relative arrival time
        task = asyncio.create_task(
            schedule_task_at_time(
                start_time + arrival_time,
                client,
                model_name,
                input_options,
                poisson_arrival_rate,
                tokenizers,
                reasoning,
                tensor_parallel_size,
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    results = []
    for finished_task in asyncio.as_completed(tasks):
        try:
            result = await finished_task
            results.append(result)
        except Exception as e:
            print(f"An error occurred: {e}")

    run_time = time.time() - start_time
    print(f"Completed {len(results)} tasks in {run_time:.2f} seconds")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=bool, default=False)
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
    tensor_parallel_size = args.tensor_parallel_size
    T = args.time_window

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    tokenizers = transformers.AutoTokenizer.from_pretrained(model_name)

    # Load a dataset to sample from
    input_options = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")[
        "train"
    ]

    records = asyncio.run(
        schedule_messages(
            client=client,
            model_name=model_name,
            input_options=input_options,
            poisson_arrival_rate=poisson_arrival_rate,
            tokenizers=tokenizers,
            reasoning=reasoning,
            T=T,
            tensor_parallel_size=tensor_parallel_size,
        )
    )

    df = pd.DataFrame(records)
    final_csv = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}_final.csv"
    df.to_csv(final_csv, index=False)
    print(f"Final results saved to {final_csv}")
