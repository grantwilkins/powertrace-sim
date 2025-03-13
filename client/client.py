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
    """
    Sends a message to the OpenAI API and records performance metrics.
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

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    tensor_parallel_size = args.tensor_parallel_size
    T = args.time_window

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    tokenizers = transformers.AutoTokenizer.from_pretrained(model_name)

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
            reasoning=reasoning,
            T=T,
            tensor_parallel_size=tensor_parallel_size,
        )
    )

    # Save final results to disk
    df = pd.DataFrame(records)
    final_csv = f"results_{model_name.split('/')[-1]}_{poisson_arrival_rate}_{tensor_parallel_size}_final.csv"
    df.to_csv(final_csv, index=False)
    print(f"Final results saved to {final_csv}")
