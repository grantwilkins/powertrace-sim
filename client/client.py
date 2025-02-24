"""
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --enable-reasoning --reasoning-parser deepseek_r1
```
"""

from openai import OpenAI
import datasets
import random
import transformers
import time
import numpy as np
import pandas as pd
import argparse
import asyncio


async def send_message(
    client: OpenAI,
    model_name: str,
    input_options: datasets.DatasetDict,
    df: pd.DataFrame,
    poisson_arrival_rate: float,
    reasoning: bool = False,
) -> dict:

    chosen_input = random.choice(input_options)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": chosen_input},
    ]
    start_time = time.time()
    loop = asyncio.get_event_loop()
    response_raw = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(model=model_name, messages=messages),
    )
    e2e_time = time.time() - start_time

    response = response_raw.choices[0]
    reasoning_content = response.reasoning_content if reasoning else ""
    content = response.message.content
    print(reasoning_content)
    print(content)

    record = {
        "Request Time": start_time,
        "Model": model_name,
        "Input": chosen_input,
        "Poisson Arrival Rate": poisson_arrival_rate,
        "Input Tokens": len(tokenizers.encode(chosen_input)),
        "Output Tokens (Response)": len(tokenizers.encode(content)),
        "Reasoning Tokens": len(tokenizers.encode(reasoning_content)),
        "E2E Latency": e2e_time,
    }

    return record


async def schedule_messages(
    client: OpenAI,
    model_name: str,
    input_options: datasets.DatasetDict,
    df: pd.DataFrame,
    poisson_arrival_rate: float,
    reasoning: bool = False,
    T: int = 600,
):
    tasks = []
    start_time = time.time()
    while time.time() < start_time + T:
        tasks.append(
            asyncio.create_task(
                send_message(
                    client=client,
                    model_name=model_name,
                    input_options=input_options,
                    df=df,
                    poisson_arrival_rate=poisson_arrival_rate,
                    reasoning=reasoning,
                )
            )
        )
        await asyncio.sleep(np.random.exponential(scale=1.0 / poisson_arrival_rate))
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=bool, default=False)
    parser.add_argument("--poisson_arrival_rate", type=float, default=1.0)
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    args = parser.parse_args()
    model_name = args.model_name
    openai_api_key = args.api_key
    openai_api_base = args.base_url
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    tokenizers = transformers.AutoTokenizer.from_pretrained(model_name)
    df = pd.DataFrame()
    assert client.chat is not None
    print(client.models.list())
    #assert model_name in client.models.list()

    input_options = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")[
        "train"
    ]["instruction"]

    T = 600  # seconds our experiment will run
    poisson_arrival_rate = args.poisson_arrival_rate
    reasoning = args.reasoning
    records = asyncio.run(
        schedule_messages(
            client=client,
            model_name=model_name,
            input_options=input_options,
            df=df,
            poisson_arrival_rate=poisson_arrival_rate,
            reasoning=reasoning,
            T=T,
        )
    )
    df = pd.DataFrame(records)
    df.to_csv(f"results_{model_name}_{poisson_arrival_rate}.csv", index=False)
