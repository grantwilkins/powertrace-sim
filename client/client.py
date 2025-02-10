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


def prepare_input(input_options: datasets.DatasetDict, message_count: int) -> dict:
    inputs = random.choices(input_options, k=message_count)
    for input in inputs:
        yield {"role": "user", "content": input}


def send_messages(
    client: OpenAI,
    model: str,
    input_options: datasets.DatasetDict,
    message_count: int,
    df: pd.DataFrame,
    poisson_arrival_rate: float,
    reasoning: bool = False,
):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(list(prepare_input(input_options, message_count)))

    start_time = time.time()
    response = client.chat.completions.create(model=model, messages=messages)
    end_time = time.time() - start_time

    print("reasoning_content:", reasoning_content)
    print("content:", content)
    for message in messages:
        if message.role == "user":
            response = response.choices.pop(0)
            reasoning_content = response.message.reasoning_content if reasoning else ""
            content = response.message.content
            df = df.append(
                {
                    "Request Time": start_time,
                    "Model": model,
                    "Input": message.content,
                    "Poisson Arrival Rate": poisson_arrival_rate,
                    "Message Count": message_count,
                    "Input Tokens": len(tokenizers.encode(message.content)),
                    "Output Tokens (Response)": len(tokenizers.encode(content)),
                    "Reasoning Tokens": len(tokenizers.encode(reasoning_content)),
                    "E2E Latency": end_time,
                },
                ignore_index=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=bool, default=False)
    parser.add_argument("--poisson_arrival_rate", type=float, default=1.0)
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    args = parser.parse_args()
    openai_api_key = "EMPTY"
    openai_api_base = args.base_url
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    tokenizers = transformers.AutoTokenizer.from_pretrained()
    df = pd.DataFrame()
    models = client.models.list()
    model = models.data[0].id
    input_options = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")[
        "train"
    ]["input"]
    T = 600  # seconds our experiment will run
    poisson_arrival_rate = args.poisson_arrival_rate
    start_time = time.time()
    end_time = start_time + (T)  # Run for T seconds
    while time.time() < end_time:
        message_count = 1
        send_messages(
            client,
            model,
            input_options,
            message_count,
            poisson_arrival_rate,
            df,
        )
        interarrival_time = np.random.exponential(scale=1.0 / poisson_arrival_rate)
        time.sleep(interarrival_time)
