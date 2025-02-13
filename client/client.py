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


def send_message(
    client: OpenAI,
    model_name: str,
    input_options: datasets.DatasetDict,
    df: pd.DataFrame,
    poisson_arrival_rate: float,
    reasoning: bool = False,
) -> pd.DataFrame:

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": random.choice(input_options)},
    ]
    start_time = time.time()
    response = client.chat.completions.create(model=model_name, messages=messages)
    end_time = time.time() - start_time

    response = response.choices[0]
    reasoning_content = response.reasoning_content if reasoning else ""
    content = response.message.content
    print(response)
    pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "Request Time": start_time,
                        "Model": model_name,
                        "Input": messages[1]["content"],
                        "Poisson Arrival Rate": poisson_arrival_rate,
                        "Input Tokens": len(tokenizers.encode(messages[1]["content"])),
                        "Output Tokens (Response)": len(tokenizers.encode(content)),
                        "Reasoning Tokens": len(tokenizers.encode(reasoning_content)),
                        "E2E Latency": end_time,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=bool, default=False)
    parser.add_argument("--poisson_arrival_rate", type=float, default=1.0)
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
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
    assert model_name in client.models.list()

    input_options = datasets.load_dataset("garage-bAInd/Open-Platypus", "default")[
        "train"
    ]["instruction"]

    T = 600  # seconds our experiment will run
    poisson_arrival_rate = args.poisson_arrival_rate
    start_time = time.time()
    end_time = start_time + (T)  # Run for T seconds
    while time.time() < end_time:
        df = send_message(
            client=client,
            model_name=model_name,
            input_options=input_options,
            df=df,
            poisson_arrival_rate=poisson_arrival_rate,
            reasoning=args.reasoning,
        )
        df.to_csv(f"{model_name}_{poisson_arrival_rate}.csv", index=False)
        interarrival_time = np.random.exponential(scale=1.0 / poisson_arrival_rate)
        time.sleep(interarrival_time)
