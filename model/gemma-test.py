from transformers import pipeline
import torch
import numpy as np

data_path = "processed_data/power_trace_data.npz"
data = np.load(data_path)

power_traces = data["power_traces"]  # shape [N, T]
tensor_parallelism = data["tensor_parallelism"]  # shape [N]
poisson_rate = data["poisson_rate"]  # shape [N]
model_size = data.get("model_size", np.zeros_like(tensor_parallelism))

# For demonstration, pick just ONE row (or the test set row) you want to feed in:
row_index = 0
this_power = power_traces[row_index]  # shape [T]
this_tp = float(tensor_parallelism[row_index])
this_rate = float(poisson_rate[row_index])
this_model_size = float(model_size[row_index])

# 3) Prepare your prompt.
#    We'll try to keep it short. If your sequence is large, you might just show partial data or summarize.
max_power = 400.0 * this_tp

# We'll provide some lines about the data
prompt = f"""
You are given:
- Poisson arrival rate = {this_rate}
- Tensor parallelism = {this_tp}
- Model size = {this_model_size} (in billions of parameters)
- A typical or partial observed power sequence (in watts): {this_power[:50].tolist()}.

Your job:
1) Generate a time-series of (say) 2400 predicted power values (integers or floats) that might match these conditions.
2) The predicted power must never exceed {max_power} watts (which is 400 * TP).
3) Format your output as a JSON array of numbers (like [value1, value2, ..., value100]).

Constraints:
- Assume the system is stable and that power will vary realistically around the scale of the existing data, but never surpass {max_power}.
- You do NOT need any user to supply arrival times or token lengths. The model can just produce a plausible sequence.

Please output ONLY the JSON array of 2400 power values, nothing else.
"""

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device="mps",
    torch_dtype=torch.bfloat16,
)

messages = [
    [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Please output ONLY the JSON array of 100 power values, nothing else..",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ],
]

output = pipe(messages, max_new_tokens=2400, temperature=0.2)
for message in output:
    print(message)
