"""
Claim:
Validate runs preserve the declared real prompts, mean rate, and Gamma arrival
shape in both the benchmark command and the scientific manifest.

Plausible wrong implementations:
- Drop burstiness and silently execute every pattern as Poisson.
- Change the mean rate when changing Gamma shape.
- Record a pattern in the manifest that differs from the executed command.
- Lose the epoch-aligned detailed request output.
"""

import validate_runner


def _workload():
    return {"dataset": "sharegpt", "num_prompts": 200, "request_rate": 4}


def test_build_validate_command_uses_real_dataset_and_arrival_rate():
    cmd = validate_runner.build_validate_command(
        "Qwen/Qwen3-8B", "http://localhost:8000/v1", 1, _workload(),
        "/data/sharegpt.json", "/tmp/validate.json")
    s = " ".join(cmd)
    # real dataset, not synthetic "random"
    assert "--dataset-name sharegpt" in s
    assert "--dataset-path /data/sharegpt.json" in s
    assert "--dataset-name random" not in s
    # synthetic Poisson arrival schedule
    assert "--request-rate 4" in s
    assert "--burstiness 1.0" in s
    assert "--num-prompts 200" in s
    # epoch-aligned detailed output (the alignment contract)
    assert "--save-detailed" in s and "--save-result" in s
    assert "--result-filename /tmp/validate.json" in s


def test_build_validate_command_preserves_nonpoisson_shape():
    workload = dict(_workload(), request_rate=2.5, burstiness=0.25)
    cmd = validate_runner.build_validate_command(
        "Qwen/Qwen3-8B", "http://localhost:8000/v1", 1, workload,
        "/data/sharegpt.json", "/tmp/validate.json")
    assert cmd[cmd.index("--request-rate") + 1] == "2.5"
    assert cmd[cmd.index("--burstiness") + 1] == "0.25"


def test_build_validate_command_avoids_doubled_v1_in_url():
    """base_url keeps /v1 for the loggers; the bench URL must not become /v1/v1."""
    cmd = validate_runner.build_validate_command(
        "Qwen/Qwen3-8B", "http://localhost:8000/v1", 1, _workload(),
        "/data/sharegpt.json", "/tmp/validate.json")
    i = cmd.index("--base-url")
    assert cmd[i + 1] == "http://localhost:8000"
    assert "--endpoint" in cmd and cmd[cmd.index("--endpoint") + 1] == "/v1/completions"


def test_build_validate_command_forwards_max_model_len():
    base = validate_runner.build_validate_command(
        "Qwen/Qwen3-8B", "http://localhost:8000/v1", 1, _workload(),
        "/data/sharegpt.json", "/tmp/validate.json")
    assert "--max-model-len" not in base  # omitted when not given (back-compat)
    cmd = validate_runner.build_validate_command(
        "Qwen/Qwen3-8B", "http://localhost:8000/v1", 1, _workload(),
        "/data/sharegpt.json", "/tmp/validate.json", max_model_len=32768)
    i = cmd.index("--max-model-len")
    assert cmd[i + 1] == "32768"


def test_build_validate_window_records_epoch_bounds():
    w = validate_runner.build_validate_window(
        _workload(), 1000.0, 1075.0, ["cmd"], {"completed": 200})
    assert w["t_start_epoch"] == 1000.0 and w["t_end_epoch"] == 1075.0
    assert w["num_prompts"] == 200
    assert w["params"]["dataset"] == "sharegpt"
    assert w["params"]["request_rate"] == 4
    assert w["params"]["burstiness"] == 1.0
    assert w["summary"]["completed"] == 200


def test_validate_window_records_executed_arrival_shape():
    workload = dict(_workload(), request_rate=2.5, burstiness=4.0)
    window = validate_runner.build_validate_window(
        workload, 1000.0, 1075.0, ["cmd"], {"completed": 200})
    assert window["params"]["request_rate"] == 2.5
    assert window["params"]["burstiness"] == 4.0
    assert window["label"].endswith("_rate2.5_burst4.0")
