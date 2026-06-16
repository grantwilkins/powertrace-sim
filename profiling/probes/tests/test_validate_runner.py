"""Pure tests for the validate runner (real-dataset workload command + window).

The live ``run`` (server/loggers/subprocess) is validated end-to-end on launch per
the acceptance gate; here we pin the command construction and the manifest window.
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
    assert "--num-prompts 200" in s
    # epoch-aligned detailed output (the alignment contract)
    assert "--save-detailed" in s and "--save-result" in s
    assert "--result-filename /tmp/validate.json" in s


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
    assert w["summary"]["completed"] == 200
