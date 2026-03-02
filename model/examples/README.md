# Examples

This directory contains runnable demonstration scripts showing how to use the PowerTrace-Sim components.

## Scripts

### `servegen_demo.py` - ServeGen Integration Demo

Demonstrates how to use ServeGen for workload generation and convert workloads to power traces.

```bash
python -m model.examples.servegen_demo
```

**Features demonstrated:**
- Creating serving configurations with `create_llama_config()`, `create_deepseek_config()`
- Generating workloads with ServeGen
- Converting request streams to power traces
- Visualizing results

### `validate_server_power.py` - Server Power Simulation Validation

Validates the server power simulator against expected behavior.

```bash
python -m model.examples.validate_server_power
```

**Output:**
- CSV file with power trace: `training_results/server_power_validation_*.csv`
- PNG plot (if matplotlib available): `training_results/server_power_validation_*.png`

**Configuration:**
- Model: Llama-3 8B
- Hardware: H100
- Tensor Parallelism: 2
- Duration: 600 seconds
- Request rate: 0.25 QPS

### `validate_datacenter_power.py` - Datacenter Simulation Validation

Validates the datacenter-level power simulation with multiple rows and racks.

```bash
python -m model.examples.validate_datacenter_power
```

**Output directory:** `training_results/dc_pilot/`

**Contents:**
- Per-row power CSVs: `row{n}_power.csv`
- Per-rack power CSVs: `row{n}__rack{id}_power.csv`
- Datacenter total: `datacenter_power.csv`
- Token throughput CSVs: `*_tokens_in.csv`, `*_tokens_out.csv`
- Sampled node/server traces

**Configuration:**
- 1 row with 10 A100 racks + 5 H100 racks
- Mixed job workload (Llama, DeepSeek variants)
- 4-hour simulation with diurnal pattern
- 0.25s output resolution

### `datacenter_24h_10mw.py` - 24-Hour Datacenter Simulation

Large-scale simulation using Azure trace replay.

```bash
python -m model.examples.datacenter_24h_10mw
```

**Requirements:**
- Azure trace file at configured path (default: `~/one_day_code.csv`)

**Output directory:** `training_results/dc_24h_10mw/`

**Contents:**
- 1-second resolution: `datacenter_power_1s.csv`
- 15-minute averages: `datacenter_power_15min.csv`

**Configuration:**
- 8 H100 racks (32 nodes)
- 75% Llama-70B + 25% Llama-8B workload mix
- Full 24-hour trace replay
- Reports peak/mean/std power in MW

### `analyze_request_timings.py` - Request Timing Analysis

Analyzes TTFT, TPOT, and latency distributions for given configurations.

```bash
python -m model.examples.analyze_request_timings \
    --model llama-3-70b \
    --hardware H100 \
    --tp 8 \
    --duration 3600 \
    --rate 1.0
```

**Arguments:**
- `--model`: Model spec (e.g., llama-3-8b, deepseek-r1-70b)
- `--hardware`: Hardware type (A100 or H100)
- `--tp`: Tensor parallelism
- `--workload`: Workload category (language, reason, multimodal)
- `--duration`: Simulation duration in seconds
- `--rate`: Request arrival rate (QPS)
- `--use-fast`: Use fast synthetic workload (no ServeGen)
- `--csv-out`: Optional path for per-request metrics CSV

**Output:**
```
Metric                 | Value
-----------------------+-----------------
TTFT (milliseconds)
  P50                 | 45.2
  P95                 | 123.4
-----------------------+-----------------
TPOT (milliseconds)
  P50                 | 12.3
  P95                 | 28.7
-----------------------+-----------------
Request Latency (seconds)
  P50                 | 2.45
  P95                 | 8.12
  P99                 | 15.67
```

## Common Patterns

### Creating a Serving Configuration

```python
from model.simulators.arrival_simulator import create_llama_config

config = create_llama_config(
    size_b=70,           # Model size in billions
    tp=8,                # Tensor parallelism
    hardware="H100"      # GPU type
)
```

### Running Server Power Simulation

```python
from model.simulators.arrival_simulator import create_llama_config
from model.simulators.server_power_simulator import ServerPowerSimulator

config = create_llama_config(size_b=8, tp=1, hardware="H100")
sim = ServerPowerSimulator(config)

result = sim.simulate_server_power(
    category="language",
    model_type="m-small",
    duration=600.0,
    rate_requests_per_sec=1.0,
    output_dt=0.25,
)

timestamps = result["timestamps"]
power_watts = result["watts"]
```

### Running Datacenter Simulation

```python
from model.simulators.datacenter_simulator import DataCenterSimulator, RowSpec

rows = [
    RowSpec(name="row1", capacity_kw=600, num_racks_a100=10, num_racks_h100=5),
]
dc = DataCenterSimulator(rows)

job_mix = {
    "llama-3-70b_h100_tp8": 0.5,
    "llama-3-8b_h100_tp1": 0.5,
}
dc.assign_jobs(job_mix)

result = dc.simulate(
    duration=3600.0,
    diurnal_min_qps_per_server=0.1,
    diurnal_max_qps_per_server=2.0,
    output_dt=1.0,
)
```
