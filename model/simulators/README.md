# Simulators

This directory contains simulation components for generating realistic LLM inference workloads and converting them to power traces at various scales.

## Files

### `arrival_simulator.py` - Request Arrival and Processing Simulation

Core module for workload generation and request processing simulation. Integrates with ServeGen for realistic workload patterns.

#### Configuration

```python
@dataclass
class ServingConfig:
    """
    Configuration for an LLM serving deployment.

    Attributes:
        model_name: Model identifier (e.g., "llama-3.1")
        model_size_b: Model size in billions of parameters
        tensor_parallelism: Number of GPUs for tensor parallelism
        hardware: GPU type ("A100" or "H100")
        max_batch_size: Maximum concurrent requests
        prefill_tps: Prefill throughput (tokens/sec)
        decode_tps: Decode throughput (tokens/sec)
        idle_power: Idle power consumption (watts)
        peak_power: Peak power consumption (watts)
        perf_db_model_name: Override for performance database lookup
    """
```

**Configuration Factories:**

```python
def create_llama_config(
    size_b: int,           # 8, 70, or 405
    tp: int = 1,           # Tensor parallelism
    hardware: str = "H100"
) -> ServingConfig:
    """Create configuration for Llama-3 models."""

def create_deepseek_config(
    size_b: int,           # 8 or 70 (distilled)
    tp: int = 1,
    hardware: str = "H100"
) -> ServingConfig:
    """Create configuration for DeepSeek-R1-Distill models."""
```

#### Request Processing

```python
@dataclass
class ServeGenRequest:
    """
    A single inference request with timing information.

    Attributes:
        request_id: Unique identifier
        arrival_time: When request arrived (seconds)
        input_tokens: Number of input/context tokens
        output_tokens: Number of output/generated tokens
        prefill_start/end: Prefill phase timing (filled by simulator)
        decode_start/end: Decode phase timing (filled by simulator)
    """

class ServingSystemSimulator:
    """
    Simulates request processing through an LLM serving system.

    Methods:
        simulate_request_processing(requests) -> List[ServeGenRequest]
            Process requests and fill in timing information
    """
```

#### Power Simulation

```python
class ServeGenPowerSimulator:
    """
    End-to-end power simulation from workload specification.

    Methods:
        simulate_power_trace(
            category: str,           # "language", "reason", "multimodal"
            model_type: str,         # "m-small", "m-mid", "m-large"
            duration: float,         # Simulation duration (seconds)
            rate_requests_per_sec: float,
            output_dt: float = 0.1,  # Output resolution
            seed: int = 0,
        ) -> Dict[str, np.ndarray]
            Returns dict with 'timestamps' and 'watts' arrays
    """
```

#### Workload Generation

```python
class FastWorkloadGenerator:
    """
    Fast synthetic workload generator (no ServeGen dependency).

    Generates Poisson arrivals with exponential token distributions.
    """

class ServeGenWorkloadGenerator:
    """
    Realistic workload generator using ServeGen.

    Supports bursty arrivals, shifting distributions, and
    category-specific patterns (language, reasoning, multimodal).
    """
```

#### Performance Sampling

```python
class PerformanceSampler:
    """
    Samples TTFT and TPOT from performance database.

    Falls back to analytical estimates if database entry not found.
    Uses fitted distributions from real benchmark data.
    """
```

### `server_power_simulator.py` - Single Server Power Simulation

Simulates power consumption for a single inference server.

```python
class ServerPowerSimulator:
    """
    Power simulation for a single LLM serving node.

    Args:
        config: ServingConfig for the model/hardware

    Methods:
        simulate_server_power(
            category: str = "language",
            model_type: str = "m-small",
            duration: float = 600.0,
            rate_requests_per_sec: float = 1.0,
            output_dt: float = 0.1,
            seed: int = 0,
            fast_workload: bool = False,
            pre_generated_requests: List[ServeGenRequest] = None,
        ) -> Dict[str, Any]

    Returns:
        {
            'timestamps': np.ndarray,     # Time points
            'watts': np.ndarray,          # Power values
            'profile_seconds': float,     # Total simulation time
            'requests': List[ServeGenRequest],  # Processed requests
        }
    """
```

### `datacenter_simulator.py` - Datacenter-Scale Power Simulation

Multi-node datacenter simulation with hierarchical organization.

#### Topology Specification

```python
@dataclass
class RowSpec:
    """
    Specification for a datacenter row.

    Attributes:
        name: Row identifier (e.g., "row1")
        capacity_kw: Power capacity in kilowatts
        num_racks_a100: Number of A100 racks
        num_racks_h100: Number of H100 racks
    """
```

#### Simulator

```python
class DataCenterSimulator:
    """
    Datacenter-scale power simulation.

    Hierarchy: Datacenter -> Rows -> Racks -> Nodes -> Servers

    Args:
        rows: List of RowSpec defining the topology

    Methods:
        assign_jobs(job_mix: Dict[str, float])
            Assign workload mix to servers (e.g., {"llama-3-70b_h100_tp8": 0.5})

        simulate(
            duration: float,
            diurnal_min_qps_per_server: float,
            diurnal_max_qps_per_server: float,
            start_of_day_sec: float = 0.0,
            output_dt: float = 1.0,
            base_seed: int = 0,
            return_profiles: bool = False,
            fast_workload: bool = True,
            num_workers: int = 8,
            diurnal_sharpness_gamma: float = 1.0,
            return_per_rack: bool = False,
            sample_row_name: str = None,
            sample_rack_id: str = None,
            sample_node_id: str = None,
            sample_server_replica_index: int = None,
            verbose: bool = False,
            global_trace_requests: List[ServeGenRequest] = None,
            trace_model_routing: Dict[str, float] = None,
        ) -> Dict[str, Any]

    Returns:
        {
            'timestamps': np.ndarray,
            'datacenter_watts': np.ndarray,
            'per_row': Dict[str, np.ndarray],
            'per_rack_by_row': Dict[str, Dict[str, np.ndarray]],  # if return_per_rack
            'sampled_*_watts': np.ndarray,  # if sample_* specified
            'datacenter_tokens_in': np.ndarray,
            'datacenter_tokens_out': np.ndarray,
            'per_row_tokens_in': Dict[str, np.ndarray],
            'per_row_tokens_out': Dict[str, np.ndarray],
        }
    """
```

## Usage Examples

### Single Server Simulation

```python
from model.simulators.arrival_simulator import create_llama_config
from model.simulators.server_power_simulator import ServerPowerSimulator

# Configure
config = create_llama_config(size_b=8, tp=1, hardware="H100")
simulator = ServerPowerSimulator(config)

# Run simulation
result = simulator.simulate_server_power(
    category="language",
    model_type="m-small",
    duration=600.0,
    rate_requests_per_sec=1.0,
    output_dt=0.1,
)

# Access results
print(f"Peak power: {result['watts'].max():.0f} W")
print(f"Mean power: {result['watts'].mean():.0f} W")
```

### Datacenter Simulation

```python
from model.simulators.datacenter_simulator import DataCenterSimulator, RowSpec

# Define topology
rows = [
    RowSpec(name="row1", capacity_kw=600, num_racks_a100=0, num_racks_h100=10),
    RowSpec(name="row2", capacity_kw=600, num_racks_a100=10, num_racks_h100=0),
]

# Create simulator
dc = DataCenterSimulator(rows)

# Assign workloads
dc.assign_jobs({
    "llama-3-70b_h100_tp8": 0.6,
    "llama-3-8b_h100_tp1": 0.4,
})

# Run simulation
result = dc.simulate(
    duration=3600.0,
    diurnal_min_qps_per_server=0.1,
    diurnal_max_qps_per_server=2.0,
    output_dt=1.0,
    num_workers=8,
)

# Access results
print(f"Peak DC power: {result['datacenter_watts'].max() / 1e6:.2f} MW")
```

### Trace Replay

```python
from model.simulators.arrival_simulator import ServeGenRequest
from model.simulators.datacenter_simulator import DataCenterSimulator, RowSpec

# Load trace
requests = [
    ServeGenRequest(i, arrival_time=t, input_tokens=inp, output_tokens=out)
    for i, (t, inp, out) in enumerate(trace_data)
]

# Configure and run
dc = DataCenterSimulator([RowSpec("row1", 500, 0, 8)])
dc.assign_jobs({"llama-3-70b_h100_tp8": 0.75, "llama-3-8b_h100_tp1": 0.25})

result = dc.simulate(
    duration=86400.0,
    global_trace_requests=requests,
    trace_model_routing={"llama-3-70b_h100_tp8": 0.75, "llama-3-8b_h100_tp1": 0.25},
)
```

## Diurnal Patterns

The datacenter simulator supports diurnal (time-of-day) load patterns:

- `diurnal_min_qps_per_server`: Minimum QPS during low-traffic periods
- `diurnal_max_qps_per_server`: Maximum QPS during peak traffic
- `start_of_day_sec`: Offset for time-of-day calculation
- `diurnal_sharpness_gamma`: Controls transition sharpness (higher = sharper)

The load follows a sinusoidal pattern peaking around midday.

## Job Configuration Format

Job names follow the pattern: `{model}_{hardware}_tp{n}`

**Supported configurations:**
- `llama-3-8b_a100_tp1`, `llama-3-8b_h100_tp1`
- `llama-3-70b_a100_tp4`, `llama-3-70b_h100_tp8`
- `deepseek-r1-distill-8b_a100_tp1`, `deepseek-r1-distill-70b_h100_tp8`
- And more (see performance database)
