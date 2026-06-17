import sys
from pathlib import Path

# profiling/agentic_traces (this package) and profiling/probes (agentic plan types).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
