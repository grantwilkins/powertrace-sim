import sys
from pathlib import Path

# profiling/jobs (campaign_config) and profiling/probes (schedule, drift guard).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
