import sys
from pathlib import Path

# timing-test/ (for build_timing_dataset) and repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
