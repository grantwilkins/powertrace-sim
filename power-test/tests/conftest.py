import sys
from pathlib import Path

# power-test/ (for join_power) and repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
