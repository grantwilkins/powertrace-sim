"""Extended ``nvidia-smi`` power logger (Tier-0 instrumentation, CAMPAIGN.md §5-A).

Each row includes stable ``index`` and ``uuid`` identity plus ``clocks.sm``
(DVFS is the largest unmodeled term and is a free field),
``clocks.mem``, ``utilization.memory`` and ``temperature.gpu``, per GPU at 4 Hz.

The bundle parser groups rows by a bounded capture window and UUID, validates a
stable UUID-to-index mapping, and rejects topology drift; it never infers samples
from anonymous row blocks.

Only the command/argv construction and the small timestamping wrapper live here.
The wrapper runs one ``nvidia-smi`` query per sample and stamps every GPU row in
that query with the same wall timestamp, so busy-node per-row ``nvidia-smi``
timestamp skew cannot violate the bundle contract.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import signal
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

# Stable identity is part of every row; ingestion groups bounded capture windows.
EXTENDED_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "temperature.gpu",
)

DEFAULT_INTERVAL_MS = 250  # 4 Hz, aligned to the engine /metrics scraper

DISPLAY_FIELDS = {
    "power.draw": "power.draw [W]",
    "clocks.sm": "clocks.current.sm [MHz]",
    "clocks.mem": "clocks.current.memory [MHz]",
    "utilization.gpu": "utilization.gpu [%]",
    "utilization.memory": "utilization.memory [%]",
    "memory.used": "memory.used [MiB]",
}

_STOP = False


def _query_fields(fields=EXTENDED_FIELDS) -> tuple[str, ...]:
    return tuple(field for field in fields if field != "timestamp")


def display_header(fields=EXTENDED_FIELDS) -> list[str]:
    return [DISPLAY_FIELDS.get(field, field) for field in fields]


def nvidia_smi_query_command(fields=EXTENDED_FIELDS) -> list[str]:
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(_query_fields(fields))}",
        "--format=csv,nounits,noheader",
    ]


def nvidia_smi_command(
    fields=EXTENDED_FIELDS, interval_ms: int = DEFAULT_INTERVAL_MS
) -> list[str]:
    """Return the argv for the live logger process."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--interval-ms",
        str(int(interval_ms)),
    ]


def write_query_rows(stream, timestamp: str, query_stdout: str) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    for row in csv.reader(StringIO(query_stdout)):
        if row:
            writer.writerow([timestamp] + [cell.strip() for cell in row])


def _timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def _stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def stream_power(interval_ms: int = DEFAULT_INTERVAL_MS) -> None:
    signal.signal(signal.SIGTERM, _stop)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(display_header())
    sys.stdout.flush()
    interval_s = int(interval_ms) / 1000.0
    while not _STOP:
        start = time.monotonic()
        timestamp = _timestamp_now()
        result = subprocess.run(
            nvidia_smi_query_command(),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        write_query_rows(sys.stdout, timestamp, result.stdout)
        sys.stdout.flush()
        remaining = interval_s - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    args = parser.parse_args()
    stream_power(args.interval_ms)


if __name__ == "__main__":
    main()
