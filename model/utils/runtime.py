from __future__ import annotations

import os
from typing import Mapping


_DEFAULT_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KMP_USE_SHM": "0",
}


def configure_threading_env(
    overrides: Mapping[str, str] | None = None,
) -> None:
    values = dict(_DEFAULT_THREAD_ENV)
    if overrides:
        values.update({str(k): str(v) for k, v in overrides.items()})
    for key, value in values.items():
        os.environ.setdefault(key, value)


__all__ = ["configure_threading_env"]
