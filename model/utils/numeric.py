from __future__ import annotations

from typing import Optional

import numpy as np


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


__all__ = ["finite_float"]
