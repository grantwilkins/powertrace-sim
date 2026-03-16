from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def derive_decode_time(itl_value: object, output_tokens: object) -> Tuple[Optional[float], str]:
    """Derive decode duration from ITL values and return the source type."""
    if isinstance(itl_value, list):
        if len(itl_value) == 0:
            return None, "list"
        try:
            arr = np.asarray(itl_value, dtype=float)
        except Exception:
            return None, "list"
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None, "list"
        return float(np.sum(arr)), "list"

    if isinstance(itl_value, (int, float)) and not isinstance(itl_value, bool):
        if not np.isfinite(float(itl_value)):
            return None, "scalar"
        try:
            out_tok = int(float(output_tokens))
        except Exception:
            return None, "scalar"
        return float(itl_value) * float(max(out_tok - 1, 1)), "scalar"

    return None, "unknown"
