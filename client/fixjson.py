import json
import os
from math import isnan
from pathlib import Path


def is_nan_like(x):
    # Treat float NaN, None, and strings like "nan"/"NaN" as NaN-like
    if x is None:
        return True
    if isinstance(x, float) and isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() == "nan":
        return True
    return False


def to_float_or_nan(x):
    # Normalize values to float; unknowns -> NaN
    try:
        if is_nan_like(x):
            return float("nan")
        if isinstance(x, str):
            return float(x.strip())
        return float(x)
    except Exception:
        return float("nan")


def fill_nans_nearest_1d(arr):
    """
    Replace NaNs in a 1D list with the nearest numeric neighbor by index.
    Tie-breaking chooses the left neighbor.
    If the entire list is NaN, returns the list unchanged.
    """
    n = len(arr)
    if n == 0:
        return arr

    a = [to_float_or_nan(v) for v in arr]
    if all(isnan(v) for v in a):
        return arr  # nothing to fill

    # Left pass: value and distance to nearest numeric on the left
    left_val = [None] * n
    left_dist = [float("inf")] * n
    last_val, last_idx = None, None
    for i in range(n):
        if not isnan(a[i]):
            last_val, last_idx = a[i], i
            left_val[i] = a[i]
            left_dist[i] = 0
        elif last_idx is not None:
            left_val[i] = last_val
            left_dist[i] = i - last_idx

    # Right pass: value and distance to nearest numeric on the right
    right_val = [None] * n
    right_dist = [float("inf")] * n
    next_val, next_idx = None, None
    for i in range(n - 1, -1, -1):
        if not isnan(a[i]):
            next_val, next_idx = a[i], i
            right_val[i] = a[i]
            right_dist[i] = 0
        elif next_idx is not None:
            right_val[i] = next_val
            right_dist[i] = next_idx - i

    # Fill NaNs by choosing closer neighbor (tie -> left)
    for i in range(n):
        if isnan(a[i]):
            ld, rd = left_dist[i], right_dist[i]
            lv, rv = left_val[i], right_val[i]
            if ld < rd and lv is not None:
                a[i] = lv
            elif rd < ld and rv is not None:
                a[i] = rv
            else:
                # tie or one side missing -> prefer left if available
                a[i] = lv if lv is not None else rv

    return a


def process_json_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)

    changed = False
    if "itls" in data and isinstance(data["itls"], list):
        original = data["itls"]
        filled = fill_nans_nearest_1d(original)
        # Only consider it changed if anything actually changed
        if filled != original:
            data["itls"] = filled
            changed = True

    if changed:
        # Write to a backup first, then atomically replace original
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        with open(backup_path, "w") as backup:
            json.dump(data, backup, indent=2, allow_nan=False)  # ensure valid JSON
        os.replace(backup_path, file_path)
        print(f"Processed: {file_path}")
    else:
        print(f"Skipped (no change or no 'itls'): {file_path}")


def main():
    for file in Path(".").glob("*.json"):
        try:
            process_json_file(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    main()
