from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BENCH_FILE_RE = re.compile(
    r"^(?P<model>.+)_tp(?P<tp>\d+)_rate(?P<rate>[\d.]+)_iter(?P<iter>\d+)_(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(?P<ext>json|csv)$"
)
SHAREGPT_JSON_RE = re.compile(
    r"^vllm-(?P<rate>[\d.]+)qps-tp(?P<tp>\d+)-(?P<served_model>.+)-(?P<date>\d{8}-\d{6})\.json$"
)
SHAREGPT_CSV_RE = re.compile(
    r"^(?P<model>.+)_tp(?P<tp>\d+)_p(?P<rate>[\d.]+)_d(?P<date>\d{8}-\d{6}|\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv$"
)
HYPHENATED_DATE_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})-(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})$"
)


@dataclass(frozen=True)
class MatchedPair:
    family: str
    dataset_dir: str
    model_name: str
    hardware: str
    tensor_parallelism: int
    rate: str
    iteration: Optional[int]
    date_key: str
    pair_key: str
    power_csv_path: str
    json_path: str


def canonical_rate(value: object) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def normalize_sharegpt_date(date_str: str) -> str:
    if re.fullmatch(r"\d{8}-\d{6}", date_str):
        return date_str
    match = HYPHENATED_DATE_RE.fullmatch(date_str)
    if not match:
        return date_str
    return (
        f"{match.group('y')}{match.group('m')}{match.group('d')}-"
        f"{match.group('h')}{match.group('mi')}{match.group('s')}"
    )


def parse_benchmark_filename(filename: str) -> Optional[Dict[str, object]]:
    match = BENCH_FILE_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "model": match.group("model"),
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "iteration": int(match.group("iter")),
        "date": match.group("date"),
        "ext": match.group("ext"),
    }


def parse_sharegpt_json_filename(filename: str) -> Optional[Dict[str, object]]:
    match = SHAREGPT_JSON_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "date": normalize_sharegpt_date(match.group("date")),
        "served_model": match.group("served_model"),
        "ext": "json",
    }


def parse_sharegpt_csv_filename(filename: str) -> Optional[Dict[str, object]]:
    match = SHAREGPT_CSV_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "model": match.group("model"),
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "date": normalize_sharegpt_date(match.group("date")),
        "ext": "csv",
    }


def parse_dataset_dir_metadata(dir_name: str, family: str) -> Optional[Tuple[str, str]]:
    if family == "benchmark":
        prefix = "benchmark-"
    elif family == "sharegpt-benchmark":
        prefix = "sharegpt-benchmark-"
    else:
        return None
    if not dir_name.startswith(prefix):
        return None
    body = dir_name[len(prefix) :]
    pieces = body.split("-")
    if len(pieces) < 2:
        return None
    hardware = pieces[-1].upper()
    model_name = "-".join(pieces[:-1])
    return model_name, hardware


def discover_dataset_dirs(data_root_dir: str, include_families: Sequence[str]) -> List[Tuple[str, str]]:
    root = Path(data_root_dir)
    discovered: List[Tuple[str, str]] = []
    if not root.exists():
        return discovered

    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        for family in include_families:
            if entry.name.startswith(f"{family}-"):
                discovered.append((family, str(entry)))
                break
    return discovered


def _benchmark_key(parsed: Dict[str, object]) -> Tuple[object, ...]:
    return (
        str(parsed["model"]),
        int(parsed["tp"]),
        str(parsed["rate"]),
        int(parsed["iteration"]),
        str(parsed["date"]),
    )


def _sharegpt_key(parsed: Dict[str, object]) -> Tuple[object, ...]:
    return (
        int(parsed["tp"]),
        str(parsed["rate"]),
        str(parsed["date"]),
    )


def discover_pairs_for_dataset(
    family: str,
    dataset_dir: str,
) -> Tuple[List[Dict[str, object]], List[MatchedPair], Dict[str, object]]:
    dataset_name = Path(dataset_dir).name
    metadata = parse_dataset_dir_metadata(dataset_name, family)
    model_from_dir, hardware_from_dir = metadata if metadata else ("unknown", "UNKNOWN")

    json_files: Dict[Tuple[object, ...], str] = {}
    power_files: Dict[Tuple[object, ...], str] = {}
    dup_json = 0
    dup_power = 0
    ignored = 0

    for file_path in sorted(Path(dataset_dir).rglob("*"), key=lambda p: str(p)):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".json", ".csv"}:
            continue

        parsed: Optional[Dict[str, object]]
        key: Optional[Tuple[object, ...]]

        if family == "benchmark":
            parsed = parse_benchmark_filename(file_path.name)
            key = _benchmark_key(parsed) if parsed else None
        elif family == "sharegpt-benchmark":
            if suffix == ".json":
                parsed = parse_sharegpt_json_filename(file_path.name)
            else:
                parsed = parse_sharegpt_csv_filename(file_path.name)
            key = _sharegpt_key(parsed) if parsed else None
        else:
            parsed = None
            key = None

        if parsed is None or key is None:
            ignored += 1
            continue

        key_path = str(file_path)
        if suffix == ".json":
            if key in json_files:
                dup_json += 1
                continue
            json_files[key] = key_path
        else:
            if key in power_files:
                dup_power += 1
                continue
            power_files[key] = key_path

    manifest_rows: List[Dict[str, object]] = []
    matched_pairs: List[MatchedPair] = []
    all_keys = sorted(set(json_files.keys()) | set(power_files.keys()), key=lambda x: str(x))

    for key in all_keys:
        json_path = json_files.get(key)
        power_path = power_files.get(key)
        status = "matched"
        if json_path is None:
            status = "power_only"
        elif power_path is None:
            status = "json_only"

        if family == "benchmark":
            key_model, tp, rate, iteration, date_key = key
            pair_key = f"{key_model}|tp={tp}|rate={rate}|iter={iteration}|date={date_key}"
        else:
            tp, rate, date_key = key
            iteration = None
            pair_key = f"tp={tp}|rate={rate}|date={date_key}"

        manifest_row = {
            "family": family,
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
            "status": status,
            "model_name": model_from_dir,
            "hardware": hardware_from_dir,
            "tensor_parallelism": int(tp),
            "rate": str(rate),
            "iteration": "" if iteration is None else int(iteration),
            "date_key": str(date_key),
            "pair_key": pair_key,
            "power_csv_path": power_path or "",
            "json_path": json_path or "",
        }
        manifest_rows.append(manifest_row)

        if status == "matched":
            matched_pairs.append(
                MatchedPair(
                    family=family,
                    dataset_dir=dataset_dir,
                    model_name=model_from_dir,
                    hardware=hardware_from_dir,
                    tensor_parallelism=int(tp),
                    rate=str(rate),
                    iteration=(None if iteration is None else int(iteration)),
                    date_key=str(date_key),
                    pair_key=pair_key,
                    power_csv_path=power_path or "",
                    json_path=json_path or "",
                )
            )

    discovery_stats = {
        "family": family,
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "num_json_files": len(json_files),
        "num_power_csv_files": len(power_files),
        "num_manifest_rows": len(manifest_rows),
        "num_matched_pairs": len(matched_pairs),
        "num_json_only": int(sum(1 for r in manifest_rows if r["status"] == "json_only")),
        "num_power_only": int(sum(1 for r in manifest_rows if r["status"] == "power_only")),
        "duplicate_json_keys_dropped": dup_json,
        "duplicate_power_keys_dropped": dup_power,
        "ignored_files": ignored,
    }
    return manifest_rows, matched_pairs, discovery_stats
