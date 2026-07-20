"""Canonical exact-arrival trace plans for open- and closed-loop replay."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
from array import array
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceRound:
    session_id: str
    round_idx: int
    ready_s: float
    prefix_tokens: int
    input_tokens: int
    output_tokens: int
    tool_wait_s: float = 0.0
    source_id: str = ""
    cached_prefix_tokens: int = 0


@dataclass(frozen=True)
class TracePlan:
    source: str
    revision: str
    rounds: tuple[TraceRound, ...]
    seed: int = 0
    horizon_s: float | None = None

    def validate(
        self, max_model_len: int | None = None, *,
        check_prefix_continuity: bool = True,
    ) -> None:
        if not self.source or not self.revision:
            raise ValueError("trace source and immutable revision are required")
        if not self.rounds:
            raise ValueError("trace plan has no rounds")
        if self.horizon_s is not None and (
            self.horizon_s <= 0
            or self.horizon_s < max(row.ready_s for row in self.rounds)
        ):
            raise ValueError("trace horizon must cover every request release")
        sessions: dict[str, list[TraceRound]] = {}
        for row in self.rounds:
            if not row.session_id:
                raise ValueError("session_id must be non-empty")
            if row.round_idx < 0 or row.ready_s < 0 or row.tool_wait_s < 0:
                raise ValueError(f"negative index/time in {row.session_id}")
            if (
                row.prefix_tokens < 0 or row.input_tokens <= 0
                or row.output_tokens <= 0 or row.cached_prefix_tokens < 0
                or row.cached_prefix_tokens > row.prefix_tokens
            ):
                raise ValueError(f"invalid token counts in {row.session_id}:{row.round_idx}")
            if max_model_len and (
                row.prefix_tokens + row.input_tokens + row.output_tokens
                > max_model_len
            ):
                raise ValueError(
                    f"{row.session_id}:{row.round_idx} exceeds max_model_len "
                    f"{max_model_len}"
                )
            sessions.setdefault(row.session_id, []).append(row)
        for session_id, rows in sessions.items():
            ordered = sorted(rows, key=lambda row: row.round_idx)
            indices = [row.round_idx for row in ordered]
            if indices != list(range(len(rows))):
                raise ValueError(f"{session_id} round_idx must be contiguous from zero")
            releases = [row.ready_s for row in ordered]
            if releases != sorted(releases):
                raise ValueError(f"{session_id} ready_s must be nondecreasing")
            if check_prefix_continuity and not _prefix_continuous(ordered):
                prior_context = 0
                for row in ordered:
                    if row.round_idx and row.prefix_tokens > prior_context:
                        raise ValueError(
                            f"{session_id}:{row.round_idx} prefix_tokens exceeds "
                            f"prior context {prior_context}"
                        )
                    prior_context = (
                        row.prefix_tokens + row.input_tokens + row.output_tokens
                    )

    def by_session(
        self, *, check_prefix_continuity: bool = True
    ) -> dict[str, tuple[TraceRound, ...]]:
        self.validate(check_prefix_continuity=check_prefix_continuity)
        grouped: dict[str, list[TraceRound]] = {}
        for row in self.rounds:
            grouped.setdefault(row.session_id, []).append(row)
        return {
            key: tuple(sorted(value, key=lambda row: row.round_idx))
            for key, value in grouped.items()
        }

    def canonical_dict(self) -> dict:
        payload = {
            "schema_version": 1,
            "source": self.source,
            "revision": self.revision,
            "seed": self.seed,
            "rounds": [asdict(row) for row in self.rounds],
        }
        if self.horizon_s is not None:
            payload["horizon_s"] = self.horizon_s
        return payload

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _prefix_continuous(rows) -> bool:
    prior_context = 0
    for row in rows:
        if row.round_idx and row.prefix_tokens > prior_context:
            return False
        prior_context = row.prefix_tokens + row.input_tokens + row.output_tokens
    return True


def load_plan(path: str | Path, max_model_len: int | None = None) -> TracePlan:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("trace plan schema_version must be 1")
    plan = TracePlan(
        source=str(payload["source"]),
        revision=str(payload["revision"]),
        seed=int(payload.get("seed", 0)),
        horizon_s=(
            float(payload["horizon_s"])
            if payload.get("horizon_s") is not None else None
        ),
        rounds=tuple(TraceRound(**row) for row in payload["rounds"]),
    )
    plan.validate(max_model_len)
    return plan


def write_plan(plan: TracePlan, path: str | Path) -> None:
    plan.validate()
    Path(path).write_text(json.dumps(plan.canonical_dict(), indent=2) + "\n")


def load_bundle_requests_json(
    path: str | Path, *, time_scale: float = 1.0, seed: int = 0
) -> TracePlan:
    """Convert one canonical bundle's request marks to independent replay rows."""
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    source_path = Path(path)
    payload = json.loads(source_path.read_text())
    required = ("input_lens", "output_lens", "request_timestamps")
    if any(key not in payload for key in required):
        raise ValueError("bundle requests need input/output lengths and timestamps")
    lengths = {len(payload[key]) for key in required}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("bundle request arrays must be nonempty and equal length")
    timestamps = [float(value) for value in payload["request_timestamps"]]
    if timestamps != sorted(timestamps):
        raise ValueError("bundle request timestamps must be nondecreasing")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    origin = timestamps[0]
    rounds = tuple(
        TraceRound(
            session_id=f"request-{index:05d}",
            round_idx=0,
            ready_s=(timestamp - origin) * float(time_scale),
            prefix_tokens=0,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            source_id=f"{source_path.parent.name}:{index}",
        )
        for index, (timestamp, input_tokens, output_tokens) in enumerate(zip(
            timestamps, payload["input_lens"], payload["output_lens"]
        ))
    )
    plan = TracePlan(
        source=f"canonical-bundle:{source_path.parent.name}",
        revision=f"sha256:{source_hash};time_scale:{float(time_scale):.12g}",
        rounds=rounds,
        seed=int(seed),
    )
    plan.validate()
    return plan


def assign_poisson_session_arrivals(
    plan: TracePlan, *, rate_rps: float, seed: int
) -> TracePlan:
    """Assign a compact deterministic open-loop start to each selected session."""
    if rate_rps <= 0:
        raise ValueError("session arrival rate must be positive")
    rng = random.Random(seed)
    arrivals = {}
    elapsed = 0.0
    for index, session_id in enumerate(plan.by_session()):
        if index:
            elapsed += rng.expovariate(rate_rps)
        arrivals[session_id] = elapsed
    rounds = tuple(
        TraceRound(**{
            **asdict(row),
            "ready_s": arrivals[row.session_id],
        })
        for row in plan.rounds
    )
    result = TracePlan(
        source=plan.source, revision=plan.revision, rounds=rounds, seed=plan.seed
    )
    result.validate()
    return result


def select_sessions(
    plan: TracePlan, *, max_sessions: int, max_rounds_per_session: int,
    min_max_context: int = 0, max_max_context: int | None = None,
    min_rounds_per_session: int = 1,
) -> TracePlan:
    """Deterministically bound replay while selecting by session context peak."""
    if (
        max_sessions <= 0 or max_rounds_per_session <= 0
        or min_rounds_per_session <= 0
    ):
        raise ValueError("session and round bounds must be positive")
    candidates = []
    for session_id, rows in plan.by_session(
        check_prefix_continuity=False
    ).items():
        if not _prefix_continuous(rows):
            continue
        if len(rows) < min_rounds_per_session:
            continue
        peak = max(row.prefix_tokens + row.input_tokens for row in rows)
        if peak < min_max_context:
            continue
        if max_max_context is not None and peak > max_max_context:
            continue
        rank = hashlib.sha256(f"{plan.seed}:{session_id}".encode()).hexdigest()
        candidates.append((rank, session_id, rows))
    chosen = sorted(candidates)[:max_sessions]
    if not chosen:
        raise ValueError("no sessions match the requested context band")
    kept_ids = {session_id for _, session_id, _ in chosen}
    round_limits = {
        session_id: {row.round_idx for row in rows[:max_rounds_per_session]}
        for _, session_id, rows in chosen
    }
    rounds = tuple(
        row for row in plan.rounds
        if row.session_id in kept_ids
        and row.round_idx in round_limits[row.session_id]
    )
    selected = TracePlan(
        source=plan.source, revision=plan.revision, rounds=rounds, seed=plan.seed
    )
    selected.validate()
    return selected


def select_context_bands(
    plan: TracePlan, bands, *, max_rounds_per_session: int,
    min_rounds_per_session: int = 1,
) -> TracePlan:
    """Select an exact seeded session count from each disjoint [low, high) band."""
    normalized = [(int(low), int(high), int(count)) for low, high, count in bands]
    for low, high, count in normalized:
        if not 0 <= low < high or count <= 0:
            raise ValueError("context bands require 0 <= low < high and count > 0")
    for index, (low, high, _) in enumerate(normalized):
        for other_low, other_high, _ in normalized[index + 1:]:
            if max(low, other_low) < min(high, other_high):
                raise ValueError("context bands must not overlap")
    selected_ids = set()
    for low, high, count in normalized:
        selected = select_sessions(
            plan, max_sessions=count,
            max_rounds_per_session=max_rounds_per_session,
            min_max_context=low, max_max_context=high - 1,
            min_rounds_per_session=min_rounds_per_session,
        )
        if len(selected.by_session()) != count:
            raise ValueError(
                f"context band [{low}, {high}) has fewer than {count} sessions"
            )
        selected_ids.update(selected.by_session())
    round_limit = {
        session_id: {
            row.round_idx for row in rows[:max_rounds_per_session]
        }
        for session_id, rows in plan.by_session(
            check_prefix_continuity=False
        ).items()
        if session_id in selected_ids
    }
    rounds = tuple(
        row for row in plan.rounds
        if row.session_id in round_limit
        and row.round_idx in round_limit[row.session_id]
    )
    result = TracePlan(
        source=plan.source, revision=plan.revision, rounds=rounds, seed=plan.seed
    )
    result.validate()
    return result


def select_densest_arrival_window(
    plan: TracePlan, *, duration_s: float, max_requests: int
) -> TracePlan:
    """Keep a bounded exact prefix of the densest contiguous open-loop window."""
    if duration_s <= 0 or max_requests <= 0:
        raise ValueError("arrival window duration and request cap must be positive")
    rows = sorted(plan.rounds, key=lambda row: row.ready_s)
    if any(row.round_idx != 0 for row in rows):
        raise ValueError("arrival-window selection requires independent requests")
    best_start, best_end = 0, 0
    end = 0
    for start, row in enumerate(rows):
        end = max(end, start)
        while end < len(rows) and rows[end].ready_s - row.ready_s <= duration_s:
            end += 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end
    chosen = rows[best_start:min(best_end, best_start + max_requests)]
    if not chosen:
        raise ValueError("arrival trace has no requests")
    origin = chosen[0].ready_s
    rebased = tuple(
        TraceRound(**{**asdict(row), "ready_s": row.ready_s - origin})
        for row in chosen
    )
    result = TracePlan(
        source=plan.source, revision=plan.revision, rounds=rebased, seed=plan.seed
    )
    result.validate()
    return result


def select_stratified_arrival_window(
    plan: TracePlan, *, duration_s: float, window_index: int, window_count: int
) -> TracePlan:
    """Select one disjoint fixed window at a quantile of one-second Fano factor."""
    if (
        duration_s <= 0 or window_count <= 0
        or not 0 <= window_index < window_count
    ):
        raise ValueError("invalid stratified arrival-window selection")
    rows = sorted(plan.rounds, key=lambda row: row.ready_s)
    if not rows:
        raise ValueError("arrival trace has no requests")
    if any(row.round_idx != 0 for row in rows):
        raise ValueError("arrival-window selection requires independent requests")
    origin = rows[0].ready_s
    n_full = int((rows[-1].ready_s - origin) // duration_s)
    windows = [[] for _ in range(n_full)]
    for row in rows:
        index = int((row.ready_s - origin) // duration_s)
        if index < n_full:
            windows[index].append(row)
    candidates = []
    for index, chosen in enumerate(windows):
        if not chosen:
            continue
        start = origin + index * duration_s
        per_second = [0] * max(1, int(math.ceil(duration_s)))
        for row in chosen:
            second = min(int(row.ready_s - start), len(per_second) - 1)
            per_second[second] += 1
        mean = sum(per_second) / len(per_second)
        variance = sum((value - mean) ** 2 for value in per_second) \
            / len(per_second)
        candidates.append((variance / mean if mean else 0.0, len(chosen), start, chosen))
    if len(candidates) < window_count:
        raise ValueError(
            f"trace has {len(candidates)} full nonempty windows, "
            f"fewer than requested {window_count}"
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    rank = min(
        len(candidates) - 1,
        ((2 * window_index + 1) * len(candidates)) // (2 * window_count),
    )
    _, _, start, chosen = candidates[rank]
    rebased = tuple(
        TraceRound(**{**asdict(row), "ready_s": row.ready_s - start})
        for row in chosen
    )
    result = TracePlan(
        source=plan.source,
        revision=(
            f"{plan.revision};window:{start:.6f}-{start + duration_s:.6f};"
            f"fano-stratum:{window_index}/{window_count}"
        ),
        rounds=rebased, seed=plan.seed, horizon_s=float(duration_s),
    )
    result.validate()
    return result


def load_canonical_csv(
    path: str | Path, *, source: str, revision: str, seed: int = 0
) -> TracePlan:
    """Load the seconds-based interchange CSV used by all source adapters."""
    required = {
        "session_id", "round_idx", "ready_s", "prefix_tokens",
        "input_tokens", "output_tokens", "tool_wait_s", "cached_prefix_tokens",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"canonical trace CSV missing columns: {sorted(missing)}")
        rounds = tuple(
            TraceRound(
                session_id=row["session_id"],
                round_idx=int(row["round_idx"]),
                ready_s=float(row["ready_s"]),
                prefix_tokens=int(row["prefix_tokens"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                tool_wait_s=float(row["tool_wait_s"]),
                source_id=row.get("source_id", ""),
                cached_prefix_tokens=int(row["cached_prefix_tokens"]),
            )
            for row in reader
        )
    plan = TracePlan(source=source, revision=revision, rounds=rounds, seed=seed)
    plan.validate()
    return plan


def load_tracelab_csv(
    path: str | Path, *, revision: str, seed: int = 0
) -> TracePlan:
    """Adapt TraceLab's millisecond CSV release to the canonical plan."""
    required = {
        "id", "input_len", "output_len", "arrival_time", "round_idx",
        "tool_wait_after_ms", "prefix_len",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"TraceLab CSV missing columns: {sorted(missing)}")
        raw = list(reader)
    if not raw:
        raise ValueError("TraceLab CSV has no rows")
    origin_ms = min(float(row["arrival_time"]) for row in raw)
    rounds = tuple(
        TraceRound(
            session_id=str(row["id"]),
            round_idx=int(row["round_idx"]),
            ready_s=(float(row["arrival_time"]) - origin_ms) / 1000.0,
            prefix_tokens=int(row["prefix_len"]),
            input_tokens=int(row["input_len"]),
            output_tokens=int(row["output_len"]),
            tool_wait_s=float(row["tool_wait_after_ms"]) / 1000.0,
            source_id=str(row["id"]),
            cached_prefix_tokens=int(row["prefix_len"]),
        )
        for row in raw
    )
    plan = TracePlan(
        source="tracelab", revision=revision, rounds=rounds, seed=seed
    )
    plan.validate(check_prefix_continuity=False)
    return plan


def load_tracelab_jsonl(
    path: str | Path, *, revision: str, seed: int = 0
) -> TracePlan:
    """Adapt the pinned sanitized TraceLab JSONL or JSONL.GZ release."""
    source_path = Path(path)
    opener = gzip.open if source_path.suffix == ".gz" else open
    sessions: OrderedDict[str, list[dict]] = OrderedDict()
    with opener(source_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            session_id = row.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"{source_path}:{line_number} missing session_id")
            sessions.setdefault(session_id, []).append(row)
    rounds = []
    for session_id, rows in sessions.items():
        ordered = sorted(
            enumerate(rows),
            key=lambda item: (int(item[1].get("round_index", item[0])), item[0]),
        )
        for round_idx, (_, row) in enumerate(ordered):
            prefix = max(int(row.get("prefix_tokens") or 0), 0)
            appended = max(int(row.get("newly_append_tokens") or 0), 1)
            output = max(int(row.get("output_tokens") or 0), 1)
            wait_ms = 0.0
            if round_idx < len(ordered) - 1:
                for tool in row.get("tools") or ():
                    latency = tool.get("tool_wall_latency_ms")
                    if latency is not None and float(latency) >= 0:
                        wait_ms += float(latency)
            rounds.append(TraceRound(
                session_id=session_id,
                round_idx=round_idx,
                ready_s=0.0,
                prefix_tokens=prefix,
                input_tokens=appended,
                output_tokens=output,
                tool_wait_s=wait_ms / 1000.0,
                source_id=str(row.get("round_id") or row.get("trace_key") or ""),
                cached_prefix_tokens=prefix if round_idx else 0,
            ))
    plan = TracePlan(
        source="tracelab", revision=revision, rounds=tuple(rounds), seed=seed
    )
    plan.validate(check_prefix_continuity=False)
    return plan


def load_burstgpt_csv(
    path: str | Path, *, revision: str, seed: int = 0
) -> TracePlan:
    """Adapt BurstGPT's seconds-based independent-request trace."""
    required = {
        "Timestamp", "Model", "Request tokens", "Response tokens", "Log Type",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"BurstGPT CSV missing columns: {sorted(missing)}")
        raw = list(reader)
    if not raw:
        raise ValueError("BurstGPT CSV has no rows")
    origin = min(float(row["Timestamp"]) for row in raw)
    ordered = sorted(
        enumerate(raw), key=lambda item: (float(item[1]["Timestamp"]), item[0])
    )
    rounds = tuple(
        TraceRound(
            session_id=f"burstgpt_{source_index}",
            round_idx=0,
            ready_s=float(row["Timestamp"]) - origin,
            prefix_tokens=0,
            input_tokens=max(int(row["Request tokens"]), 1),
            output_tokens=max(int(row["Response tokens"]), 1),
            source_id=(
                f"row:{source_index}:model:{row['Model']}:"
                f"type:{row['Log Type']}"
            ),
        )
        for source_index, row in ordered
    )
    plan = TracePlan(
        source="burstgpt", revision=revision, rounds=rounds, seed=seed
    )
    plan.validate()
    return plan


def load_stratified_burstgpt_csv(
    path: str | Path, *, revision: str, duration_s: float,
    window_index: int, window_count: int, seed: int = 0,
) -> TracePlan:
    """Select a fixed Fano stratum without loading the full trace."""
    if (
        duration_s <= 0 or window_count <= 0
        or not 0 <= window_index < window_count
    ):
        raise ValueError("invalid stratified arrival-window selection")
    required = {
        "Timestamp", "Model", "Request tokens", "Response tokens", "Log Type",
    }

    def rows():
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"BurstGPT CSV missing columns: {sorted(missing)}"
                )
            yield from enumerate(reader)

    timestamps = (float(row["Timestamp"]) for _, row in rows())
    try:
        origin = next(timestamps)
    except StopIteration:
        raise ValueError("BurstGPT CSV has no rows") from None
    end = origin
    for timestamp in timestamps:
        origin = min(origin, timestamp)
        end = max(end, timestamp)
    n_full = int((end - origin) // duration_s)
    seconds = max(1, int(math.ceil(duration_s)))
    totals = array("Q", [0]) * n_full
    per_second = array("Q", [0]) * (n_full * seconds)
    for _, row in rows():
        relative = float(row["Timestamp"]) - origin
        index = int(relative // duration_s)
        if index >= n_full:
            continue
        second = min(int(relative - index * duration_s), seconds - 1)
        totals[index] += 1
        per_second[index * seconds + second] += 1
    candidates = []
    for index, total in enumerate(totals):
        if not total:
            continue
        mean = total / seconds
        offset = index * seconds
        variance = sum(
            (per_second[offset + second] - mean) ** 2
            for second in range(seconds)
        ) / seconds
        candidates.append((variance / mean, total, index))
    if len(candidates) < window_count:
        raise ValueError(
            f"trace has {len(candidates)} full nonempty windows, "
            f"fewer than requested {window_count}"
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    rank = min(
        len(candidates) - 1,
        ((2 * window_index + 1) * len(candidates)) // (2 * window_count),
    )
    chosen_index = candidates[rank][2]
    relative_start = chosen_index * duration_s
    start = origin + relative_start
    end = start + duration_s
    chosen = []
    for source_index, row in rows():
        timestamp = float(row["Timestamp"])
        if not start <= timestamp < end:
            continue
        chosen.append((timestamp, source_index, row))
    chosen.sort(key=lambda item: (item[0], item[1]))
    rounds = tuple(
        TraceRound(
            session_id=f"burstgpt_{source_index}",
            round_idx=0,
            ready_s=timestamp - start,
            prefix_tokens=0,
            input_tokens=max(int(row["Request tokens"]), 1),
            output_tokens=max(int(row["Response tokens"]), 1),
            source_id=(
                f"row:{source_index}:model:{row['Model']}:"
                f"type:{row['Log Type']}"
            ),
        )
        for timestamp, source_index, row in chosen
    )
    plan = TracePlan(
        source="burstgpt",
        revision=(
            f"{revision};window:{relative_start:.6f}-"
            f"{relative_start + duration_s:.6f};"
            f"fano-stratum:{window_index}/{window_count}"
        ),
        rounds=rounds,
        seed=seed,
        horizon_s=float(duration_s),
    )
    plan.validate()
    return plan
