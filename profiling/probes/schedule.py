"""Pure probe-schedule construction (CAMPAIGN.md §3 / §5-B).

A *schedule* is a deterministic data structure describing what state to pin the
engine at and for how long. It contains every number that determines the
measured operating point (concurrency ladders, input lengths, context sizes,
hold durations, server knobs) and **no I/O** — so the whole of it is unit-tested
offline. The thin live layer (``probe_runner`` / ``bench_driver``) only executes
the schedule against a running server; it adds no new state decisions.

Tier-1 probes (one anchor model per GPU type):
  idle_hold, decode_staircase, prefill_staircase, context_holds,
  transients, mixed_grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Rough per-sequence rates used only to size --num-prompts so a level runs ~hold_s
# (benchmark_serving is count-bounded, not wall-time-bounded). The actual dwell is
# measured and recorded as the level's epoch window, so these need only be ballpark.
_DECODE_TOK_PER_S = 75.0
_PREFILL_TOK_PER_S = 4000.0


def estimate_num_prompts(concurrency: int, input_len: int, output_len: int,
                         hold_s: float) -> int:
    """Number of prompts to keep `concurrency` busy for ~hold_s seconds."""
    if concurrency <= 0:
        return 0
    prefill_s = input_len / _PREFILL_TOK_PER_S
    decode_s = max(output_len - 1, 0) / _DECODE_TOK_PER_S
    per_req_s = max(prefill_s + decode_s, 0.05)
    waves = max(1, math.ceil(hold_s / per_req_s))
    return int(concurrency * waves)


@dataclass(frozen=True)
class RequestSpec:
    """One request shape sent during a level."""
    input_len: int
    output_len: int
    prefix_len: int = 0
    ignore_eos: bool = True


@dataclass(frozen=True)
class ProbeLevel:
    """A single held operating point."""
    level: int
    concurrency: int          # 0 == idle (no traffic) -> --max-concurrency
    hold_seconds: float
    request: RequestSpec
    label: str
    num_prompts: int = 0      # total requests for this level (sized to hold_s)


@dataclass
class ProbeSchedule:
    """An ordered list of levels plus the server config they require."""
    probe_type: str
    levels: list[ProbeLevel]
    server_overrides: dict = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return float(sum(l.hold_seconds for l in self.levels))


def _pow2_ladder(max_n: int) -> list[int]:
    """[1, 2, 4, ...] up to and including max_n exactly."""
    ladder, n = [], 1
    while n < max_n:
        ladder.append(n)
        n *= 2
    ladder.append(int(max_n))
    return ladder


def build_decode_staircase(
    max_num_seqs: int, hold_s: float = 45.0,
    prompt_len: int = 8, output_len: int = 2048,
) -> ProbeSchedule:
    """Concurrency ladder 1..max_num_seqs; short prompt, long decode, ignore_eos.

    Identifies the decode saturation curve, e_w_decode, and the power cap.
    """
    levels = [
        ProbeLevel(
            level=i,
            concurrency=n,
            hold_seconds=hold_s,
            request=RequestSpec(input_len=prompt_len, output_len=output_len,
                                ignore_eos=True),
            label=f"decode_N{n}",
            num_prompts=estimate_num_prompts(n, prompt_len, output_len, hold_s),
        )
        for i, n in enumerate(_pow2_ladder(max_num_seqs))
    ]
    return ProbeSchedule("decode_staircase", levels)


def build_prefill_staircase(
    input_lens=(256, 1024, 4096, 16384, 65536), hold_s: float = 45.0,
) -> ProbeSchedule:
    """Concurrency-1, max_tokens=1, chunked-prefill OFF -> clean pure-prefill dwells.

    Identifies e_f_prefill, the attention-L^2 term and the TTFT-vs-length curve.
    """
    levels = [
        ProbeLevel(
            level=i,
            concurrency=1,
            hold_seconds=hold_s,
            request=RequestSpec(input_len=int(n), output_len=1, ignore_eos=True),
            label=f"prefill_{int(n)}",
            num_prompts=estimate_num_prompts(1, int(n), 1, hold_s),
        )
        for i, n in enumerate(input_lens)
    ]
    return ProbeSchedule(
        "prefill_staircase", levels,
        server_overrides={"enable_chunked_prefill": False},
    )


def build_context_holds(
    contexts=(2048, 8192, 32768, 131072), batch: int = 8,
    hold_s: float = 45.0, output_len: int = 256,
) -> ProbeSchedule:
    """Primed long prefix then steady decode at fixed batch, varying context.

    Identifies e_kv (KV-read energy) — the term short prompts cannot.
    """
    max_ctx = int(max(contexts))
    levels = [
        ProbeLevel(
            level=i,
            concurrency=batch,
            hold_seconds=hold_s,
            request=RequestSpec(input_len=8, output_len=output_len,
                                prefix_len=int(ctx), ignore_eos=True),
            label=f"context_{int(ctx)}",
            num_prompts=estimate_num_prompts(batch, int(ctx) + 8, output_len, hold_s),
        )
        for i, ctx in enumerate(contexts)
    ]
    return ProbeSchedule(
        "context_holds", levels,
        server_overrides={
            "max_model_len": max_ctx + output_len + 64,
            "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        },
    )


def build_transients(
    concurrency: int = 64, on_s: float = 20.0, off_s: float = 20.0, repeats: int = 4,
    output_len: int = 2048,
) -> ProbeSchedule:
    """Sharp idle->batch->idle steps to identify the lag-filter constant alpha."""
    levels = []
    for r in range(repeats):
        levels.append(ProbeLevel(
            level=2 * r, concurrency=0, hold_seconds=off_s,
            request=RequestSpec(input_len=8, output_len=output_len),
            label=f"transient_idle_{r}",
        ))
        levels.append(ProbeLevel(
            level=2 * r + 1, concurrency=concurrency, hold_seconds=on_s,
            request=RequestSpec(input_len=8, output_len=output_len, ignore_eos=True),
            label=f"transient_on_{r}",
            num_prompts=estimate_num_prompts(concurrency, 8, output_len, on_s),
        ))
    return ProbeSchedule("transients", levels)


def build_idle_hold(hold_s: float = 60.0) -> ProbeSchedule:
    """No traffic; identifies p_idle."""
    return ProbeSchedule("idle_hold", [
        ProbeLevel(level=0, concurrency=0, hold_seconds=hold_s,
                   request=RequestSpec(input_len=8, output_len=1), label="idle")
    ])


def build_mixed_grid(
    n_points: int = 16, seed: int = 0,
    decode_range=(1, 128), prefill_range=(256, 16384),
    hold_s: float = 45.0, output_len: int = 512,
) -> ProbeSchedule:
    """Latin-hypercube over (decode concurrency floor, prefill injection length).

    Identifies the prefill x decode interaction. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    # Latin hypercube on the unit square.
    base = (np.arange(n_points)[:, None] + rng.uniform(size=(n_points, 2))) / n_points
    pts = np.empty_like(base)
    for d in range(2):
        pts[:, d] = base[rng.permutation(n_points), d]

    d_lo, d_hi = decode_range
    p_lo, p_hi = prefill_range
    levels = []
    for i in range(n_points):
        conc = int(round(d_lo + pts[i, 0] * (d_hi - d_lo)))
        conc = max(1, conc)
        # log-spaced prefill injection length
        inp = int(round(p_lo * (p_hi / p_lo) ** pts[i, 1]))
        levels.append(ProbeLevel(
            level=i, concurrency=conc, hold_seconds=hold_s,
            request=RequestSpec(input_len=inp, output_len=output_len, ignore_eos=True),
            label=f"mixed_{i}_c{conc}_p{inp}",
            num_prompts=estimate_num_prompts(conc, inp, output_len, hold_s),
        ))
    return ProbeSchedule("mixed_grid", levels)


BUILDERS = {
    "idle_hold": build_idle_hold,
    "decode_staircase": build_decode_staircase,
    "prefill_staircase": build_prefill_staircase,
    "context_holds": build_context_holds,
    "transients": build_transients,
    "mixed_grid": build_mixed_grid,
}
