"""Continuous-batching scheduler simulation (vLLM V1 semantics).

Replays a recorded arrival sequence through first-come-first-served
admission with chunked prefill (decode-first token budget) and a KV-cache
capacity limit, advancing time by the iteration-time model. Emits each
request's time to first token, per-token latencies, and end-to-end latency.

Engine policy constants are recorded serving configuration (vLLM V1
defaults; per-run flags like ``--max-num-seqs`` come from the serve
scripts), not fitted values. The simulation itself has no fitted constants;
all timing comes from the supplied iteration-time function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from iteration_time import HARDWARE_PROFILES, iteration_time_s, iteration_work, kv_bytes_per_token
from model.training_data.moe_routing import RoutingLaw


@dataclass
class EngineConfig:
    max_num_seqs: int = 256
    chunk_budget_tokens: int = 2048
    kv_capacity_tokens: float = 0.0  # 0 -> derive from hardware and weights
    gpu_memory_utilization: float = 0.9


@dataclass
class _Sequence:
    index: int
    arrival_s: float
    n_in: int
    n_out: int
    initial_context: int = 0
    prefilled: int = 0
    generated: int = 0
    admitted_s: float = float("nan")
    first_token_s: float = float("nan")
    token_times: list = field(default_factory=list)

    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= self.n_in

    @property
    def finished(self) -> bool:
        return self.prefill_done and self.generated >= self.n_out

    @property
    def context_tokens(self) -> int:
        return self.initial_context + self.prefilled + self.generated


def derive_kv_capacity_tokens(arch: Mapping[str, object], *, hardware: str,
                              tp: int, gpu_memory_utilization: float = 0.9) -> float:
    """KV token capacity from device memory minus resident weights."""
    total = gpu_memory_utilization * HARDWARE_PROFILES[hardware]["hbm_capacity"] * tp
    free = total - float(arch["w_bytes"])
    if free <= 0:
        raise ValueError("Weights exceed the requested memory budget")
    return free / kv_bytes_per_token(arch)


def simulate_requests(
    requests: Sequence[tuple], *, arch: Mapping[str, object], hardware: str,
    tp: int, eff_flops: float, eff_bw: float, t_launch_s: float,
    t_sample_s: float = 0.0, engine: EngineConfig = EngineConfig(),
    transformer_bw_scale: float = 1.0,
    iteration_seconds: Callable[..., float] | None = None,
    iteration_trace: list | None = None,
    routing_law: RoutingLaw | None = None,
) -> list[dict]:
    """Simulate `(arrival_s, n_in, n_out)` requests; return per-request timing.

    Request tuples may include a fourth ``initial_context`` field for cached
    prefixes; ``n_in`` is then the number of prompt tokens actually executed.

    `iteration_seconds(decode_batch, context_mean, chunk_tokens,
    chunk_context)` may be injected for tests; by default it composes the
    architecture work model with the roofline time model.

    If `iteration_trace` is a list, one record per engine iteration is
    appended: (t_start_s, t_end_s, decode_batch, context_mean,
    chunk_tokens, chunk_context, n_chunks, prefill_logits, weight_bytes) —
    the phase-resolved executed work on wall time that the power stage consumes.
    """
    default_timing = iteration_seconds is None
    if engine.max_num_seqs < 1 or engine.chunk_budget_tokens < 1:
        raise ValueError("Engine sequence and token budgets must be positive")

    kv_capacity = engine.kv_capacity_tokens or derive_kv_capacity_tokens(
        arch, hardware=hardware, tp=tp,
        gpu_memory_utilization=engine.gpu_memory_utilization)
    order = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    pending = [_Sequence(i, *requests[i]) for i in order]
    for seq in pending:
        if seq.n_in < 1 or seq.n_out < 1 or seq.initial_context < 0:
            raise ValueError("Requests need at least one prompt and one output token")
    waiting: list[_Sequence] = []
    running: list[_Sequence] = []
    done: list[_Sequence] = []
    clock = 0.0
    arrival_cursor = 0

    def kv_in_use() -> float:
        # Admission reserves room for the full prompt plus the next token
        # (block reservation); afterwards usage tracks the grown context.
        return float(sum(
            max(seq.context_tokens, seq.initial_context + seq.n_in) + 1
            for seq in running
        ))

    while len(done) < len(pending):
        while arrival_cursor < len(pending) and pending[arrival_cursor].arrival_s <= clock:
            waiting.append(pending[arrival_cursor])
            arrival_cursor += 1
        # Admit waiting sequences first-come-first-served while a seat and
        # KV room for the full prompt plus one generated token exist.
        while (
            waiting
            and len(running) < engine.max_num_seqs
            and kv_in_use() + waiting[0].initial_context + waiting[0].n_in + 1
            <= kv_capacity
        ):
            waiting[0].admitted_s = clock
            running.append(waiting.pop(0))
        if not running:
            if arrival_cursor >= len(pending):
                break
            clock = max(clock, pending[arrival_cursor].arrival_s)
            continue

        # Compose one iteration: every prefill-complete sequence decodes one
        # token (decode-first), then the remaining token budget feeds prefill
        # chunks in arrival order.
        decoding = [seq for seq in running if seq.prefill_done]
        budget = engine.chunk_budget_tokens - len(decoding)
        chunks: list[tuple] = []  # (sequence, tokens, prior context)
        for seq in running:
            if seq.prefill_done or budget <= 0:
                continue
            tokens = min(budget, seq.n_in - seq.prefilled)
            chunks.append((seq, tokens, seq.initial_context + seq.prefilled))
            budget -= tokens
        decode_batch = len(decoding)
        context_mean = (sum(seq.context_tokens for seq in decoding) / decode_batch
                        if decode_batch else 0.0)
        chunk_tokens = sum(tokens for _, tokens, _ in chunks)
        chunk_context = (
            sum(tokens * prior for _, tokens, prior in chunks) / chunk_tokens
            if chunk_tokens else 0.0
        )
        prefill_logits = sum(
            seq.prefilled + tokens >= seq.n_in for seq, tokens, _ in chunks
        )
        iteration_started = clock
        work = iteration_work(
            arch, decode_batch=decode_batch, context_mean=context_mean,
            prefill_chunks=[(tokens, prior) for _, tokens, prior in chunks],
            prefill_logits=prefill_logits, routing_law=routing_law)
        if default_timing:
            duration = iteration_time_s(
                work, hardware=hardware, tp=tp, eff_flops=eff_flops,
                eff_bw=eff_bw, t_launch_s=t_launch_s,
                t_sample_s=t_sample_s,
                transformer_bw_scale=transformer_bw_scale)
        else:
            duration = iteration_seconds(
                decode_batch, context_mean, chunk_tokens, chunk_context)
        clock += duration
        if iteration_trace is not None:
            iteration_trace.append((iteration_started, clock, decode_batch,
                                    context_mean, chunk_tokens, chunk_context,
                                    len(chunks), prefill_logits,
                                    work["gemm_bytes"]))

        for seq, tokens, _ in chunks:
            seq.prefilled += tokens
            if seq.prefill_done:
                # The iteration that consumes the final prompt chunk emits
                # the first output token (vLLM prefill produces token one).
                seq.generated = 1
                seq.first_token_s = clock
                seq.token_times.append(clock)
        for seq in decoding:
            seq.generated += 1
            seq.token_times.append(clock)
        for seq in list(running):
            if seq.finished:
                running.remove(seq)
                done.append(seq)

    output = []
    for seq in sorted(done, key=lambda s: s.index):
        itls = [b - a for a, b in zip(seq.token_times, seq.token_times[1:])]
        output.append({
            "index": seq.index, "arrival_s": seq.arrival_s,
            "admitted_s": seq.admitted_s,
            "n_in": seq.n_in, "n_out": seq.n_out,
            "initial_context": seq.initial_context,
            "ttft_s": seq.first_token_s - seq.arrival_s,
            "itl_s": itls,
            "decode_duration_s": sum(itls),
            "e2e_s": seq.token_times[-1] - seq.arrival_s,
        })
    return output
