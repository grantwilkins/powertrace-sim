"""Deterministic direct-token replay for canonical trace plans."""

from __future__ import annotations

import hashlib
import json
import time


def deterministic_tokens(label: str, count: int, vocab_size: int, seed: int) -> list[int]:
    """Stable non-special token IDs without hidden process-local randomness."""
    if vocab_size <= 32:
        raise ValueError("tokenizer vocabulary must contain more than 32 tokens")
    key = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    state = int.from_bytes(key[:8], "little")
    span = vocab_size - 16
    result = []
    for _ in range(count):
        state = (6364136223846793005 * state + 1442695040888963407) % 2**64
        result.append(16 + state % span)
    return result


class TokenSession:
    """Construct exact-length prompts while preserving each declared prefix."""

    def __init__(self, session_id: str, vocab_size: int, seed: int):
        self.session_id = session_id
        self.vocab_size = vocab_size
        self.seed = seed
        self.history: list[int] = []

    def prompt(self, row) -> list[int]:
        if row.round_idx == 0:
            self.history = deterministic_tokens(
                f"{self.session_id}:prefix", row.prefix_tokens,
                self.vocab_size, self.seed,
            )
        elif row.prefix_tokens > len(self.history):
            raise ValueError(
                f"{self.session_id}:{row.round_idx} prefix_tokens "
                f"{row.prefix_tokens} exceeds prior context {len(self.history)}"
            )
        prefix = self.history[:row.prefix_tokens]
        new_tokens = deterministic_tokens(
            f"{self.session_id}:{row.round_idx}:input", row.input_tokens,
            self.vocab_size, self.seed,
        )
        return prefix + new_tokens

    def commit(self, row, prompt: list[int], output_tokens: list[int]) -> None:
        if len(output_tokens) != row.output_tokens:
            raise ValueError("returned output token IDs do not match planned length")
        self.history = prompt + output_tokens


def request_record(
    row, *, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int,
    cached_prompt_tokens: int, request_timestamp: float, planned_ready_epoch: float,
    ttft: float, tpot: float, prefix_cache: bool, prompt_sha256: str,
    output_sha256: str,
) -> dict:
    return {
        "session_id": row.session_id,
        "turn_idx": int(row.round_idx),
        "input_len": int(prompt_tokens),
        "output_len": int(completion_tokens),
        "reasoning_tokens": int(reasoning_tokens),
        "cached_prompt_tokens": int(cached_prompt_tokens),
        "expected_cached_tokens": (
            int(row.cached_prefix_tokens) if prefix_cache else 0
        ),
        "prefix_tokens": int(row.prefix_tokens),
        "new_input_tokens": int(row.input_tokens),
        "planned_output_tokens": int(row.output_tokens),
        "request_timestamp": float(request_timestamp),
        "planned_ready_epoch": float(planned_ready_epoch),
        "arrival_delay_s": float(request_timestamp - planned_ready_epoch),
        "ttft": float(ttft),
        "itl": float(tpot),
        "post_gap_s": float(row.tool_wait_s),
        "prefix_cache": int(bool(prefix_cache)),
        "source_id": row.source_id,
        "prompt_sha256": prompt_sha256,
        "output_sha256": output_sha256,
    }


def build_requests_json(records: list[dict]) -> dict:
    """Bundle-compatible arrays plus exact replay/accounting provenance."""
    keys = (
        ("input_lens", "input_len"),
        ("output_lens", "output_len"),
        ("ttfts", "ttft"),
        ("itls", "itl"),
        ("request_timestamps", "request_timestamp"),
        ("session_ids", "session_id"),
        ("turn_idx", "turn_idx"),
        ("planned_ready_epoch", "planned_ready_epoch"),
        ("arrival_delay_s", "arrival_delay_s"),
        ("prefix_tokens", "prefix_tokens"),
        ("new_input_tokens", "new_input_tokens"),
        ("planned_output_tokens", "planned_output_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("cached_prompt_tokens", "cached_prompt_tokens"),
        ("expected_cached_tokens", "expected_cached_tokens"),
        ("post_gap_s", "post_gap_s"),
        ("prefix_cache", "prefix_cache"),
        ("source_ids", "source_id"),
        ("prompt_sha256", "prompt_sha256"),
        ("output_sha256", "output_sha256"),
    )
    return {array: [row[field] for row in records] for array, field in keys}


def _usage_details(usage: dict) -> tuple[int, int, int, int]:
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise ValueError("stream usage must include prompt_tokens and completion_tokens")
    prompt = int(usage["prompt_tokens"])
    completion = int(usage["completion_tokens"])
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    if not 0 <= cached <= prompt or not 0 <= reasoning <= completion:
        raise ValueError("invalid cached/reasoning token accounting")
    return prompt, completion, cached, reasoning


def validate_exact_accounting(
    row, *, planned_prompt_tokens: int, prompt_tokens: int,
    completion_tokens: int, cached_tokens: int, prefix_cache: bool,
    cache_block_tokens: int,
) -> None:
    if prompt_tokens != planned_prompt_tokens:
        raise ValueError(
            f"server prompt_tokens={prompt_tokens}, planned={planned_prompt_tokens}"
        )
    if completion_tokens != row.output_tokens:
        raise ValueError(
            f"server completion_tokens={completion_tokens}, "
            f"planned={row.output_tokens}"
        )
    expected_cached = row.cached_prefix_tokens if prefix_cache else 0
    if abs(cached_tokens - expected_cached) > cache_block_tokens:
        raise ValueError(
            f"server cached_tokens={cached_tokens}, expected={expected_cached} "
            f"within one {cache_block_tokens}-token cache block"
        )


async def send_round(
    http, base_url, model, row, prompt, prefix_cache, cache_block_tokens=16
) -> dict:
    """Send one exact-length completion and require authoritative usage."""
    url = base_url.rsplit("/v1", 1)[0].rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": row.output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    t_send = time.time()
    first_activity = None
    last_activity = None
    usage = None
    output_token_ids: list[int] = []
    async with http.post(url, json=payload) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(
                f"replay request {row.session_id}:{row.round_idx} failed "
                f"HTTP {response.status}: {body[:500]}"
            )
        async for raw in response.content:
            for encoded in raw.decode("utf-8", "strict").splitlines():
                line = encoded.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or ():
                    text = choice.get("text")
                    delta = choice.get("delta") or {}
                    token_ids = choice.get("token_ids") or delta.get("token_ids") or ()
                    if isinstance(token_ids, int):
                        token_ids = (token_ids,)
                    output_token_ids.extend(int(token) for token in token_ids)
                    active = bool(
                        text or delta.get("content") or delta.get("reasoning_content")
                        or token_ids
                    )
                    if active:
                        now = time.time()
                        first_activity = first_activity or now
                        last_activity = now
    if usage is None:
        raise ValueError("completion stream ended without usage accounting")
    prompt_details = usage.get("prompt_tokens_details") or {}
    if prefix_cache and "cached_tokens" not in prompt_details:
        raise ValueError("cache-on replay requires usage.prompt_tokens_details.cached_tokens")
    prompt_count, output_count, cached_count, reasoning_count = _usage_details(usage)
    validate_exact_accounting(
        row, planned_prompt_tokens=len(prompt), prompt_tokens=prompt_count,
        completion_tokens=output_count, cached_tokens=cached_count,
        prefix_cache=prefix_cache, cache_block_tokens=cache_block_tokens,
    )
    if len(output_token_ids) != output_count:
        raise ValueError(
            f"server returned {len(output_token_ids)} output token IDs for "
            f"{output_count} completion tokens"
        )
    if first_activity is None or last_activity is None:
        raise ValueError("non-empty completion had no timed output activity")
    ttft = first_activity - t_send
    tpot = (last_activity - first_activity) / max(output_count - 1, 1)
    record = request_record(
        row, prompt_tokens=prompt_count, completion_tokens=output_count,
        reasoning_tokens=reasoning_count, cached_prompt_tokens=cached_count,
        request_timestamp=t_send, planned_ready_epoch=0.0,
        ttft=ttft, tpot=tpot, prefix_cache=prefix_cache,
        prompt_sha256=hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        output_sha256=hashlib.sha256(
            json.dumps(output_token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    record["_output_token_ids"] = output_token_ids
    return record
