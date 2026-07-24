"""Instrumented vLLM NIXL prefill/decode proxy.

The request protocol follows vLLM 0.22's NixlConnector integration proxy. Each
JSONL event uses wall-clock nanoseconds so it aligns with request and power data.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path


def prefill_payload(payload: dict) -> dict:
    request = payload.copy()
    request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    request["stream"] = False
    request["max_tokens"] = 1
    request.pop("stream_options", None)
    request.pop("min_tokens", None)
    request.pop("min_completion_tokens", None)
    return request


def transferred_payload(payload: dict, response: dict) -> dict:
    params = response.get("kv_transfer_params")
    if not params:
        raise ValueError("prefill response omitted kv_transfer_params")
    request = payload.copy()
    request["kv_transfer_params"] = params
    return request


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", buffering=1)

    def write(self, request_id: str, event: str, **fields) -> None:
        row = {"wall_ns": time.time_ns(), "request_id": request_id,
               "event": event, **fields}
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def create_app(prefill_url: str, decode_url: str, events: Path):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    log = EventLog(events)
    prefill = httpx.AsyncClient(timeout=None, base_url=prefill_url.rstrip("/") + "/v1")
    decode = httpx.AsyncClient(timeout=None, base_url=decode_url.rstrip("/") + "/v1")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await prefill.aclose()
        await decode.aclose()

    async def handle(endpoint: str, incoming: Request):
        request_id = incoming.headers.get("x-request-id") or uuid.uuid4().hex
        payload = await incoming.json()
        headers = {"X-Request-Id": request_id}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        log.write(request_id, "proxy_received")
        log.write(request_id, "prefill_sent")
        response = await prefill.post(endpoint, json=prefill_payload(payload), headers=headers)
        response.raise_for_status()
        prefill_response = response.json()
        log.write(request_id, "prefill_completed")
        decode_payload = transferred_payload(payload, prefill_response)

        async def stream():
            log.write(request_id, "decode_sent")
            first = True
            async with decode.stream("POST", endpoint, json=decode_payload,
                                     headers=headers) as result:
                result.raise_for_status()
                async for chunk in result.aiter_bytes():
                    if first:
                        log.write(request_id, "decode_first_byte")
                        first = False
                    yield chunk
            log.write(request_id, "decode_completed")

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"X-Request-Id": request_id},
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle("/completions", request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.prefill_url, args.decode_url, args.events),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
