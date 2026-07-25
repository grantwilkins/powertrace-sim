# Disaggregated campaign data

`gpt-oss-20b-a100-pd-confirmatory-35692922/` is the accepted cache-disabled
five-cell GPT-OSS-20B A100 disaggregated campaign from Slurm job `35692922`.
The job completed with `DISAGGREGATED_CAMPAIGN_OK` on 2026-07-24.

The retained evidence contains run metadata, GPU topology, pinned-runtime
identity, the preflight result, exact proxy events, and each cell's requests,
raw 250 ms power samples, role-separated engine telemetry, and timing
boundaries. These files are byte-for-byte copies of the immutable Sherlock run
root at
`/scratch/users/gfw/ptsim/runs/gpt-oss-20b-a100-pd-confirmatory-35692922`.

Transient prefiller, decoder, proxy, and benchmark logs are not retained.

`gpt-oss-20b-a100-pd-transfer-35765103/` is the accepted cache-disabled
four-cell GPT-OSS-20B A100 disaggregated transfer campaign from Slurm job
`35765103`. The job completed with `DISAGGREGATED_TRANSFER_OK` on 2026-07-25.

The retained evidence contains run metadata, GPU topology, pinned-runtime
identity, repository provenance, the effective runner, exact proxy events, and
each cell's requests, raw 250 ms power samples, role-separated engine
telemetry, and traffic boundaries. These files are byte-for-byte copies of the
immutable Sherlock run root at
`/scratch/users/gfw/ptsim/runs/gpt-oss-20b-a100-pd-transfer-35765103`.

Every cell holds the 250 ms sampling contract across 5089 samples, with a
median spacing of 0.250 s and a maximum of 0.307 s. This is the first transfer
campaign to do so: jobs `35741640` and `35763582` were both rejected at
`rate-0p5-calibration` for a single multi-second gap caused by streaming the
meter onto Lustre, fixed in `cb1eed5`.

The immutable request plans are retained under `plans/`, so the campaign's
recorded SHA-256 pins evidence that is present rather than absent. Each plan
recomputes to its stored and campaign-reported hash: `probe.json`
`f9bfe39e1de3e70fd339fe51f464f2c50571ad3d2a15ef23ca76933d81e4bc7d` for
`stage-probe`, `calibration.json`
`91ff6030d73340fdb48500095a36b6cd56f12e8b6d87465c79af21a755c76d1a` for
`rate-0p5-calibration`, and `heldout.json`
`8d266db47071ea263f7ed160b2a9d58a973e9c4ad2e36f176491f45a5bc5375a` for both
`rate-0p5-heldout` and its exact `rate-0p5-replay`.

Transient prefiller, decoder, proxy, and benchmark logs are not retained.

## Token and cache evidence

Power-model fitting needs token and cache covariates joined to the meter. This
campaign carries them at three resolutions, all on a common wall clock.

Per request, in each cell's `requests.json` as index-aligned arrays: `input_lens`
and `output_lens` for prompt and generated token counts, `ttfts`, `itls` for the
full per-token inter-token latency history, `request_timestamps`, and
`request_ids`. `input_lens` equals the plan's `prompt_len` request for request.

Per request, in `plans/*.json`: `prompt`, `prompt_len`, `output_len`, and the
immutable `offset_s` arrival schedule, alongside the plan `seed` and
`interval_s`. Prompts are randomly sampled token soup, chosen so that no two
requests share a prefix.

Per role at 250 ms, in `engine_prefill.csv` and `engine_decode.csv`, on the same
cadence as `power.csv`: `prompt_tokens_total`, `generation_tokens_total`, and
`iteration_tokens_total_sum` with its `_count` are monotonic counters spanning
the whole engine lifetime, so a cell's totals are the difference across its
traffic boundaries, not the final value. `gpu_cache_usage_perc` gives KV block
occupancy, `num_requests_running` and `num_requests_waiting` give queue state,
and the NIXL family records transferred bytes, transfer and post time,
descriptor counts, failures, and expiries.

`prefix_cache_queries_total` and `prefix_cache_hits_total` are identically zero
in every cell for both roles. That is the cache-disabled control the campaign
asserts, and it means no cached-token term is confounding these cells; the
counters are retained as the evidence of it rather than as a fitted covariate.

Per request phase boundaries are in `proxy_events.jsonl` as
`{request_id, event, wall_ns}` over `proxy_received`, `prefill_sent`,
`prefill_completed`, `decode_sent`, `decode_first_byte`, and
`decode_completed`, which separate NIXL handoff from decoder compute.
