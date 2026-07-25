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

The immutable request plans are not retained. Each is large, is regenerable
from its recorded seed, and has its SHA-256 pinned by the campaign: probe
`f9bfe39e1de3e70fd339fe51f464f2c50571ad3d2a15ef23ca76933d81e4bc7d`,
calibration
`91ff6030d73340fdb48500095a36b6cd56f12e8b6d87465c79af21a755c76d1a`, and
held-out with its exact replay
`8d266db47071ea263f7ed160b2a9d58a973e9c4ad2e36f176491f45a5bc5375a`.

Transient prefiller, decoder, proxy, and benchmark logs are not retained.
