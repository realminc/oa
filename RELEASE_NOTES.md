# OA v0.15.2 — Experimental Rust and Python preview

## Summary

Patch release for v0.15.1. Fixes shared-memory bank conflicts in the
wide tiled F32 and F16 GEMM kernels and vectorizes shared-memory tile
loads across the tiled matrix family, improving measured `mat_mul_nt`
throughput on Intel Iris Xe (ADL GT2) by up to ~50% on representative
NLP/ALM shapes at the compiled-replay boundary. Updates the `vkrust`
dependency to 0.13.4 and serializes precision mode across kernel-census
tests. No public API changes.

The SDK matmul tutorial now reports both execution boundaries —
`eager_submit_readback` and `replay_submit_timed` — so device kernel
time is visible separately from submit/readback overhead.

## Verification

Correctness-gated OARS benchmarks at `ca3fc85c` on Intel Iris Xe
(ADL GT2), Mesa 26.3.1, Vulkan 1.4.363: 18 matmul shapes pass their
oracles with zero unexpected fallbacks; the full ML profile suite, all
15 NLP training tutorials, and the video encode/decode suite pass;
the dated benchmark ledger in the source repository carries the raw
tables. Interleaved A/B measurements isolate the kernel deltas
from the integrated GPU's DVFS window; VTune characterization records
the remaining stall profile. Absolute numbers vary with device thermal
state.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Cooperative-matrix instructions are not available on this
device class; measured F32 throughput is bounded by the shared-memory
schedule. Additional device qualification and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Verify the published artifact
checksums before installing from GitHub Releases.
