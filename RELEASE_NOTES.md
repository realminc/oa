# OA v0.11.0 — Rust development preview

## Summary

v0.11.0 advances the Experimental JIT compiler work above v0.10.3. An
engine-level census records which fusion candidates are declined and why on
real workloads (OA-JIT-03a/b/c), and NLP kernel ordering is now driven by that
measured evidence. JIT specialization compilation moves onto a bounded
background worker so compilation no longer blocks execution (OA-JIT-04).

Backward-anchored fusion lands its first implemented contract: activation
adjoints fuse into linear data-backward chains, L1 ownership for
backward-anchored fusion is decided, the linear-backward producer prologue is
generalized, and the norm adjoint is fused (OA-JIT-05/08). DNN provider
kernels are admitted before lowered nodes are inserted, and the elementwise
JIT route can be forced per run with a named workload set.

## Compatibility and migration

OA's public Rust/Python API is unchanged. JIT remains Experimental and
disabled by default. Rust (`oarust`, namespace `oa`), Python (`oapython`,
import `oa`), and their lockfile version metadata agree on 0.11.0.

## Performance

No release performance claim is made. Recorded evidence is Haswell
regression checking and real-workload census/coverage data on a single
Intel iGPU; it is not a benchmark and says nothing about other devices.

## Verification

Local gates on the release commit: `cargo fmt --check`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-features`. The realminc/oa tag pipeline additionally
builds and tests the exact tag and verifies version-source agreement and
tag ancestry on main.

## Known limitations

JIT is Experimental and off by default; no end-to-end training improvement
is claimed. Evidence is single-device Intel Haswell only. Fusion admission
remains conservative; declined candidates retain source execution.

## Upgrade

Bump `oarust`/`oapython` to 0.11.0. No migration required. Cached
`.vkcache` artifacts are disposable and rebuild as needed.
