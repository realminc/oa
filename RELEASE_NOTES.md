# OA v0.14.1 — Experimental Rust and Python preview

## Summary

Patch release for v0.14.0. Fixes a throughput regression where every
`Engine::current()` clone ran the `.vkcache` persistence path on drop,
serializing the pipeline cache and rewriting the cache file per defaulted
operation. Persistence now runs once, when the last `Engine` clone drops.
Measured on Iris Xe (ADL GT2), Mesa 26.3.1, `tu_nlp_bpe_transformer`
F32: wall time recovered from 32.24 ms/step to 7.7 ms/step (1.99K to
8.3K samples/s), restoring the v0.13 wall–GPU gap of ~16%. Also fixes
engine clones restoring the previous logger on drop, which deselected
the active logger after the first defaulted call.

## Verification

`cargo fmt --check`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, `cargo test --lib --all-features`
(243 passed), `cargo test --test runtime --test core --all-features`
(201 passed) pass on the release commit. `tu_nlp_bpe_transformer`
verified end to end on the release candidate: cache persistence fires
once at engine destruction and convergence is unchanged.

# OA v0.14.0 — Experimental Rust and Python preview

## Summary

OA introduces the thread-local default engine and the paired default/explicit
constructor convention across the Rust surface. Simple programs no longer
create or thread an `Engine`; `Engine::set_default` selects the device once
per thread, and `..._on(&engine, ...)` forms remain for explicit engine
selection. This release also consolidates the internal device API and adds
occupancy-gated wide-tile GEMM routes.

## Added and improved

- `Engine::set_default` installs a chosen engine as the thread's default and
  `Engine::current` returns a handle to it. The lazy per-thread default is
  the sole engine path; no global engine exists.
- Every engine-first value constructor ships a defaulted twin:
  `matrix::ones`, `Matrix::new_from_f32`/`new_from_slice`, generated nn layer
  constructors, `ml::load_checkpoint`, audio/image decode, video-frame and
  texture uploads, and SDK model and dataset constructors. The explicit
  `..._on(&engine, ...)` form is the implementation; the short form resolves
  the thread's default.
- Sessions, capture, replay, presentation, and device queries keep an
  explicit `&Engine`.
- `runtime/device.rs` is the single composed device API; the internal
  command surface is consolidated under `runtime/device/logical`.
- Occupancy-gated 128x64 wide-tile and FP16-storage wide GEMM variants
  derive their admission gate from the device contract.
- Trading observation capacity rises to a 128M element budget with packed
  market-history support.
- SDK tutorials, examples, benchmarks, and tests use defaulted
  constructors; the matrix-multiplication and NLP tutorials no longer
  thread an engine.

## Compatibility and migration

Breaking: engine-first value constructors are renamed to the `_on`
spelling. `matrix::ones(&engine, shape)` becomes
`matrix::ones_on(&engine, shape)`; the defaulted `matrix::ones(shape)` is
new. `Matrix::new_from_f32`/`new_from_slice` follow the same split.
Session and device APIs are unchanged. Python keeps engine-first
signatures; `Engine` remains required there.

The Rust crate is `oarust`, the Python distribution is `oapython` and both
use the `oa` namespace. Core remains the default Rust build profile.
GPU, queue, codec and driver capability determine route availability.

## Verification

`cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features
-- -D warnings` and `cargo test --workspace --all-features` pass on the
release commit. Device-admission tests remain environment-gated through
`OA_TEST_DEVICE_INDEX`.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Additional device qualification and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Rename engine-first value
constructors to their `_on` forms or adopt the defaulted spellings under
`Engine::set_default`. Verify the published artifact checksums before
installing from GitHub Releases.
