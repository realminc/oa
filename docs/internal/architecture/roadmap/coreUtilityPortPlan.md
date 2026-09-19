# Core Utility Port Plan

**Status:** In Progress

**Updated:** 2026-09-12

**Architecture:** [OA Rust Architecture](../oaArchitecture.md)

This document tracks the incremental port of `oa/core/` utility headers from the
OA C++ donor into the `oa` Rust crate under `src/rs/core/`. Entries are ordered
by impact on tutorials and examples, not by C++ source order.

## Already Ported

| Rust module | Donor header | Notes |
|---|---|---|
| `core::cli` / `Cli<C>` | `oa/core/cli.h` | YAML + argv precedence; `ParseCliValue` trait |
| `core::config` | `oa/core/config.h` | `CheckpointConfig`, `LogConfig` |
| `core::log_metrics` / `LogMetrics` | `oa/core/log.h` | JSONL scalar sink; thread-safe |
| `core::time` | `oa/core/time.h` | `Stopwatch`, `ScopedTimer`, `Timestamp`, `Datetime` |
| `core::perf_stat` / `PerfStat` | `oa/core/perfStat.h` | Min/max/mean/σ accumulator |
| `core::path` / `Path` | `oa/core/paths.h` | Canonical OA path conventions (var, cache, home) |
| `core::filesystem` / `Filesystem` | `oa/core/filesystem.h` | Read/write/mkdir/exists/list |
| `core::memory` | `oa/core/memory` (Vulkan + host) | AVX2/AVX-512 streaming copy; secure erase |
| `core::dtype` / `DType`, `Element` | `oa/core/type.h` | F32/F16/BF16/I32/U32/U8 |
| `core::error` / `Error`, `Result` | `oa/core/status.h` | Backend-neutral error contract |
| `core::image` / `Image` | `oa/core/image.h` | FP32 planar NCHW/CHW value |
| `core::matrix` / `Matrix` | `oa/core/matrix.h` | Core tensor value |
| `core::op` | `oa/core/op.h` | Operation contract vocabulary |
| `core::autograd` | `oa/core/autograd.h` | Reverse-mode tape |
| `core::operation` | schema-generated | Generated operation metadata |
| `core::vlm` | `oa/core/vlm.h` | Vision-language model helpers |
| `core::json` | internal | JSONL push helpers |

## Porting Queue

Ordered by tutorial-reduction impact. Each row is one or a few Rust modules.

### Tier 1 — Tutorial / example essentials

These are referenced directly in every tutorial and reduce the most boilerplate.

#### `core/callback.rs` + iterator base

**Donor:** `oa/core/callback.h`, `oa/core/iterator.h`

`Callback` is the Keras-style hook interface used by `TrainingLoop` (and any
future `ItBatch` / `ItInference`). The Rust equivalent is a sealed trait with
three optional methods:

```rust
pub trait Callback {
    fn on_begin(&mut self, iter: &dyn IteratorContext) {}
    fn on_step(&mut self,  iter: &dyn IteratorContext) {}
    fn on_end(&mut self,   iter: &dyn IteratorContext) {}
}
```

`IteratorContext` exposes `index()`, `is_done()`, and `total()` through a thin
object-safe trait so callbacks remain decoupled from `TrainingLoop`'s concrete
type. The C++ `Iterator` base (stateful `isDone` / `next` cursor) maps to a
private session contract within `TrainingLoop`; it is not exposed as a public
type.

**Scope:** `src/rs/core/callback.rs`

#### `core/env_flag.rs`

**Donor:** `oa/core/envFlag.h`

`EnvFlag` reads OA's canonical env knobs (`OA_DISABLE_COOPMAT`,
`OA_FORCE_PRECISION`, etc.) through a uniform API: bool toggle, string override,
integer override, and `set_if_unset` for programmatic defaults. `NumericMode`
(`Fast` / `Stable` / `Deterministic`) and `apply_numeric_mode` are defined here
and consumed by `EngineConfig`.

Tutorials benefit from `EnvFlag::is_set` and `EnvFlag::get_int` for runtime
knobs without depending on the engine.

**Scope:** `src/rs/core/env_flag.rs`

### Tier 2 — Internal correctness primitives

Required before porting runtime internals and before any tutorial that references
the validation layer.

#### `BufferAccess` enum

**Donor:** `oa/core/bufferAccess.h`

Three-variant enum (`Read`, `Write`, `ReadWrite`) used by 30+ operation sites.
Added to the existing `src/rs/core/op.rs` module rather than a new file.

#### `core/validation.rs`

**Donor:** `oa/core/validation.h`

`ValidationSeverity` (Verbose → Fatal), `Validation` singleton (enable/disable,
env init, severity filter, optional callback, debug counters). Macro equivalents
are Rust attribute macros or `cfg(debug_assertions)` conditional helpers:

```rust
#[cfg(debug_assertions)]
macro_rules! oa_validate { ... }
```

The debug-counter API maps to `cfg(debug_assertions)`-gated atomics so they
compile out in release with zero cost.

**Scope:** `src/rs/core/validation.rs`

### Tier 3 — I/O and data-loading

#### `core/mapped_file.rs`

**Donor:** `oa/core/mappedFile.h`

Read-only whole-file `mmap` with RAII. On Linux uses `memmap2` (already in
dependency tree); other platforms fall back to a heap-owned `Vec<u8>`. Used by
the safetensor reader, audio codec loader, and future checkpoint I/O.

**Scope:** `src/rs/core/mapped_file.rs`

### Tier 4 — Cosmetic / branding

#### `core/constant.rs`

**Donor:** `oa/core/constant.h`

`REALM_BANNER`, `COMPACT_BANNER`, `VIEWPORT_TITLE`, `brand_viewport()`.
Required by CLI tools and the viewer application startup path.

**Scope:** `src/rs/core/constant.rs`

### Deferred

| Header | Reason deferred |
|---|---|
| `thread.h` (`Thread`, `ThreadPool`, `Channel`, `Task`) | Runtime internals; blocked on Stage 6 multi-device work. `std::thread` + `crossbeam-channel` adequate for current tutorials. |
| `math.h` (`Fixed<D>`) | Financial domain only; no current tutorial dependency. |
| `determinism.h` | Superseded by `EnvFlag` + `NumericMode` which own the same state. |

## Exit gate for this batch

```
python3 -m unittest discover -s test/py -v
python3 tools/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```
