# OA Rust Architecture

**Status:** Canonical

**Updated:** 2026-09-07

**Roadmap:** [Rust port roadmap](roadmap/portRoadmap.md)

**Compatibility:** [OA compatibility ledger](../porting/oaCompatibility.md)

This document defines the target architecture of OA's Rust implementation.
Current source proves implementation status; the roadmap orders planned work;
subsystem documents may add detail without redefining these boundaries.

## 1. Product boundary

OA is a GPU-first semantic computing library implemented on Vulkan. It owns:

- typed device-resident values;
- stateless numerical, ML, vision, audio, render, and crypto operations;
- stateful sessions for external and iterative processes;
- semantic and executable graph compilation;
- Vulkan execution, memory, synchronization, diagnostics, and profiling;
- Rust and Python surfaces generated from shared operation contracts where
  those surfaces are mechanical.

OA does not expose a public object for every Vulkan handle, hide a CPU backend
behind failed GPU work, or reproduce the C++ source hierarchy mechanically.
Applications, examples, tutorials, and product-specific policy remain outside
the library boundary.

## 2. Architectural invariants

1. **One local execution owner.** `Engine` owns the Vulkan instance, selected
   logical devices, memory, queues, kernels, scheduling, and profiling.
2. **No public runtime facade.** Process setup and hardware discovery are
   construction responsibilities of `Engine`; a second owning `Runtime` would
   create ambiguous lifetime and sharing policy.
3. **Composition for optional systems.** Presentation, codecs, transport,
   collectives, and training sessions borrow an engine. They do not own, wrap,
   subclass, or recreate one.
4. **No hidden ownership.** Convenience scopes may select an engine-owned
   semantic recorder, but they never create or retain engines, devices,
   submissions, or resources.
5. **No hidden execution.** Recording, planning, submission, waiting, readback,
   and session shutdown are explicit failure-bearing boundaries.
6. **One operator source of truth.** A normalized operation schema owns
   signatures, validation, inference, effects, differentiation, lowering
   identity, language surfaces, documentation, and tests.
7. **Two graph levels.** Semantic operations and values remain separate from
   concrete Vulkan dispatch, transfer, render, media, and synchronization work.
8. **Semantics survive storage sharing.** Zero-copy interoperability preserves
   dtype, shape, layout, pixel, audio, timing, color, topology, and readiness
   metadata.
9. **Kernel policy is private.** Capabilities, measurements, and workload
   properties select kernels during lowering; callers do not select vendor or
   shader routes through the semantic API.
10. **One proven path precedes breadth.** A public capability exists only after
    one complete vertical slice passes an independent oracle or conformance
    gate and the applicable Vulkan validation.

## 3. Public contract kinds

Every public contract is one of:

| Kind | Responsibility | Examples |
|---|---|---|
| Value | Data plus semantic metadata; no active process | `Buffer`, `Matrix`, `Image`, `Audio`, `VideoFrame` |
| Operation | Stateless transformation, query, or explicit one-shot effect | `matrix::add`, `vision::resize` |
| Session | Stateful lifecycle, protocol, stream, or iterative process | `VideoDecoder`, `Presenter`, `TrainingSession` |

Values may be cheap handles or views over owned storage. Operations do not own
engines or long-lived mutable state. Sessions borrow dependencies explicitly
and define their state transitions, failure state, and terminal operation.

## 4. Source and public module structure

The physical source tree expresses dependency and ownership:

```text
src/rs/
  lib.rs                    curated public facade
  core.rs
  core/                     foundational values and metadata
  runtime.rs
  runtime/                  execution, memory, graph, Vulkan
  matrix.rs + matrix/       matrix operations
  image.rs + image/         image operations
  audio.rs + audio/         audio values, operations, sessions
  video.rs + video/         video values, operations, sessions
  vision.rs + vision/       vision operations
  render.rs + render/       render operations and sessions
  ml.rs + ml/               ML operations and sessions
  crypto.rs + crypto/       crypto operations
```

`core/` is initially an internal Rust module. It owns foundational semantic
values such as buffer, matrix, image, dtype, shape, layout, and checked size
metadata. It is not a miscellaneous utility directory and owns no Vulkan
device, queue, allocator, scheduler, or session.

`lib.rs` explicitly re-exports the admitted root vocabulary:

```rust
mod core;

pub use core::{Buffer, DType, Image, Matrix, Shape};
pub use runtime::{DeviceId, Engine, Event};
```

Wildcard public re-exports are rejected. Source placement in `core/` does not
by itself create a public `oa::core` namespace. Python may provide an identity
alias such as `oa.core.Matrix` only when compatibility requires it; the root
`oa.Matrix` remains canonical.

A crate-local `core` module shadows Rust's built-in `core` name for unqualified
internal paths. Code that needs the language crate uses `::core::...`.

## 5. Values and storage

The target value model is conceptual composition, not inheritance:

```text
Buffer       byte range, placement, allocation identity, readiness
  Matrix     dtype, shape, strides, offset, numeric semantics
  Image      extent, format, layout, planes, color semantics
  Audio      channels, samples, sample rate, channel layout
  VideoFrame planes, coded/visible extent, timestamps, readiness
  Texture    sampled/storage/render usage and image or buffer backing
```

An owning resource cannot outlive the engine service required to destroy it.
The first implementation must choose and test an explicit representation for
that edge; it must not rely on raw handle copying or destructor ordering by
convention. Views keep the underlying allocation alive and validate that byte
ranges, strides, formats, and alias relationships remain valid.

Identifiers with distinct meanings use transparent newtypes. Byte sizes,
offsets, alignments, element counts, and Vulkan-width integers use checked
arithmetic and fallible conversion at their owning boundary.

## 6. Engine, device, and completion

The target ownership is:

```text
Engine
  Vulkan entry and instance
  selected physical and logical devices
  per-device queues and memory
  kernel artifacts and pipeline caches
  semantic and executable compilation services
  scheduler and profiler
```

`DeviceId` identifies a device within one engine and is not a raw Vulkan handle.
Public `Device` information, if admitted, is a borrowed view or immutable value;
it does not independently own a logical device.

One engine may select one or several local physical devices. Remote machines
never pretend to be local devices and no local allocator spans unrelated
logical devices.

Submission returns an `Event` identifying exact completion on its originating
engine and execution epoch. Polling and waiting validate that association.
Dropping an event does not wait. Dropping a recorder restores selection only.
Dropping a session releases host state only; explicit `close`, `flush`, `drain`,
or `abort` reports terminal failure.

The public construction shape is expected to begin with:

```rust
let engine = Engine::builder()
    .devices(DeviceSelection::All)
    .build()?;
```

This syntax remains Planned until the first vertical slice proves its lifetime
and error behavior.

## 7. Authoring, graphs, and execution

Rust and Python provide eager authoring over the same semantic contracts used
for compiled execution. Eager authoring does not imply one immediate Vulkan
submission per operation.

The semantic graph contains:

- operation and value identities;
- typed attributes;
- shapes, dtypes, layouts, aliases, mutations, and effects;
- control dependencies and autograd provenance;
- placement constraints without Vulkan handles.

The executable graph contains:

- compute and indirect dispatch;
- upload, copy, transfer, and readback;
- render, media, and presentation work;
- queue selection, barriers, ownership transfer, and completion;
- concrete resources, pipelines, and push/binding layouts.

Compilation proceeds conceptually as:

```text
validate semantic contracts
  -> functionalize aliases and mutation
  -> construct backward work when requested
  -> decompose and fuse
  -> select placement and precision
  -> select or tune kernels
  -> plan transient memory
  -> schedule queues and synchronization
  -> record reusable executable work
```

One semantic operation may lower to several executable nodes. One fused
executable node retains the identity of every contributing semantic operation.

## 8. Operations and Rust syntax

Stateless operations use lowercase Rust modules:

```rust
let sum = matrix::add(&left, &right)?;
let resized = vision::resize(&image, extent)?;
```

Type-local constructors and convenience methods may delegate to the same
schema-owned implementation when they add no second validation or lowering
path. C++ `FnMatrix` and similar namespace shapes are not transliterated.

Operator traits are deferred until the failure model is proven. An operator
must not hide a panic, a submission, a wait, or a fallback. If an operator
records an infallible semantic node while validation is deferred, that deferred
failure and its diagnostic boundary must be explicit in the graph contract.

## 9. Schema and generated surfaces

One normalized operation record owns every mechanically derivable surface:

```text
operation schema
  -> Rust declarations and metadata
  -> Python bindings and signatures
  -> validation and inference
  -> autograd metadata
  -> semantic and kernel registry rows
  -> reference documentation
  -> positive and negative contract tests
```

Generated code has stable ordering, collision checks, stale-output cleanup,
ownership banners, schema/generator versions where identity persists, and an
idempotent regeneration gate. A build script writes only below Cargo's
`OUT_DIR`; checked-in generated source is updated by an explicit generation
command and verified for drift in CI.

## 10. Shader and kernel boundary

Shipping GPU programs remain first-class Slang sources under `src/slang`.
Reflection and a small OA attribute schema describe entry points, bindings,
workgroup geometry, capabilities, dtypes, layouts, specialization parameters,
and candidate identity where reliable.

Generated metadata answers what a kernel is and requires. The runtime planner
answers whether it should run for a particular operation, shape, device, and
workload. Runtime measurement and autotuning decisions are not encoded as
static shader attributes.

Development compilation, precompiled release artifacts, and caches converge on
one validated artifact representation. Shader compilation or metadata failure
fails the owning build/generation operation; it is never silently ignored.

## 11. Unsafe, concurrency, and failure

Safe semantic code sits above small unsafe implementation islands:

```text
unsafe Vulkan / allocator / FFI boundary
  -> safe runtime contracts
  -> safe semantic values and operations
```

Every unsafe operation has a stated validity, synchronization, aliasing, and
lifetime invariant. Resource owners define destruction order explicitly.
`Send` and `Sync` are admitted per type from Vulkan external-synchronization
requirements and OA locking policy, never assumed globally.

Recoverable failures use `Result<T, Error>` with preserved context. Public
library code does not panic for device absence, invalid inputs, unsupported
capabilities, allocation failure, compilation failure, or runtime execution
failure. There is no hidden CPU fallback.

## 12. Evidence and status

Every operation needs an independent oracle, property test, differential
reference, or conformance stream. GPU kernels cover zero, odd, minimal,
boundary, alias, reuse, and poison cases where meaningful. Vulkan core and
synchronization validation are separate gates; GPU-assisted validation is used
for shader memory-safety checkpoints on applicable hardware.

Documents and claims use these statuses:

- **Canonical:** current cross-module target contract.
- **Shipped:** implemented and verified by named evidence.
- **Experimental:** implemented but unstable or incompletely qualified.
- **Planned:** accepted direction with a roadmap dependency and acceptance gate.
- **Research:** alternatives and evidence without an implementation promise.

The presence of a module, type, shader, generated file, or successful compile
does not make a capability Shipped.
