# OA Rust Architecture

**Status:** Canonical

**Updated:** 2026-09-10

**Roadmap:** [Rust port roadmap](roadmap/portRoadmap.md)

**Compatibility:** [OA compatibility ledger](../porting/oaCompatibility.md)

**Source mapping:** [Rust source and module structure](oaSourceStructure.md)

This document defines the target architecture of OA's Rust implementation.
Current source proves implementation status; the roadmap orders planned work;
subsystem documents may add detail without redefining these boundaries.

## 1. Product boundary

OA is a GPU-first semantic computing library implemented on Vulkan. It owns:

- typed device-resident values;
- stateless numerical, ML, vision, audio, render, and cryptography operations;
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
4. **One retained execution lifetime.** Rust constructs an `Engine` explicitly.
   Values may retain an opaque handle to that same engine's services so storage
   and pending work remain valid; they never create a second engine or device
   owner. Python may expose one binding-owned process engine for its established
   `oa.FnMatrix` convenience surface.
5. **Simple eager authoring, visible blocking.** Ordinary domain operations
   validate, record, and return values without requiring submit/wait ceremony.
   Host observation may flush, submit, and wait; `try_*` observation never
   waits. Explicit submission remains a failure-bearing advanced boundary, and
   destruction never executes or waits.
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
11. **One spatial-math convention and formula authority.** `vlm` uses
    right-handed `+X` right, `+Y` up, camera-forward `-Z`, row-major storage,
    row-vector multiplication, and Vulkan `[0, 1]` clip depth. Raster Y belongs
    to viewport state. Consumers use named VLM operations instead of locally
    transposing, sign-correcting, or patching spatial matrices.

## 3. Public contract kinds

Every public contract is one of:

| Kind | Responsibility | Examples |
|---|---|---|
| Value | Data plus semantic metadata; no active process | `Buffer`, `Matrix`, `Image`, `Audio`, `VideoFrame` |
| Operation | Stateless transformation, query, or explicit one-shot effect | `matrix::add`, `image::resize` |
| Session | Stateful lifecycle, protocol, stream, or iterative process | `VideoDecoder`, `MediaPlayer`, `Presenter`, `TrainingSession` |

Values may be cheap handles or views over owned storage. Operations do not own
engines or long-lived mutable state. Sessions borrow dependencies explicitly
and define their state transitions, failure state, and terminal operation.

## 4. Source and public module structure

The physical source tree expresses dependency and ownership:

```text
src/rs/
  lib.rs                    curated public facade
  core.rs
  core/                     public values, metadata, errors, primitives
    memory.rs + memory/     host-memory contracts and private ISA paths
    vlm.rs + vlm/           packed host spatial values and formulas
  runtime.rs
  runtime/
    dispatch.rs             generic executable compute descriptions
    engine.rs               public engine implementation and orchestration
    event.rs                public completion contract
    executable_graph.rs     private owned nodes and resource hazards
    plan.rs                 public immutable captured execution
    session.rs              private eager recording and batch ownership
    storage.rs              safe value-storage boundary
    shader/                 private Slang artifacts, reflection, metadata
    vk/                     private Vulkan handles, memory, execution, timing
  matrix.rs + matrix/       matrix operations
  image.rs + image/         image codecs and stateless image operations
  audio.rs + audio/         audio values, codecs, operations, device sessions
  video.rs + video/         video-frame values, codecs, and device sessions
  media.rs + media/         cross-track sources, clocks, transport, and playback
  vision.rs + vision/       image/video interpretation operations
  render.rs + render/       render operations and sessions
  ml.rs + ml/               ML operations and sessions
  cryptography.rs + cryptography/  hashing, secure memory, and PQC operations
src/py/
  lib.rs + *.rs             PyO3 bindings over the Rust public API
sdk/py/
  python/oa/                importable Python package and type information
  test/ + tutorials/        Python verification and SDK workflows
```

`core` is the public foundation module. It owns foundational semantic values
such as buffer, matrix, image, dtype, shape, and layout; checked size and
identity metadata; and OA's backend-neutral `Error`, `ErrorKind`, and `Result`
contracts. It is not a miscellaneous utility directory and owns no Vulkan
device, queue, allocator, scheduler, logging sink, or session. Stateful
diagnostic output is application-owned or composed beneath `Engine`; `core`
may own only backend-neutral diagnostic vocabulary.

`core::memory` is a bounded backend-neutral host-memory policy surface. It owns
checked byte-slice copy, explicit one-way streaming copy, secure erasure, and
equality contracts. Architecture-specific SIMD remains private; GPU
allocations, mapped-range lifetimes, upload rings, flushes, and transfer
submission remain runtime responsibilities. OARS does not replace Rust's
`core`, `alloc`, or `std` containers and synchronization vocabulary.

`lib.rs` explicitly re-exports the admitted root vocabulary:

```rust
pub mod core;
pub mod audio;

pub use core::{Buffer, DType, Error, ErrorKind, Image, Matrix, Result, Shape};
pub use audio::{Audio, AudioCapture, AudioEncoder, AudioPlayer};
pub use runtime::{DeviceId, Engine, Event};
```

The example includes Planned types and describes the target facade; source and
subsystem status documents decide what currently exists. Wildcard public
re-exports are rejected. Each type has one owning module and implementation;
the crate root explicitly re-exports admitted principal values and sessions as
identity aliases. Thus `oa::Matrix` and `oa::core::Matrix`, or
`oa::AudioPlayer` and `oa::audio::AudioPlayer`, identify the same item rather
than parallel implementations. Stateless functions remain on their owning
lowercase module and are not duplicated at the root.

Callers may introduce local OpenMaya-style abbreviations without expanding the
OA public surface:

```rust,ignore
use oa::{audio as oaa, core as oac, ml as oaml, vision as oacv};

let player = oaa::AudioPlayer::open(&engine, config)?;
```

OA does not publish separate `oaa`, `oac`, `oaml`, or `oacv` crates or alias
modules. Python mirrors the owning modules and may expose the same admitted
root class identity, so `oa.AudioPlayer is oa.audio.AudioPlayer`.

The crate-local `core` module shadows Rust's built-in `core` name for
unqualified internal paths. Code that needs the language crate uses
`::core::...`.

Domain nesting is not ontology. Audio and Video are media domains, but remain
public siblings of `media`: `media` coordinates timed tracks and transport
rather than owning every audio/video value or operation. Likewise, Vision
consumes Image and VideoFrame values; it does not own their codecs or storage.
Render consumes Image and VideoFrame through Texture or other typed resource
views; it does not redefine those source values. The detailed target is
[OA Rust media boundary](../media/oaMedia.md).

## 5. Values and storage

The target value model is conceptual composition over shared storage, not
inheritance:

```text
Buffer / BufferView     byte range, placement, allocation identity, readiness
  ├─ Matrix             dtype, shape, strides, offset, numeric semantics
  │    ├─ Audio         planar samples, sample rate, channel layout
  │    └─ Image         when one dense matrix represents admitted pixels
  ├─ Image planes       extent, format, layout, color semantics
  │    └─ VideoFrame    coded/visible extent, timestamps, decode readiness
  └─ other typed views  encoded bytes, vertices, indices, or native resources

Texture                 sampled/storage/render usage over admitted image or
                        buffer backing; never an alias for Image or VideoFrame
```

These edges mean “is backed by” or “exposes a checked view of,” never “is a.”
`Matrix` is OA's N-dimensional dense numerical value and replaces a separate
Tensor vocabulary; it is not the universal storage base. Compressed packets,
multi-plane video images, render textures, and topology-bearing geometry must
not be forced into Matrix merely because some of their storage is numeric.

An owning resource retains the internal services required to destroy its
storage and complete already-produced work, even if the public `Engine` handle
is dropped first. This is shared lifetime of the same execution owner, not an
independent runtime facade. Views keep the underlying allocation alive and
validate that byte ranges, strides, formats, and alias relationships remain
valid.

Identifiers with distinct meanings use transparent newtypes. Byte sizes,
offsets, alignments, element counts, and Vulkan-width integers use checked
arithmetic and fallible conversion at their owning boundary. See
[OA Rust values and storage](../core/oaValues.md) for the per-value contracts
and admission gate.

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
  structured logging session
```

Each engine owns one structured logging session configured by `LogOptions`.
Weak thread-local selection routes namespaced logging macros on the owning
thread and restores the previous live selection for nested engines. An
unassociated thread falls back to standard error. Logging never owns runtime
services, submits work, waits, or establishes completion; explicit flush and
close boundaries report retained sink failures.

`DeviceId` identifies a device within one engine and is not a raw Vulkan handle.
Public `Device` information, if admitted, is a borrowed view or immutable value;
it does not independently own a logical device.

One engine may select one or several local physical devices. Remote machines
never pretend to be local devices and no local allocator spans unrelated
logical devices.

Explicit submission returns an `Event` identifying exact completion on its
retained originating device and execution epoch. `Event::is_complete` queries
that timeline and `Event::wait` is an explicit advanced blocking boundary.
Eager values retain the completion needed by host observation, so ordinary
math does not traffic in events. Dropping an event does not wait; the engine's
retirement service keeps submitted resources and their device graph alive
until completion. Dropping a recorder restores selection only. Dropping a
session releases host state only; explicit `close`, `flush`, `drain`, or
`abort` reports terminal failure.

The Experimental one-device construction shape is:

```rust
let engine = Engine::builder()
    .devices(DeviceSelection::Automatic)
    .build()?;
```

`DeviceSelection::Index` selects an exact physical-device ordinal reported by
the active Vulkan loader. Multi-device `All` selection remains Planned until
Stage 5 proves explicit transfer and per-device ownership.

## 7. Authoring, graphs, and execution

Rust and Python provide eager authoring over the same semantic contracts used
for compiled execution. An eager operation returns its output value. The
engine-owned private execution session records non-empty operations into an
eager batch. Blocking host observation submits the producing batch and waits;
`Engine::checkpoint` explicitly submits pending eager work and returns its
exact event. This batching policy does not change the domain-operation surface.

Host observation such as `Matrix::read::<T>` or `Matrix::read_f32` flushes the
producing work and waits for its exact completion before exposing values.
Non-blocking observation uses an explicitly named `try_*` API. Advanced callers
may capture work and use explicit engine submission and events for overlap,
profiling, local multi-device placement, or distributed orchestration.

The Experimental reusable path is:

```rust
let (plan, output) = engine.capture(|| {
    let product = matrix::mat_mul_nt(&input, &weight)?;
    matrix::add(&product, &bias)
})?;
let event = engine.submit(&plan)?;
event.wait()?;
let values = output.read_f32()?;
```

Capture is isolated and never submits or waits. It currently requires an empty
eager session, rejects nesting and empty captures, and restores eager recording
after success, error, or unwind. Plans retain their captured outputs and stable
read-only Matrix input identities and may be submitted repeatedly only through
their originating engine. Shape- and dtype-identical input rebinding is
explicit, never waits, rejects aliases, and invalidates the compiled command.
Unchanged untimed submissions reuse one simultaneously submittable recording.

Whole-plan device timing is an explicit instrumented submission rather than a
property of all execution:

```rust
let event = engine.submit_timed(&plan)?;
event.wait()?;
let device_duration = event.device_duration()?;
```

The event owns the measurement for that replay. Device duration does not include
host recording, queue submission, waiting, or readback time and is not a
host/device clock correlation.

The semantic graph contains:

- operation and value identities;
- typed attributes;
- shapes, dtypes, layouts, aliases, mutations, and effects;
- control dependencies and autograd provenance;
- placement constraints without Vulkan handles.

`SemanticGraph::debug_report_json` serializes that backend-independent evidence
under the donor `oa.semantic_graph.v2` report schema. This is a diagnostic
diff surface, not a persisted executable-plan descriptor or cache ABI.

The executable graph contains:

- compute and indirect dispatch;
- upload, copy, transfer, and readback;
- render, media, and presentation work;
- queue selection, barriers, ownership transfer, and completion;
- concrete resources, pipelines, and push/binding layouts.

Domain lowering emits generic executable descriptions such as
`ComputeDispatch`; it does not add `submit_matrix_add`, `submit_image_resize`,
or other operation-specific methods to `Engine`. The engine owns generic
record/submit paths. Kernel identities and pipeline lookup are generated from
shader metadata as that registry lands.

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
let one = matrix::ones(&engine, [2, 3])?;
let two = matrix::full(&engine, [2, 3], 2.0)?;
let sum = matrix::add(&one, &two)?;
let product = matrix::mat_mul_nt(&one, &two)?; // [2, 3] × [2, 3]ᵀ -> [2, 2]
let values = sum.read_f32()?;
```

Type-local constructors and convenience methods may delegate to the same
schema-owned implementation when they add no second validation or lowering
path. Rust does not transliterate C++ namespace casing or emulate namespace
reopening: Core `oa::FnMatrix::add` becomes `oa::matrix::add`, while the ML
extension `oa::FnMatrix::gelu` becomes `oa::ml::matrix::gelu`. Python retains
`oa.FnMatrix` compatibility and may hide its binding-owned process engine,
matching the established three-line authoring surface.

The first Experimental BLAS slice names the physical semantic convention
directly: `matrix::mat_mul_nt` accepts `[M, K]` and `[N, K]` and produces
`[M, N]`. It submits through the same generic engine boundary as elementwise
operations; applications do not manually submit or wait unless they enter a
future explicit orchestration surface.

Operator traits are deferred until the failure model is proven. An operator
must not hide a panic, wait, or fallback. If an operator
records an infallible semantic node while validation is deferred, that deferred
failure and its diagnostic boundary must be explicit in the graph contract.

Stateful ML layers implement an object-safe `oa::ml::Module` trait and own one
constructor-populated `ModuleRegistry`. Direct parameters, non-trainable
buffers, and owned `Rc<dyn Module>` children share one local namespace.
Recursive paths, traversal, parameter counts, train/eval propagation, and
future persistence derive from that single tree. Duplicate identity is an
error; an optimizer must never update one stable parameter twice through two
paths. The registry owns structure, not execution: modules and parameters
retain Matrix handles while `Engine` remains the sole runtime owner.

Rust does not reproduce the C++ `Module` inheritance hierarchy or `Nn*` type
prefixes. Trait composition and the existing `oa::ml::nn` namespace express
those roles directly.

The public translation rule is uniform:

| OA C++ | Rust | Role |
|---|---|---|
| `oa::Matrix` | `oa::Matrix` | Semantic value identity is preserved. |
| Core `oa::FnMatrix::add` | `oa::matrix::add` | Core numerical operation module replaces the `Fn*` namespace. |
| ML `oa::FnMatrix::gelu` | `oa::ml::matrix::gelu` | ML ownership replaces C++ namespace reopening. |
| `oa::FnLoss::crossEntropy` | `oa::ml::loss::cross_entropy` | Losses retain their distinct ML operation family. |
| `oa::Image` | `oa::Image` | Reusable pixel value; not owned by Vision, Render, or UI. |
| `oa::FnImage::resize` | `oa::image::resize` | Stateless image transformation. |
| `oa::Audio` | `oa::Audio` | Finite planar audio value over Matrix storage. |
| `oa::FnAudio::normalize` | `oa::audio::normalize` | Stateless audio transformation. |
| `oa::VideoFrame` | `oa::VideoFrame` | Timed, readiness-bearing frame value. |
| `oa::FnVideo::*` | `oa::video::*` | Stateless video/bitstream transformations only. |
| `oa::FnDetection::*` | `oa::vision::*` | Image/video interpretation. |
| `oa::AudioEncoder` | `oa::audio::AudioEncoder` | Stateful session type and methods. |
| `oa::AudioCapture` | `oa::audio::AudioCapture` | Stateful device-input session. |
| `oa::AudioPlayer` | `oa::audio::AudioPlayer` | Stateful incremental decode/output session. |
| `oa::VideoPlayer` | `oa::media::MediaPlayer` where synchronized A/V is intended | Cross-track playback session rather than a Vision owner. |

One-shot verbs use verb modules or functions such as `audio::decode_file`.
Noun modules such as `decoder` or `encoder` are reserved for stateful session
implementations when that distinction prevents ambiguity. File boundaries may
remain private beneath a facade; only explicit re-exports define public API.
The complete migration spelling and placement rules live in
[C++ to Rust API translation](../porting/oaCppToRust.md).

Principal public values and stateful sessions may additionally be explicitly
re-exported from `lib.rs`:

```rust,ignore
let player = oa::AudioPlayer::open(&engine, config)?;
let same_type = oa::audio::AudioPlayer::open(&engine, config)?;
```

This is one definition with two identity paths. Root aliases preserve the
compact C++/Python class vocabulary; owning modules preserve discovery and
domain structure. Free operations such as `audio::decode_file` and
`matrix::add` remain module-only.

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

The OA C++ operation schemas, language-neutral algorithms, Slang kernels, and
their tests are the donor authority for the Rust port. Before creating a Rust
operation or shader, the port records the donor path and classifies the change
as verbatim reuse, mechanical ABI/module adaptation, Rust ownership/API
redesign, or a justified replacement. Rust-native ownership does not justify
reimplementing an already proven numerical kernel. Replacements require the
same oracle plus differential and performance evidence against the donor; the
portable donor path remains available until that gate passes.

## 10. Shader and kernel boundary

The detailed contract is [OA Rust Compute Kernel System](../compute/oaComputeKernel.md).

Shipping GPU programs remain first-class Slang sources under `src/slang`.
Reflection and a small OA attribute schema describe entry points, bindings,
workgroup geometry, capabilities, dtypes, layouts, specialization parameters,
candidate identity, writable-range partitioning, and collision policy where
reliable. Non-atomic writes require a proved exclusive physical domain; atomic
and reduction routes record their storage and numeric-order contract.

Each compiled shader module owns exactly one stage entry point named `main`.
Semantic and physical identity comes from its stable kernel ID, OA attributes,
source/artifact identity, and compiled module hash—not from a globally unique
Slang function name. A source containing several independently dispatchable
kernels is split into one module per kernel before it enters the artifact
registry.

Generated metadata answers what a kernel is and requires. The runtime planner
answers whether it should run for a particular operation, shape, device, and
workload. Runtime measurement and autotuning decisions are not encoded as
static shader attributes.

Development compilation, precompiled release artifacts, and caches converge on
one validated artifact representation. Shader compilation or metadata failure
fails the owning build/generation operation; it is never silently ignored.

Target-independent Slang compilation, reflection, artifact identity, and
kernel metadata live under `runtime/shader`. Vulkan shader modules, descriptor
layouts, compute pipelines, and Vulkan pipeline caches live under `runtime/vk`.
There is no public shader subsystem and no second Vulkan-shaped engine facade.

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

Compute execution is detailed in [OA Rust Compute Architecture](../compute/oaCompute.md),
and performance claims follow [OA Rust Performance Evidence](../performance/oaPerformance.md).

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
