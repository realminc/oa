# OA Rust Port Roadmap

**Status:** Planned

**Updated:** 2026-09-07

**Architecture:** [OA Rust Architecture](../oaArchitecture.md)

This roadmap orders dependencies. It is not a release promise or a list of
features that currently ship.

## Current position

Stage 0 is the `v0.1.0` repository-contract checkpoint. Stage 1, the one-device
Matrix-add vertical slice, is complete as an Experimental foundation.
Builder-based automatic or exact-index
device selection, Vulkan 1.3 timeline-semaphore and synchronization2 feature
negotiation, one compute-capable logical device, VMA-backed host-visible
storage, command-pool ownership, empty command-buffer recording, checked FP32
initialization and readback, zero-extent behavior, and
storage/device/instance lifetime retention are Experimental prerequisites.
Timeline-backed checkpoint submission, non-blocking event drop, asynchronous
command-buffer retirement, exact completion query/wait, and event-after-engine
lifetime behavior are also Experimental. The bounds-checked FP32 add kernel now
has deterministic Slang-to-SPIR-V compilation, reflected ABI validation,
Vulkan 1.3 `spirv-val` validation, embedding, descriptor-indexing capability
negotiation, and private compute-pipeline construction. Executable add
recording, a generic `ComputeDispatch` and engine submission path, checked
direct asynchronous dispatch, result-owned completion, host-observation
waiting, non-blocking `try_read_f32`, and independent host-oracle coverage are
Experimental. Stage 2 now has one normalized schema generating 19 out-of-place
elementwise Rust functions, 19 `f32` physical variants, and one `i32` add
variant with stable kernel/artifact lookup, Slang kernels, and hardware-oracle
tests. The first Stage 4 slice has also landed ahead of reusable execution: a
schema-generated FP32 `matrix::mat_mul_nt` preserving the OA
`[M, K] × [N, K] -> [M, N]` convention, with a 64×64×16 shared-memory
physical tile, odd/zero-shape CPU-oracle coverage, and a runnable SDK tutorial.
It is not yet a performance-qualified GEMM routing system. A shared
engine-owned bindless
descriptor heap serves all generated pipelines. A private engine-owned
execution session records non-empty eager operations, joins their owned graph
snapshots, and submits one hazard-planned command buffer at blocking host
observation or `Engine::checkpoint`. `try_read` never submits or waits. The
graph recorder derives per-buffer RAW, WAR, and WAW barriers while omitting
read-after-read barriers. Isolated capture and immutable engine-associated
`ExecutionPlan` replay are Experimental: capture rejects pending eager work,
nesting, failure, and empty work; replay rejects foreign engines and returns an
exact event. Explicit timed replay records a fresh Vulkan timestamp query pair
around the complete executable graph and exposes its wrap-corrected device
duration through that exact event. An Experimental six-shape MatMul recording
suite now correctness-gates captured-plan replay, runs at least seven fresh
processes, and preserves raw timing and build/device provenance. Command-buffer
caching, calibrated clocks,
mutable stable input slots, semantic graph identity, broadcasting, mutation
contracts, and broader generated documentation remain incomplete.
Engine-owned structured console/file logging, weak thread-local selection,
custom component tags, release compile-out for trace/debug call sites, and
explicit failure-bearing flush/close boundaries are Experimental.
Existing unrelated modules and shaders remain design scaffolding unless a
later status document names their implementation and verification evidence.

## Stage 0 — Repository contract

### Outcome

- repository-owned `AGENTS.md`;
- template-owned shared rules with the Rust contract strengthened at its owner;
- canonical architecture separated from research notes;
- explicit compatibility ledger for C++ concepts;
- `core/`, `runtime/`, and domain operation boundaries established;
- `Engine` selected as the sole local execution owner;
- first vertical slice specified with acceptance evidence.

### Exit gate

- documentation links resolve;
- rules do not impose contradictory C++ and Rust public syntax;
- no document presents scaffolded APIs as shipped;
- formatting, lint, test, and generation commands are defined;
- unrelated user work remains untouched.

## Stage 1 — One-device Matrix add

### Required path

```text
Engine::builder
  -> Vulkan entry and instance
  -> physical-device discovery and selection
  -> logical device and compute queue
  -> allocator and storage buffer
  -> checked host upload
  -> FP32 matrix::add returning Matrix
  -> executable compute dispatch
  -> asynchronous completion retained by the result
  -> checked host observation waits for exact completion
  -> independent host oracle
```

### Contract decisions proved here

- engine, device, allocation, matrix, plan, and event lifetimes;
- device and event identity;
- safe runtime boundary above `ash` and `vk-mem`;
- matrix shape, dtype, stride, byte-size, and alias validation;
- semantic versus executable operation representation;
- one generic engine submission boundary rather than per-operation engine
  methods;
- shader artifact, reflection, binding, and push-data contract;
- error propagation and destruction order;
- direct-dispatch bounds and zero-work behavior.

### Test matrix

- zero-sized input according to the admitted zero-size contract;
- one element;
- odd element counts around the workgroup width;
- several multidimensional shapes;
- mismatched shapes and dtypes;
- byte-size and dispatch-count overflow;
- alias and output reuse cases;
- allocation and shader/pipeline failure propagation;
- repeated submission and event association;
- deterministic fresh-process CPU comparison.

### Runtime evidence

- exact Vulkan SDK/registry, loader, device, driver, enabled capabilities, and
  build provenance;
- core validation with zero unexpected messages;
- synchronization validation with zero unexpected messages;
- GPU-assisted validation for shader bounds on an applicable real GPU;
- no CPU execution fallback.

This stage admits only the tested device/capability path. It does not establish
multi-vendor or multi-device support.

## Stage 2 — Operation schema seed (Experimental checkpoint complete)

The first normalized schema now generates:

- 19 out-of-place elementwise Rust signatures and rustdoc;
- 19 `f32` physical variants plus one `i32` add proof with exact dtype routing;
- stable private kernel identities and embedded artifact metadata;
- bounds-checked Slang kernels with reflected OA attributes and push ABI;
- an external odd-size hardware test using independent golden values;
- positive and negative generator/schema tests.

Regeneration is deterministic and idempotent. A normal Cargo build does not
modify checked-in files. The handwritten add registry and shader route have
been removed. Validation and shape inference remain shared handwritten lowering
helpers until their schema-generated fixture layer lands.

## Stage 3 — Reusable execution (Experimental baseline complete)

The baseline now captures an isolated eager recording into an immutable plan
and explicitly re-records and submits its retained executable graph repeatedly.
Remaining work must establish:

- semantic capture ownership;
- executable plan lifetime;
- stable resource retention;
- event epochs and dependency chaining;
- observable graph breaks, compilation, and fallback counters;
- explicit readback and inspection boundaries.

Exact whole-plan device timing and a correctness-gated fresh-process MatMul
recording suite are Experimental checkpoints. Clock-domain calibration,
phase/node timestamps, physical artifact telemetry, accepted baselines, and
cross-implementation comparison remain separate dependencies.

Operator overloading remains deferred until this stage proves where validation
and lowering failures are reported without panics.

## Stage 4 — Matrix foundation

Grow the Matrix surface by complete schema-owned slices:

1. creation/upload and fill;
2. elementwise arithmetic (out-of-place FP32 baseline complete) and broadcasting;
3. reduction;
4. GEMM baseline (Experimental FP32 `mat_mul_nt` tiled route and reusable-plan
   measurement complete; routing, specialized variants, and qualification remain);
5. autograd seed.

Each operation requires its own oracle and edge-case pack. Kernel variants and
tuning enter only after the baseline semantic route is stable.

## Stage 5 — Multi-device local execution

Prove explicit transfer between two local devices before adding placement
automation. Admit transport paths in evidence order:

1. same logical device or device-group path with queried peer capabilities;
2. compatible external memory plus explicit external synchronization;
3. bounded host staging correctness path.

Remote transport is not part of this stage.

## Stage 6 — Image and vision pipeline

Add `Image` storage views and metadata, then one complete upload → resize →
readback slice. Preserve extent, format, layout, color, readiness, and alias
semantics. Reuse the same engine, schema, graph, event, and shader systems.

## Stage 7 — ML inference seed

Add one small end-to-end inference workload using the established Matrix
contracts. Establish capability-filtered kernel candidates and a correctness
baseline before autotuning or vendor-specialized routes.

## Stage 8 — Stateful media and presentation

Add one session at a time after its state machine, borrowed-engine lifetime,
external synchronization, and explicit close/drain behavior are specified.
Vulkan Video and WSI are implementation backends, not the public media model.

## Deferred until their dependencies exist

- training and optimizer sessions;
- generalized autograd;
- distributed execution and collectives;
- remote workers and satellites;
- audio device graphs;
- broad rendering and UI;
- release performance comparisons;
- Python packaging beyond the first schema parity proof.

## Rejected migration patterns

- translating the C++ directory tree or class hierarchy line by line;
- maintaining two execution owners or a default global owner;
- publishing TODO-backed APIs to reserve names;
- handwritten registries that duplicate schemas or shader reflection;
- build scripts that rewrite source or ignore generator failure;
- expanding domain breadth before the first vertical slice is proven.
