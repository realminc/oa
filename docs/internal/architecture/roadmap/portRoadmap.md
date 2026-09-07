# OA Rust Port Roadmap

**Status:** Planned

**Updated:** 2026-09-07

**Architecture:** [OA Rust Architecture](../oaArchitecture.md)

This roadmap orders dependencies. It is not a release promise or a list of
features that currently ship.

## Current position

Stage 0 is the `v0.1.0` repository-contract checkpoint. Stage 1, the one-device
Matrix-add vertical slice, is next. Existing modules and shaders remain design
scaffolding unless a later status document names their implementation and
verification evidence.

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
  -> FP32 matrix::add semantic operation
  -> executable compute dispatch
  -> explicit submit returning Event
  -> explicit wait
  -> checked readback
  -> independent host oracle
```

### Contract decisions proved here

- engine, device, allocation, matrix, plan, and event lifetimes;
- device and event identity;
- safe runtime boundary above `ash` and `vk-mem`;
- matrix shape, dtype, stride, byte-size, and alias validation;
- semantic versus executable operation representation;
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

## Stage 2 — Operation schema seed

Extract Matrix add into the first normalized operation schema and generate:

- Rust signature and semantic metadata;
- validation and inference fixtures;
- kernel association metadata;
- reference documentation;
- positive and negative contract tests.

Regeneration must be deterministic and idempotent. A normal Cargo build does
not modify checked-in files. The handwritten Stage 1 seam is removed after the
generated route passes the same oracle and validation gates.

## Stage 3 — Reusable execution

Add immutable execution plans and explicit repeated submission over the same
operation contract. Establish:

- semantic capture ownership;
- executable plan lifetime;
- stable resource retention;
- event epochs and dependency chaining;
- observable graph breaks, compilation, and fallback counters;
- explicit readback and inspection boundaries.

Operator overloading remains deferred until this stage proves where validation
and lowering failures are reported without panics.

## Stage 4 — Matrix foundation

Grow the Matrix surface by complete schema-owned slices:

1. creation/upload and fill;
2. elementwise arithmetic and broadcasting;
3. reduction;
4. GEMM baseline;
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
