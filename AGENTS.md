# OA Rust Agent Contract

This file is the mandatory repository-specific entry point for automated
contributors. Shared engineering policy remains owned by the linked rules in
`template/.cursor/rules`; do not copy those rules into this repository.

## Read first

Before changing architecture or public behavior, read:

1. `docs/internal/architecture/oaArchitecture.md` — canonical Rust target.
2. `docs/internal/architecture/roadmap/portRoadmap.md` — dependency order and
   current implementation stage.
3. `docs/internal/porting/oaCompatibility.md` when translating an OA C++
   concept.
4. The relevant linked `.cursor/rules/*.mdc`, subsystem document, source, and
   tests.

The C++ repository is evidence for behavior and hard-won constraints, not a
source-layout or implementation template. If the Rust architecture, C++
architecture, a plan, and live code disagree, report the conflict.

## Repository-specific invariants

- `Engine` is the sole local owner of Vulkan instances, logical devices,
  memory, queues, kernels, scheduling, and profiling. Do not add a public
  `Runtime` owner or second execution facade.
- `core` is the public foundation module for semantic values, checked metadata,
  shared error/result contracts, and other backend-neutral primitives. `lib.rs`
  explicitly re-exports the admitted common vocabulary at the crate root.
- Raw Vulkan, allocator, OS, and third-party errors remain private to their
  adapters and are translated into the backend-neutral `core::Error` contract.
- Values preserve semantics when storage is shared. Operations are stateless;
  stateful external and iterative processes are sessions that borrow an engine.
- Semantic operations remain separate from executable Vulkan work.
- Eager domain operations return semantic values and do not require a public
  submit/wait ceremony. Explicit engine submission remains available for
  capture, orchestration, and profiling and returns an exact `Event`. Host
  observation may flush and wait; `try_*` observation never waits. `Drop` never
  submits, waits, drains, reads back, or finalizes a session.
- One operation schema owns every mechanically derivable Rust, Python,
  validation, autograd, registry, documentation, and test surface.
- Kernel selection is private lowering policy and remains vendor-neutral at the
  public API.
- Port behavior through complete vertical slices. Do not translate the C++
  repository line by line or create a parallel framework.

## Current stage

The repository remains Experimental. No GPU operation is Shipped. The
one-device runtime, schema-generated out-of-place `f32` elementwise family,
`i32` Matrix-add dtype proof, and FP32 `matrix::mat_mul_nt` with a generated
64×64×16 tiled kernel and runnable SDK oracle are implemented checkpoints.
Broader integer operations, broadcasting, in-place mutations, GEMM routing and
specialized variants, low-precision storage, and broader domains remain
incomplete. Engine-owned structured logging and explicit whole-plan Vulkan
device timing are Experimental. The fresh-process MatMul recording runner and
six-shape suite are Experimental; accepted baselines, cross-implementation
comparison, calibrated host/device clocks, structured runtime metrics, and
graph diagnostics remain Planned.
The private executable-graph foundation snapshots resolved compute dispatches,
records multiple nodes in one primary command buffer, and derives per-buffer
RAW, WAR, and WAW barriers. A private engine-owned execution session batches
eager work until blocking observation or `Engine::checkpoint`; `try_read`
neither submits nor waits. Isolated capture and immutable engine-associated
plan replay are Experimental. Timed replay uses an independently owned query
pair per submission. Semantic graph identity, stable mutable plan slots,
compiled command caching, calibrated clocks, and non-compute graph nodes remain
Planned.
Planned APIs and scaffold modules are not capability claims.

## Required baseline

Run the narrowest relevant proof followed by:

```bash
python3 -m unittest discover -s tools/gen/fn/tests -v
python3 -m unittest discover -s tools/profiling/tests -v
python3 tools/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Use the linked Rust, Vulkan, kernel, validation, documentation, Git, and rule
contracts for their additional scoped gates. Preserve unrelated user work.
