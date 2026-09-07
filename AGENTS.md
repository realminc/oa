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
- `core/` is an internal source boundary for foundational semantic values and
  metadata. `lib.rs` explicitly re-exports the admitted root API; physical
  placement does not automatically create a public `oa::core` namespace.
- Values preserve semantics when storage is shared. Operations are stateless;
  stateful external and iterative processes are sessions that borrow an engine.
- Semantic operations remain separate from executable Vulkan work.
- Submission and completion are explicit and return an exact `Event`; `Drop`
  never submits, waits, drains, reads back, or finalizes a session.
- One operation schema owns every mechanically derivable Rust, Python,
  validation, autograd, registry, documentation, and test surface.
- Kernel selection is private lowering policy and remains vendor-neutral at the
  public API.
- Port behavior through complete vertical slices. Do not translate the C++
  repository line by line or create a parallel framework.

## Current stage

The repository is a structural prototype. No GPU operation is Shipped. Planned
APIs and scaffold modules are not capability claims. The first acceptance
target is the Stage 1 one-device Matrix-add slice in the roadmap.

## Required baseline

Run the narrowest relevant proof followed by:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Use the linked Rust, Vulkan, kernel, validation, documentation, Git, and rule
contracts for their additional scoped gates. Preserve unrelated user work;
`tools/gen/shader/generate.py` was already modified before this checkpoint.
