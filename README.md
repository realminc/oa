# oa

Rust implementation of OA's GPU-first semantic computing architecture.

The project is currently establishing its ownership model, source boundaries,
generation contracts, and first executable Vulkan slice. Existing modules and
shaders are structural prototypes unless named by a verified checkpoint.

## Direction

OA keeps Vulkan explicit through `ash`, uses Slang for GPU programs, and places
a safe semantic API above narrowly contained unsafe runtime code.

The target public surface is language-like:

```rust
use oa::{matrix, Engine, Matrix};

// Planned syntax; not implemented yet.
let engine = Engine::builder().build()?;
let sum = matrix::add(&left, &right)?;
let event = engine.submit()?;
engine.wait(event)?;
```

Foundational values live physically under an internal `core/` module and are
explicitly re-exported from `lib.rs`. Stateless operations live in lowercase
domain modules. Stateful codecs, streams, presentation, training, and transport
remain explicit session types that borrow an engine.

`Engine` is the sole local owner of Vulkan devices, memory, queues, kernels,
scheduling, and profiling. Recording, submission, completion, readback, and
session shutdown remain explicit.

## Status

No GPU operation is currently classified as Shipped. The first planned
checkpoint is one-device FP32 Matrix add from checked upload through explicit
submission and readback to an independent host oracle.

## Documentation

- [Documentation index](docs/README.md)
- [Canonical architecture](docs/internal/architecture/oaArchitecture.md)
- [Port roadmap](docs/internal/architecture/roadmap/portRoadmap.md)
- [C++ to Rust compatibility ledger](docs/internal/porting/oaCompatibility.md)

## Development

The intended baseline gates are:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

These baseline gates establish source hygiene. The first vertical slice adds
the independent correctness and Vulkan validation gates needed for a capability
claim; a successful compile or empty test run is not such a claim.
