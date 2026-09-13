# OA

OA is a GPU-first semantic computing engine written in Rust. One explicit
`Engine` owns Vulkan devices, memory, queues, scheduling, kernels, and
profiling; typed values and domain operations build on that owner without
exposing backend machinery through the public API.

This repository is the new primary OA implementation. The earlier C++ codebase
continues separately as the donor and compatibility reference.

## Three lines of compute

Rust:

```rust
use oa::{matrix, Engine};

let engine = Engine::new()?;
let one = matrix::ones(&engine, [2, 3])?;
let two = matrix::full(&engine, [2, 3], 2.0)?;
let total = matrix::add(&one, &two)?;

assert_eq!(total.read_f32()?, [3.0; 6]);
# Ok::<(), oa::Error>(())
```

Python preview—the same module ownership and native runtime:

```python
import oa

engine = oa.Engine()
one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
total = oa.matrix.add(one, two)

assert total.read_f32() == [3.0] * 6
```

Python abbreviations are ordinary local imports, not parallel APIs:

```python
import oa.core as oac
import oa.matrix as oam
```

## What works today

OA is an experimental, executable rewrite—not a structure-only stub. Current
checked vertical slices include:

- Vulkan 1.3 device selection, VMA-backed storage, bindless descriptors,
  asynchronous retirement, timeline events, eager batching, semantic capture,
  immutable execution plans, replay, rebinding, hazard analysis, and Vulkan
  timestamp evidence;
- schema-owned Matrix elementwise, reduction, indexing, RNG, transpose, gather,
  tiled FP32 matrix multiplication, broadcasting, and reverse-mode operations;
- ML modules, autograd, Adam/AdamW/SGD/Muon, training iterators, callbacks,
  checkpoints, RNN/GRU/Transformer/MoE/Mamba-3 NLP tutorials, reinforcement
  learning, VQ, and the in-progress Animation Language Model stack;
- Image operations and codecs, planar Audio and codecs/DSP/sessions, Video
  containers/decoding, Vision detection/metrics, Render value foundations,
  cryptographic hashes/Merkle/PQC, secure memory, and Vulkan batch hashing;
- a native PyO3 binding preview for Engine, FP32 Matrix creation, addition,
  `mat_mul_nt`, metadata, and synchronized readback.

No GPU operation is classified as stable yet. Capability claims are tied to
the compatibility ledger, independent oracles, and recorded hardware evidence.

## Architecture

```text
semantic values and operations
             │
             ▼
      semantic operation graph
             │ private lowering
             ▼
      executable Vulkan graph
             │
             ▼
 Engine-owned queues, memory, events, profiling
```

- Values carry semantics; shared storage does not erase type identity.
- Operations are stateless. Stateful external or iterative work is a session.
- The semantic graph stays separate from executable backend work.
- Eager operations return values; explicit submit/wait is reserved for capture,
  orchestration, profiling, multi-device, and distributed work.
- One operation schema owns derivable Rust, Python, validation, autograd,
  registry, documentation, and test surfaces.
- Slang kernels are embedded in the binary; runtime users do not ship loose
  `.spv` files.

## Performance

The Rust rewrite does not assume safety costs speed. OA contains measured
low-level paths where a distinct contract earns them—for example native-target
small copies reached up to 1.82× stock Rust and explicit one-way streaming
reached 1.16–1.77× for tested 1 KiB–4 MiB chunks. Portable ordinary copy remains
within the recorded 3% band in the complete sweep. These are platform-local
experimental results, not universal claims; the full distributions, clocks,
temperatures, binaries, and counterexamples are retained in the
[memory comparison](docs/internal/performance/oaMemoryComparison.md).

The [VLM comparison](docs/internal/performance/oaVlmComparison.md) likewise
publishes wins, parity, and remaining C++ gaps. GPU matrix multiplication is
correctness-gated but not yet performance-qualified against the full donor
routing system.

## Build

Requirements: Rust 1.98, Python 3, `slangc`, `spirv-val`, Vulkan 1.3, and a
compatible driver. Linux builds use Clang/LLD for native linking while `rustc`
and LLVM compile Rust.

```bash
cargo build --release
cargo run --release --example core_mat_mul_intro
```

Cargo keeps intermediate artifacts in `target/`. OA’s staging tool copies only
runnable executables into `bin/<profile>/`:

```bash
python3 tools/build/stage.py --profile release --target core_mat_mul_intro
./bin/release/sdk/tutorials/core/core_mat_mul_intro
```

## Python preview

```bash
cd sdk/py
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop
.venv/bin/python -m unittest discover -s test -v
```

The Python package is intentionally a separate extension crate over the same
Rust library. It does not create a second runtime or CPU implementation.

## Verification

```bash
python3 -m unittest discover -s test/py -v
python3 tools/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Hardware tests are explicitly ignored by the default harness and run serially
on admitted devices. See [test organization](test/README.md).

## Documentation

- [Documentation index](docs/README.md)
- [Canonical architecture](docs/internal/architecture/oaArchitecture.md)
- [Port roadmap](docs/internal/architecture/roadmap/portRoadmap.md)
- [C++ donor compatibility ledger](docs/internal/porting/oaCompatibility.md)
- [ML port inventory](docs/internal/porting/oaMlPortInventory.md)

OA is licensed under the Business Source License 1.1. See `LICENSE`.
