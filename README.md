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
use oa::{matrix, Engine};

// Engine construction is implemented as an Experimental foundation.
let engine = Engine::builder().build()?;
let one = matrix::ones(&engine, [2, 3])?;
let two = matrix::full(&engine, [2, 3], 2.0)?;
let sum = matrix::add(&one, &two)?;

// Host observation is the synchronization boundary.
let values = sum.read_f32()?;
assert_eq!(values, [3.0; 6]);
```

The planned Python binding preserves OA's familiar facade and process engine:

```python
import oa

one = oa.FnMatrix.ones([2, 3])
two = oa.FnMatrix.full([2, 3], 2.0)
sum = oa.FnMatrix.add(one, two)
```

Foundational values, checked metadata, and shared failure contracts live in the
public `core` module. Common vocabulary such as `Matrix`, `Image`, `Error`, and
`Result` is also explicitly re-exported from `lib.rs`. Stateless operations live
in lowercase domain modules. Stateful codecs, streams, presentation, training,
and transport remain explicit session types that borrow an engine.

`Engine` is the sole local execution owner. Rust construction stays explicit;
values retain the internal engine lifetime they need, and ordinary operations
infer it from their inputs. Explicit submit/event controls are reserved for
capture, profiling, multi-device, distributed, and other orchestration paths.
Blocking is visible at host observation (`read::<T>` and `read_f32`) while the
corresponding `try_*` calls remain non-blocking. Destruction never submits or
waits.

## Status

The experimental runtime foundation can automatically or explicitly select one
Vulkan 1.3 compute device, require and enable timeline semaphores plus
synchronization2 and the descriptor-indexing features required by the kernel
ABI, and create dense `f32` and `i32` matrices in checked VMA-backed,
host-visible storage. One dynamic `Matrix` carries a runtime `DType`; sealed
Rust element types provide checked upload and readback without making the
storage owner generic.

The first normalized operation schema owns 19 out-of-place elementwise
operations: `add`, `sub`, `mul`, `div`, `scale`, `neg`, `abs`, `log`, `sqrt`,
`pow`, `add_scalar`, `sub_scalar`, `div_scalar`, `exp`, `sin`, `cos`,
`reciprocal`, `clamp_max`, and `clamp_min`. Explicit generation emits their Rust
functions, stable private kernel IDs and artifacts, bounds-checked Slang entry
points, and an external hardware-oracle test. The build compiles and reflects
every schema entry, validates each ABI and Vulkan 1.3 SPIR-V artifact, and embeds
it. Generated kernels share one engine-owned bindless descriptor heap while
retaining separate private pipelines. All 19 operations admit `f32`; `add`
also has one generated exact-dtype `i32` route. Mixed dense dtypes fail rather
than promoting implicitly.

A second normalized matrix schema owns the Experimental FP32
`matrix::mat_mul_nt` baseline. It preserves OA's `[M, K] × [N, K] -> [M, N]`
weight-layout convention, generates its public function, stable private kernel
identity, bounds-checked Slang module, and hardware-oracle test, and lowers
through the same generic engine submission path. Its current physical route is
an FP32 64×64×16 shared-memory tile adapted from OA's established GEMM
arithmetic. It remains Experimental and is not a performance-qualified routing
system.

Each Rust operation validates and returns a matrix while its direct lowerer
creates a generic compute-dispatch description. One engine submission path
resolves that description, dispatches asynchronously, and retains its
timeline-backed completion in the result; the engine contains no per-operation
entry points. Submitted command buffers retain every referenced allocation
through asynchronous retirement. Host readback is an explicit observation
boundary that waits; typed `try_read::<T>` does not. Binary broadcasting and
in-place mutation are not yet admitted.

No GPU operation is currently classified as Shipped. The active Experimental
checkpoint is the one-device dense elementwise and FP32 `mat_mul_nt` path from
checked initialization through asynchronous dispatch and synchronized host
observation to schema-owned independent golden oracles. The `i32` proof
currently covers addition only.

Build and stage the public matmul tutorial, then run its independent CPU
validation with:

```bash
cargo build --release --example core_mat_mul_intro
python3 tools/build/stage.py --profile release --target core_mat_mul_intro
./bin/release/sdk/tutorials/core/core_mat_mul_intro
```

Cargo keeps intermediate artifacts under `target/`. The staging step copies
only runnable binaries into OA's `bin/{debug,release}/` layout.

The Experimental MatMul benchmark companion measures captured-plan replay with
whole-graph Vulkan timestamps. Its six checked-in workloads use fresh
processes, independent constant-input correctness checks, fixed warmup and
cooldown, raw logs, and machine-readable provenance:

```bash
cargo build --release --example core_mat_mul_bench
python3 tools/build/stage.py --profile release --target core_mat_mul_bench
python3 tools/profiling/suite.py
```

Canonical recording requires a clean release tree and a resolved Vulkan device
and registry. The runner does not yet accept baselines or establish a release
performance claim.

## Documentation

- [Documentation index](docs/README.md)
- [Canonical architecture](docs/internal/architecture/oaArchitecture.md)
- [Port roadmap](docs/internal/architecture/roadmap/portRoadmap.md)
- [C++ to Rust compatibility ledger](docs/internal/porting/oaCompatibility.md)

## Development

Builds require Python 3, `rustfmt`, `slangc`, and `spirv-val` on `PATH`.
`PYTHON`, `SLANGC`, and `SPIRV_VAL` may name explicit executables. Missing
tools, stale generated sources, compilation failure, reflected ABI drift, and
Vulkan 1.3 SPIR-V validation failure stop the build.

The intended baseline gates are:

```bash
python3 -m unittest discover -s tools/gen/fn/tests -v
python3 -m unittest discover -s tools/profiling/tests -v
python3 tools/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

These baseline gates establish source hygiene. The first vertical slice adds
the independent correctness and Vulkan validation gates needed for a capability
claim; a successful compile or empty test run is not such a claim.
