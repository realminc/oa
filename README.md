<p align="center">
  <img src="https://raw.githubusercontent.com/realminc/oa/main/sdk/asset/docs/readme/oaSpaceCathedral.jpg" width="100%" alt="OA — one Rust and Vulkan foundation for compute, ML, media, and intelligent systems">
</p>

# OA

OA is a GPU-first semantic computing engine written in Rust. One explicit
`Engine` owns Vulkan devices, memory, queues, scheduling, kernels, and
profiling; typed values and domain operations build on that owner without
exposing backend machinery through the public API.

This repository is the new primary OA implementation. The earlier C++ codebase
continues separately as the donor and compatibility reference.

[![Release](https://img.shields.io/github/v/release/realminc/oa?include_prereleases&label=preview)](https://github.com/realminc/oa/releases)
[![CI](https://github.com/realminc/oa/actions/workflows/ci.yml/badge.svg)](https://github.com/realminc/oa/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/oarust?label=crates.io)](https://crates.io/crates/oarust)
[![PyPI](https://img.shields.io/pypi/v/oapython?label=pypi)](https://pypi.org/project/oapython/)
[![License](https://img.shields.io/badge/license-BSL--1.1-3b3b3b)](LICENSE)

<p align="center">
  <a href="https://github.com/realminc/oa/blob/main/sdk/asset/docs/readme/realmIdentityAscii.mp4">
    <img src="https://raw.githubusercontent.com/realminc/oa/main/sdk/asset/docs/readme/realmIdentityAscii.gif" width="100%" alt="Realm ASCII identity display forming under a right-to-left hover, rippling under three liquid presses, then returning to a sine wave">
  </a>
</p>

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

- Vulkan device selection with Strict 1.3 and bounded Compatibility 1.2
  compute profiles, VMA-backed storage, profile-specific descriptors,
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
- native PyO3 bindings for Engine/Event, typed dense Matrix construction and
  operations, the current functional ML spine, Image codecs/transforms, Audio
  codecs/DSP/features, metadata, and explicit synchronized host observation.

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

## Distribution names

The project and API namespace are `oa`. The language-specific distribution
names are `oarust` on crates.io and `oapython` on PyPI; Python uses `import oa`.

The Rust package is `oarust`, with library name `oa`:

```toml
[dependencies]
oarust = "0.8.6"
```

The crate contains the library and its required build inputs, including the
source support modules exposed through `oa::sdk`. Runnable SDK programs are
separate release downloads; they are not part of the registry package.

Rust code uses `use oa::...`. Python installs with
`python -m pip install oapython==0.8.6` and uses `import oa`.
Tagged CI verifies and publishes both packages, the SDK executables and native
Linux packages, source and Rust API documentation archives, and checksums.
The old `oarust` 0.0.1 package is a documentation-only name reservation.

## Build

Requirements: Rust 1.98, Python 3, rustfmt, CMake, `slangc` 2026.14,
`spirv-val`, native audio/window-system development libraries, and a compatible
Vulkan driver. Strict execution requires Vulkan 1.3; bounded Compatibility
compute supports admitted Vulkan 1.2 devices. Compatibility presentation is
not yet qualified. Linux builds use Clang/LLD for native linking while `rustc`
and LLVM compile Rust. SDL3 builds statically from the pinned Cargo source
by default (`bundled-sdl`); `--no-default-features` uses system SDL3 instead.

```bash
cargo build --release
cargo run --release --example core_mat_mul_intro
```

Cargo keeps intermediate artifacts and build-script shader output in `target/`.
Optional generator previews also belong under `target/gen/`; live schema-owned
`.gen.rs` and Slang sources remain in their owning source directories. OA’s
staging tool copies only runnable executables into `bin/<profile>/`:

```bash
python3 tool/build/stage.py --profile release --target core_mat_mul_intro
./bin/release/sdk/tutorials/core/core_mat_mul_intro
```

VS Code and Zed expose matching **Build** and **Clean Build** tasks for Debug
and Release, each with an optional Python variant. The shared command-line
workflow is also available directly:

```bash
python3 tool/build/workflow.py --profile debug
python3 tool/build/workflow.py --profile release --clean --python
```

It stages Rust executables and test runners under `bin/debug/` or
`bin/release/`. `--python` creates the repository-root `.venv`, installs
Maturin there if needed, and installs the matching native extension. Clean
builds remove only the selected Cargo profile and staged output; with
`--python`, they also recreate the venv. On Linux, before rebuilding, the
workflow stops this user's running executables from that profile's staged tree.

## Python preview

The PyPI distribution is `oapython`; the imported package is `oa`:

```bash
python -m pip install oapython
```

Python follows the Rust module graph directly:

```python
import oa

engine = oa.Engine()
one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
total = oa.matrix.add(one, two)
assert total.to_list() == [3.0] * 6
```

For a source checkout, run from the repository root:

```bash
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop
.venv/bin/python -m unittest discover -s test/py -v
```

The native PyO3 implementation lives in `src/py`. Root `pyproject.toml` and
`uv.lock` own the checkout-wide Python environment and distribution metadata;
the importable `oa` package, PyO3 crate, examples, and tutorials live in
`sdk/py`. Neither creates a second runtime or CPU implementation.

## Release artifacts

Every public release is assembled by the tagged CI workflow and publishes:

- an exact Rust source archive with the resolved `Cargo.lock`;
- a Linux x86-64 SDK archive containing the runnable tutorials, benchmarks,
  and applications staged from the same Release build;
- matching `oa-sdk` packages for Debian, RPM, and Arch Linux;
- one portable CPython 3.10+ ABI3 wheel, published to PyPI and then downloaded
  back from PyPI before attachment to GitHub;
- dependency and toolchain evidence plus one checksum manifest covering every
  downloadable artifact.

The Rust crate currently links into its consumers and does not expose a stable
C ABI. Consequently these releases do not label an internal Rust `dylib` as an
OA runtime `.so`; a separate `oa` runtime system package is Planned for the
checkpoint that introduces a supported dynamic-library boundary. The Python
wheel does contain and test its native ABI3 extension.

GitHub-hosted validation proves the host and compilation contracts. Tests that
require a real GPU remain explicit capability gates and are not silently
reclassified as CPU-Vulkan validation.

## Verification

```bash
.venv/bin/python -m unittest discover -s test/py -v
python3 tool/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Hardware tests are explicitly ignored by the default harness and run serially
on admitted devices. See [test organization](test/README.md).

## SDK and source reference

- [Rust tutorials](https://github.com/realminc/oa/tree/main/sdk/rs/tutorials)
- [Rust examples](https://github.com/realminc/oa/tree/main/sdk/rs/examples)
- [Python tutorials](https://github.com/realminc/oa/tree/main/sdk/py/tutorials)
- [Python examples](https://github.com/realminc/oa/tree/main/sdk/py/examples)
- [Python SDK guide](https://github.com/realminc/oa/tree/main/sdk/py/README.md)
- [Test organization](https://github.com/realminc/oa/tree/main/test/README.md)

OA is licensed under the Business Source License 1.1. See `LICENSE`.
