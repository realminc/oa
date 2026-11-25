<p align="center">
  <a href="https://github.com/realminc/oa/blob/main/oasdk-dev/asset/docs/readme/realmIdentityAscii.mp4">
    <img src="https://raw.githubusercontent.com/realminc/oa/main/oasdk-dev/asset/docs/readme/realmIdentityAscii.gif" width="100%" alt="Realm ASCII identity display forming under a right-to-left hover, rippling under three liquid presses, then returning to a sine wave">
  </a>
</p>

# OA

**GPU-first semantic computing in Rust. Compute, ML, media, and presentation on Vulkan behind one explicit `Runtime`.**

OA is an experimental, executable runtime: typed values and domain operations
build on one owner of devices, memory, queues, scheduling, kernels, and
profiling without exposing backend machinery through the public API. The
Cargo package is **`oarust`**; the library and Python import namespace are
**`oa`**.

[![Release](https://img.shields.io/github/v/release/realminc/oa?include_prereleases&label=preview)](https://github.com/realminc/oa/releases)
[![CI](https://github.com/realminc/oa/actions/workflows/ci.yml/badge.svg)](https://github.com/realminc/oa/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/oarust?label=crates.io&cacheSeconds=300)](https://crates.io/crates/oarust)
[![PyPI](https://img.shields.io/pypi/v/oapython?label=pypi&cacheSeconds=300)](https://pypi.org/project/oapython/)
[![License](https://img.shields.io/badge/license-BSL--1.1-3b3b3b)](LICENSE)

[Realm](https://realm.software/) · [Developer guide](https://dev.realm.software/oa)

This repository is the primary OA implementation. The earlier C++ codebase
continues separately as the donor and compatibility reference. OA is a
preview: capability claims are tied to the compatibility ledger, independent
oracles, and recorded hardware evidence rather than to API surface alone.

## Install

```toml
[dependencies]
oarust = "0.17"
```

```bash
python -m pip install oapython
```

Rust code uses `use oa::...`; Python uses `import oa`.

## A few lines of compute

```rust
use oa::matrix;

let one = matrix::ones([2, 3])?;
let two = matrix::full([2, 3], 2.0)?;
let total = matrix::add(&one, &two)?;

assert_eq!(total.read_f32()?, [3.0; 6]);
# Ok::<(), oa::Error>(())
```

No `Runtime` value appears: each thread has a lazy default runtime that owns
the device work. Python resolves the same default:

```python
import oa

one = oa.matrix.ones([2, 3])
two = oa.matrix.full([2, 3], 2.0)
total = oa.matrix.add(one, two)

assert total.read_f32() == [3.0] * 6
```

## Default and explicit runtimes

Every runtime-first value constructor ships in a pair. The short form resolves
the thread's default runtime; the `_on` form selects one explicitly.
`Runtime::set_default(runtime)` is the multi-device path: install the node's
runtime once per thread and ordinary calls resolve to it.

| Purpose | Defaulted | Explicit runtime |
| --- | --- | --- |
| Allocate | `matrix::ones(shape)` | `matrix::ones_on(&runtime, shape)` |
| Upload | `Matrix::new_from_f32(shape, data)` | `Matrix::new_from_f32_on(&runtime, shape, data)` |
| Layer | `nn::Linear::new_with_seed(in, out, seed)` | `nn::Linear::new_with_seed_on(&runtime, in, out, seed)` |
| Decode | `audio::decode_file(path)` | `audio::decode_file_on(&runtime, path)` |

Operations on existing values infer their owner from storage and never take
an runtime. Sessions—training, capture, codec, presentation—keep an explicit
`&Runtime`: ownership is part of their contract, not a convenience.

Python uses the same convention with an optional trailing `runtime` keyword
instead of `_on`: `oa.matrix.ones([2, 3])` or
`oa.matrix.ones([2, 3], runtime=runtime)`, and `oa.Runtime.set_default(runtime)`
installs the thread's default.

## What works today

Checked vertical slices include:

- Vulkan device selection constrained to the compiled Core 1.3 or bounded
  Compatibility 1.2 profile, VMA-backed storage, profile-specific descriptors,
  asynchronous retirement, timeline events, eager batching, semantic capture,
  immutable execution plans, replay, rebinding, hazard analysis, and Vulkan
  timestamp evidence;
- schema-owned Matrix elementwise, reduction, indexing, RNG, transpose, gather,
  tiled FP32 matrix multiplication, broadcasting, and reverse-mode operations;
- ML modules, autograd, Adam/AdamW/SGD/Muon, training iterators, callbacks,
  checkpoints, RNN/GRU/Transformer/MoE/Mamba-3 NLP tutorials, reinforcement
  learning, VQ, and the in-progress Animation Language Model stack;
- Image operations and codecs, planar Audio and codecs/DSP/sessions, Video
  containers/decoding/encoding, Vision detection/metrics, Render value
  foundations, cryptographic hashes/Merkle/PQC, secure memory, and Vulkan
  batch hashing;
- native PyO3 bindings for Runtime/Event, typed dense Matrix construction and
  operations, the functional ML spine, Image codecs/transforms, Audio
  codecs/DSP/features, metadata, and explicit synchronized host observation.

No GPU operation is classified as stable yet.

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
 Runtime-owned queues, memory, events, profiling
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
The crate contains the library and its required build inputs. Runnable
tutorials, benchmarks, and applications moved to the standalone oasdk source
companion in v0.17.0 and are not part of this tree or the registry package.
Tagged CI verifies and publishes both packages, the `oa` CLI and native
Linux packages, source and Rust API documentation archives, and checksums.
The old `oarust` 0.0.1 package is a documentation-only name reservation.

## Build

Requirements: Rust 1.98, Python 3, rustfmt, CMake, `slangc` 2026.14,
`spirv-val`, native audio/window-system development libraries, and a compatible
Vulkan driver. The default Core build requires Vulkan 1.3 and the required
bindless features. An explicit Compatibility build supports admitted Vulkan 1.2
devices with bounded bindings. Compatibility presentation is
not yet qualified. Linux builds use Clang/LLD for native linking while `rustc`
and LLVM compile Rust. SDL3 builds statically from the pinned Cargo source
by default (`bundled-sdl`); `--no-default-features` uses system SDL3 instead.

```bash
cargo build --release
cargo run -p sdk --release --example tu_core_mat_mul_intro
```

The compact media applications share OA's GPU viewer:

```bash
cargo run --release --bin oa -- view image.png
cargo run --release --bin oa -- record --camera 1
cargo run --release --bin oa -- capture --window --hide-cursor
```

`oa record` previews the selected camera; Space or the on-screen controls
record and finish, Ctrl+S saves a copy under Videos/OA. The first camera route
requires hardware H.264 encoding and records video without microphone audio.
Camera and screen previews share the GPU 2D canvas—drag to pan, Ctrl+wheel to
zoom, 1–5 for channel views. `oa capture` records a desktop-selected screen
through the portal/PipeWire route; DMA-BUF zero-copy capture remains planned.

For older hardware, use `OA_BUILD_PROFILE=compatibility cargo build --release`.
Each build compiles and embeds one shader catalog and one binding backend;
the variable selects at compile time, not runtime.

The repository workflow stages executables under `bin/<profile>/`:

```bash
python3 tool/oa build --profile release
python3 tool/oa build --profile release --clean --python
```

`--python` creates the repository-root `.venv`, installs Maturin, and installs
the matching native extension.

## Python preview

Python follows the Rust module graph directly, with the runtime explicit:

```bash
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop
.venv/bin/python -m unittest discover -s test/py -v
```

The native PyO3 implementation lives in `src/py/pyo3`, the importable `oa`
package in `src/py/oa`, and Python tutorials in `oasdk-dev/src/py`. Neither creates a
second runtime or CPU implementation.

## Release artifacts

Every public release is assembled by the tagged CI workflow and publishes:

- an exact Rust source archive with the resolved `Cargo.lock`;
- a Linux x86-64 archive containing the `oa` runtime CLI staged from the
  same Release build;
- matching `oa` packages for Debian, RPM, and Arch Linux;
- one portable CPython 3.10+ ABI3 wheel, published to PyPI and then downloaded
  back from PyPI before attachment to GitHub;
- dependency and toolchain evidence plus one checksum manifest covering every
  downloadable artifact.

The Rust crate currently links into its consumers and does not expose a stable
C ABI; the Python wheel contains and tests its native ABI3 extension.
GitHub-hosted validation proves the host and compilation contracts. Tests that
require a real GPU remain explicit capability gates and are not silently
reclassified as CPU-Vulkan validation.

## Verification

```bash
.venv/bin/python -m unittest discover -s test/py -v
python3 tool/oa gen fn --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Hardware tests are explicitly ignored by the default harness and run serially
on admitted devices. See [test organization](test/README.md).

## Source reference

- [Test organization](https://github.com/realminc/oa/tree/main/test/README.md)

Runnable tutorials, benchmarks, and examples moved to the standalone
oasdk source companion in v0.17.0 and are not part of this source tree.

OA is licensed under the Business Source License 1.1. See `LICENSE`.
