<p align="center">
  <img src="sdk/asset/documentation/readme/oaSpaceCathedral.jpg" width="100%" alt="OA — one Vulkan-native foundation for compute, ML, media, and mobile">
</p>

# OA

**Powerful accelerators. Surrounded by glue.**

OA—One API, Open Architecture—is a unified, GPU-first C++20 and Python library:
one execution system for numerical computing, machine learning, vision, audio,
media, data, crypto, rendering, interfaces, and plotting.

A useful intelligent system does not end at matrix multiplication. It captures or
decodes data, transforms it, trains or executes a model, evaluates the result,
preserves state, and then displays, plays, encodes, or transmits the output.
Conventional workflows split that path across libraries, allocators, queues,
runtimes, and language boundaries. Every handoff can add integration work,
conversion, allocation, copying, synchronization, deployment dependencies, or an
opaque fallback.

OA answers with one library, one engine, one resource model, and one completion
contract. `oa::Engine` owns the Vulkan device, memory, queues, scheduling, kernels,
and profiling. Semantic values preserve their domain meaning, stateless `oa::Fn*`
operations transform them, and sessions own stateful activity such as training,
playback, capture, presentation, and MCP control.

[![Release](https://img.shields.io/github/v/release/realminc/oa?include_prereleases&label=preview)](https://github.com/realminc/oa/releases)
[![CI](https://github.com/realminc/oa/actions/workflows/ci.yml/badge.svg)](https://github.com/realminc/oa/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/oapython?label=pypi)](https://pypi.org/project/oapython/)
[![License](https://img.shields.io/badge/license-BSL--1.1-3b3b3b)](LICENSE)

> **Development preview.** The API and artifact formats remain pre-1.0 and may
> change. Read the [latest release notes](docs/external/releases/v0.7.21.md) for
> shipped scope, verification, compatibility, and known limitations.

## One runtime. Two front ends.

Python is an authoring surface over the same native OA values, operations, and
Vulkan execution as C++. Matrices, gradients, optimizer state, and graph execution
remain inside OA rather than moving through a second host implementation between
operations. Devices enter through Vulkan capability checks and qualified execution
routes, never through a brand allow-list; unsupported paths fail closed instead of
silently changing the workload.

## Quick start

### C++

```cpp
#include <oa/oa.h>

OA_MAIN("ExampleCoreMatrix") {
	auto one = oa::FnMatrix::ones({2, 3});
	auto two = oa::FnMatrix::full({2, 3}, 2.0F);
	auto sum = oa::FnMatrix::add(one, two);

	oa::Array<oa::F32, 6> values{};
	if (not oa::FnMatrix::copyToHost(sum, values.data(), sizeof(values)).isOk()) {
		return 1;
	}
	for (const oa::F32 value : values) {
		if (oa::abs(value - 3.0F) > 1e-06F) {
			return 1;
		}
	}

	oa::puts("Matrix addition verified: every value is 3");
	return 0;
}
```

This is the runnable [SDK C++ matrix example](sdk/cpp/examples/core/matrix.cpp).
`OA_MAIN` provides one lexical engine owner; the checked `copyToHost` sink is the
completion boundary for the recorded matrix work. Advanced applications can own
`oa::Engine` directly when they need explicit capture, submission, and event control.

### Python

Install the Linux preview wheel as `oapython`; import it as `oa`:

```bash
python -m pip install oapython
```

```python
import oa

one = oa.FnMatrix.ones([2, 3])
two = oa.FnMatrix.full([2, 3], 2.0)
sum = oa.FnMatrix.add(one, two)

values = oa.FnMatrix.copyToHost(sum)
assert len(values) == 6
assert all(abs(value - 3.0) <= 1e-06 for value in values)

print("Matrix addition verified: every value is 3")
```

This is the runnable [SDK Python matrix example](sdk/py/examples/core/matrix.py).
Import is host-only. The first device-backed request creates the binding host lazily;
Python calls the same C++ values and Vulkan kernels rather than a NumPy or CPU fallback.

## Build from source

Requirements: Linux, CMake 3.20+, Ninja, a C++20 compiler, the Vulkan SDK/loader,
Slang, and vcpkg. Vulkan 1.4 is the release target; individual routes remain
capability-gated.

```bash
cmake --preset release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
cmake --install build/release --prefix ~/.local
```

Applications consume OA through CMake:

```cmake
find_package(oa CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE oa::oa)
```

### Binary packages

Each [GitHub prerelease](https://github.com/realminc/oa/releases) builds runtime and SDK
tarballs plus `.deb`, `.rpm`, and `.pkg.tar.zst` packages. These packages are built on
Ubuntu 24.04 and require glibc 2.39 or newer.

On Arch Linux:

```bash
paru -S oa-bin oa-sdk-bin       # release binaries
paru -S oa-git oa-sdk-git       # current source
```

## Architecture

```text
C++ / Python / Android
          │
          ▼
 semantic values + oa::Fn* operations + sessions
          │
          ▼
 oa::Engine ── capture → oa::ExecutionPlan
            └─ submit  → oa::Event
          │
          ▼
 private semantic lowering + Vulkan execution graph
          │
          ▼
 embedded Slang → SPIR-V kernel manifest
```

The public API does not expose Vulkan graphs, queues, allocators, routers, pipeline
objects, or a public `oa::ExecutionSession`. Values carry domain semantics even when storage can be
shared. Operations are stateless transformations. Sessions own stateful external or
iterative processes. `oa::Engine` is the sole local execution owner; presentation, media,
training, and MCP services borrow it through composition.

Public headers live under `source/cpp/include/oa/`; implementations and shaders
under `source/cpp/lib/oa/`; vendored C/C++ dependencies under
`source/cpp/thirdparty/`; Python mirrors the public module boundaries under
`source/py/`.

## Documentation

- [Developer documentation](https://dev.realm.software/)
- [GitHub releases](https://github.com/realminc/oa/releases)
- [Release notes](docs/external/releases/README.md)
- [Changelog](CHANGELOG.md)
- [OA foundation benchmark](docs/external/benchmarks/oaStd.md)
- [NLP training benchmark](docs/external/benchmarks/oaNlpSuite.md)
- [Desktop/mobile NLP validation](docs/external/benchmarks/oaMobileLab.md)
- [C++ tutorials](sdk/cpp/tutorials)
- [Python tutorials](sdk/py/tutorials)

## License

[Business Source License 1.1](LICENSE). Source is available for reading, modification,
non-production use, and the production uses permitted by OA's Additional Use Grant. Each
version converts to Apache-2.0 on its stated Change Date. Commercial licensing:
`realminc.depravity737@passinbox.com`.

Copyright © 2025–2026 Lukasz Biernat, trading as Realm.

OA vendors or integrates permissively licensed components. Release packages include
OA's license, the attribution manifest, and available dependency copyright files. See
[NOTICE.md](NOTICE.md) for the exact dependency boundary, including components used only
by tests or build tooling.
