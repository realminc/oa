# OA host-memory primitives

**Status:** Experimental correctness checkpoint; matched performance harness implemented

**Updated:** 2026-09-08

`core::memory` owns the small backend-neutral host-memory surface whose
semantics or measured implementation differ from an ordinary Rust operation.
It does not replace Rust's `core`, `alloc`, or `std` libraries.

## Public contract

- `copy` copies between equal-length, non-overlapping byte slices and reports a
  length mismatch through OA's `Error` contract rather than panicking.
- `copy_streaming` has the same result but additionally declares that the CPU
  will not consume the destination soon.
- `equal` may exit at the first mismatch.
- `equal_constant_time` visits all bytes for equal public lengths before
  producing its result. It is not a system-wide constant-time claim.
- `zero_secure` uses observable stores and a compiler fence to resist
  dead-store elimination. It cannot erase copies, caches, swap, or device
  storage.

Raw pointers are private implementation boundaries. Their unsafe contracts
name range validity, initialization, aliasing, lifetime, and destination reuse.
Public callers use slices.

## Implementation

Ordinary dynamic copies use compiler-known fixed blocks through 256 bytes on
AVX2 builds. Baseline x86-64 builds use the platform copy intrinsic directly,
with no custom size or runtime-ISA dispatcher. This preserves the checked
public error contract while letting the compiler lower fixed sizes and the
platform choose the dynamic copy implementation. It deliberately trades the
portable tiny-copy gains for removal of the portable middle-size regressions.
AVX-512F builds inline a cached 257-byte through 4-KiB path that aligns the
destination and copies four 64-byte vectors per iteration. AVX2-targeted builds
admit that path through runtime AVX-512F feature detection only at 2–4 KiB,
where the call overhead can be amortized. Other lengths and targets reach the
platform bulk implementation before traversing the small-copy dispatch tree.
Length-error construction is cold and both public wrappers are inlined.
`Error` retains its payload in a private box: `Error` and `Result<()>` are one
pointer wide on the tested target. The category, message, source chain,
`Display`/`Debug` output, and `Send + Sync` contract are preserved. Constructing
an error adds one allocation; successful copies allocate nothing. Compactness
is a tested implementation property, not a stable public-layout/ABI promise.
Native dispatch and explicit streaming policy remain workload-dependent;
neither is an always-faster guarantee. The ordinary portable path has only the
length check and copy intrinsic on success, not a second dispatch tree.

On x86-64, runtime capability checks admit AVX-512F plus AVX-512BW first and
AVX2 second. An explicit streaming copy uses aligned non-temporal stores from
1 KiB through 4 MiB, then executes a store fence before publication. Outside
that window, sizes above 256 bytes use the platform routine directly, avoiding
the cached SIMD path's cold-arena cost. Smaller sizes use the ordinary small
copy policy. Architecture-specific
intrinsics remain private below `core::memory`.

Vulkan mapped uploads use streaming copy because their destination is
published to GPU work and is not consumed by the CPU first. Mapping ownership,
flush/invalidate behavior, allocation selection, and eventual upload-ring
reuse remain owned by `runtime::vk`.

## Evidence and limitations

Unit and external API tests cover zero length, every size through 1,024 bytes,
varied source and destination alignment, both streaming-policy boundaries,
guard-byte preservation, length rejection without writes, mismatch positions,
and exact secure erasure. Linux guard-page tests place source and destination
at both ends of mappings, covering all sizes through 1,088 bytes and larger
boundaries. Any access beyond a protected allocation edge faults. Vulkan
hardware tests cover the integrated mapped upload/readback call path.

The release `core_memory_bench` executable retains OA C++'s case inventory:
sizes, misalignment offsets, 64-byte allocation, working sets, iteration
counts, five warmups, 21 measurements, rotating order, correctness gates,
decimal GB/s, and CSV fields. Harness v2 selects each policy outside the timed
loop and obscures both slices on each iteration to prevent loop unswitching
and load hoisting. Historical v1 timings are not directly comparable with v2.
`tools/profiling/memory_rust.py` records paired stock-Rust/OARS ratios from the
same process with raw samples, clocks, and thermal evidence. The older paired
`tools/profiling/memory_compare.py`
runner alternates languages for at least seven measured fresh-process pairs and
retains every raw stream plus clock and thermal observations. Rust
`copy_from_slice` versus OARS `copy` or `copy_streaming` is the primary Rust
acceptance comparison. C++ `std::memcpy` and OA C++ `oa::memcpy` remain separate
porting references; explicit streaming and compiler-sized rows are never
relabeled as ordinary runtime copy.

No release performance result is accepted yet. Development measurements and
their scope belong in [the memory report](../performance/oaMemoryComparison.md).
The fixed-state wrapper selects and verifies the `performance` power
profile/governor, CPU min=max 2.6 GHz, and Xe
min=max 1.0 GHz, then restores the prior state. The reference laptop currently
reports `lap-detected`, so the wrapper correctly rejects canonical timing.
The user-approved observed-clock protocol remains available for local
development measurements.

AddressSanitizer passes the public copy tests, guard-page tests, and memory
unit tests, including direct AVX2/AVX-512 routes. Reproduce on the pinned
toolchain with:

```bash
RUSTC_BOOTSTRAP=1 \
  RUSTFLAGS='-Zsanitizer=address -C target-cpu=native -C link-arg=-fuse-ld=lld' \
  cargo test --release --target x86_64-unknown-linux-gnu \
    --test core memory
```

The bootstrap setting enables rustc's experimental sanitizer instrumentation
for this diagnostic command only; normal builds do not set it. The standard
library and native dependencies are not rebuilt with sanitizer instrumentation.

Still Planned:

- Miri qualification and a supported sanitizer CI profile;
- a release-qualified Rust-versus-stock-Rust artifact across the complete sweep;
- AArch64-specific qualification;
- persistent mapped upload/readback rings and BAR-aware allocation policy;
- broader fault, fuzz, and supported-platform qualification.
