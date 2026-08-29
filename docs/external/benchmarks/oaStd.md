# OA foundation benchmark

**Status:** OA v0.7.19 fixed-host checkpoint evidence

**Measured:** 2026-08-27

This report compares OA's owned foundation primitives with the host C++
standard-library implementation for the exact operations exercised by
`BenchStd`. It is a workload-specific engineering result, not a claim that one
library is universally faster, safer, or more portable.

The container measured in this checkpoint is now published as `oa::Vector`;
the naming-only change from the former pre-1.0 `oa::Vec` spelling does not alter
the implementation or the recorded workload.

## Current result

The Release comparison used Clang 22.1.8, glibc 2.44, and Linux
7.1.8-arch1-3 on an Intel Core i5-1145G7. Each process was pinned to CPU 2 and
used 32,768 general items, 8,192 hash items, five warmups, and 21 alternating
OA/host samples. `intel_pstate` used the `powersave` governor with a 400 MHz to
4.4 GHz range. The package sensor read 70 C before and 66 C after the campaign.
Seven fresh processes completed and every correctness preflight and checksum
passed. Values are medians of the seven per-process `oa/std` ratios; a value
below 1.0 favors OA.

| Case | Median `oa/std` | Seven-process range | Result on this workload |
|---|---:|---:|---|
| Steady clock | 1.026x | 1.016–1.029x | OA's checked wrapper is 2.6% slower |
| Mutex lock/unlock | 1.193x | 1.167–1.225x | OA's checked ABI call remains 19.3% slower |
| Array indexed read | 1.001x | 0.998–1.002x | parity |
| Span traversal | 1.000x | 0.999–1.003x | parity |
| Atomic relaxed add | 1.000x | 1.000–1.059x | parity median |
| Optional lifecycle | 0.999x | 0.998–1.001x | parity |
| Algorithm sort | 1.003x | 0.993–1.011x | parity |
| Find/count algorithms | 0.939x | 0.938–0.939x | OA used 93.9% of host time |
| C `strlen` / byte | 0.994x | 0.981–1.039x | parity |
| C `strchr` miss / byte | 1.004x | 0.985–1.006x | parity |
| C `strcmp` equal / byte | 1.000x | 0.993–1.004x | parity |
| C `strncmp` equal / byte | 1.009x | 0.999–1.013x | parity to 0.9% slower |
| `StringView` find / byte | 1.009x | 0.984–1.016x | parity to 0.9% slower |
| Scalar sin/tanh | 1.000x | 0.972–1.069x | parity median |
| `Vector`, reserved scalar push | 1.593x | 1.589–1.701x | OA remains 59.3% slower |
| `Vector`, geometric growth | 1.062x | 1.029–1.132x | OA remains 6.2% slower |
| `Vector`, bulk append / element | 1.000x | 0.998–1.001x | parity |
| `String`, reserved push | 0.923x | 0.906–0.996x | OA used 92.3% of host time |
| `String`, geometric growth | 0.876x | 0.865–0.892x | OA used 87.6% of host time |
| `String`, bulk append / byte | 1.004x | 1.001–1.012x | parity to 0.4% slower |
| Variant emplace/visit | 1.001x | 0.999–1.002x | parity |
| Function invoke | 1.000x | 0.999–1.000x | parity |
| `HashMap` insertion | 0.341x | 0.309–0.346x | OA used 34.1% of host time |
| `HashSet` insertion | 0.351x | 0.278–0.356x | OA used 35.1% of host time |
| `HashMap` successful find | 0.879x | 0.861–0.882x | OA used 87.9% of host time |
| `HashSet` successful find | 0.751x | 0.675–0.762x | OA used 75.1% of host time |
| `HashMap` collision insertion | 0.661x | 0.628–0.760x | OA used 66.1% of host time |
| `HashMap` collision miss | 0.634x | 0.601–0.647x | OA used 63.4% of host time |
| `SharedPtr` make/release, single-thread process state | 0.999x | 0.785–1.042x | parity median; noisy allocation path |
| `SharedPtr` copy/release, single-thread process state | 2.627x | 2.512–2.701x | host uses a process-state non-atomic shortcut |
| `SharedPtr` make/release, after thread creation | 0.968x | 0.944–1.010x | OA used 96.8% of host time |
| `SharedPtr` copy/release, after thread creation | 1.003x | 0.988–1.017x | parity with the thread-safe host path |

The shared-owner cases explicitly separate the host's single-thread process
state from its behavior after thread creation. On this glibc/libstdc++ host,
copy/release avoids atomic reference-count operations until a thread has
existed; OA retains thread-safe counts in both states. After thread creation,
the OA and host copy paths are at parity. Adopting the host shortcut would make
the ownership result depend on undocumented process-global state and is not
accepted for the portable foundation contract.

The reserved `Vector` case now consumes a prebuilt input sequence and validates
every element outside the timed section. That removes the earlier source-value
code-generation asymmetry and exposes a larger OA scalar-push gap. Disassembly
attributes the retained difference to the optimizer's treatment of OA's
reallocation control-flow graph, not to allocation or a weaker host safety
contract. Bulk append is the intended high-throughput path and is at parity.
Outlining checks, unchecked push APIs, allocator overreads, public pthread
layouts, spin loops, and non-atomic shared counts were rejected because they
either regressed another workload or weakened portability, progress, or
failure semantics.

## Formatting checkpoint

OA exposes one brace-formatting and record-output family:

```cpp
#include <oa/core/std/print.h>

const oa::String message = oa::format("epoch={} loss={:.4f}", epoch, loss);
OA_RETURN_IF_ERROR(oa::print("epoch={} loss={:.4f}", epoch, loss));
OA_RETURN_IF_ERROR(oa::print(oa::PrintStream::Error, "failed: {}", reason));
OA_RETURN_IF_ERROR(oa::write("\rprogress={:.1f}%", percent));
OA_RETURN_IF_ERROR(oa::flush());
```

`oa::print` appends exactly one newline; `oa::write` appends none. Neither
implicitly flushes, so visible same-line progress uses `\r`, `oa::write`, and
the explicit `oa::flush` shown above. The functions return `oa::Status` for
recoverable host-output failures. Format strings use sequential `{}` or
`{:spec}` fields and must be character-array literals; runtime text is accepted
as data and is not reparsed as a format program. The supported surface includes
OA strings and views, C strings, booleans, characters, integers, enums,
floating-point values, pointers, escaped braces, alignment, sign, alternate
form, zero padding, bounded width and precision, integer bases, and the
`f`/`e`/`g` floating families.

The 2026-08-28 dev checkpoint compares OA's bounded, literal brace formatter
against `std::format` for identical output. It used Release Clang 22.1.8,
glibc 2.44, Linux 7.1.8-arch1-3, the same Intel Core i5-1145G7, CPU 3 affinity,
AC power, the `powersave` governor, and the balanced power profile. Each fresh
process used 32,768 items, five warmups, and 21 alternating samples. All seven
processes passed full-output preflight and timed checksums. Ratios are the
median of the seven per-process OA/standard-library ratios; lower is better.

| Case | Median `oa/std` | Seven-process range | Result on this workload |
|---|---:|---:|---|
| signed decimal integer | 0.589x | 0.577-0.603x | OA used 58.9% of host time |
| padded hex + bool | 0.668x | 0.661-0.688x | OA used 66.8% of host time |
| fixed-precision float | 0.913x | 0.901-0.919x | OA used 91.3% of host time |
| padded hex + bool + fixed float | 1.003x | 0.975-1.011x | parity within spread |

Inlining lets the compiler fold fixed literal syntax; records through 128
bytes use bounded inline staging, and floating conversion writes into caller
storage rather than an intermediate owning string. These changes close the
previous mixed-record gap without changing output, precision, locale,
overflow, or failure behavior. OA still pins floating conversion to the C
numeric locale and caps format width, precision, and total output. The logger,
CLI, and user-facing C++ SDK applications, examples, and tutorials now use this
same brace-formatting route. Logger records no longer have fixed 4096-byte or
metrics-JSONL 256-byte truncation points; metrics tags are JSON-escaped and
non-finite values serialize as `null`.

## Host-memory checkpoint

`BenchMemory` separately qualifies the raw-byte primitives now owned by
`oa/core/std/memory.h`. It used the same Tiger Lake host, Clang 22.1.8, glibc
2.44, Linux 7.1.8, five warmups, 21 rotating samples, and seven fresh
processes. Each implementation passed a full-content, overlap, alignment, and
guard-byte oracle before timing. Ratios are OA/libc latency; lower is better.

### Dynamic copy with reusable buffers

| Size | Median OA/libc | Seven-process range | Result |
|---:|---:|---:|---|
| 8 B | 0.498x | 0.448-0.540x | 50.2% less latency |
| 32 B | 0.663x | 0.632-0.679x | 33.7% less latency |
| 256 B | 0.968x | 0.864-1.085x | parity within spread |
| 4 KiB | 1.059x | 1.018-1.088x | dispatcher cost before libc |
| 64 KiB | 1.003x | 1.001-1.024x | parity |
| 1 MiB | 1.007x | 0.984-1.045x | parity |
| 16 MiB | 1.022x | 1.003-1.038x | near parity |
| 64 MiB | 1.025x | 0.910-1.066x | parity within spread |

Fixed compile-time OA and compiler-copy rows reduce to the same instruction
sequences. The dynamic dispatcher is retained for the stable 8-byte and
32-byte wins; ordinary large copies continue through libc.

### Cache-cold one-way streaming

This sweep rotates chunks through 256 MiB source and destination arenas, 32
times the host's 8 MiB last-level cache. `runtime` is ordinary `oa::memcpy`;
`stream` is the explicit one-way `oa::memcpyStream` policy. Throughput is the
median absolute result across the seven processes.

| Chunk | Runtime OA/libc (range) | Stream OA/libc (range) | libc | OA stream |
|---:|---:|---:|---:|---:|
| 256 B | 0.922x (0.891-0.991x) | 0.940x (0.904-1.025x) | 7.641 GB/s | 7.959 GB/s |
| 512 B | 1.007x (0.975-1.055x) | 1.016x (0.975-1.054x) | 8.358 GB/s | 8.571 GB/s |
| 1 KiB | 1.012x (0.958-1.121x) | 0.855x (0.808-0.984x) | 8.809 GB/s | 10.053 GB/s |
| 2 KiB | 1.006x (0.989-1.036x) | 0.653x (0.608-0.731x) | 7.396 GB/s | 11.300 GB/s |
| 4 KiB | 1.000x (0.987-1.024x) | 0.955x (0.818-1.037x) | 11.168 GB/s | 11.693 GB/s |
| 8 KiB | 1.006x (0.971-1.030x) | 0.882x (0.826-0.900x) | 10.871 GB/s | 12.467 GB/s |
| 16 KiB | 0.992x (0.977-1.019x) | 0.853x (0.824-0.876x) | 10.937 GB/s | 13.053 GB/s |
| 64 KiB | 1.008x (0.975-1.044x) | 0.792x (0.765-0.856x) | 10.880 GB/s | 13.587 GB/s |
| 256 KiB | 0.990x (0.973-1.030x) | 0.861x (0.787-0.904x) | 10.925 GB/s | 12.925 GB/s |
| 1 MiB | 1.000x (0.985-1.017x) | 0.831x (0.809-0.860x) | 10.798 GB/s | 13.023 GB/s |
| 2 MiB | 0.988x (0.953-1.042x) | 0.839x (0.769-0.857x) | 11.248 GB/s | 13.348 GB/s |
| 4 MiB | 1.015x (0.963-1.036x) | 0.829x (0.780-0.882x) | 11.129 GB/s | 13.492 GB/s |
| 8 MiB | 0.982x (0.921-1.034x) | 0.972x (0.906-1.023x) | 16.106 GB/s | 16.255 GB/s |
| 16 MiB | 0.996x (0.980-1.136x) | 0.990x (0.918-1.078x) | 16.170 GB/s | 16.284 GB/s |
| 64 MiB | 0.989x (0.973-1.071x) | 0.993x (0.955-1.078x) | 15.735 GB/s | 15.819 GB/s |

The qualified x86 policy uses exact-bounds non-temporal stores from 1 KiB
through 4 MiB and ordinary copy elsewhere. Stable wins appear at 1 KiB, 2 KiB,
8 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 2 MiB, and 4 MiB. The best row uses
65.3% of libc time at 2 KiB. The 64 KiB through 4 MiB rows use 79.2-83.9%,
apart from 256 KiB at 86.1%. The 4 KiB process range crosses parity and is
classified as flat. Outside the window, the fallback restores parity.

### Equality, ordering, and overlap

| Operation and input | 8 B | 256 B | 4 KiB | 64 KiB |
|---|---:|---:|---:|---:|
| `memEqual`, equal | 1.714x (1.593-2.666x) | 1.367x (1.360-1.499x) | 1.187x (1.051-1.196x) | 0.639x (0.638-0.732x) |
| `memEqual`, early mismatch | 1.629x (1.629-1.640x) | 0.778x | 0.787x (0.779-0.843x) | 0.894x (0.824-1.033x) |
| `memEqual`, late mismatch | 1.629x (1.629-1.630x) | 1.058x (1.000-1.079x) | 1.207x (1.064-1.250x) | 0.645x (0.643-0.713x) |
| `memEqualConstantTime`, equal | 1.617x (1.571-1.713x) | 1.402x (1.363-1.548x) | 0.563x (0.554-0.603x) | 0.664x (0.592-0.665x) |

The constant-work row has a stricter security contract than libc's early-exit
comparison, so it is not a semantic speedup claim. `memcmp` and both overlapping
`memmove` directions were 1.000x median at all four sizes; complete move ranges
were within 0.932-1.041x. Platform fill and zero remain faster than the rejected
hand-vectorized implementations.

A force-all non-temporal policy was rejected at 1.636x libc latency for 256 B
and about 1.14x for 8-64 MiB. An eight-vector AVX-512 unroll also failed the
seven-process no-regression gate. The accepted adaptive policy is applied only
to semantically one-way `UploadRing` and mapped BAR uploads. These CPU results
do not claim GPU, PCIe, submission, or end-to-end model throughput.

## Correctness gate

All 271 focused Release tests pass. The Debug sanitizer profiles contain one
additional assertion-only test, so combined ASAN+UBSAN and TSan each pass
272 tests. ASAN enabled leak detection and strict string checks; ASAN and UBSAN
used halt-on-first-error; no finding was suppressed. TSan reported no race.
GoogleTest warns that its fork-based death-test harness observes two runtime
threads under TSan, but the complete suite finishes successfully.

The tests cover every admitted primitive family, including collision,
tombstone, rehash, adversarial sort order, aliased and over-aligned container
growth, throwing construction and relocation, invalid text ranges,
reference-count bounds, invalid atomic orders, signed extrema, deadline
overflow, and concurrent final strong/weak release. The complete product and
public-header closure also build from the isolated source checkpoint.

At the 4.29 checkpoint, LLVM coverage for every then-current
`source/cpp/include/oa/core/std` header was 77.72% of lines, 78.24% of
functions, and 63.95% of branches. This is broad family coverage, not complete
path coverage or a remeasurement after the 4.30 memory additions. Aborting
death-test children cannot flush their profiles, so fail-closed branches are
systematically undercounted.
Ordinary-path gaps remain concentrated in Chrono, String/StringView, Array,
allocator variants, and less-used shared/weak operations. Longer fuzz,
allocation-fault, and hostile-input campaigns remain separate qualification
work.

The benchmark runs a full-content/terminator preflight outside each timed
section, then validates the timed OA/host result checksum before reporting the
pair. Unknown CLI arguments are rejected rather than silently ignored.

## Reproduce

```sh
cmake --build build/release --target BenchStd TestOaStd -j
ctest --test-dir build/release -R '^TestOaStd$' --output-on-failure
taskset -c 2 ./bin/release/test/core/std/benchStd \
  --items 32768 --warmups 5 --samples 21

# Formatting evidence above used CPU 3 on the recorded host.
taskset -c 3 ./bin/release/test/core/std/benchStd \
  --items 32768 --warmups 5 --samples 21

cmake --build build/release --target BenchMemory TestMemory -j
taskset -c 2 ./bin/release/test/core/memory/benchMemory --quick --copy
taskset -c 2 ./bin/release/test/core/memory/benchMemory --quick --primitives
taskset -c 2 ./bin/release/test/core/memory/benchMemory --streaming
```

Run the benchmark as at least seven fresh processes under a fixed power policy
and recorded thermal conditions. A release comparison requires immutable
before/after commits, alternating samples, median and spread, and passing
correctness gates. A single process is useful for profiling but is not release
evidence.

## Evidence boundary

These numbers qualify one compiler, host library, machine, workload, and date.
They do not establish cross-platform superiority, complete security,
allocator-failure behavior, race freedom, hostile-input resilience, or
linked-binary independence from `libstdc++` or `libc++`. Those claims require
their own race, fault-injection, fuzz, compiler, linker, and platform matrices.
