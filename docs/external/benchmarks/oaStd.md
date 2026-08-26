# OA foundation benchmark

**Status:** OA v0.7.19 fixed-host engineering evidence

**Measured:** 2026-08-26

This report compares OA's owned foundation primitives with the host C++
standard-library implementation for the exact operations exercised by
`BenchStd`. It is a workload-specific engineering result, not a claim that one
library is universally faster, safer, or more portable.

## Current result

The Release comparison used Clang 22.1.8 and glibc 2.44 on the local x86-64
development laptop. Each process was pinned to CPU 2 and used 32,768 general
items, 8,192 hash items, five warmups, and 21 alternating OA/host samples.
Seven fresh processes completed and every checksum passed. Values are medians
of the seven per-process `oa/std` ratios; a value below 1.0 favors OA.

| Case | Median `oa/std` | Seven-process range | Result on this workload |
|---|---:|---:|---|
| Algorithm sort | 0.994x | 0.980–1.035x | parity |
| `HashMap` insertion | 0.459x | 0.446–0.468x | OA used 45.9% of host time |
| `HashMap` successful find | 0.871x | 0.793–0.906x | OA used 87.1% of host time |
| `SharedPtr` make/release | 1.579x | 1.271–1.610x | host implementation remains faster |
| `SharedPtr` copy/release | 1.919x | 1.720–2.016x | host implementation remains faster |

The accepted hot-path pass reduced the sort median from 1.590x to 0.994x and
the successful hash lookup median from 1.617x to 0.871x. Absolute current
medians were 38.295 ns per sort benchmark operation, 10.407 ns per hash
insertion, and 1.202 ns per successful lookup. Shared ownership remains an open
performance target; a packed 64-bit counter experiment was rejected after it
regressed copy/release to 2.655x in the focused run.

## Correctness gate

All 209 focused foundation tests pass, including collision, tombstone,
rehash, adversarial sort-order, move-only lifetime, and concurrent final
strong/weak release contracts. The benchmark validates an independent result
checksum before timing each OA/host pair.

## Reproduce

```sh
cmake --build build/release --target BenchStd TestOaStd -j
ctest --test-dir build/release -R '^TestOaStd$' --output-on-failure
taskset -c 2 ./bin/release/test/core/std/benchStd \
  --items 32768 --warmups 5 --samples 21
```

Run the benchmark as at least seven fresh processes under a fixed power policy
and recorded thermal conditions. A release comparison requires immutable
before/after commits, alternating samples, median and spread, and passing
correctness gates. A single process is useful for profiling but is not release
evidence.

## Evidence boundary

These numbers qualify one compiler, host library, machine, workload, and date.
They do not establish cross-platform superiority, security, allocator-failure
behavior, race freedom, or linked-binary independence from `libstdc++` or
`libc++`. Those claims require their own sanitizer, race, fault-injection,
compiler, linker, and platform matrices.
