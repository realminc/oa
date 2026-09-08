# Rust std / OARS memory comparison

**Status:** Experimental, observed-clock development evidence; not release qualification

**Updated:** 2026-09-08

The current portable full sweep has no ordinary-copy median more than 3%
slower than stock Rust. Native-target copying retains measured small/medium
gains, and explicit one-way streaming gains 1.16–1.77× at the tested
1-KiB–4-MiB chunks. This is **not** an always-faster claim: ordinary cached
copy still regresses in the cold 4-KiB streaming workload, and individual
process results vary.

The primary baseline is Rust `copy_from_slice`, not C++ `std::memcpy` or OA
C++ `oa::memcpy`. Those remain porting references, not Rust acceptance criteria.

## Compact Error checkpoint (current)

Clean source `4b432a3ddf63237e305d545452f1e6031bd5a92b` keeps the portable
copy intrinsic and native algorithms from the preceding checkpoint, but moves
the private Error payload behind a Box. On this target, `Error` and `Result<()>`
are now pointer-sized. Public kinds, messages, formatting, source chaining,
and Send/Sync behavior remain unchanged. Constructing an error adds one heap
allocation; successful copies do not allocate. This is a private representation
choice, not a stable ABI guarantee.

The portable benchmark's OARS stack frame shrinks from 0x58 to 0x28 bytes,
matching stock. Both hot loops still contain fifteen instructions and call
the same `memcpy@GLIBC_2.14`. This verifies the frame change, not a universal
explanation of timing differences: register choices and code placement differ.

All three sweeps passed correctness gates with two warmup and seven fresh
measured processes, three-second cooldowns, and the 65 °C package start gate.
The full portable sweep covers 68 aligned/misaligned runtime cases plus
fixed-size diagnostics: **65 paired median speedups within 0.97–1.03, three
above 1.03, none below 0.97**. The former median gap is closed on this sweep;
the descriptive band is not a statistical-equivalence or universal guarantee.

| Portable case | Rust std ns | OARS ns | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|
| 64 B, misaligned | 2.733 | 2.728 | 1.000× | 0.995–1.005 |
| 128 B, aligned | 3.535 | 3.524 | 0.997× | 0.990–1.069 |
| 128 B, misaligned | 3.898 | 3.580 | 1.005× | 0.898–1.122 |
| 256 B, aligned | 3.952 | 3.932 | 0.997× | 0.957–1.012 |
| 256 B, misaligned | 15.174 | 14.836 | 1.002× | 0.974–1.026 |
| 512 B, aligned | 6.802 | 6.743 | 0.999× | 0.980–1.021 |
| 512 B, misaligned | 8.267 | 8.387 | 0.986× | 0.965–1.002 |
| 4 KiB, aligned | 41.946 | 40.726 | 1.014× | 0.984–1.038 |

The native copy **quick** sweep covers 28 runtime cases: 16 paired medians
above 1.03, twelve within 0.97–1.03, none below 0.97. It is narrower than the
historical native full sweep below; do not transfer that full coverage to this
new binary or compare absolute timings across quick/full protocols.

| Native aligned case | Rust std ns | OARS ns | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|
| 8 B | 3.514 | 1.928 | 1.823× | 1.812–1.840 |
| 64 B | 2.722 | 2.335 | 1.164× | 1.157–1.172 |
| 128 B | 3.504 | 2.340 | 1.497× | 1.486–2.418 |
| 256 B | 3.888 | 3.267 | 1.190× | 1.180–1.190 |
| 512 B | 6.707 | 5.059 | 1.343× | 1.263–1.405 |
| 4 KiB | 36.047 | 28.497 | 1.274× | 1.259–1.292 |
| 64 KiB | 1814.016 | 1803.146 | 1.002× | 0.994–1.011 |

The native streaming quick sweep uses a 64-MiB arena and thirteen chunk sizes.
Throughput is decimal GB/s; speedup compares explicit streaming against stock.

| Chunk | Rust std | OARS ordinary | OARS streaming | Stream speedup | p10–p90 |
|---:|---:|---:|---:|---:|---:|
| 256 B | 6.633 | 6.932 | 6.519 | 0.989× | 0.950–1.071 |
| 512 B | 7.062 | 6.932 | 6.698 | 0.961× | 0.904–1.121 |
| 1 KiB | 6.950 | 6.337 | 9.301 | 1.361× | 1.222–1.681 |
| 2 KiB | 6.489 | 7.277 | 11.193 | 1.766× | 1.640–1.907 |
| 4 KiB | 10.447 | 7.358 | 12.150 | 1.200× | 1.122–1.217 |
| 8 KiB | 10.863 | 10.601 | 12.378 | 1.162× | 1.137–1.211 |
| 16 KiB | 11.087 | 11.224 | 12.954 | 1.177× | 1.149–1.202 |
| 64 KiB | 10.920 | 10.931 | 13.380 | 1.202× | 1.171–1.233 |
| 256 KiB | 10.530 | 10.893 | 13.110 | 1.172× | 1.145–1.221 |
| 1 MiB | 11.108 | 10.892 | 13.135 | 1.173× | 1.138–1.196 |
| 4 MiB | 10.605 | 10.797 | 13.061 | 1.170× | 1.124–1.269 |
| 8 MiB | 14.460 | 14.524 | 14.360 | 0.999× | 0.979–1.018 |
| 16 MiB | 14.488 | 14.517 | 14.349 | 1.000× | 0.974–1.053 |

Ordinary copying at cold 4 KiB has a paired median of 0.713× stock
(p10–p90 0.638–0.765), despite winning reused-buffer copying. Explicit streaming
remains appropriate only when the CPU will not consume the destination soon.
There is no established streaming benefit below 1 KiB or above 4 MiB.

[Current distributions and raw streams](evidence/memory-boxed-error-2026-09-08.tar.gz)
retain all three sweeps, including outliers. Package snapshots span 52–64 °C.
The 2.6-GHz ceiling and observed-clock admission remain development conditions:
firmware reports `lap-detected`, the profile/governor remain balanced/powersave,
and GPU clocks are not locked. Every artifact retains `canonical=false`.

- Portable executable SHA-256:
  `c4e2ab252ea692570764393bf07a565541c26a42985fec1d77ac56938c20b53c`.
- Native executable SHA-256 (both native sweeps):
  `8acd275910196a12055d59d30d9f6f24de07dd43c4a10eaf87759f0dddd9f3d9`.
- Archive SHA-256:
  `4a6dd7e44f78cf1bd09cf4ea038746dc41f63f764a86930877431ed01f73d834`.
- Default all-feature tests: 55 Rust tests pass, 29 hardware-dependent tests
  remain ignored; 23 Python tests pass. Formatting, Clippy, and generation
  drift checks pass. Portable and native core integration tests and native
  Error unit tests pass under AddressSanitizer; std/dependencies are not rebuilt
  with instrumentation. Ignored GPU tests were not rerun.

The [test layout](../../../test/README.md) now follows
`test/rs/<module>/test_*.rs` and `test/py/<module>/test_*.py`. Explicit Cargo
suites and a Python inventory gate prevent silently orphaned Rust tests.

## Earlier portable cleanup checkpoint

Source `41854758496b7b67cb7e36f01b9802acf85d40a8` removes the custom ordinary-copy
dispatcher from baseline x86-64 builds. After the existing checked length
boundary, these builds use the platform copy intrinsic directly. This gives
up the earlier portable tiny-copy gains and runtime-gated cached 4-KiB route;
it does not claim a new algorithm that beats the platform everywhere.
Native-target copying is unchanged: rebuilding produced the same complete
executable SHA-256 as the native measurements below.

The full portable sweep passed all correctness gates and recorded seven fresh
measured processes over 68 aligned/misaligned runtime cases, plus separate
fixed-size diagnostics. Sixty ordinary-copy median ratios were within
0.97–1.03, none exceeded 1.03, and eight remained below 0.97. Thus the former
large portable penalty is reduced, **not completely eliminated**.

| Case | Rust std ns | OARS ns | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|
| 64 B, misaligned | 2.764 | 2.755 | 0.997× | 0.988–1.004 |
| 128 B, aligned | 3.554 | 3.572 | 0.996× | 0.886–1.002 |
| 128 B, misaligned | 3.526 | 3.914 | 0.901× | 0.893–1.061 |
| 256 B, aligned | 3.918 | 3.899 | 1.004× | 0.978–1.051 |
| 256 B, misaligned | 14.993 | 15.198 | 0.975× | 0.974–1.027 |
| 512 B, aligned | 6.814 | 6.886 | 1.017× | 0.976–1.146 |
| 512 B, misaligned | 8.417 | 8.517 | 0.990× | 0.943–1.025 |
| 4 KiB, aligned | 40.713 | 39.886 | 1.023× | 1.006–1.039 |

The remaining below-0.97 cases are aligned 129 bytes and misaligned
65/96/127/128/129/192/257 bytes. Their paired medians range from 0.900 to
0.953. This full-sweep protocol differs from the earlier quick sweep's
iteration counts and allocation sequence; do not calculate a before/after
latency speedup from their absolute nanoseconds. Each stock/OARS pair within
one process uses the same buffers and workload.

Disassembly of this exact portable binary shows fifteen instructions per hot
loop for both OARS and stock Rust, with the same `memcpy@GLIBC_2.14` target.
The OARS function still reserves a larger stack frame for its cold error
conversion (0x58 versus 0x28 bytes), and instruction placement differs.
Caller layout/error representation is a next investigation, not a proven
explanation of every residual difference. No Error representation or public
failure contract was changed in this checkpoint.

The separate portable streaming quick sweep also passed seven measured
processes over the 64-MiB arena. Tested 1-KiB–4-MiB chunks achieved paired
median speedups of 1.15–1.75. The 512-byte median remains 0.944 (p10–p90
0.850–1.020); this does not establish a streaming benefit outside the admitted
non-temporal window.

[Raw cleanup distributions and streams](evidence/memory-portable-cleanup-2026-09-08.tar.gz)
retain the unchanged harness-v2 protocol: two warmup processes, seven measured
processes, three-second cooldowns, the 65 °C start gate, and observed-clock
development admission. Package snapshots ranged from 54 to 65 °C. The same
firmware/power limitations below still apply; this is not release evidence.

- Portable executable SHA-256:
  `96a894b498b16fcda5a8b20a3a2e85944f3e44f6d8ea079a4d617a47778a3045`.
- Cleanup archive SHA-256:
  `957f8339667a46f918032232ee625944156706e8d8aa66f7f3356b736b2e4992`.
- Formatting, all-feature Clippy/tests, generator/profiling tests, generation
  drift, and portable AddressSanitizer API/guard-page checks passed.

## Historical method and evidence

The historical native tables below measure clean source commit
`7647ad08b26cd4bcc3ceb0e46bf3147ed02b4df6`; the earlier portable table measures
`1ac3cd28a4affc2c2ce3716cbda6dc7ed7dd48c1`. The latter changes the portable
small-copy cutoff to 128 bytes. Rebuilding its native executable produced the
exact same SHA-256 as the measured native binary, so that evidence remains
applicable byte-for-byte. Each sweep has two warmup
processes, seven fresh measured processes, three-second cooldowns, a 65 °C
package-temperature start gate, correctness checks, and raw stdout/stderr.
Each process retains five inner warmups, 21 samples, and rotating policy order.

Harness v2 selects the implementation outside the timed loop and passes both
slices through `black_box` on every iteration. This prevents compiler load
hoisting and asymmetric loop dispatch. It also initializes allocated bytes
before forming Rust slices. The size inventory, alignments, iteration counts,
and CSV contract retain continuity with OA C++, but the old v1 timings are
**not directly comparable**. The superseded v1 report remains in Git history
at `d698928`; its ratios must not be used to quantify this change.

Latency/throughput columns are medians of process medians. Speedups are the
median of seven **paired** stock/OARS latency ratios from the same process.
Their p10–p90 range describes observed spread, not a confidence interval.
Consequently a displayed speedup need not equal the ratio of displayed medians.
A speedup above one favors OARS.

[Raw JSON distributions, provenance, and CSV/stderr streams](evidence/memory-2026-09-08-v2.tar.gz)
are retained together. The archive also preserves the earlier `e80d28f`
checkpoint's full copy, full streaming, and portable sweeps. The following
tables use `final-native-copy.json`, `final-native-streaming.json`, and
`portable-128-copy.json`. `final-portable-copy.json` records the superseded
32-byte cutoff experiment, not the final portable implementation. Archive SHA-256:
`a379dcfb7ae8614391b2171e637e973ba4abbf2e3c52478ebd88e5a16a7f81f0`.

## Historical ordinary copy: native target

The full sweep covers 34 sizes, aligned and misaligned, for 68 runtime cases
plus separate compiler-sized diagnostics. Of the 68 ordinary-copy cases, 54
had median speedups above 1.03 and 14 fell within 0.97–1.03. None fell below
0.97. This descriptive band is not proof of statistical equivalence.

Aligned, reused-buffer results:

| Bytes | Rust std ns | OARS ns | Paired speedup | p10–p90 |
|---:|---:|---:|---:|---:|
| 8 | 3.502 | 1.896 | 1.850× | 1.830–1.857 |
| 16 | 3.498 | 1.899 | 1.845× | 1.829–1.862 |
| 32 | 2.750 | 1.897 | 1.445× | 1.296–1.659 |
| 64 | 2.723 | 2.317 | 1.175× | 1.172–1.197 |
| 128 | 3.498 | 2.599 | 1.344× | 1.331–1.371 |
| 256 | 3.902 | 3.266 | 1.192× | 1.149–1.201 |
| 257 | 5.149 | 3.684 | 1.385× | 1.303–1.523 |
| 512 | 6.767 | 4.638 | 1.451× | 1.423–1.480 |
| 1 KiB | 11.433 | 7.762 | 1.471× | 1.457–1.606 |
| 4 KiB | 39.925 | 28.105 | 1.421× | 1.387–1.447 |
| 64 KiB | 1794.631 | 1797.587 | 0.998× | 0.984–1.000 |
| 1 MiB | 60446.215 | 60637.930 | 1.003× | 0.966–1.007 |
| 64 MiB | 4600244.000 | 4761240.250 | 0.984× | 0.933–0.992 |

The corresponding misaligned 512-byte, 1-KiB, and 4-KiB median speedups are
1.548×, 2.045×, and 1.262×. All offsets and distributions are in the archive.
These results support the cached medium-copy path on this CPU, not a benefit
for every cache state, application, or processor.

## Historical one-way streaming: native target

The final quick sweep rotates through a 64-MiB arena with 13 chunk sizes;
the earlier checkpoint's full 256-MiB/15-chunk sweep is also archived.
This policy assumes the CPU will not consume the destination soon.
Throughput is decimal GB/s.

| Chunk | Rust std | OARS ordinary | OARS streaming | Stream speedup | p10–p90 |
|---:|---:|---:|---:|---:|---:|
| 256 B | 7.788 | 7.800 | 7.538 | 0.969× | 0.934–1.079 |
| 512 B | 8.533 | 8.373 | 8.823 | 1.027× | 0.947–1.071 |
| 1 KiB | 8.465 | 8.069 | 10.009 | 1.196× | 1.095–1.366 |
| 2 KiB | 7.180 | 8.310 | 11.981 | 1.700× | 1.642–1.762 |
| 4 KiB | 10.794 | 8.210 | 13.054 | 1.201× | 1.168–1.213 |
| 8 KiB | 11.391 | 11.352 | 13.185 | 1.163× | 1.155–1.187 |
| 16 KiB | 11.221 | 10.839 | 13.420 | 1.190× | 1.153–1.223 |
| 64 KiB | 11.281 | 11.217 | 13.314 | 1.182× | 1.156–1.203 |
| 256 KiB | 11.231 | 11.339 | 13.197 | 1.184× | 1.114–1.234 |
| 1 MiB | 11.324 | 11.457 | 13.427 | 1.194× | 1.162–1.209 |
| 4 MiB | 11.175 | 11.172 | 13.324 | 1.186× | 1.143–1.202 |
| 8 MiB | 15.042 | 15.148 | 14.881 | 0.997× | 0.986–1.025 |
| 16 MiB | 14.966 | 15.081 | 15.019 | 1.002× | 0.993–1.042 |

The ordinary cached 4-KiB route loses about 24% throughput in this workload,
despite winning the reused-buffer test. Streaming wins about 20% instead.
Cache intent must remain explicit; the library cannot infer future destination
reuse from two slices. There is no demonstrated streaming win below 1 KiB or
above 4 MiB.

## Earlier portable 128-byte cutoff (superseded)

The earlier portable quick sweep covers 28 runtime cases with no native CPU
flags. Aligned results show that fallback to the platform copier still incurs
length-check/dispatch and compiler-codegen costs:

| Bytes | Rust std ns | OARS ns | Paired speedup | p10–p90 | Misaligned speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.500 | 2.021 | 1.730× | 1.586–1.863 | 1.751× |
| 32 | 2.800 | 2.052 | 1.371× | 1.350–1.490 | 1.364× |
| 64 | 2.774 | 2.618 | 1.059× | 0.984–1.073 | 0.846× |
| 128 | 3.561 | 4.082 | 0.876× | 0.754–0.982 | 0.705× |
| 256 | 3.889 | 4.277 | 0.907× | 0.806–0.912 | 0.998× |
| 512 | 6.725 | 7.572 | 0.887× | 0.839–0.948 | 0.955× |
| 4 KiB | 36.175 | 32.030 | 1.134× | 1.095–1.145 | 1.130× |
| 64 KiB | 1797.274 | 1800.095 | 0.999× | 0.996–1.084 | 0.998× |

Across aligned/misaligned cases, nine medians exceed 1.03, thirteen are within
0.97–1.03, and six are below 0.97. Portable 64–512-byte copies remain an
optimization target, especially misaligned 128-byte copies at about 30% lower
throughput than stock. The 32-byte cutoff experiment improved that case but
regressed aligned 64-byte copies; the final 128-byte cutoff retains the earlier
small-copy behavior while improving the 256-byte fallback. These tradeoffs
remain visible rather than being averaged into a universal win.
Native CPU targeting stays opt-in:
silently enabling it would make distributed binaries unsafe on older CPUs.

## Platform, build, and qualification

- Intel Core i5-1145G7, x86-64, AVX2 and AVX-512F/BW; Linux 7.2.2.
- rustc 1.98.1 / LLVM 22.1.8; Clang and LLD 22.1.8.
- Native flags: `-C target-cpu=native -C link-arg=-fuse-ld=lld`.
- Portable build: repository defaults, no `RUSTFLAGS`; Clang/LLD still selected.
- Historical native executable SHA-256:
  `c66d7a055883880be5ed4f18bd05f83dfd7fd4b0d8cbdd4a76aaaf2d291bd285`.
- Earlier portable executable SHA-256:
  `f728c49eaa0f1e03afe86d6ee8728c9b9dcb3e85fb02c03684c05e747f365854`.

The CPU ceiling was 2.6 GHz; most post-process policy-0 snapshots were near
2.6 GHz, while idle/cooldown snapshots were lower. These before/after snapshots
do not prove continuous clock stability. Recorded package temperatures across
the historical native/cutoff sweeps ranged from 51 to 60 °C. The machine remained
`balanced`/`powersave` with firmware `lap-detected`; GPU clocks were not
locked. No clock/power settings were changed for these observed-state runs.

The user-approved observed-clock protocol admits this as local development
evidence, not canonical fixed-state or cross-machine qualification.
`canonical=false` remains in every artifact. The executable attempts to pin
itself to the first allowed CPU, but the runner does not record pin success for
every process. Desktop activity was not eliminated; spread must remain visible.

Formatting, all-feature Clippy/tests, generator drift/tests, and profiling
tests pass. Native memory unit/API tests and Linux guard pages also pass under
AddressSanitizer. The sanitizer does not rebuild std/native dependencies;
Miri and broader platform qualification remain planned. Ignored hardware
Vulkan tests were not rerun as part of this CPU benchmark checkpoint.

## Reproduce

Build and measure native copy, keeping the same flags in the recorder's
environment for provenance:

```bash
RUSTFLAGS='-C target-cpu=native -C link-arg=-fuse-ld=lld' \
  cargo build --release --example core_memory_bench
RUSTFLAGS='-C target-cpu=native -C link-arg=-fuse-ld=lld' \
  python3 tools/profiling/memory_rust.py \
    --binary target/release/examples/core_memory_bench \
    --mode copy --observed-clocks --output /absolute/results/native-copy.json
```

Use `--mode streaming --quick` for the final 64-MiB streaming protocol.
For the portable quick sweep, rebuild without `RUSTFLAGS` and record
`--mode copy --quick` without that environment override.

For canonical admission, omit `--observed-clocks` and invoke the recorder
through `tools/profiling/stable_clocks.sh --cpu-khz 2600000 --gpu-mhz 1000 --`.
The wrapper must actually establish and verify the performance profile,
governor, and fixed clocks; the firmware limitation is not silently waived.
