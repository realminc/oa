# OA C++ / GLM / OARS VLM comparison

**Status:** Experimental CPU-capped development evidence; canonical measurement blocked

**Updated:** 2026-09-08

This report owns cross-language VLM performance evidence. It does not change
the VLM semantic contract.

## Current vector / quaternion fast-path checkpoint

The broader pass retains selective inlining for Vec3/Vec4 normalization,
Vec3 projection, quaternion rotation, and checked look-at construction.
Projection's existing scaled fallback is a private cold, non-inlined helper.
Vec2 normalization, vector length, quaternion normalization, and Mat4 multiplication
retain their previous policies. Arithmetic, failure checks, extreme-value
fallbacks, and public layouts are unchanged. No custom memcpy, unsafe code,
fast-math, reduced precision, or new ISA requirement was introduced.

These are **native-build development results**, not a universal 3× claim or
an accepted whole-library regression baseline. The saved Rust baseline already
includes the earlier Mat3 improvement; the speedups below are additional gains
against that baseline, not against stock Rust or GLM.

All 100 workloads (50 cases × FP32/FP64) were rerun in alternating C++ / saved
Rust / candidate Rust process order: two warmup rounds, seven measured rounds,
three-second inter-round cooldown, CPU affinity `[2]`, and a 65 °C package
admission gate. Each process uses 65,536 items, three inner warmups, and eleven
inner samples. Every cross-language and baseline/candidate checksum gate passed.

Latency columns are medians of process medians in ns/item. Speedups are medians
of the seven paired baseline/candidate ratios. p10–p90 is observed spread, not
a confidence interval; it need not match the ratio of displayed latency medians.

| Case | Rust before | Rust after | OA C++ | GLM | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|---:|---:|
| Quaternion rotate, FP32 | 34.345 | 7.639 | 7.000 | 5.522 | 4.504× | 4.150–4.555 |
| Quaternion rotate, FP64 | 33.808 | 6.805 | 7.740 | 5.754 | 4.983× | 4.693–5.101 |
| Vec3 project, FP32 | 38.203 | 9.468 | 5.217 | 3.369 | 4.047× | 3.477–4.069 |
| Vec3 project, FP64 | 39.336 | 9.870 | 5.664 | 3.721 | 3.990× | 3.967–4.244 |
| Vec3 normalize, FP32 | 16.239 | 6.062 | 5.038 | 3.232 | 2.678× | 2.205–2.701 |
| Vec3 normalize, FP64 | 16.433 | 6.872 | 4.930 | 4.196 | 2.389× | 2.342–2.682 |
| Vec4 normalize, FP32 | 7.152 | 6.176 | 5.739 | 3.907 | 1.160× | 1.003–1.526 |
| Vec4 normalize, FP64 | 8.448 | 6.833 | 5.542 | 4.463 | 1.233× | 1.226–1.616 |
| Look-at, FP32 | 92.213 | 89.917 | 46.920 | 24.282 | 1.025× | 1.022–1.048 |
| Look-at, FP64 | 99.629 | 97.937 | 55.387 | 28.025 | 1.017× | 0.996–1.036 |
| TRS compose, FP32 | 28.125 | 28.143 | 12.536 | 35.780 | 1.000× | 0.872–1.027 |
| TRS compose, FP64 | 30.071 | 30.097 | 17.726 | 55.143 | 0.999× | 0.838–1.005 |
| Viewport unproject, FP32 | 141.995 | 141.565 | 124.957 | 6.963 | 0.999× | 0.987–1.031 |
| Viewport unproject, FP64 | 127.724 | 128.581 | 137.483 | 50.140 | 0.994× | 0.969–1.261 |

Across the full suite, FP32 has four paired median speedups above 1.03, 46
within 0.97–1.03, and none below 0.97. FP64 has eight, 37, and five respectively.
The five lower FP64 medians remain unresolved, not dismissed as noise:

| FP64 case | Rust before | Rust after | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|
| Mat4 transpose | 7.479 | 8.920 | 0.876× | 0.767–0.964 |
| Mat4 × Vec4 | 8.092 | 8.678 | 0.947× | 0.905–1.002 |
| Transform direction | 8.589 | 8.878 | 0.947× | 0.815–1.007 |
| Transform point | 8.541 | 8.641 | 0.966× | 0.869–1.025 |
| Vec4 add | 2.040 | 2.136 | 0.945× | 0.875–0.998 |

Their operation bodies were not edited, but that does not establish the cause
or rule out changes in compiled code or layout. No across-the-board
non-regression claim is made. The FP64 normal-matrix row also improved without
a direct algorithm edit (146.715 → 124.041 ns/item, paired 1.182×); attribution
likewise remains unproven. Mat4 determinant and multiply still have substantial
gaps to OA C++: final FP32 latencies are 62.902 versus 7.569 and 33.304 versus
11.709 ns/item respectively. The earlier Mat3 gain is retained, not multiplied
by the new vector/quaternion speedups.

### Rejected and superseded broader policies

- `9003c97e5f32c85e223f2acb13ff615e3eda74ba` (`candidate.json`) also hinted
  matrix helpers and split quaternion normalization into fast/cold helpers.
  It passed correctness but FP64 TRS composition regressed from 30.525 to
  62.454 ns/item, paired 0.484× (p10–p90 0.448–0.590). Those policies were removed.
- `eeebb26af7ae045db5cb972ddabd26f76cebed4e` (the intermediate `final.json`)
  restored TRS behavior but retained broader vector hints. FP32 look-at was
  0.963×, Vec2 normalization 0.961×, and viewport unprojection 0.957×. The retained
  source removes the vector-length hint, limits the normalization hint to Vec3/4,
  and gives look-at its own hint. The final measured report is `accepted.json`;
  its filename does not confer canonical or whole-suite regression qualification.

### Provenance and verification

- Rust baseline: `1c3d84a8a7fa746ae61850c865d440f179babd36`, executable SHA-256
  `2e4c8121989797ab71fcc67378b24772c22abf41319e993a290ca377fb4eb34a`.
- Retained Rust source: `6d1dfb2fc0601a7bfd7ace8b61582cb576157511`, executable SHA-256
  `41032e7272487c946232e5f0aa66d0009c472349ba2bf2951da6f3e852017200`.
- OA C++ source: `b770c237621eb2ae9c353568e5d2a87649e853ff`, unchanged executable
  SHA-256 `4ab898c23f31ccf7d1412e7153e853393db9e8de27591e7bd8fb212e2890b00c`.
- All source records are clean. Both Rust executables use
  `-C target-cpu=native -C link-arg=-fuse-ld=lld`. C++ uses
  `-std=c++20 -O3 -DNDEBUG -march=native -fuse-ld=lld -DOA_VLM_HAS_GLM=1`, with
  the real `assert.cpp` linked from source. GLM 1.1.0 and its manifest hash are
  unchanged from the historical Mat3 checkpoint below.
- Intel Core i5-1145G7; rustc 1.98.1 / LLVM 22.1.8; Clang/LLD 22.1.8.

The CPU remains capped at 2.6 GHz, not locked there. Package snapshots span
49–60 °C. Balanced/powersave, firmware `lap-detected`, unlocked GPU clocks, and
varying sampled CPU frequencies remain recorded limitations. `canonical=false`
is preserved. Portable performance and other processors are unmeasured.

All baseline gates pass: 68 Rust tests, 25 Python tests, formatting, all-feature
Clippy, generator drift, and whitespace checks. Thirty hardware tests remain
ignored; this host-only change does not claim new GPU validation. All 29 VLM
tests pass under portable and native AddressSanitizer builds; std/native
dependencies are not rebuilt with instrumentation. Nine added tests cover
Vec2/3/4 and quaternion normalization in both precisions across zero,
non-finite, subnormal, signed-scale, and extreme values, plus scaled/invalid
vector projection. Existing independent matrix and rotation oracles remain.

[Raw broader-pass evidence](evidence/vlm-fast-paths-2026-09-08.tar.gz) contains
all three reports and their complete stdout/stderr streams. Every stream hash,
100-case checksum gate, per-implementation distribution, and paired speedup was
reproduced directly from the archive. Archive SHA-256:
`e497c505feb6476d00d41214b987b51db682fca45de8a2eb60265b0be0f9c416`.
Reproduction uses the commands below with the new source and the saved
Mat3-checkpoint report supplied as `--baseline-oars-record`.

## Historical Mat3 optimization checkpoint

The retained change is an inlining hint for `Mat3` multiplication. Its arithmetic,
loop order, packed layout, and public contract are unchanged. `Mat4` and its
checked inverse retain their previous implementation. No custom memory copier,
unsafe code, fast-math, reduced precision, or extra ISA requirement was added.
Timings below are from native-target builds on this laptop; portable performance
and other CPUs remain unmeasured.

The final comparison reruns a saved clean-source Rust baseline alongside the
candidate and freshly built OA C++/GLM reference. Each of nine rounds runs all
three executables, reversing their order on alternate rounds. Two rounds warm
up; seven are measured, with three-second cooldowns between rounds and a 65 °C
package admission gate before each process. All inherit CPU affinity `[2]`.
The workload remains 50 cases × two precisions, 65,536 items, three inner
warmups, and eleven inner samples. Every checksum comparison passed.

Latency columns are medians of process medians in ns/item. Speedups are medians
of the seven **paired** baseline/candidate ratios; p10–p90 is observed spread,
not a confidence interval. A speedup above one favors the candidate and need
not equal the ratio of the displayed latency medians.

| Case | Rust before | Rust after | OA C++ | GLM | Paired speedup | p10–p90 |
|---|---:|---:|---:|---:|---:|---:|
| Mat3 multiply, FP32 | 32.893 | 9.586 | 8.124 | 9.484 | 3.402× | 2.842–3.487 |
| Mat3 multiply, FP64 | 33.915 | 11.312 | 9.740 | 11.130 | 2.984× | 2.684–3.150 |
| Mat4 multiply, FP32 | 33.419 | 33.620 | 12.491 | 21.723 | 1.004× | 0.927–1.042 |
| Mat4 multiply, FP64 | 36.500 | 36.837 | 17.345 | 17.506 | 1.000× | 0.921–1.138 |
| Viewport unproject, FP32 | 142.869 | 143.461 | 126.766 | 6.979 | 1.004× | 0.903–1.011 |
| Viewport unproject, FP64 | 130.064 | 129.473 | 140.515 | 50.965 | 1.007× | 0.932–1.286 |

This closes most of the Mat3-multiply gap, not the whole VLM gap: OARS is still
about 16–18% higher latency than OA C++ in these two rows. In the complete
suite, FP32 has three paired median speedups above 1.03, 47 within 0.97–1.03,
and none below 0.97. FP64 has five, 43, and two respectively. The two lower
medians are unchanged `mat3_determinant` (0.967×, p10–p90 0.893–1.046) and
`vec3_distance` (0.968×, 0.733–0.998). They remain visible; this is not an
across-the-board non-regression guarantee. Do not attribute changes in untouched
operations to the Mat3 algorithm without further evidence.

Checked normalization, quaternion rotation, vector projection, general Mat4
determinant, and Mat4 multiplication remain optimization targets. For example,
final FP32 normalization is 16.350 ns/item versus OA C++ 4.989; determinant is
62.929 versus 7.351. Their checks and mathematical contracts were not weakened.

### Rejected broader candidate

Source `80ebe43ced290e99fe442e588cffb26615872343` interchanged the column/reduction
loops and hinted inlining for both matrix sizes, without changing each output's
accumulation order. It passed correctness and improved Mat3 and Mat4 multiply,
but FP64 viewport unprojection regressed to 152.740 ns/item from 128.654
(paired speedup 0.837×, p10–p90 0.819–0.915). Its `candidate.json` and full raw
streams are retained. The final source removes that broader change; its Mat3
gain comes from inlining alone. The faster standalone Mat4 row is not accepted
as a win for its dependent checked-inverse workload.

### Provenance and qualification

- Rust baseline: `c06d1133067698fbf19b152e9f0000a88012612d`.
- Final Rust source: `1c3d84a8a7fa746ae61850c865d440f179babd36`.
- OA C++ source: `b770c237621eb2ae9c353568e5d2a87649e853ff`.
- All source records are clean. Baseline and candidate use identical
  `-C target-cpu=native -C link-arg=-fuse-ld=lld` flags.
- Intel Core i5-1145G7; rustc 1.98.1 / LLVM 22.1.8; Clang/LLD 22.1.8.
- GLM 1.1.0 from `/home/empyrealm/Code/3d/glm-master`. The 432-file `glm/`
  manifest SHA-256 is `43da209efa40ab8761f458e06654f3d1f97511f6d0ccbe0d5541c12d9ab2dad7`:
  sorted root-relative paths, each line `<file SHA-256>  <path>\n`.
- Baseline Rust executable SHA-256:
  `dce68fa83b1190c3a92f797841d59e560bbd380f35553c823ae55b1bf7f36401`.
- Final Rust executable SHA-256:
  `2e4c8121989797ab71fcc67378b24772c22abf41319e993a290ca377fb4eb34a`.
- OA C++ executable SHA-256:
  `4ab898c23f31ccf7d1412e7153e853393db9e8de27591e7bd8fb212e2890b00c`.

The CPU ceiling was 2.6 GHz, not a fixed clock. Recorded post-process
frequencies varied; they do not prove continuous loaded-clock stability.
Final package snapshots span 57–63 °C. The machine remained balanced/powersave
with firmware `lap-detected`, and GPU clocks were not locked. These are
user-admitted development observations, not canonical release evidence;
`canonical=false` is preserved.

Formatting, all-feature Clippy/tests, Python tests, and generator drift checks
pass: 59 Rust tests and 25 Python tests pass, with 30 hardware tests ignored.
Twenty VLM tests pass under both portable and native AddressSanitizer builds;
std/native dependencies are not rebuilt with instrumentation. The four new
matrix-product tests compare independent dot products, including signed zero,
subnormals, infinities, overflow, and NaN classification. GPU tests were not
rerun for this host-only checkpoint.

[Raw baseline, rejected candidate, and final evidence](evidence/vlm-mat3-2026-09-08.tar.gz)
contains the three JSON reports and all stdout/stderr streams. All 100 case
distributions and paired speedups were reproduced from those archived streams;
the candidate/final per-stream hashes were verified. Archive SHA-256:
`c7133650666a1124890626e3813e27c2c2b7daa5ddf75d5176d94a175a9ab04a`.

### Reproduce this development comparison

Build the C++ reference from the recorded OA tree, linking its real contract
handler and keeping GLM external (no C++ repository source changes):

```bash
clang++ -std=c++20 -O3 -DNDEBUG -march=native -fuse-ld=lld \
  -DOA_VLM_HAS_GLM=1 -Isource/cpp/include \
  -isystem /absolute/glm-1.1.0 \
  test/cpp/core/vlm/benchVlm.cpp source/cpp/lib/oa/core/assert.cpp \
  -o /absolute/results/oa-cpp-bench
```

Build each Rust source with the native flags above and save each executable
outside Cargo's output tree. First record the baseline with `vlm_compare.py`;
then run the same command against the candidate, adding the baseline record:

```bash
taskset -c 2 python3 tools/profiling/vlm_compare.py \
  --oa-repo /absolute/oa --oars-repo /absolute/oars \
  --oa-binary /absolute/results/oa-cpp-bench \
  --oars-binary /absolute/results/candidate-native \
  --baseline-oars-record /absolute/results/baseline.json \
  --oa-build-flags '-std=c++20 -O3 -DNDEBUG -march=native -fuse-ld=lld -DOA_VLM_HAS_GLM=1; GLM 1.1.0; assert.cpp linked from source' \
  --oars-rustflags '-C target-cpu=native -C link-arg=-fuse-ld=lld' \
  --output /absolute/results/final.json --exploratory
```

The baseline run omits `--baseline-oars-record`, points `--oars-binary` at the
saved baseline executable, and writes `baseline.json`. The optional record
reruns that exact binary with the current workload; it does not compare against
its old timings. Its checksum/hash/build/source admission and paired-ratio
calculation are exercised by the tests and the complete final run.

## Workload and oracle

OA C++ and OARS execute the same 50 named cases for FP32 and FP64 over 65,536
deterministically generated items. Both use three inner warmups, 11 inner
samples, a median nanoseconds-per-item estimator, and the same checksum
reduction. Cases retain their `arithmetic` or `hardened` classification.

`tools/profiling/vlm_compare.py` alternates OA C++/OARS process order, defaults
to two warmup pairs and seven measured pairs, and validates every OARS checksum
against OA C++ before accepting timing. OA C++ separately validates its result
against GLM 1.1.0. The C++ checksum serialization uses 17 significant digits so
the FP64 `2e-11` tolerance survives the process boundary.

## Canonical command

```bash
RUSTFLAGS='-C target-cpu=native -C link-arg=-fuse-ld=lld' \
  cargo build --release --example core_vlm_bench
tools/profiling/stable_clocks.sh --cpu-khz 2600000 --gpu-mhz 1000 -- \
  python3 tools/profiling/vlm_compare.py \
    --oa-repo /home/empyrealm/Code/GitHub/oa \
    --oars-repo /home/empyrealm/Code/GitHub/oars \
    --oa-binary /absolute/clean/oa/benchVlm \
    --oars-binary /absolute/clean/oars/core_vlm_bench \
    --oars-rustflags '-C target-cpu=native -C link-arg=-fuse-ld=lld' \
    --output /absolute/result/vlm.json
```

Admission requires clean immutable repositories, exact executable hashes, the
`performance` power profile and governor, CPU min=max 2.6 GHz, Xe min=max
1.0 GHz, the 65 °C start gate, no degraded profile, no GPU throttling, and no
requested or observed clock drift. Raw child output and process observations
remain beside the JSON artifact.

## Historical development baseline (before this pass)

A 2026-09-08 run covered the full FP32 and FP64 suite with two warmup pairs,
seven measured pairs, alternating language order, and every checksum gate
passing. Active-core observation remained effectively at 2.6 GHz; the recorded
policy-0 snapshot median was 2.600 GHz. This is accepted as a platform-local
development baseline. It is not canonical release evidence because both trees
were dirty, cooldown was zero, the profile was `balanced`, governors were
`powersave`, clocks were not requested as min=max, and firmware reported
`lap-detected`.

Within that diagnostic only, the median OARS/OA-C++ ratio across operations was
1.163× for FP32 and 1.102× for FP64. Using the existing 0.95/1.03 classification
bands, FP32 had 10 faster, 8 parity, and 32 slower rows; FP64 had 13, 10, and 27.
The largest recurring gaps were checked projection, general determinant and
matrix multiply, quaternion rotation, and hardened normalization.

Representative FP32 medians in ns/item are:

| Case | OA C++ | GLM | OARS | OARS / OA |
|---|---:|---:|---:|---:|
| `mat4_vec4` | 5.907 | 5.866 | 3.804 | 0.644 |
| `mat4_transpose` | 6.255 | 6.097 | 4.520 | 0.723 |
| `transform_point` | 5.799 | 5.603 | 4.307 | 0.743 |
| `mat3_determinant` | 2.527 | 2.485 | 2.195 | 0.869 |
| `mat4_mul` | 12.260 | 21.834 | 33.416 | 2.726 |
| `vec3_normalize` | 5.060 | 3.352 | 16.338 | 3.229 |
| `quat_rotate` | 7.063 | 5.687 | 34.534 | 4.889 |
| `vec3_project_checked` | 5.309 | 3.687 | 38.439 | 7.240 |
| `mat4_determinant` | 7.541 | 7.294 | 62.795 | 8.327 |

Representative FP64 medians in ns/item are:

| Case | OA C++ | GLM | OARS | OARS / OA |
|---|---:|---:|---:|---:|
| `mat3_determinant` | 5.089 | 5.279 | 2.643 | 0.519 |
| `vec4_add` | 3.588 | 3.713 | 2.152 | 0.600 |
| `vec4_dot` | 3.222 | 3.157 | 2.036 | 0.632 |
| `mat4_inverse_checked` | 60.453 | 42.405 | 154.067 | 2.549 |
| `normal_matrix_checked` | 57.437 | 11.214 | 146.850 | 2.557 |
| `vec3_normalize` | 5.149 | 4.244 | 16.426 | 3.190 |
| `quat_rotate` | 7.859 | 5.970 | 34.130 | 4.343 |
| `vec3_project_checked` | 6.266 | 5.074 | 40.103 | 6.400 |
| `mat4_determinant` | 10.774 | 10.320 | 70.809 | 6.572 |

Lower latency is better. The aggregate ratios and these rows are valid for
choosing the next local optimization targets; they are not portable or release
speed claims.

No fixed-state result may replace this status until the reference laptop stops
reporting `lap-detected` and the complete canonical command passes.
