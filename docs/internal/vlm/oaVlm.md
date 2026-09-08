# OA Rust Vulkan Linear Math

**Status:** Experimental; matched OA/GLM benchmark implemented

**Updated:** 2026-09-08

**Canonical architecture:**
[OA Rust Architecture](../architecture/oaArchitecture.md#2-architectural-invariants)

VLM means Vulkan Linear Math. It is a compact host-side spatial-math layer,
not a vision-language model and not an alternative to the device-resident
`Matrix` value. It owns no engine, allocation, graph, Vulkan handle, or
session. The implementation uses only Rust's standard library.

VLM does not call `core::memory`: its small typed values are consumed by the CPU
immediately. Ordinary value assignment stays compiler-managed; byte-copy calls
and non-temporal stores are not substituted for arithmetic or value movement.
Bulk dataset movement and mapped GPU uploads belong to their storage owners,
which can choose the memory policy from the destination's actual reuse contract.

## Convention and ownership

VLM is the formula authority for OA spatial vectors, rotations, affine
transforms, view matrices, projections, normal transforms, and viewport
conversion. Its fixed convention is:

- right-handed coordinates;
- `+X` right, `+Y` up, and camera-forward `-Z`;
- row-major matrix storage;
- row-vector multiplication, so `value * (a * b)` applies `a` then `b`;
- Vulkan normalized-device-coordinate depth `[0, 1]`;
- top-origin viewport conversion, with raster orientation owned by viewport
  state rather than projection matrices.

External axis, unit, handedness, and quaternion-order conversion occurs once at
the format boundary. Consumers must not add local transposes, sign corrections,
or projection element patches.

## Public surface

`oa::core::vlm` owns the implementation and `oa::vlm` is its public identity
re-export. Values are `#[repr(C)]`, packed scalar aggregates. `Vec2`, `Vec3`,
`Vec4`, `Quat`, `Mat3`, `Mat4`, and `Viewport` default to `f32`; their `D*`
aliases select `f64`. `Quat` storage is vector-first `(x, y, z, w)`.

Rust methods replace C++ overload families. Arithmetic operators remain
immediate host scalar math. Undefined operations—including zero/non-finite
normalization, singular inverse, degenerate camera bases, invalid projections,
and projective division near zero—return `None`. Scalar division itself retains
ordinary IEEE floating-point behavior. No operation panics, submits work,
waits, or falls back to a different backend.

The current Experimental surface includes:

- vector arithmetic, geometry, components, interpolation, robust
  normalization, and approximate comparison;
- quaternion construction, composition, inversion, rotation, shortest-path
  interpolation, all six explicit Euler orders, matrix conversion, from-to,
  and look rotation;
- packed 3×3 and 4×4 matrix arithmetic, row-vector transforms, determinant,
  checked inverse, affine/normal matrices, TRS, and signed-scale/shear/reflection
  decomposition;
- checked pose and look-at views, centered/shifted/off-center perspective and
  orthographic projection, finite/infinite reversed Z, and top-origin viewport
  project/unproject;
- spherical/Cartesian conversion.

## Evidence and remaining work

`test/rs/core/test_vlm.rs` is an external-style independent contract suite. It covers
packed sizes, `f32` and `f64`, zero/non-finite/subnormal/extreme normalization,
right-handed orientation, row-vector composition, quaternion/matrix parity,
all Euler orders, shortest-path interpolation, inverse and singular failure,
affine recomposition with reflection and shear, camera-forward `-Z`, Vulkan
depth endpoints, odd-size viewport round trips, and large-world double
precision.

Mat3 multiplication explicitly permits inlining at the call site; Mat4 retains
the compiler's default inlining policy. Both retain their original loop nesting
and `+0`, increasing-inner-index multiply/add order, with no reassociation,
fast-math, new ISA requirement, or public-layout change. Independent 3×3/4×4 `f32`/`f64`
dot-product tests check exact non-NaN results, including signed zeros, subnormals,
overflow, and infinities; NaN classification, not payload identity, is checked.

Vec3/Vec4 normalization, Vec3 projection, quaternion rotation, and checked
look-at construction also carry targeted inlining hints. Vec2 normalization
and vector length retain the default policy. Projection's existing scaled
fallback is a private cold, non-inlined helper; its arithmetic and rejection
behavior are unchanged. Quaternion normalization remains unsplit. These are
measured implementation choices, not a general rule to inline every method.
Normalization tests cover all four value families in both precisions across
zero, subnormal, signed-scale, extreme, and non-finite inputs. Portable and
native correctness are checked separately from native-only timing evidence;
the owning performance report records remaining FP64 regressions explicitly.

This checkpoint remains Experimental. Migration of render, animation,
simulation, and format-boundary consumers; automated duplicate-formula audits;
compiler/package qualification; and broader scalar/SIMD optimization remain
Planned. GPU batch spatial math remains deferred until a device-resident
workload proves its crossover point.

## Matched OA/GLM performance evidence

`core_vlm_bench` implements the same 50 named FP32/FP64 workloads, input
generation, item count, checksum reduction, hardened/arithmetic labels, inner
warmups, inner samples, and nanoseconds-per-item boundary as OA C++'s
`BenchVlm`. `tools/profiling/vlm_compare.py` runs the existing paired OA
C++/GLM executable and OARS executable in alternating fresh-process order. It
requires identical case sets and validates each OARS checksum against OA C++ at
the established FP32/FP64 tolerance before retaining timing.

The canonical invocation is:

```bash
tools/profiling/stable_clocks.sh --cpu-khz 2600000 --gpu-mhz 1000 -- \
  python3 tools/profiling/vlm_compare.py \
    --oa-repo /home/empyrealm/Code/GitHub/oa \
    --oars-repo /home/empyrealm/Code/GitHub/oars \
    --oa-binary /absolute/clean/oa/benchVlm \
    --oars-binary /absolute/clean/oars/core_vlm_bench \
    --oars-rustflags '-C target-cpu=native -C link-arg=-fuse-ld=lld' \
    --output /absolute/result/vlm.json
```

The outer runner defaults to two warmup process pairs, seven measured pairs,
and three seconds of cooldown. It rejects dirty repositories, missing fixed
clock/profile state, clock drift, different cases/contracts, and checksum
mismatch. OA C++ prints 17 significant checksum digits so the FP64 cross-process
oracle is not weakened by text rounding.

For a Rust optimization comparison, `--baseline-oars-record /absolute/baseline.json`
adds the saved Rust executable from a prior passing, clean-source record to each
round. The recorder verifies its hash and matching Rust flags, preserves its
original source identity, alternates baseline/candidate order, checks both
against C++, and records paired baseline/candidate speedup distributions. Save
the baseline executable outside Cargo's output path before rebuilding. Raw
streams are hashed and inherited CPU affinity is recorded. Non-finite checksums
and non-positive/non-finite timings fail admission.

Current paired diagnostic status and eventual accepted evidence belong in the
[VLM comparison report](../performance/oaVlmComparison.md), not this subsystem
contract.
