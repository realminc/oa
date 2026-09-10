# OA Rust Numeric Stability

**Status:** Canonical policy; only the documented Experimental FP32/I32 slices are implemented

**Updated:** 2026-09-09

Numeric behavior belongs to semantic operation contracts and immutable physical
plans. A faster route is not equivalent if it changes dtype, accumulation,
reduction order, overflow, NaN handling, color conversion, quantization or
determinism outside the admitted policy.

## Current policy

- Dense Matrix storage currently admits `f32`, plus `i32` for Matrix add and
  `u32` for admitted index/state operations.
- Generated FP32 elementwise work has no implicit promotion.
- `i32` add wraps modulo 2^32 by contract.
- The FP32 matmul and current ML kernels accumulate in FP32.
- Training replay requires graph-resident state/RNG advances and reports source
  fallback; similar printed metrics do not prove bitwise determinism.
- FP16, BF16, FP64 device compute, mixed precision and packed quantization are
  not general OARS capabilities.

## Required separation

Every operation or plan records independently:

| Fact | Examples |
|---|---|
| semantic dtype/encoding | dense `f32`, future BF16, Q4 block encoding |
| physical storage | FP16 input, packed bytes plus FP32 scales |
| compute precision | FP16/BF16/FP32 arithmetic mode |
| accumulator precision | FP32 reduction for BF16 inputs |
| output precision | FP32 or rounded BF16 |
| order policy | fixed tree, atomic/unordered, deterministic split reduction |
| exceptional policy | NaN/Inf, integer overflow, divide-by-zero, saturation |
| selected artifact | exact kernel/specialization and compiler provenance |

The public semantic contract chooses allowed behavior; the router may choose
only candidates inside it.

## Stable forms

Algorithms with numerically stable donor implementations should be ported,
including max-subtracted softmax/log-sum-exp, stable cross-entropy, two-pass or
otherwise qualified variance, guarded normalization and scaled accumulation.
Replacing them requires differential evidence across adversarial magnitudes,
not merely matching typical training loss.

Fusion must preserve operation order where the contract makes it observable.
Contracting multiply-add, reassociating reductions, eliminating a rounding
boundary, folding conversion into normalization or changing color transfer
math may require a distinct numeric mode rather than being treated as a free
optimization.

## Determinism modes

The target should distinguish:

- `Reproducible`: same executable/build/device/driver and fixed inputs produce
  the required repeatability contract;
- `DeterministicOrder`: kernels use a fixed reduction/update order and reject
  candidates using unordered atomics;
- `Fast`: permits explicitly documented order variation within numeric
  tolerances.

Names remain internal until a public need is proven. A deterministic request
filters candidates before tuning. Cache identity includes the mode. Cross-GPU
or cross-driver bitwise identity is never inferred from same-device evidence.

## Oracle and tolerance policy

Tolerance is operation- and dtype-specific, never a repository-wide epsilon.
Each schema should name an oracle and error metric:

- exact comparison for integers, encodings and operations that promise exact
  results;
- ULP/absolute/relative bounds for elementwise floating-point work;
- dimension- and conditioning-aware bounds for reductions, GEMM and convolution;
- distributional/fixed-seed tests for RNG;
- per-channel code-value or linear-light error for image/video conversion;
- signal/error-energy and clipping checks for audio;
- end-to-end loss/logit/perplexity gates for reduced-precision models.

Always test zeros, signed zero where relevant, subnormals according to the
device policy, minimum/maximum finite values, NaN/Inf, cancellation, long
reductions and adversarial input scale. Record whether the oracle itself uses
higher precision or a trusted independent implementation.

## Vision and media

Vision tolerance is not generic tensor tolerance. A resize-normalize
microfusion must be compared to the exact source chain under the same:

- interpolation coordinates and edge mode;
- color range, primaries, transfer and YCbCr matrix;
- chroma siting and bit depth;
- channel order and alpha policy;
- rounding/clamping point, output layout and dtype.

Measure structural or perceptual metrics only in addition to, not instead of,
per-pixel contract bounds.

## Low precision gate

Before a low-precision route becomes a default:

1. define storage conversion including ties, NaN/Inf and overflow;
2. define compute/accumulation and reduction order;
3. prove packed access alignment, tails and concurrent-write safety;
4. compare against a higher-precision oracle over shape and magnitude packs;
5. run an end-to-end domain-quality workload;
6. demonstrate a measured benefit with the exact route and no unexpected
   fallback.

## Reporting

Evidence records maximum and distribution of error, failing index/input when
possible, seed, process/build/device/driver, exact candidate, fallback counters
and whether contraction/fast-math was enabled. Printed loss, accuracy or image
appearance alone is not numeric proof.

## Primary references

- [IEEE 754-2019 overview](https://standards.ieee.org/ieee/754/6210/)
- [Vulkan floating-point controls](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#limits-fp)
