# OARS BLAS Compute

**Status:** Experimental FP32 `mat_mul_nt` baseline; routed BLAS is Planned

**Updated:** 2026-09-09

OARS does not expose a vendor BLAS handle or a second execution runtime. Matrix
operations remain semantic Rust functions and lower through the same engine,
graph, event, memory and profiling system as every other domain. This document
records the physical linear-algebra scope; [OaBlasLt](oaBlasLt.md) owns plan and
selection policy.

## Current truth

The only admitted GEMM operation is:

```text
matrix::mat_mul_nt(left [M,K], right [N,K]) -> output [M,N]
```

It supports dense FP32, validates shape/overflow/zero work, and selects one
generated 64x64x16 tiled Slang kernel with 256 threads. Odd and zero shapes have
CPU-oracle coverage and captured replay has a fresh-process benchmark runner.

OARS currently has no general transpose/layout combinations, alpha/beta,
strided batch, grouped GEMM, split-K, Stream-K, prepacking, low-precision GEMM,
public BLAS ABI, or candidate router. Existing ML linear kernels and the two
exact-shape OaDna fusions are separate proven routes, not evidence of a general
BLAS implementation.

## Semantic families

The target admits complete vertical slices, not one giant public `gemm` with
unimplemented flags:

- matrix multiply orientations required by Matrix and autograd;
- vector/matrix and rank-k operations when a real consumer exists;
- strided batched and grouped problems;
- reductions and non-GEMM primitives under their semantic Matrix names;
- sparse/quantized operands as distinct value contracts;
- fused linear forms created by OaDna, not user-selected physical APIs.

The public surface describes math and layout. Internal problem descriptors may
look BLAS-like, but vendor enums and solution IDs never escape lowering.

## Physical families

| Family | Intended workload | Required distinctions |
|---|---|---|
| portable tiled | general dense baseline | tails, alignment, strides, accumulation |
| small-M/N | vectors, decode, narrow projections | occupancy versus launch/workgroup cost |
| batched/grouped | many independent problems | pointer/offset tables and heterogeneous shapes |
| split-K | large K with too few output tiles | reduction workspace and deterministic order |
| Stream-K-like | uneven tile waves across compute units | dynamic/static schedule, workspace and fairness |
| cooperative matrix | devices with a qualified matrix capability | exact component types, scopes and tile shapes |
| packed/quantized | encoded weight values | scales, blocks, zero points and accumulation |
| fused epilogue | matmul plus legal elementwise tail | liveness, auxiliary outputs and training data |

Every family retains a portable source route. “Portable” means semantically
available on the admitted Vulkan capability floor, not equally fast on every
GPU.

## Epilogues

OaDna may request an epilogue tree containing output scale, bias, residual,
activation, clamp, conversion and auxiliary output. OaBlasLt accepts only trees
advertised by a candidate. Arbitrary shader expression injection and public
vendor-algorithm selection are rejected.

Training requires exact saved values or a proven recomputation policy. A fused
forward cannot eliminate pre-activation data needed by its backward contract.

## Routing order

```text
semantic Matrix operation or OaDna partition
  -> complete MatmulProblem
  -> reject illegal candidates
  -> exact tuning-cache hit
  -> otherwise cold-start heuristic
  -> optional bounded correctness-gated tuning
  -> immutable MatmulPlan
  -> executable graph nodes
```

Route failure returns the portable source path only when that path is itself
admitted and the fallback is explicit in diagnostics. It never falls back to a
CPU implementation after GPU work fails.

## Cross-vendor policy

Optimization keys are hardware facts first: subgroup behavior, shared memory,
register/occupancy limits, matrix capabilities, memory topology and driver
identity. Device/vendor IDs may choose qualified artifact packs, but public
behavior stays vendor-neutral. Mobile schedules additionally account for
bandwidth, thermals, unified memory and smaller workgroups; discrete-GPU packs
may favor larger tiles and deeper staging.

The goal is not to reproduce thousands of vendor kernels immediately. A small
set of structurally different schedules, a strong filter/heuristic, bounded
tuning and workload-driven expansion should cover the first useful envelope.

## Evidence

A new BLAS route needs an independent host or trusted-library oracle, exact
layout/stride/alias tests, odd tails, zero work, overflow, alignment, alpha/beta
special cases where admitted, deterministic-mode tests, Vulkan validation and
fresh-process performance evidence. Route and fallback counters must identify
the exact candidate.

C++ OA performance tables are donor history only. They do not establish OARS
parity or regression baselines.

## Primary references

- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html)
- [hipBLASLt Stream-K](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.0/how-to/how-to-use-streamk.html)
