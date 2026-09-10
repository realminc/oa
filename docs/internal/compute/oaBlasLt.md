# OaBlasLt — Rust Matmul Planning, Tuning and Execution

**Status:** Planned architecture above an Experimental single-kernel baseline

**Updated:** 2026-09-09

OaBlasLt is a private planning service owned by `Engine`. It converts one fully
specified matrix multiplication problem into an immutable executable plan. It
is not a public vendor-compatibility API, a second engine, or a place for ML
semantics.

## Problem descriptor

A `MatmulProblem` must include every behavior-affecting fact:

```text
M, N, K and batch/group dimensions
operand/output shapes, byte offsets, strides and transpose/layout operations
input, output, compute and accumulator types
alpha/beta and output initialization semantics
aliasing and overlap constraints
epilogue tree and auxiliary outputs
training/recompute and observable-intermediate requirements
determinism, rounding, overflow and NaN policy
workspace limit and allocation class
device/placement and capture/replay constraints
```

Checked Rust constructors validate arithmetic, ranges and mutually dependent
fields once. Newtypes distinguish bytes, elements, dimensions, alignments and
stable IDs where confusion is dangerous. An immutable descriptor is hashable
and safe to share; Rust improves host-side correctness and lifetime clarity but
does not make the GPU kernel faster by itself.

## Preferences

Private preferences include maximum workspace, required determinism, tuning
budget, cold-start latency class and whether weight prepacking is allowed.
Tests and benchmarks may request an exact candidate. Applications do not select
AMD/NVIDIA/Intel routes through Matrix APIs.

## Plan contract

An immutable `MatmulPlan` owns or retains:

- normalized problem and exact device/capability fingerprint;
- selected candidate and SPIR-V/pipeline identity;
- workspace formula/allocation requirements;
- packed-weight identity and validity rules if applicable;
- direct/indirect dispatch sequence and synchronization requirements;
- numeric/determinism contract;
- semantic provenance and explicit fallback reason;
- selector/tuning provenance and cache version.

The plan is published only after all candidates, resources and command
preflight succeed. Repeated execution performs no graph analysis or heuristic
query. Inputs can change only through validated stable slots; a changed problem
invalidates the plan.

## Candidate selection

1. Build the complete problem.
2. Enumerate schema-generated candidates for its semantic family.
3. Reject by dtype/layout/stride/alignment/shape/tail/alias/capability,
   determinism and workspace.
4. Consult an exact versioned tuning-cache key.
5. Rank a bounded top-K using a deterministic cold-start model.
6. If permitted, run each candidate through correctness before timing.
7. Measure equivalent device work, record median/spread, and select only a
   statistically meaningful winner.
8. Pin the exact winner in the immutable plan.

No tuning runs on every call. No invalid candidate is timed “to see what
happens.” Heuristic failure and no-applicable-candidate are separate diagnostic
states.

## Cache identity

The cache key includes:

- normalized full problem and numeric policy;
- device UUID/PCI identity as available, driver and Vulkan conformance/version;
- enabled capability profile and relevant queried limits;
- OARS schema/generator/kernel-pack version;
- Slang/compiler flags, exact SPIR-V module hash and specialization values;
- workspace and determinism preferences;
- tuning protocol/oracle version.

The value records candidate identity, workspace, correctness result, raw timing
artifact reference and selection reason. A vendor-library solution index is
not portable across library releases or architectures; OARS applies the same
principle to its own cache.

## Cold-start model

The first model should be transparent and deterministic: classify shape
regime, estimate tile waves, memory traffic, reduction/workspace cost and
capability fit, then rank a small candidate set. It should be testable against
checked-in fixtures and never silently mutate from field telemetry.

Later learned ranking is acceptable only if its feature schema, model artifact,
version, fallback and reproducibility are explicit. It still cannot override
legality or correctness.

## Weight prepacking

Prepacking is legal when the semantic weight is immutable for the plan. Cache
identity includes source storage/version, dtype, shape, packing candidate,
device and compiler/kernel version. Mutation invalidates the packed value.
Packing time and memory are reported separately and amortization is measured on
the actual reuse count.

## Advanced schedules

- split-K requires a reduction strategy, workspace and determinism contract;
- Stream-K-like scheduling balances tile work across compute units and may use
  static or device-coordinated dynamic assignment;
- persistent kernels need explicit termination, fairness, occupancy and
  watchdog constraints;
- grouped problems need checked device-resident descriptors and per-problem
  bounds;
- cooperative-matrix routes require exact queried SPIR-V/Vulkan component and
  scope support;
- fused epilogues must preserve auxiliary outputs, saved-for-backward values and
  observable intermediates.

## Implementation order

1. Generalize current `mat_mul_nt` into `MatmulProblem` without changing its
   selected kernel.
2. Add portable candidates for small-M and a second tile; implement legality
   filtering and exact route diagnostics.
3. Add a deterministic cold heuristic and in-memory plan cache.
4. Add persistent versioned cache and correctness-gated bounded tuning.
5. Add epilogue trees and OaDna integration.
6. Add batched/grouped and split-K/Stream-K-like families.
7. Add low-precision/cooperative-matrix device packs after dtype qualification.

Each phase is a complete correctness checkpoint. The router must exist before
large kernel-pack expansion so new kernels have one canonical admission path.

## Lessons from vendor stacks

cuDNN and hipDNN separate graph recognition, engine configuration and immutable
execution plan. cuBLASLt and hipBLASLt expose problem descriptors, preferences,
heuristic results and reusable algorithm choices. MIOpen combines solver
applicability with find/performance databases and kernel caches. OARS should
adopt these separations while keeping the selector private and integrated with
its cross-domain graph.

## Primary references

- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [hipBLASLt tuning utility](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.14.0/how-to/how-to-use-hipblaslt-tuning-utility.html)
- [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.1.1/how-to/how-to-use-hipblaslt-offline-tuning.html)
- [MIOpen documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
