# Vendor Compute Audit and OARS Gap Analysis

**Status:** Research; informs plans but makes no OARS capability claim

**Updated:** 2026-09-10

This audit compares OARS with public CUDA/cuDNN/cuBLAS, ROCm/hipDNN/MIOpen/
hipBLASLt and oneDNN Graph designs. It separates mechanisms from kernel count:
even with perfect portable kernels, OARS still needs selection, persistence,
packaging, observability and device qualification.

## Evidence and provenance

The ROCm code audit used a local `rocm-libraries-develop` super-repository
snapshot. The snapshot was not itself a Git worktree, so an immutable commit
cannot be recorded. Findings below name repository-relative paths and the
2026-09-09 audit date; re-audit against a commit before using them as release
evidence.

Public comparisons cite current primary vendor/project documentation. Product
support changes over time; exact installed versions and hardware capabilities
remain the authority for tests.

## Architecture comparison

| Concern | Mature vendor/project approach | OARS today | OARS target |
|---|---|---|---|
| semantic scope | cuDNN/hipDNN/oneDNN Graph compile tensor/DNN partitions; BLAS libraries own matmul problems | one multi-domain semantic graph exists, but executable operations are mostly Matrix/ML | one graph preserves Matrix/Image/Video/Audio/Render/Crypto metadata and forms cross-domain partitions |
| engine/provider model | graph -> applicable engines/configs/knobs -> immutable plan | closed private DNN analyzer; exact source routes and two replacements | closed Rust plan/provider core first; optional versioned plugin ABI only if justified |
| kernel selection | large curated libraries, applicability filters, heuristics and offline/online tuning | exact generated kernel ID | legal candidate filter + cold heuristic + bounded tuning + exact plan pinning |
| persistence | vendor kernel/plan/heuristic caches and pre-tuned databases | Vulkan driver pipeline behavior plus in-process command cache; no OARS tuning database | versioned artifact, pipeline and tuning caches with device/driver/compiler keys |
| launch amortization | CUDA/HIP graphs, reusable executable plans, vendor handles/caches | eager batching and cached Vulkan command replay | retain portable replay; add variants, indirect/device-generated work only when measured |
| memory/workspace | explicit workspace queries/preferences and internal virtual tensors | buffer pool and graph transient arenas; DNN workspace policy is effectively zero | joint virtual-value, transient, stable-slot and provider-workspace planning |
| low precision | extensive FP16/BF16/TF32/FP8/integer/packed kernels | FP32 baseline; limited I32/U32 semantics | explicit representation and numeric policies plus qualified device packs |
| observability | logging, engine/solution IDs, find/tuning tools, vendor profilers | deterministic semantic/executable reports and fallback counters are a strong seed | exact candidate/cost/cache/tune reason and per-phase timing |
| qualification | vendor-specific matrices, packaged databases/kernels | narrow local hardware evidence | capability/device-class CI with unsupported paths fail-closed |

## ROCm findings

### hipDNN

hipDNN uses a graph API with dynamically loaded engine plugins. Plugins report
applicable engine IDs; engine configuration and knobs produce immutable plans;
execution receives caller-owned tensor and workspace pointers.

The local source is more advanced than the older “limited graph” summary in
parts of its design documentation:

- `projects/hipdnn/backend/src/descriptors/EngineHeuristicDescriptor.cpp`
  obtains applicable engines and applies an ordered policy list;
- policy order is environment override, descriptor override, then built-in
  Config followed by StaticOrdering;
- `projects/hipdnn/backend/src/heuristics/SelectionHeuristic.cpp` treats plugin
  policies as untrusted and validates returned IDs are a unique subset of input
  candidates;
- `AutotuneRankingStore`, `AutotuneCacheKey`, frontend timed-run helpers and
  cross-process tests show that ranking/persistence is a first-class subsystem;
- plugin SDK `IEngine`/plan-builder/plan-style responsibilities separate
  applicability, workspace, knobs, immutable execution and runtime bindings.

Lesson for OARS: implement explicit candidate/applicability/result types and
validate selector output even if all providers are initially in-process Rust.
Do not copy the dynamic plugin complexity before a real external-provider use
case exists.

### hipDNN providers

The audited hipBLASLt provider accepts only specific matmul layouts/dtypes and
epilogues. It builds a complete descriptor, requests heuristic algorithms under
a workspace preference, caches the selected algorithm/workspace, then executes
with runtime pointers. Its row-major adaptation is internal.

The MIOpen provider validates convolution solutions/workspace during planning
but may delay benchmarking-dependent algorithm choice until buffers exist at
first execution, guarded and cached. This reveals an important plan boundary:
some tuning requires real addresses/data and must be an explicit preparation
phase, never accidental work on steady replay.

### MIOpen

MIOpen has solver applicability, FindDb, system/user performance databases,
immediate/find modes and a versioned compiled-kernel cache. Its local docs under
`projects/miopen/docs/conceptual/` describe user entries taking precedence and
version changes invalidating old user databases.

Lesson for OARS: selection metadata, tuned parameters and compiled artifacts are
different caches with different invalidation. Cold-process behavior is a
product feature and must be benchmarked separately from steady kernel time.

### hipBLASLt and Tensile

hipBLASLt combines a full matmul descriptor, heuristic solution search, tuning
utilities and large architecture-specific Tensile/assembly libraries. Offline
solution indices are not stable across architecture or library release.
Stream-K families redistribute tile work across compute units and trade
workspace/coordination for more consistent utilization.

Lesson for OARS: never persist a bare ordinal; store exact artifact/problem/
device identity. Add structurally different schedules before multiplying tile
sizes. A small-M and Stream-K-like family can cover gaps that hundreds of
ordinary tiles cannot.

## NVIDIA findings

cuDNN Graph separates operation graphs, engines, engine configurations/knobs,
heuristic modes and execution plans. Autotuning selects among supported plans.
cuBLASLt similarly exposes problem/preferences and heuristic algorithms; its
heuristic query can itself be material enough that documentation recommends
querying once and reusing results.

CUDA Graphs instantiate reusable executable graphs to reduce repeated CPU
launch work. CUDA also supports restricted device graph launch and conditional
nodes. Therefore “OARS has GPU graphs and CUDA does not” is false.

NVIDIA's 2026 CUDA Rust announcement adds two research frontends rather than a
new portable runtime: `cuda-oxide` lowers explicit Rust SIMT kernels to PTX,
while `cutile-rs` lowers logical Rust tile programs through CUDA Tile IR. Their
most relevant mechanisms are checked launch contracts, per-thread or per-tile
exclusive mutable partitions, ownership retained across lazy execution, and
compile-time specialization without public runtime policy arguments. Both
projects are early-stage and CUDA/NVIDIA-specific.

OARS adopts those mechanisms as design evidence, not dependencies. Its existing
private `PreparedDispatch` is the correct Vulkan seam to strengthen; future
candidate metadata should make physical write partitions and collision policy
machine-readable. OaTile remains a Slang/SPIR-V construction system, and CUDA
Rust may serve only as an operation-specific experimental oracle unless a
separately approved backend proves enough value to justify duplicated runtime,
artifact, synchronization, packaging, and qualification work. See the
[detailed CUDA Rust assessment](oaCudaRust.md).

OARS's opportunity is broader semantic integration: vendor DNN and BLAS
libraries generally optimize their partition, while OARS can jointly eliminate
intermediates and schedule a color-convert/resize/normalize/ML/render pipeline.
CUDA's graph control, update maturity and kernel breadth are presently ahead.

## Intel/oneDNN findings

oneDNN Graph defines explicit fusion patterns for matmul, attention, gated MLP,
convolution, interpolation, normalization, reductions, reorder/typecast and
quantization. This supports the same conclusion as the OA donor: conversion and
normalization are optimization-worthy graph semantics, not ML-only helpers.

OARS should use one semantic pattern vocabulary across devices rather than an
Intel-specific public API. Intel qualification still needs actual Vulkan device
capability and profiler evidence; oneDNN CPU/GPU support does not prove the
equivalent OARS route.

## Vulkan gap and opportunity

Portable Vulkan already supplies reusable command buffers, descriptor-based
resource binding, timeline semaphores, synchronization2, timestamps and
indirect dispatch. These are enough for a strong host-amortized baseline.

Optional mechanisms can close narrower gaps:

- `VK_EXT_device_generated_commands` permits device-authored supported command
  streams with layout/preprocess/address/synchronization requirements;
- `VK_AMDX_shader_enqueue` models execution-graph dispatch on supporting AMD
  devices only;
- persistent scheduler kernels can implement bounded work queues without a
  special extension, at occupancy/fairness/watchdog/debugging cost.

The correct order is optimize captured replay and fusion first, then measure
whether host submission or dynamic device control remains material. Extensions
cannot compensate for missing kernels, tuning or plan persistence.

## What OARS is missing besides kernels

Prioritized dependency order:

1. **Complete problem/candidate model.** General `MatmulProblem`, cross-domain
   region descriptors and generated provider admission.
2. **Capability fingerprint.** Exact device/driver/features/limits identity used
   by legality, cache and reports.
3. **Selector.** Deterministic legality filter, ranked candidates, explicit
   failure taxonomy and exact route telemetry.
4. **Persistent caches.** Separate OARS tuning winners, compiled pipeline data
   and prepacked constants with strict invalidation.
5. **Preparation lifecycle.** Explicit compile/tune/prepack transaction before
   steady replay; no surprise work inside execute.
6. **Workspace/transient integration.** Provider workspace joins global
   liveness, alias and retirement planning.
7. **Cross-domain schema coverage.** Image/vision conversion, resize, normalize,
   layout and dtype operations plus generated pattern fixtures.
8. **Device-class packs.** Portable, mobile, integrated and discrete/
   accelerator candidates qualified on named hardware.
9. **Packaging.** Artifact/database versioning, cache location/limits,
   corruption recovery and reproducible offline pack creation.
10. **Profiling loop.** Phase timing, exact fallback/candidate counters,
    roofline/counter investigations and workload-driven variant budgets.
11. **Graph breadth.** Transfer, image, indirect, external-sync, render and video
    nodes with exact Vulkan synchronization.
12. **CI qualification.** Independent oracles, validation modes and thermal/
    performance gates across the supported device matrix.

Items 1-6 should precede a large kernel-writing campaign. Otherwise new
kernels accumulate without one reliable way to admit, choose, cache or explain
them.

## What OARS may do better

Already demonstrated foundations:

- one owner for device, memory, graph, submission and retirement;
- backend-neutral semantic evidence separated from Vulkan work;
- exact event/resource lifetime instead of destructor synchronization;
- deterministic handle-free semantic/executable/training reports;
- fused nodes retain many-to-one semantic provenance;
- replay-safe optimizer/RNG state can remain on the GPU;
- Rust types and ownership constrain host-side plan/resource mistakes.

Potential advantage, not yet demonstrated:

- optimize and schedule mixed Image/Video/Vision/Matrix/ML/Render pipelines in
  one compiler with one memory lifetime plan;
- use Slang to share audited algorithms across portable and device packs;
- keep public APIs vendor-neutral while still using measured per-device routes.

These become advantages only after cross-domain operations, providers and
measurements ship.

## Primary references

- [hipDNN documentation](https://rocm.docs.amd.com/projects/hipdnn/en/latest/)
- [hipDNN architecture](https://rocm.docs.amd.com/projects/hipdnn/en/latest/conceptual/architecture.html)
- [hipDNN plugin development](https://rocm.docs.amd.com/projects/hipdnn/en/latest/how-to/develop-plugins.html)
- [MIOpen documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [MIOpen find modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html)
- [hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html)
- [cuDNN Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [NVIDIA CUDA Rust announcement](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
- [NVlabs cuda-oxide](https://github.com/NVlabs/cuda-oxide)
- [NVlabs cutile-rs](https://github.com/NVlabs/cutile-rs)
- [oneDNN Graph fusion patterns](https://uxlfoundation.github.io/oneDNN/graph_fusion_patterns.html)
- [Vulkan device-generated commands](https://registry.khronos.org/vulkan/specs/latest/pdf/vkspec.pdf)
