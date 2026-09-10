# OA Rust Compute Architecture

**Status:** Canonical target; the implemented one-device Vulkan path is Experimental

**Updated:** 2026-09-10

This document owns the compute subsystem boundary in OARS. Compute is not a
synonym for ML or DNN: GPU-lowered Matrix, Image, Vision, Audio, Video, Render,
Crypto, and future scientific work shares the same semantic compiler and
executable Vulkan runtime. [OaDna](oaDna.md) is the private cross-domain
optimization planner inside that compiler.

## Shipped, Experimental, and Planned

Nothing in the GPU compute surface is Shipped yet.

Experimental source currently proves:

- one selected Vulkan 1.3 device and one compute queue;
- schema-generated dense `f32` elementwise operations, `i32` Matrix add, and
  one FP32 `matrix::mat_mul_nt` 64x64x16 tiled kernel;
- an engine-owned bindless storage-buffer heap, exact timeline events, eager
  batching, RAW/WAR/WAW planning, isolated capture, and immutable replay;
- one cached simultaneous-use command buffer for unchanged untimed plans;
- whole-plan Vulkan timestamp measurement for explicitly timed replay;
- a private donor-backed semantic compatibility analyzer and two exact-shape
  inference replacements: QKV projection+bias and gate/up+SwiGLU;
- stable training resource frames, graph-resident replay RNG counters, and
  in-place AdamW state represented as semantic SSA aliases;
- generated physical-write contracts and prepared-dispatch binding checks for
  Core reductions/Scale and ML LayerNorm/RMSNorm/core losses, with unclassified
  candidates reported explicitly as `null`.

Planned work includes general GEMM routing, candidate selection and autotuning,
low precision and quantization, image/vision kernels, vision microfusions,
transfer/image/indirect/render/media graph nodes, multi-queue scheduling,
multi-device execution, qualified cross-vendor performance packs, explicit
physical write-domain coverage for every remaining candidate, byte-range and
alias proofs, and complete prepared-dispatch evidence.

The C++ OA implementation is donor evidence, not OARS shipping evidence. Its
kernel counts, timing results, and supported-device claims must not be copied
as Rust status.

## End-to-end path

```text
domain operation schema
  -> generated Rust operation and semantic contract
  -> SemanticGraph values, operations, effects, aliases, and metadata
  -> OaDna pattern partitioning and fusion eligibility
  -> capability/numeric/workspace filter
  -> kernel-family candidate selection or cached tuning result
  -> validate physical write domain, collision policy, bindings, and dispatch
  -> generic executable nodes and physical resources
  -> lifetime, transient-memory, queue, and synchronization planning
  -> reusable Vulkan command recording
  -> explicit timeline Event
```

Only the first, simpler form of this path exists today. Current domain
lowering chooses an exact generated kernel ID before recording a
`ComputeDispatch`; it does not yet enumerate or tune candidates.

## Ownership boundaries

| Layer | Owns | Never owns |
|---|---|---|
| Domain module | semantic validation, result metadata, operation identity | Vulkan handles or route selection |
| Operation schema | mechanically derivable Rust/Python surface, compatibility roles, validation and test fixtures | runtime measurements or device policy |
| Semantic graph | typed values, layouts, aliases, mutation, effects, autograd and control provenance | pipelines, descriptors, barriers |
| OaDna | legal partitions, cross-domain fusions, training/liveness constraints, candidate requests | device/queue lifetime or public vendor knobs |
| OaBlasLt/OaTile and other private providers | complete physical problem, candidates, plans, workspace, numeric policy | public Matrix/Image semantics |
| Executable graph | retained resources, concrete nodes, hazards, schedules and semantic provenance | public graph editing |
| `Engine` | devices, memory, descriptors, pipelines, scheduling, profiling, submission and retirement | duplicated domain APIs |
| `runtime/vk` | Ash handles, command encoding and Vulkan synchronization | semantic meaning |

`Engine` remains the sole local execution owner. Values retain opaque access to
the same engine services needed for storage and pending work; they do not own a
second runtime. Destructors release ownership only and never submit, wait,
read back, drain, or close a session.

## Semantic work versus executable work

The semantic graph must preserve everything needed to prove a transformation:

- operation and value identity;
- shape, dtype, stride, offset, layout, color, timing, channel and topology
  metadata as applicable;
- aliases, mutations, external values, virtual values and observable outputs;
- numeric/determinism policy, saved-for-backward data and side effects;
- control dependencies and semantic owners of fused work.

The executable graph owns physical decisions:

- kernel artifact and specialization identity;
- storage/image bindings, push data and direct or indirect dimensions;
- binding ranges, physical write partitions, and collision/reduction policy;
- workspace and transient allocation;
- queues, barriers, ownership transfer and completion;
- reusable recording and profiling instrumentation.

One semantic operation may lower to several executable nodes. One executable
microfusion may implement several semantic operations, but it retains all of
their identities. A fusion is invalid if it changes rounding, color semantics,
alias visibility, externally observed intermediates, readiness, training data,
or failure behavior.

## Cross-domain microfusion

OARS should first remove avoidable memory traffic between common adjacent
operations. These are not DNN-only patterns:

| Domain | Semantic region | Intended physical result |
|---|---|---|
| Image/Vision | color conversion -> resize -> scale/bias normalize -> layout conversion -> dtype conversion | one bounds-checked preprocessing kernel when metadata and sampling policy match |
| Video/Vision | NV12 or P010 planes -> color conversion -> resize -> normalize -> model layout | one or a small pipeline preserving range, matrix, chroma siting, visible extent and readiness |
| Matrix/ML | matmul -> bias -> activation or residual | a planned GEMM epilogue |
| Matrix/ML | Q/K/V projections sharing one input | grouped projection or shared-input schedule |
| Audio | sample-format conversion -> channel mix -> gain/normalize | one bandwidth-oriented kernel when clipping and rounding agree |
| Render/Image | swizzle -> transfer-function conversion -> pack/unpack | one format kernel when image layout transitions permit it |
| Crypto/Data | parse/convert -> batched transform -> compact | one device-resident region only when secret-data and bounds policy remain valid |

The portable source chain remains the correctness fallback. Recognition alone
does not authorize replacement; provider admission, device support, exact
metadata, liveness, aliasing, workspace, numeric mode and an independent oracle
must all pass.

## Cross-vendor performance strategy

One universal kernel cannot be optimal on an Adreno phone, an Intel integrated
GPU, a desktop NVIDIA GPU, and an AMD accelerator. Conversely, generating the
Cartesian product of every tile and knob is not maintainable. OARS uses a
bounded hierarchy:

1. A portable correctness kernel exists for every admitted operation.
2. A small set of reusable schedule families covers materially different
   workloads: direct tiled, small-M, split-K or Stream-K-like, grouped/batched,
   persistent, reduction, and bandwidth microfusion.
3. Capability predicates select by facts such as subgroup width, shared-memory
   budget, cooperative-matrix support, alignment, storage features and queue
   properties. Vendor/device IDs may select a qualified pack, but do not enter
   the public API.
4. A cold-start heuristic ranks a bounded top set using the full problem and a
   queried device profile.
5. Optional correctness-gated autotuning measures only those candidates and
   persists the winner under a versioned exact key.
6. Exact plan replay pins the selected artifact until invalidation; it never
   reruns selection on every operation call.
7. Workload telemetry informs which shapes deserve new schedules. It must not
   silently upload user data or turn a benchmark override into product policy.

Slang provides source reuse, interfaces, generics, specialization and SPIR-V
generation; it does not make one SPIR-V binary optimal everywhere. Candidate
artifacts remain explicit and validated. See [the kernel system](oaComputeKernel.md),
[OaTile](oaTile.md), and [OaBlasLt](oaBlasLt.md).

## Submission and overhead

Current eager operations accumulate in a private `ExecutionSession`. Blocking
observation or `Engine::checkpoint` joins pending nodes, records one primary
command buffer, submits it on the compute queue and signals the next timeline
value. Zero-work operations record no dispatch and do not flush unrelated work.
`try_read` never submits or waits.

Captured untimed plans reuse one simultaneously submittable command recording.
That is the principal portable Vulkan mechanism for amortizing host record
cost. Future work should prioritize, in evidence order:

1. eliminate repeated planning, pipeline lookup and allocation from steady
   replay;
2. pre-record useful command variants and use stable parameter/resource frames;
3. batch small nodes or replace them with legal microfusions;
4. add indirect dispatch when dimensions are device-produced;
5. evaluate `VK_EXT_device_generated_commands` only on queried devices where it
   reduces end-to-end cost;
6. evaluate vendor extensions such as `VK_AMDX_shader_enqueue` only as optional
   device packs, never as the cross-vendor baseline;
7. use persistent work queues only for measured workloads, with explicit
   occupancy, fairness, watchdog, memory-ordering and termination proofs.

The previously discussed roughly 0.03-0.04 ms Vulkan versus 0.01 ms CUDA submit
figures are anecdotal and are not accepted OARS evidence. The comparison must
use equivalent pre-recorded work, synchronization, clocks, validation state,
fresh processes and host boundaries before it enters a status document.

## Numeric contract

Keep separate:

1. semantic dtype and encoding;
2. physical input/output storage;
3. compute and accumulator precision;
4. reduction order and determinism policy;
5. selected kernel and specialization.

Current dense mixed-dtype promotion is not supported. Quantized weights are
future encoded semantic values with payload, scales, block policy and logical
shape; they are not byte matrices pretending to be a dense scalar dtype. See
[numeric stability](oaNumericStability.md) and
[quantization and dtypes](oaQuantizationAndDtypes.md).

## What OARS already does well

The strongest current foundation is architectural rather than benchmarked
speed: one engine owns the whole lifetime, semantic and executable graphs are
separate, eager and captured execution share lowering, exact events and resource
retention are explicit, graph reports preserve fused provenance, and training
state/RNG can remain inside replay. That is a credible base for optimizing an
entire mixed-domain pipeline rather than handing isolated tensor partitions to
separate libraries.

This does not establish parity with mature vendor stacks. The missing selector,
kernel breadth, low-precision routes, database, profiler feedback, packaging,
device qualification and failure recovery are listed in the
[vendor compute audit](oaVendorComputeAudit.md).

NVIDIA's first CUDA Rust SIMT and tile projects provide useful independent
evidence for checked launch tokens and disjoint writable partitions, but they
do not change OARS's Vulkan/Slang boundary. The dated comparison, adopted
principles, and rejected integration paths are recorded in the
[CUDA Rust assessment](oaCudaRust.md).

## Acceptance gates

A compute checkpoint requires:

- a schema-owned semantic contract and independent oracle;
- zero, odd/tail, boundary, invalid, alias, reuse and poison cases where
  meaningful;
- deterministic generation and exact reflection/SPIR-V validation;
- separate core, synchronization and applicable GPU-assisted validation;
- exact route, candidate, fallback, device, driver, compiler, build and dirty
  state in evidence;
- at least seven correctness-gated fresh-process measurements with median and
  spread for a performance claim;
- no claim that a generated row, recognized pattern, compiled pipeline or
  passing host test is by itself Shipped.

## Primary references

- [Vulkan specification, device-generated commands](https://registry.khronos.org/vulkan/specs/latest/pdf/vkspec.pdf)
- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [Slang reflection](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/09-reflection.html)
