# OA Rust Compute Architecture

**Status:** Canonical architecture; one-device elementwise execution is Experimental

**Updated:** 2026-09-07

This document owns the Rust compute subsystem contract. The C++ compute system
provides behavioral evidence and hard-won implementation constraints, but its
classes and source layout are not a porting template.

## Current executable path

The implemented path is:

```text
matrix operation schema
  -> generated Rust domain function
  -> shared semantic validation and direct lowering
  -> runtime::ComputeDispatch
  -> private ExecutionSession::record
  -> owned ExecutableGraph snapshots
  -> blocking observation or Engine::checkpoint
  -> one joined hazard-planned graph
  -> runtime/vk command recording
  -> one Vulkan compute queue
  -> timeline Event retained by the output Matrix
  -> read::<T> waits at host observation
```

The current Experimental lowerer records each non-empty operation into the
engine-owned private execution session. Ordinary APIs return values and never
require callers to invoke `submit` or `wait`. Blocking host observation submits
the pending batch and waits for its exact event; `Engine::checkpoint` is the
explicit submission boundary.

## Ownership boundaries

| Layer | Owns now | Must not own |
|---|---|---|
| `matrix` and other domain modules | semantic validation, output shape/dtype, operation identity, lowering | Vulkan handles, queues, or per-operation engine methods |
| operation schema and generator | mechanical API, contracts, kernel identity, shader source, tests | runtime policy or measured route choice |
| `runtime::ComputeDispatch` | backend-neutral executable bindings, access declarations, push values, workgroups | mathematical semantics or Vulkan handles |
| private `ExecutableGraph` | owned concrete buffer bindings, copied push values, resource hazards, ordered nodes | public graph editing or semantic inference |
| private `ExecutionSession` | pending eager graphs, written-storage readiness, batch transfer at submission | device, queue, allocator, or public lifecycle ceremony |
| `Engine` and its private handle | device services, submission epochs, retirement, future scheduling/profiling | duplicated domain operations |
| `runtime/vk` | Ash handles, descriptors, pipelines, command recording, queue submission, timeline synchronization | public matrix/image/audio semantics |

`Engine` is the only local execution owner. A value retains the same private
engine services needed to keep storage and pending work alive; it does not own
or create another runtime.

The Vulkan device currently owns one bindless storage-buffer descriptor heap
shared by all generated compute pipelines. Buffers own descriptor indices;
recorded command buffers retain the buffers they reference until their exact
timeline epoch retires.

## Semantic and executable work

The semantic layer owns:

- operation identity and typed inputs/outputs;
- shapes, dtypes, layouts, aliases, mutation, and effects;
- differentiation and numeric policy when admitted;
- errors visible to the caller.

The executable layer owns:

- selected kernel identity and artifact;
- concrete storage bindings and declared access;
- push-constant bytes and dispatch dimensions;
- queue, synchronization, resource lifetime, and completion.

One semantic operation may eventually lower to several executable nodes, and a
fused node may preserve several semantic owners. Therefore `Engine` accepts a
generic dispatch description; it never grows `submit_matrix_add`,
`submit_image_resize`, or equivalent domain-specific forwarding methods.

## Submission, synchronization, and observation

Every non-empty current compute operation records an owned graph snapshot into
the private execution session. At blocking observation or an explicit
checkpoint, the session joins all pending nodes, records one command buffer,
submits it to the same compute queue, and signals the next timeline value.
Within that graph the recorder tracks each retained buffer and inserts a Vulkan
buffer barrier for write-to-read, read-to-write, and write-to-write conflicts;
read-to-read needs no barrier. State is retained across unrelated nodes so a
dependency is not lost merely because another resource was used between its
producer and consumer.

All current accesses are storage-buffer accesses on one compute queue. Their
graph barriers therefore use `COMPUTE_SHADER` for both stages,
`SHADER_STORAGE_READ` and/or `SHADER_STORAGE_WRITE` for the declared accesses,
the complete logical buffer range, and no queue-family transfer. A submission
still waits on the preceding timeline value at `ALL_COMMANDS`, which owns the
inter-graph visibility edge in the one-queue prototype. Transfers, images,
indirect dispatch, queue changes, and subranges require distinct executable
node and synchronization contracts; the compute barrier is not generalized to
them.

Zero-element operations record no dispatch and their empty output is
immediately host-ready. They do not force an unrelated eager batch to submit.

`Matrix::read::<T>` and its `read_f32` convenience wrapper are blocking
host-observation boundaries. They validate the requested Rust element type,
then wait for the output's producer before invalidating and reading mapped
storage. `Matrix::try_read::<T>` and `try_read_f32` return `NotReady` instead
of waiting. `Drop` never
submits, waits, drains, maps, or reads back.

Recorded output storage uses a shared private readiness state. It moves from
recorded, to submitted with an exact event, to observable after completion. A
submission or recording failure makes production failure persistent for later
observation. `try_read` returns `NotReady` for both recorded and incomplete
submitted work and never flushes the session.

Isolated capture can transfer the pending graph and written-storage bindings
into a public structurally immutable `ExecutionPlan`. Capture never submits or
waits and plan replay returns an exact event. Untimed replay caches one
simultaneously submittable primary command; read-only Matrix inputs have stable
captured identities and may be rebound under exact shape, dtype, ownership, and
no-alias validation. Rebinding invalidates the recording once. Mutable outputs
and general semantic value identity remain incomplete.

## Numeric contract

Keep these facts separate:

1. semantic dtype (`DType::F32`, `DType::I32`, future `DType::Bf16`);
2. physical storage width and encoding (`f32`, `bf16`, packed Q4/Q8);
3. compute and accumulator precision;
4. selected physical kernel or specialization.

The current matrix slice admits dense `f32` and `i32` storage through one
runtime-typed `Matrix`. All generated elementwise operations admit `f32`, while
`matrix::add` also admits exact same-dtype `i32`; mixed dense dtypes fail without
implicit promotion. Signed `i32` addition wraps modulo 2^32, including at the
minimum and maximum boundaries. The sealed Rust `Element` mapping owns checked
host upload and readback and does not make device storage generic. Packed
quantization is a separate semantic representation because its payload, scale
planes, block size, and logical stride are not one dense scalar per element.
Low-precision storage, reductions, broader integer arithmetic, quantization,
and numeric modes remain Planned. Kernel metadata uses normalized lowercase
dtype tokens; see
[the kernel system](oaComputeKernel.md#dtype-vocabulary).

## Current versus target behavior

| Concern | Current Experimental behavior | Target dependency |
|---|---|---|
| Eager execution | engine-owned batching at observation/checkpoint | scheduling diagnostics and broader executable nodes |
| Reuse | immutable captured graph re-recorded per submission | compiled command caching and stable slots |
| Kernel selection | exact schema-generated kernel ID | capability- and measurement-filtered candidates |
| Dependencies | per-buffer graph hazards plus a serialized inter-submit timeline chain | multi-queue executable resource-hazard graph |
| Memory | checked VMA-backed host-visible storage | upload/readback rings and transient planning |
| Profiling | explicit whole-plan device duration on timed replay | calibrated clocks, phase/node timestamps, and statistics |
| Devices | one selected physical device | explicit local transfer before automated placement |

## Porting from C++ OA

Preserve:

- the single engine owner;
- semantic versus executable graph separation;
- generic dispatch records and exact events;
- shared bindless descriptors and resource retention;
- schema-owned operations and generated kernel identity;
- explicit host observation and fail-closed capability checks.

Do not mechanically reproduce `ExecutionSession`, `ExecutableGraph`, `Stream`,
pipeline registry classes, or C++ access facades. Add each layer only when its
Rust ownership, failure, and verification contract is required by the roadmap.

## Acceptance gates

A compute checkpoint requires:

- public contract tests plus an independent oracle or conformance reference;
- zero, odd, minimal, boundary, invalid, alias, reuse, and poison cases where
  meaningful;
- deterministic schema generation and an empty drift check;
- reflected descriptor, push-constant, entry-point, attribute, and workgroup
  validation;
- `spirv-val` for every configured artifact;
- core, synchronization, and applicable GPU-assisted validation run separately;
- explicit device, driver, loader/registry, tool, build, and dirty-state evidence;
- performance claims only through the protocol in
  [OA performance evidence](../performance/oaPerformance.md).

No module, generated function, compiled shader, or passing host-only test is by
itself a Shipped capability.
