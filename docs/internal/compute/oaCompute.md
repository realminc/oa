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
  -> EngineHandle::submit
  -> runtime/vk command recording
  -> one Vulkan compute queue
  -> timeline Event retained by the output Matrix
  -> read::<T> waits at host observation
```

The current Experimental lowerer submits each operation immediately. Eager
submission is an implementation bridge, not the long-term public contract.
Ordinary APIs return values and never require callers to invoke `submit` or
`wait`. The target engine-owned execution session may batch eager work without
changing that surface.

## Ownership boundaries

| Layer | Owns now | Must not own |
|---|---|---|
| `matrix` and other domain modules | semantic validation, output shape/dtype, operation identity, lowering | Vulkan handles, queues, or per-operation engine methods |
| operation schema and generator | mechanical API, contracts, kernel identity, shader source, tests | runtime policy or measured route choice |
| `runtime::ComputeDispatch` | backend-neutral executable bindings, access declarations, push values, workgroups | mathematical semantics or Vulkan handles |
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

Every non-empty current compute operation records one command buffer, submits
to the same compute queue, and signals the next timeline value. A submission
waits on the preceding value at `ALL_COMMANDS`, which serializes the current
one-queue prototype and provides the visibility edge between chained matrix
operations. This is correct for the admitted path but is not the target
fine-grained graph scheduler.

Zero-element operations record no dispatch. They still take an engine
checkpoint so their output owns an exact readiness state.

`Matrix::read::<T>` and its `read_f32` convenience wrapper are blocking
host-observation boundaries. They validate the requested Rust element type,
then wait for the output's producer before invalidating and reading mapped
storage. `Matrix::try_read::<T>` and `try_read_f32` return `NotReady` instead
of waiting. `Drop` never
submits, waits, drains, maps, or reads back.

The future executable graph must derive RAW, WAR, and WAW dependencies from
declared resource access. Any cross-queue path additionally owns stage/access
masks, queue-family transfer, and completion edges; source order is not proof.

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
| Eager execution | direct asynchronous submission per operation | engine-owned batching session |
| Kernel selection | exact schema-generated kernel ID | capability- and measurement-filtered candidates |
| Dependencies | serialized timeline chain on one compute queue | executable resource-hazard graph |
| Memory | checked VMA-backed host-visible storage | upload/readback rings and transient planning |
| Profiling | no admitted timestamp/statistics API | calibrated timestamps and phase counters |
| Reuse | pipelines persist; commands retire after one use | immutable compiled plans and replay |
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
