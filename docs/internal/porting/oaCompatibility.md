# OA C++ to Rust Compatibility Ledger

**Status:** Planned

**Updated:** 2026-09-07

**Rust architecture:** [OA Rust Architecture](../architecture/oaArchitecture.md)

This ledger records which OA concepts survive the Rust implementation. It is a
semantic migration map, not a file-by-file port checklist.

## Decision classes

- **Preserve:** retain the behavior and invariant.
- **Redesign:** retain the requirement through a Rust-native contract.
- **Defer:** keep the dependency and acceptance gate without publishing an API.
- **Reject:** do not reproduce the old abstraction.
- **Historical:** leave evidence in the C++ repository; do not copy it.

## Foundational architecture

| OA concept | Decision | Rust direction |
|---|---|---|
| `oa::Engine` as sole execution owner | Preserve | `Engine` owns Vulkan, devices, memory, queues, kernels, scheduling, and profiling. |
| Runtime wrapper/default engine | Redesign | Rust constructs `Engine` explicitly and values retain its opaque internal lifetime. Python may own one implicit process engine for the established `oa.FnMatrix` facade; neither surface introduces a second runtime type. |
| `core/` source boundary | Preserve and expose | Public `core` owns foundational values, checked metadata, errors/results, and backend-neutral primitives; `lib.rs` explicitly re-exports the common root vocabulary. |
| Public `oa.core` namespace | Adopt | Rust exposes `oa::core`; Python mirrors it as `oa.core`. Explicit root re-exports remain identity aliases, not duplicate implementations. |
| Value / operation / session classification | Preserve | Rust structs/enums, module functions, and explicit stateful session structs. |
| Shared storage with semantic value types | Preserve | Composition and checked zero-copy views; no inheritance requirement. |
| C++ Pimpl and access bridges | Redesign | Private fields, crate visibility, typed ownership, and narrow safe runtime interfaces. |
| Engine inheritance and secondary contexts | Reject | Composition and explicit borrowing only. |

## Public API

| OA concept | Decision | Rust direction |
|---|---|---|
| `oa::FnMatrix`, `oa::FnImage`, other `Fn*` namespaces | Redesign by language | Rust uses `matrix::`, `image::`, `vision::`, and other lowercase operation modules. Python retains admitted `oa.Fn*` compatibility facades generated from the same schema. |
| PascalCase value and session types | Preserve | Normal Rust type naming. |
| camelCase methods and parameters | Reject | Rust `snake_case`. |
| C++ `in`/`out`/`inOut` parameter prefixes | Reject | Borrowing, mutable borrowing, and ownership express access mode. |
| Root semantic vocabulary | Preserve | Curated explicit re-exports from `lib.rs`. |
| One `Matrix` for dense scalar dtypes | Preserve with Rust boundary typing | `Matrix` retains runtime `DType`; sealed `Element` implementations make host upload/readback generic over admitted Rust primitives without making device storage generic. |
| `QuantMatrix` separate from dense `Matrix` | Preserve | Packed payload, scale planes, block policy, and logical layout remain one distinct semantic encoded value; Q4/Q8 are not `DType` variants. |
| Convenience methods | Redesign | Delegate to the same schema-owned operation; never create a second path. |
| Operator overloads | Defer | Admit only after graph and error behavior are proven without panics. |
| Public declarations backed by TODOs | Reject | Keep planned APIs in documentation until implementation and contract tests exist. |
| C++/Python identical spelling | Redesign | Preserve semantic identity while using idiomatic Rust and Python spelling. |

## Ownership and errors

| OA concept | Decision | Rust direction |
|---|---|---|
| Eager `Fn*` calls return values | Preserve | Rust domain operations and Python `Fn*` calls return semantic values without mandatory submit/wait calls. |
| Explicit `submit` and `Event` completion | Preserve as advanced control | Capture, overlap, profiling, multi-device, and distributed paths return engine-associated events. Ordinary host observation flushes and waits for the exact producer; `try_*` observation does not wait. |
| Destructor never submits or waits | Preserve | `Drop` releases state only and cannot report completion failure. |
| OA `Status` / `Result<T>` | Redesign | `core::Error`, `core::ErrorKind`, and `core::Result<T>` preserve contextual sources without exposing backend error types; common paths are re-exported at the crate root. |
| OA logging foundation | Redesign | Runtime exposes Rust-native levels, compact extensible components, options, and namespaced macros. Each `Engine` owns its sink; weak thread-local selection and stderr fallback avoid a process-global logger. |
| Universal host/device timer | Redesign progressively | Explicit timed plan submission returns device duration through its exact `Event`; host intervals remain separately named `Instant` regions until a unified statistics contract is justified. |
| `UniquePtr` / `SharedPtr` translation | Redesign | Borrow or own directly; use `Box` or `Arc` only for proved lifetime needs. |
| Raw owning Vulkan handles in semantic values | Reject | Opaque, typed, lifetime-safe resource ownership. |
| Hidden CPU fallback | Reject | Unsupported GPU work fails explicitly. |

## Graphs and generation

| OA concept | Decision | Rust direction |
|---|---|---|
| Semantic graph | Preserve | Domain values, operations, attributes, effects, aliases, control, and autograd provenance. |
| Executable Vulkan graph | Preserve progressively | The private Rust graph snapshots concrete compute dispatches and plans buffer RAW/WAR/WAW barriers. Immutable capture/replay is Experimental; semantic identity, command caching, and non-compute nodes remain deferred. |
| Private execution session | Preserve with Rust ownership | One engine-owned session now batches eager graph snapshots and tracks output readiness; the C++ access facades and context hierarchy are not reproduced. |
| Public execution plan | Preserve with Rust ownership | `Engine::capture` returns an immutable engine-associated plan plus the closure result; `Engine::submit(&plan)` replays asynchronously and returns an exact event. |
| One operation schema | Preserve | Generate Rust, Python, validation, autograd, registry, docs, and tests from one record. |
| Existing C++ generator implementation | Redesign | Reuse schema knowledge where sound; generators may be rewritten for deterministic Rust output. |
| Handwritten kernel and operation registries | Reject | Extend the schema or reflected shader metadata instead. |
| Generated output written during normal builds | Reject | Build scripts use `OUT_DIR`; checked-in output changes only through explicit generation. |

## Vulkan and shaders

| OA concept | Decision | Rust direction |
|---|---|---|
| Vulkan as the execution backend | Preserve | `ash` remains the thin binding layer. |
| Volk loader | Reject | `ash::Entry`, `ash::Instance`, and `ash::Device` own dispatch tables. |
| VMA general allocator | Preserve initially | `vk-mem` stays private behind OA memory contracts and may be replaced by measured path. |
| Upload/readback rings and transient planning | Defer | Add after the general buffer path proves ownership and completion. |
| Slang source and SPIR-V | Preserve | First-class `src/slang` source with one `main` entry per compiled module, validated reflection, and embedded artifacts. |
| Shader attributes as sole operation schema | Reject | Attributes describe kernel facts; operation semantics remain schema-owned. |
| Kernel routing in public APIs | Reject | Internal capability- and measurement-based lowering only. |

## Domains

| OA domain | Decision | Entry condition |
|---|---|---|
| Matrix | Preserve first | One-device elementwise slice plus FP32 `mat_mul_nt` baseline and schema-owned SDK oracle. |
| Image and Vision | Preserve | Matrix/storage/event contracts are stable. |
| ML inference | Preserve | Matrix baseline, GEMM baseline, and schema generation work. |
| Autograd and training | Defer | Semantic graph and reusable plan contracts are proven. |
| Audio | Defer | Image/vision-style semantic storage and session patterns exist. |
| Video | Defer | Session state, image planes, external synchronization, and completion are specified. |
| Render and UI | Defer | Image/resources plus presentation borrowing are proven. |
| Crypto | Defer | Core dispatch, secret-data policy, independent vectors, and side-channel scope are defined. |
| Local multi-device | Defer | One-device execution and explicit transfer are correct. |
| Distributed execution | Defer | Local multi-device and session/transport contracts exist. |

## Documentation migration rule

Before copying an OA document:

1. identify the semantic contract or evidence it owns;
2. classify each relevant concept using this ledger;
3. remove C++ implementation narration and historical release claims;
4. write Rust ownership, failure, concurrency, and acceptance behavior;
5. label unimplemented direction Planned rather than Shipped;
6. link to the original OA document for history instead of duplicating it.

Documents with no dependency in the current roadmap stay in the C++ repository
until their Rust prerequisite becomes active.
