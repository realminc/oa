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
| Runtime wrapper/default engine | Reject | Construction and discovery belong to `Engine::builder`; no second owner or singleton. |
| `core/` source boundary | Preserve | Internal `core/` owns foundational values and metadata; `lib.rs` explicitly re-exports admitted root types. |
| Public `oa.core` namespace | Redesign | Rust root paths are canonical; a Python identity alias may exist only for compatibility. |
| Value / operation / session classification | Preserve | Rust structs/enums, module functions, and explicit stateful session structs. |
| Shared storage with semantic value types | Preserve | Composition and checked zero-copy views; no inheritance requirement. |
| C++ Pimpl and access bridges | Redesign | Private fields, crate visibility, typed ownership, and narrow safe runtime interfaces. |
| Engine inheritance and secondary contexts | Reject | Composition and explicit borrowing only. |

## Public API

| OA concept | Decision | Rust direction |
|---|---|---|
| `oa::FnMatrix`, `oa::FnImage`, other `Fn*` namespaces | Redesign | `matrix::`, `image::`, `vision::`, and other lowercase Rust operation modules. |
| PascalCase value and session types | Preserve | Normal Rust type naming. |
| camelCase methods and parameters | Reject | Rust `snake_case`. |
| C++ `in`/`out`/`inOut` parameter prefixes | Reject | Borrowing, mutable borrowing, and ownership express access mode. |
| Root semantic vocabulary | Preserve | Curated explicit re-exports from `lib.rs`. |
| Convenience methods | Redesign | Delegate to the same schema-owned operation; never create a second path. |
| Operator overloads | Defer | Admit only after graph and error behavior are proven without panics. |
| Public declarations backed by TODOs | Reject | Keep planned APIs in documentation until implementation and contract tests exist. |
| C++/Python identical spelling | Redesign | Preserve semantic identity while using idiomatic Rust and Python spelling. |

## Ownership and errors

| OA concept | Decision | Rust direction |
|---|---|---|
| Explicit `submit` and `Event` completion | Preserve | Fallible submission returns an engine-associated event; waiting is explicit. |
| Destructor never submits or waits | Preserve | `Drop` releases state only and cannot report completion failure. |
| OA `Status` / `Result<T>` | Redesign | Typed Rust `Result<T, Error>` with contextual sources. |
| `UniquePtr` / `SharedPtr` translation | Redesign | Borrow or own directly; use `Box` or `Arc` only for proved lifetime needs. |
| Raw owning Vulkan handles in semantic values | Reject | Opaque, typed, lifetime-safe resource ownership. |
| Hidden CPU fallback | Reject | Unsupported GPU work fails explicitly. |

## Graphs and generation

| OA concept | Decision | Rust direction |
|---|---|---|
| Semantic graph | Preserve | Domain values, operations, attributes, effects, aliases, control, and autograd provenance. |
| Executable Vulkan graph | Preserve | Concrete dispatch, transfer, media, render, queue, barrier, and completion work. |
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
| Slang source and SPIR-V | Preserve | First-class `src/slang` source with validated reflection and artifacts. |
| Shader attributes as sole operation schema | Reject | Attributes describe kernel facts; operation semantics remain schema-owned. |
| Kernel routing in public APIs | Reject | Internal capability- and measurement-based lowering only. |

## Domains

| OA domain | Decision | Entry condition |
|---|---|---|
| Matrix | Preserve first | One-device add vertical slice and schema seed. |
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
