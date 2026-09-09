# OA C++ to Rust Compatibility Ledger

**Status:** Planned

**Updated:** 2026-09-09

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
| OA foundation memory primitives | Preserve selectively | `core::memory` owns checked slice copy, explicit one-way streaming copy, fixed-work equality, and secure erasure. Private x86-64 AVX2/AVX-512 paths preserve the qualified OA policy; Rust `std` remains the default foundation rather than being replaced wholesale. |
| Complete OA C++ foundation/STL replacement | Reject | Rust uses `core`, `alloc`, and `std` for ordinary containers, ownership, synchronization, paths, and I/O. Add an OA primitive only for a distinct contract or correctness-gated measured benefit. |
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
| Explicit `submit` and `Event` completion | Preserve as advanced control | Capture, overlap, profiling, multi-device, and distributed paths return engine-associated events. Ordinary host observation flushes and waits for the exact producer; `try_*` observation does not wait. Execution plans and training programs can query or wait for their latest replay after the returned event is dropped; consuming `reset` is the Rust wait-and-release boundary. |
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
| Semantic graph | Preserve | Domain values, operations, attributes, effects, aliases, control, and generation-safe autograd forward/backward provenance. |
| Executable Vulkan graph | Preserve progressively | The private Rust graph snapshots concrete compute dispatches, plans buffer RAW/WAR/WAW barriers, and caches unchanged untimed command recording. Read-only Matrix inputs have stable captured identities and checked rebinding; Matrix values retain semantic identity. Deterministic `oa.semantic_graph.v2` and handle-free `oa.execution_graph.v3` reports expose semantic ownership and normalized executable/hazard evidence without making the graph editable. General semantic graph compilation and non-compute nodes remain deferred. |
| OaDna / private `Dnn*` planner | Preserve progressively | Port the canonical semantic-value planner, generated provider admission, partition diagnostics, and admitted lowering into private Rust runtime machinery. Do not author a second ML graph or expose provider policy publicly. |
| Private execution session | Preserve with Rust ownership | One engine-owned session now batches eager graph snapshots and tracks output readiness; the C++ access facades and context hierarchy are not reproduced. |
| Public execution plan | Preserve with Rust ownership | `Engine::capture` returns a structurally immutable engine-associated plan plus the closure result; `Engine::submit(&plan)` replays asynchronously and returns an exact event. Diagnostics expose handle-free graph/cache evidence. Rust additionally admits explicit shape/dtype/owner/alias-checked rebinding of read-only Matrix inputs. |
| One operation schema | Preserve | Generate Rust, Python, validation, autograd, registry, docs, and tests from one record. |
| Existing C++ generator implementation | Preserve behavior, redesign implementation | Port the schema content, generated contract, and determinism checks; the generator itself may be rewritten idiomatically in Rust or Python. |
| Handwritten kernel and operation registries | Reject | Extend the schema or reflected shader metadata instead. |
| Generated output written during normal builds | Reject | Build scripts use `OUT_DIR`; checked-in output changes only through explicit generation. |

## Vulkan and shaders

| OA concept | Decision | Rust direction |
|---|---|---|
| Vulkan as the execution backend | Preserve | `ash` remains the thin binding layer. |
| Volk loader | Reject | `ash::Entry`, `ash::Instance`, and `ash::Device` own dispatch tables. |
| VMA general allocator | Preserve initially | `vk-mem` stays private behind OA memory contracts and may be replaced by measured path. |
| Upload/readback rings and transient planning | Defer | Add after the general buffer path proves ownership and completion. |
| Slang source and SPIR-V | Preserve with provenance | Port proven OA shader algorithms rather than independently rewriting them. Mechanical module/import, bindless ABI, naming, and attribute adaptation is allowed and recorded beside the owning operation metadata. First-class `src/slang` source retains one `main` entry per compiled module, validated reflection, embedded artifacts, and fail-closed push-block sizing from exact SPIR-V. `OUT_DIR` is a compile-time boundary, never a runtime shader path. |
| Embedded pipeline preload | Preserve progressively | Every schema-owned shader is embedded and every generated pipeline is created during engine construction. C++-equivalent preload opt-out and persistent Vulkan pipeline-cache configuration remain Planned. |
| Shader attributes as sole operation schema | Reject | Attributes describe kernel facts; operation semantics remain schema-owned. |
| Kernel routing in public APIs | Reject | Internal capability- and measurement-based lowering only. |

## Domains

| OA domain | Decision | Entry condition |
|---|---|---|
| Vulkan Linear Math (VLM) | Preserve with Rust-native failure syntax | Experimental packed `f32`/`f64` values and fixed spatial convention use standard-library scalar math; checked C++ output parameters become `Option` results. Consumer migration and qualification remain open. |
| Matrix | Preserve first | One-device elementwise slice plus FP32 `mat_mul_nt` baseline and schema-owned SDK oracle. |
| Image and Vision | Preserve | Matrix/storage/event contracts are stable. |
| ML inference | Preserve | Matrix baseline, GEMM baseline, and schema generation work. |
| Autograd and training | Preserve progressively | The execution baseline now supports Experimental Embedding → stacked Elman Rnn → reshape → Linear → cross-entropy → complete BPTT and in-place AdamW over stable parameter/moment storage. Fixed-shape `TrainingProgram` capture adds stable gradient/input slots, graph-resident AdamW state, cached command replay, and semantic forward/backward tape ranges. Donor-backed `TrainingLoop` connects eager or captured completion to fixed/variable epochs, workload/loss accounting, borrowed Rust-native metrics/callbacks, explicit stop/error control, and GPU timestamps. Rust-native `Module`/`ModuleRegistry` composition provides recursive ownership without reproducing the C++ inheritance root. The canonical Char-RNN and Char-Transformer rows pass their exact C++ 300-step workload and quality gates; Transformer now runs through captured replay. Generalized autograd, RNG replay state, dynamic specialization, built-in callback policies, and live training sessions remain deferred. |
| Audio | Preserve progressively | Experimental planar FP32 `Audio` composition, checked channel metadata, WAV/FLAC/MP3 synchronous decode, and WAV-F32 encode/save use the existing Matrix storage and host-observation contract. DSP schemas/kernels and stateful capture/playback/streaming encode remain deferred. |
| Video | Defer | Session state, image planes, external synchronization, and completion are specified. |
| Render and UI | Defer | Image/resources plus presentation borrowing are proven. |
| Crypto | Preserve progressively | Experimental CPU Keccak-f[1600], SHAKE-128/256, KMAC-256, typed 32-byte hashes, and arbitrary-leaf Merkle proofs are direct donor-backed ports with KATs and secure temporary erasure. Vulkan batch hashing and ML-DSA remain deferred. |
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
