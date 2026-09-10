# OA C++ to Rust Compatibility Ledger

**Status:** Planned

**Updated:** 2026-09-10

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
| `oa::FnMatrix`, `oa::FnImage`, other `Fn*` namespaces | Redesign by language | Rust uses ownership-preserving lowercase operation modules: Core `FnMatrix` becomes `oa::matrix`, its ML extension becomes `oa::ml::matrix`, `FnLoss` becomes `oa::ml::loss`, and media operations use `image`, `audio`, `video`, or `vision`. Python retains admitted `oa.Fn*` compatibility facades generated from the same schema. |
| PascalCase value and session types | Preserve | Normal Rust type naming. |
| camelCase methods and parameters | Reject | Rust `snake_case`. |
| C++ `in`/`out`/`inOut` parameter prefixes | Reject | Borrowing, mutable borrowing, and ownership express access mode. |
| Root semantic vocabulary | Preserve | Curated explicit re-exports from `lib.rs`. Principal values and sessions may appear at both their owning module path and the root as the same item; stateless functions remain module-only. |
| OpenMaya-style abbreviated modules | Support as caller aliases | Rust callers may write `use oa::{audio as oaa, core as oac, ml as oaml, vision as oacv}` and Python callers may alias imports. OA does not publish duplicate abbreviated crates or modules. |
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
| Image | Preserve progressively | Experimental dense `Image` composition validates Matrix rank, NCHW/NHWC/CHW/HWC/HW layout, and Gray/GrayAlpha/RGB/RGBA/BGR/BGRA channel order. All 50 donor tensor-native operations are schema-owned under `oa::image`: geometric, pixel, filter, and normalization. Typed color/resize-normalize/segmentation compositions plus JPEG/PNG/WebP/BMP/TGA one-shot codecs are implemented. Native-plane, Texture, transfer/range, and broader-layout work remains gated. |
| Vision | Preserve and narrow | The complete donor `FnDetection` surface is Experimental: IoU, deterministic NMS, confusion/counting, detection AP/mAP, and segmentation metrics remain GPU-resident semantic Matrix operations. Vision does not retain donor umbrella ownership of image/video codecs and sessions; tracking and richer typed detections remain Planned. |
| ML inference | Preserve | Matrix baseline, GEMM baseline, and schema generation work. |
| Autograd and training | Preserve progressively | The execution baseline now supports Experimental Embedding → stacked Elman Rnn → reshape → Linear → cross-entropy → complete BPTT and in-place optimizer updates over stable parameter/state storage. Donor-backed masked cross-entropy plus mean Smooth L1, MSE, L1, and BCE add schema-owned scalar loss roots with logits- or prediction-only adjoints. Schema-owned Philox uniform/normal and inverted Dropout add exact-seed eager execution, forward/backward mask regeneration, and graph-resident per-operation replay counters. The donor two-pass FP32 global gradient-norm clip records one variadic semantic mutation without CPU observation. Fixed-shape `TrainingProgram` capture adds stable gradient/input slots, graph-resident optimizer/RNG state, cached command replay, and semantic forward/backward tape ranges. Donor-backed `ItTraining` connects eager or captured completion to fixed/variable epochs, workload/loss accounting, borrowed Rust-native metrics/callbacks, explicit stop/error control, GPU timing distributions, and progress/summary/CSV/validation/checkpoint/phase/schedule policies. Its attached cloneable `TrainingSession` adds bounded safe-point commands, optimistic revisions, typed live parameters, pause/resume/stop, handler requests, terminal snapshots, and observer-independent results without moving the Vulkan iterator across threads. Its eager route accepts the object-safe `Optimizer` contract plus donor-backed `NoOpOptimizer`, FP32 SGD/momentum, Adam, and Muon; captured replay remains honestly AdamW-specific. Adam, AdamW, exact no-momentum SGD, and Muon persistence share the native `.oam` codec, bounded checkpoint manager, and restore-best callback. Rust-native `Module`/`ModuleRegistry` composition provides recursive ownership and Dropout train/eval behavior without reproducing the C++ inheritance root. The canonical Char-RNN and Char-Transformer rows pass their exact C++ 300-step workload and quality gates through the iterator lifecycle. Generalized autograd, serialized/shared RNG state, dynamic specialization, captured Muon, and remaining optimizer persistence remain deferred. |
| Audio | Preserve progressively | Experimental planar FP32 `Audio`, checked channel metadata, WAV/FLAC/MP3 one-shot decode, WAV-F32 encode/save, schema-owned DSP/feature transforms, PCM-S16 packet encode, and lock-free capture/playback sessions are implemented. Cross-backend device qualification, compressed streaming encode, and low-latency effects remain deferred. |
| Video | Preserve progressively | Experimental `VideoFrame` retains one packed Image with single-frame extent, timing, source color metadata, Matrix readiness, and root/module identity. The bounded unfragmented-MP4 `VideoDemuxer` reads/seeks H.264, H.265, AV1, and VP9 packet streams. Exact Vulkan Video profile/format/session/image ownership is private; typed H.264 VUI/HRD/scaling-list conversion plus one Intel/Mesa IDR qualification prove canonical VCL packing, result status, cross-family NV12 readback, and exact FFmpeg pixel-oracle agreement. HEVC SPS/PPS scaling lists and SPS short-/long-term reference records are retained as resolved typed values but are not yet lowered into a session-parameter object. Native frame planes, remaining codec parameters, reusable DPB/session state, encoding, and public synchronization remain deferred. |
| Media | Redesign from donor composition | Own cross-track sources, clocks, transport, and synchronized playback as a sibling coordinator over Audio and Video, not their parent namespace. |
| Render and UI | Preserve progressively | Experimental packed RGBA8 `Texture` has exact upload/readback and codec-file sinks. Native image backing, Texture GPU operations, Renderer, Presenter, UI, and Plot remain dependency-ordered. |
| Cryptography | Preserve progressively | Experimental donor-complete public surface: CPU Keccak/SHAKE/KMAC and Merkle values, borrowed secure host storage, typed CPU ML-DSA-65, plus schema-owned U8 Vulkan SHAKE/Keccak/power-of-two Merkle operations. Incomplete donor device ML-DSA remains rejected. |
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
