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
| Semantic graph | Preserve | Domain values, operations, attributes, effects, read-only and mutation aliases, control, and generation-safe autograd forward/backward provenance. Transactional composite lowering retains physical children beneath one generated parent operation and rolls back abandoned work. |
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
| Embedded pipeline preload | Preserve progressively | Every schema-owned shader is embedded and retains one stable generated table slot. Engine construction derives requirements from SPIR-V capabilities, enables available Int64/Int8/8-bit-storage features, creates only device-admitted pipelines, and fails executable-graph construction before state mutation when a selected kernel is unavailable. This eliminates the eager-registry validation mismatch on the qualified Intel device. C++-equivalent preload opt-out and persistent Vulkan pipeline-cache configuration remain Planned. |
| Shader attributes as sole operation schema | Reject | Attributes describe kernel facts; operation semantics remain schema-owned. |
| Kernel routing in public APIs | Reject | Internal capability- and measurement-based lowering only. |

## Domains

| OA domain | Decision | Entry condition |
|---|---|---|
| Vulkan Linear Math (VLM) | Preserve with Rust-native failure syntax | Experimental packed `f32`/`f64` values and fixed spatial convention use standard-library scalar math; checked C++ output parameters become `Option` results. Consumer migration and qualification remain open. |
| Matrix | Preserve first | One-device elementwise/reduction/index slice plus FP32 `mat_mul_nt` baseline and schema-owned SDK oracle. Donor last-axis gather/reverse and categorical `sample_logits` greedy/dense/TopK/nucleus routes are Experimental with replay-safe Philox state. |
| Image | Preserve progressively | Experimental dense `Image` composition validates Matrix rank, NCHW/NHWC/CHW/HWC/HW layout, and Gray/GrayAlpha/RGB/RGBA/BGR/BGRA channel order. All 50 donor tensor-native operations are schema-owned under `oa::image`: geometric, pixel, filter, and normalization. Typed color/resize-normalize/segmentation compositions plus JPEG/PNG/WebP/BMP/TGA one-shot codecs are implemented. Native-plane, Texture, transfer/range, and broader-layout work remains gated. |
| Vision | Preserve and narrow | The complete donor `FnDetection` surface is Experimental: IoU, deterministic NMS, confusion/counting, detection AP/mAP, and segmentation metrics remain GPU-resident semantic Matrix operations. Vision does not retain donor umbrella ownership of image/video codecs and sessions; tracking and richer typed detections remain Planned. |
| ML inference | Preserve progressively | Matrix/GEMM, recurrent, attention, MoE, and the complete parameter-owning FP32 Mamba-3 SISO/MIMO paths are connected. The SDK-owned ALM core adds a temporal Conv1d VQ-VAE, dense/MoE/hybrid causal token prior, native frozen CLIP text tower and byte-BPE parser, autoregressive generation, motion decode, donor-compatible CLIP v1 model files, and ALM v3 product bundles through one `.oam` codec. External CLIP weight translation, ALM data/training applications, KV cache, broader dtypes, and mobile-specific providers remain gated. |
| Autograd and training | Preserve progressively | The execution baseline now supports Experimental embedding → recurrent/attention/SSM module composition → Linear → cross-entropy → complete reverse mode and in-place optimizer updates over stable parameter/state storage. Mamba-3 preprocessing adds one eight-output tape node which merges reached adjoints, zero-fills unused branches, and records one two-stage deterministic backward; grouped SISO adds one eleven-input node routed through six-stage short, nine-stage chunked, or two-stage bounded generic reverse, while shared-state MIMO adds one fifteen-input node. All routes have complete finite-difference evidence. Donor-backed gated RMSNorm completes the differentiable block. Donor-backed masked cross-entropy plus mean Smooth L1, MSE, L1, and BCE add schema-owned scalar loss roots with logits- or prediction-only adjoints. Schema-owned Philox uniform/normal and inverted Dropout add exact-seed eager execution, forward/backward mask regeneration, and graph-resident per-operation replay counters. The donor two-pass FP32 global gradient-norm clip records one variadic semantic mutation without CPU observation. Fixed-shape `TrainingProgram` capture adds stable gradient/input slots, graph-resident optimizer/RNG state, cached command replay, and semantic forward/backward tape ranges. Donor-backed `ItTraining` connects eager or captured completion to fixed/variable epochs, workload/loss accounting, borrowed Rust-native metrics/callbacks, explicit stop/error control, GPU timing distributions, and progress/summary/CSV/validation/checkpoint/phase/schedule policies. Its attached cloneable `TrainingSession` adds bounded safe-point commands, optimistic revisions, typed live parameters, pause/resume/stop, handler requests, terminal snapshots, and observer-independent results without moving the Vulkan iterator across threads. Its eager route accepts the object-safe `Optimizer` contract plus donor-backed `NoOpOptimizer`, FP32 SGD/momentum, Adam, and Muon; captured replay remains honestly AdamW-specific. Adam, AdamW, exact no-momentum SGD, and Muon persistence share the native `.oam` codec, bounded checkpoint manager, and restore-best callback. Rust-native `Module`/`ModuleRegistry` composition provides recursive ownership and Dropout train/eval behavior without reproducing the C++ inheritance root. The complete canonical 15-row Char/Byte/BPE NLP matrix passes its 300-step workload, exact deterministic quality/generation gates, and fresh-owner checkpoint reload through the iterator lifecycle. The additional Empyrealm-Core fidelity row preserves its nested module topology while sharing the donor-identical Mamba-3 providers. Generalized autograd beyond admitted structured nodes, serialized/shared RNG state, dynamic specialization, captured Muon, and remaining optimizer persistence remain deferred. |
| Audio | Preserve progressively | Experimental planar FP32 `Audio`, checked channel metadata, WAV/FLAC/MP3 one-shot decode, WAV-F32 encode/save, schema-owned DSP/feature transforms, PCM-S16 packet encode, and lock-free capture/playback sessions are implemented. Cross-backend device qualification, compressed streaming encode, and low-latency effects remain deferred. |
| Video | Preserve progressively | Experimental `VideoFrame` retains one packed Image, packed RGBA8 Texture, host-planar YUV420 backing, or private native decoded-image slot with single-frame extent, timing, source color metadata, producer readiness, and root/module identity. `video::from_texture` plus the bounded Annex-B family complete the donor `FnVideo` operation surface without a compatibility class. The bounded unfragmented-MP4 `VideoDemuxer` reads/seeks H.264, H.265, AV1, and VP9 packet streams and exposes exact typed profiles from `avcC`/`hvcC`/`av1C`/`vpcC`. The streaming `VideoMuxer` writes H.264/H.265 MP4 with explicit AVC/HEVC configuration, distinct presentation/decode timing and signed composition offsets, variable sample timing, 64-bit chunk-offset selection, optional native PCM-S16 audio, and explicit finalize/abandon behavior; donor streams reopen in OARS and decode through FFmpeg. The public Engine-owned `VideoDecoder` admits progressive H.264 Baseline/Main/High, H.265 Main, AV1 Main without film grain, and VP9 Profile 0, all 8-bit 4:2:0, with reusable transactional DPB sessions, result-status verification, display-order buffering, byte-exact 60-frame host-output FFmpeg differentials, and a synchronous native-output route on the recorded Intel/Mesa device. AV1 and VP9 additionally decode hidden pictures and resolve show-existing frames through logical-to-physical DPB maps. Native frames omit implicit host materialization and retain their slots through live clones plus registered same-Engine consumer events without exposing Vulkan handles. Decoder-owned retained-slot YUV420 readback explicitly restores codec ownership and matches all four complete donor streams byte-for-byte against FFmpeg. The composed `VideoPlayer` adds immediate first-frame presentation, manual and paced advance, looping/EOS, timestamp and absolute display-frame seek, reset, bounded shared-backing history, exact backward/signed frame stepping with deterministic replay, counters, and explicit close over the host-output decoder route. Audio-synchronized media transport, direct Render/ML native-plane consumers, visible crop/stride metadata, broader AV1/VP9 syntax, broader AVC/HEVC bitstream shapes, encoding, and public asynchronous synchronization remain deferred. |
| Media | Redesign from donor composition | Own cross-track sources, clocks, transport, and synchronized playback as a sibling coordinator over Audio and Video, not their parent namespace. |
| Render and UI | Preserve progressively | Experimental packed RGBA8 `Texture` has exact upload/readback and codec-file sinks. Native image backing, Texture GPU operations, Renderer, Presenter, UI, and Plot remain dependency-ordered. |
| Cryptography | Preserve progressively | Experimental donor-complete public surface: CPU Keccak/SHAKE/KMAC and Merkle values, borrowed secure host storage, typed CPU ML-DSA-65, plus schema-owned U8 Vulkan SHAKE/Keccak/power-of-two Merkle operations. Incomplete donor device ML-DSA remains rejected. |
| Local multi-device | Defer | One-device execution and explicit transfer are correct. |
| Distributed execution | Defer | Local multi-device and session/transport contracts exist. |

The Video row's deferred `encoding` item refers to stateful encoder sessions
and packet production. Exact backend-neutral H.264 High and H.265 Main encode
capability/format queries are shipped as the admission evidence for that next
vertical slice; they do not make `encoder_sessions_available()` true.

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
