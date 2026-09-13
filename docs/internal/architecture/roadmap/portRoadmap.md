# OA Rust Port Roadmap

**Status:** Planned

**Updated:** 2026-09-12

**Architecture:** [OA Rust Architecture](../oaArchitecture.md)

This roadmap orders dependencies. It is not a release promise or a list of
features that currently ship.

## Current position

Stage 0 is the `v0.1.0` repository-contract checkpoint. Stage 1, the one-device
Matrix-add vertical slice, is complete as an Experimental foundation.
Builder-based automatic or exact-index
device selection, Vulkan 1.3 timeline-semaphore and synchronization2 feature
negotiation, one compute-capable logical device, VMA-backed host-visible
storage, command-pool ownership, empty command-buffer recording, checked FP32
initialization and readback, zero-extent behavior, and
storage/device/instance lifetime retention are Experimental prerequisites.
Timeline-backed checkpoint submission, non-blocking event drop, asynchronous
command-buffer retirement, exact completion query/wait, and event-after-engine
lifetime behavior are also Experimental. The bounds-checked FP32 add kernel now
has deterministic Slang-to-SPIR-V compilation, reflected ABI validation,
Vulkan 1.3 `spirv-val` validation, embedding, descriptor-indexing capability
negotiation, and private compute-pipeline construction. Executable add
recording, a generic `ComputeDispatch` and engine submission path, checked
direct asynchronous dispatch, result-owned completion, host-observation
waiting, non-blocking `try_read_f32`, and independent host-oracle coverage are
Experimental. Stage 2 now has one normalized schema generating 19 out-of-place
elementwise Rust functions, 19 `f32` physical variants, and one `i32` add
variant with stable kernel/artifact lookup, Slang kernels, and hardware-oracle
tests. The first Stage 4 slice has also landed ahead of reusable execution: a
schema-generated FP32 `matrix::mat_mul_nt` preserving the OA
`[M, K] × [N, K] -> [M, N]` convention, with a 64×64×16 shared-memory
physical tile, odd/zero-shape CPU-oracle coverage, and a runnable SDK tutorial.
It is not yet a performance-qualified GEMM routing system. A shared
engine-owned bindless
descriptor heap serves all generated pipelines. A private engine-owned
execution session records non-empty eager operations, joins their owned graph
snapshots, and submits one hazard-planned command buffer at blocking host
observation or `Engine::checkpoint`. `try_read` never submits or waits. The
graph recorder derives per-buffer RAW, WAR, and WAW barriers while omitting
read-after-read barriers. Isolated capture and immutable engine-associated
`ExecutionPlan` replay are Experimental: capture rejects pending eager work,
nesting, failure, and empty work; replay rejects foreign engines and returns an
exact event. Explicit timed replay records a fresh Vulkan timestamp query pair
around the complete executable graph and exposes its wrap-corrected device
duration through that exact event. An Experimental six-shape MatMul recording
suite now correctness-gates captured-plan replay, runs at least seven fresh
processes, and preserves raw timing and build/device provenance. Untimed plans
cache one simultaneously submittable command, validated read-only Matrix inputs
retain stable slot identity across rebinding, and diagnostic snapshots report a
normalized executable identity plus graph, cache, submission, rebinding, and
fallback counters. Calibrated clocks, mutable output slots, broadcasting,
mutation contracts, and broader generated documentation remain incomplete.
The Vulkan device now also owns a bounded exact-size pool for completed
host-visible storage buffers. Final buffer ownership returns allocations and
their bindless descriptor slots only after command retirement; full uploads
overwrite reused bytes, descriptor pressure evicts pooled entries, and device
shutdown drains the pool before descriptor and allocator destruction. Sealed
stable-resource capture now qualifies semantic/observed liveness and exact Rust
owners, then materializes disjoint transient intervals into shared private
buffer arenas with recomputed hazards. Replaced allocations bypass the pool.
This is an Experimental Matrix arena slice, not the planned upload/readback
rings or a cross-placement transient allocator.
Matrix values now carry a stable private semantic identity used by the first
reverse-mode ML slice.
The first host-memory foundation slice is Experimental. `core::memory` has
checked exact-length ordinary and one-way streaming copies, an inlined dynamic
small-copy dispatcher, runtime-gated AVX2/AVX-512 non-temporal stores in the
qualified 1 KiB through 4 MiB window, ordinary and fixed-work equality, and
secure erasure. Vulkan mapped uploads use the one-way policy before allocation
flush. Exact-bounds, alignment, mismatch, guard-byte, and public API tests pass;
the matched OA C++/OARS harness and alternating fresh-process recorder are
implemented. Fixed-clock acceptance remains blocked by the reference
laptop's degraded performance profile, and this does not replace the planned
upload ring.
Engine-owned structured console/file logging, weak thread-local selection,
custom component tags, release compile-out for trace/debug call sites, and
explicit failure-bearing flush/close boundaries are Experimental. The startup
banner now records engine version plus indexed queried compute-device,
driver/conformance, memory, PCI identity, and queue-family information.
The first ML training seed is Experimental: U32 class/token-index storage,
FP32 Linear, Embedding, and stacked Elman Rnn, zero-copy differentiable reshape,
stable mean cross-entropy, a consumed thread-affine GradientTape,
finite-difference-checked parameter gradients, deterministic repeated-index
scatter-add and complete BPTT, and in-place AdamW parameter/moment updates over
stable storage pass the serial hardware contract. Constructor-owned recursive
Module registration now supplies deterministic dotted parameter/buffer traversal,
duplicate-identity rejection, train/eval propagation, and one composed
character-model optimizer path. Fixed-shape `TrainingProgram` replay now retains
stable input/gradient/parameter/moment state, advances AdamW state on the GPU,
and reuses one cached command for the complete canonical Char-Transformer gate.
This is not generalized autograd. The complete canonical 5×3 Char/Byte/BPE
comparison matrix passes its exact 300-step corpus, loss, accuracy,
fixed-prompt generation, and fresh-owner checkpoint gates. Recurrent streaming
state, shared and serialized RNG streams, dynamic specialization, and qualified
performance remain Planned. The separate Empyrealm-Core fidelity tutorial is
connected without copying the donor's currently identical renamed shaders.
The donor-backed `ItTraining`
connects eager and captured completion to epoch/work accounting, loss metrics,
ordered callback control, exact GPU timing distributions, and progress/summary,
CSV, validation, checkpoint/restore-best, early-stop, phase, and learning-rate-schedule policies. Its
object-safe optimizer behavior seam and eager constructor now admit the donor
no-op owner, FP32 SGD with optional momentum, and FP32 Adam; their checked
hardware oracles pass on Intel Iris Xe with Mesa 26.2.2 and Vulkan 1.4.354.
Adam, AdamW, exact no-momentum SGD, and Muon state share the native `.oam` checkpoint
path and bounded checkpoint-manager policy. The
prepare/record path now owns automatic
second-step capture, cached replay, preparation-node rejection, and explicit
safe recapture. Plan rejection preserves and eagerly submits the source step,
then disables capture until recapture is requested. Successful automatic
capture pre-records the reusable untimed command transactionally, so its first
replay is a cache hit. Schema-owned replay roles now reject host-stepped AdamW
and require exactly one graph-state advance before all replay updates. Successful
programs expose ordered semantic-validation-through-command-recording stage
evidence and the donor-compatible `oa.training_compilation.v2` JSON report;
captured semantic and executable state are also available through normalized
`oa.semantic_graph.v2` and handle-free `oa.execution_graph.v3` reports.
Schema-owned Philox uniform/normal and inverted Dropout now select replay
kernels with graph-resident per-operation counters and fail-closed pairing
validation.
Existing unrelated modules and shaders remain design scaffolding unless a
later status document names their implementation and verification evidence.

## Stage 0 — Repository contract

### Outcome

- repository-owned `AGENTS.md`;
- template-owned shared rules with the Rust contract strengthened at its owner;
- canonical architecture separated from research notes;
- explicit compatibility ledger for C++ concepts;
- `core/`, `runtime/`, and domain operation boundaries established;
- `Engine` selected as the sole local execution owner;
- first vertical slice specified with acceptance evidence.

### Exit gate

- documentation links resolve;
- rules do not impose contradictory C++ and Rust public syntax;
- no document presents scaffolded APIs as shipped;
- formatting, lint, test, and generation commands are defined;
- unrelated user work remains untouched.

## Stage 1 — One-device Matrix add

### Required path

```text
Engine::builder
  -> Vulkan entry and instance
  -> physical-device discovery and selection
  -> logical device and compute queue
  -> allocator and storage buffer
  -> checked host upload
  -> FP32 matrix::add returning Matrix
  -> executable compute dispatch
  -> asynchronous completion retained by the result
  -> checked host observation waits for exact completion
  -> independent host oracle
```

### Contract decisions proved here

- engine, device, allocation, matrix, plan, and event lifetimes;
- device and event identity;
- safe runtime boundary above `ash` and `vk-mem`;
- matrix shape, dtype, stride, byte-size, and alias validation;
- semantic versus executable operation representation;
- one generic engine submission boundary rather than per-operation engine
  methods;
- shader artifact, reflection, binding, and push-data contract;
- error propagation and destruction order;
- direct-dispatch bounds and zero-work behavior.

### Test matrix

- zero-sized input according to the admitted zero-size contract;
- one element;
- odd element counts around the workgroup width;
- several multidimensional shapes;
- mismatched shapes and dtypes;
- byte-size and dispatch-count overflow;
- alias and output reuse cases;
- allocation and shader/pipeline failure propagation;
- repeated submission and event association;
- deterministic fresh-process CPU comparison.

### Runtime evidence

- exact Vulkan SDK/registry, loader, device, driver, enabled capabilities, and
  build provenance;
- core validation with zero unexpected messages;
- synchronization validation with zero unexpected messages;
- GPU-assisted validation for shader bounds on an applicable real GPU;
- no CPU execution fallback.

This stage admits only the tested device/capability path. It does not establish
multi-vendor or multi-device support.

## Stage 2 — Operation schema seed (Experimental checkpoint complete)

The first normalized schema now generates:

- 19 out-of-place elementwise Rust signatures and rustdoc;
- 19 `f32` physical variants plus one `i32` add proof with exact dtype routing;
- stable private kernel identities and embedded artifact metadata;
- bounds-checked Slang kernels with reflected OA attributes and push ABI;
- an external odd-size hardware test using independent golden values;
- positive and negative generator/schema tests.

Regeneration is deterministic and idempotent. A normal Cargo build does not
modify checked-in files. The handwritten add registry and shader route have
been removed. Validation and shape inference remain shared handwritten lowering
helpers until their schema-generated fixture layer lands.

## Stage 3 — Reusable execution (Experimental baseline complete)

The baseline now captures an isolated eager recording into a structurally
immutable plan, caches unchanged untimed command recording, and admits validated
read-only Matrix input rebinding with cache invalidation. A donor-backed,
Vulkan-independent `SemanticGraph` structural port now preserves typed value,
view, operation, attribute, access, alias, mutation, control-dependency, and
autograd-range provenance. Matrix schemas generate explicit compatibility
contracts; Matrix dispatches record semantic values and metadata-view lineage,
captured plans retain the graph, and construction validates semantic-to-executable
ownership. Compatibility identities remain distinct from exact OA donor hashes
until the complete donor behavior contract lands. A private donor-backed DNN
analyzer now consumes the graph, recognizes schema-authorized candidate regions,
and reports portable versus recognized partitions. Its first physical providers
mechanically port the exact OA FP32 `[1024,32]` inference QKV projection+bias
replacement and `M=1024, N=64, K=32` Linear+Linear+SwiGLU gate/up replacement,
including many-to-one semantic provenance, eliminated-intermediate lifetime
proof, and explicit source-path fallback. Training capture retains the proven
source lowering. Current ML schemas emit compatibility contracts; their
dispatches preserve scalar semantic attributes and a captured Transformer
forward reaches DNN analysis without compatibility compute nodes. Reached tape
nodes attach forward outputs to generation-checked contiguous backward ranges,
and DNN analysis can distinguish their training operations. Private in-place
AdamW operations functionalize each state, parameter, and moment write as a
fresh semantic value aliasing its prior SSA version while the execution binding
retains the same storage.
Remaining work must establish:

- exact OA schema contract identities beyond the Matrix compatibility seed;
- semantic mutation coverage for future public and stateful in-place writers;
- schema/candidate-owned access ranges, physical write domains, collision
  policies, and generated overlap rejection evidence;
- complete private prepared-dispatch evidence joining capabilities, bindings,
  aliases, workspace, dispatch geometry, and reflected ABI before command
  encoding;
- additional capability/lifetime-qualified DNN providers beyond QKV and gate/up;
- mutable output bindings and shape-specialized variants;
- multi-queue, multi-device, and distributed scheduling contracts;
- persisted graph descriptors distinct from diagnostic JSON reports.

The current mixed semantic/compatibility capture is a checkpoint, not the
completed OA graph port. Extend semantic bindings to the remaining domains,
then extend the donor-backed, capability/lifetime-qualified DNN physical
lowering instead of adding tutorial-specific fused execution paths.

Exact whole-plan device timing and a correctness-gated fresh-process MatMul
recording suite are Experimental checkpoints. Clock-domain calibration,
phase/node timestamps, physical artifact telemetry, accepted baselines, and
cross-implementation comparison remain separate dependencies.

Operator overloading remains deferred until this stage proves where validation
and lowering failures are reported without panics.

## Independent foundation checkpoint — VLM (Experimental)

The host-only `core::vlm` checkpoint ports OA's packed `f32`/`f64` spatial
values and formula authority without depending on the Vulkan runtime. It uses
Rust's standard library, preserves the fixed OA row-vector and Vulkan-depth
convention, and replaces C++ output-parameter checks with `Option` results that
leave no partially updated value. Its external contract suite covers packed
layout, robust normalization, every Euler order, quaternion/matrix parity,
inverse and singular behavior, signed-scale/shear/reflection decomposition,
camera-forward `-Z`, Vulkan depth, odd viewport extents, and large-world `f64`.

Consumer migration and source-audit enforcement remain Planned. A matched
50-case OA C++/GLM/OARS benchmark and alternating fresh-process recorder are
implemented; the current noncanonical diagnostic identifies checked
projection, determinant/multiply, quaternion rotation, and normalization as
optimization targets. SIMD specialization remains deferred until canonical
fixed-state evidence and a measured consumer workload justify it.

## Stage 4 — Matrix foundation

Grow the Matrix surface by complete schema-owned slices:

1. creation/upload and fill;
2. elementwise arithmetic (out-of-place FP32 baseline complete) and broadcasting;
3. reduction;
4. GEMM baseline (Experimental FP32 `mat_mul_nt` tiled route and reusable-plan
   measurement complete; routing, specialized variants, and qualification remain);
5. autograd seed.

Each operation requires its own oracle and edge-case pack. Kernel variants and
tuning enter only after the baseline semantic route is stable.

## Stage 5 — ML training seed (Experimental first slice)

The first complete paths now prove:

```text
Linear -> mean cross-entropy -> reverse-mode gradients -> AdamW -> lower loss
Embedding -> reshape -> Linear -> cross-entropy -> scatter-add gradient
Embedding -> Rnn -> reshape -> Linear -> cross-entropy -> complete BPTT
Embedding -> LayerNorm -> reshape -> Linear -> cross-entropy -> complete adjoint
Token/position Embedding -> causal TransformerBlock -> Linear -> 300-step LM
```

It uses the established Matrix, eager recording, executable graph, embedded
shader, and Event contracts without a second tensor or execution owner. The
current tape admits Linear, Embedding, LayerNorm, GELU, standard composed
scaled dot-product attention with causal and optional additive masking,
equal-shape FP32 addition, stacked Elman Rnn, and zero-copy reshape chains
terminating in cross-entropy, and the optimizer updates
FP32 values out of place. Recursive Module traversal now binds a composed
character model to that optimizer exactly once per parameter. LayerNorm uses a
two-pass last-dimension variance calculation and has a complete parameter/input
adjoint checked through an embedding predecessor. Complete this stage with:

1. port exact OA operation identities over the connected Matrix compatibility
   seed, attach ML semantic contracts, and qualify DNN partitions for physical lowering;
2. extend semantic bindings to all domains and extend stable-resource frames,
   replay-safe RNG transformation, transient alias
   materialization, compilation-stage diagnostics, and graph reports into the
   existing Rust `ExecutionSession`/`ExecutionPlan` ownership;
3. replace prototype ML shader bodies with provenance-recorded ports of the OA
   kernels and routes, beginning with AdamW/AdamwMany4, Linear/GEMM, loss,
   normalization, recurrent, and attention families;
4. port remaining Matrix views, broadcasting, reductions, deterministic random
   creation, and the matmul orientations required by generated backward rules;
5. port schema-owned generalized reverse traversal and gradient accumulation;
6. add recurrent streaming state (tutorial-level character RNN convergence is
   Experimental and complete);
7. graph-resident AdamW step state, stable gradient/input slots, fixed-shape
   training-program replay, and private per-operation RNG counters are
   Experimental; add serialized/shared RNG state, dynamic
   specialization, and multi-buffered input staging;
8. extend the Experimental donor-backed `ItTraining` lifecycle from exact
   eager/captured completion, epoch/work accounting, loss metrics, ordered
   callback control, GPU timing distributions, progress/summary presentation,
   automatic prepare/record capture, safe recapture, and source-preserving
   capture fallback to checkpoint, evaluation, scheduling, and the connected
   bounded `TrainingSession` safe-point control plane using the same
   Rust-native traits and borrowing;
9. the Experimental canonical 5×3 Char/Byte/BPE NLP comparison matrix and
	distinct Empyrealm-Core fidelity row are connected through the same
	operation/module/training owners without duplicating the Mamba-3 providers.
	The Mamba fused preprocess, multi-output tape node, grouped-SISO and
	shared-state MIMO primitives, short/chunked/bounded-generic adjoints, gated
	normalization, parameter-owning module, and explicit recurrent step/cache are
	connected; broader dtypes, mobile qualification, and canonical performance
	remain gated.

The source-by-source authority and current gap are recorded in
[OA ML port inventory](../../porting/oaMlPortInventory.md). No new ML shader is
admitted without checking that ledger and the live OA donor first.

The complete NLP suite and performance comparison are release gates, not
implementation-fragment tests.

## Independent host-domain checkpoint — Audio and Cryptography (Experimental)

Two bounded host-facing slices reuse the existing Matrix/runtime foundation
without changing ML lowering or the generated operation surface:

- `Audio` composes a non-empty planar FP32 `[channels, samples]` Matrix with a
  non-zero sample rate and checked speaker layout. Synchronous WAV/PCM, FLAC,
  and MP3 decode uses a private, feature-bounded Symphonia adapter and uploads
  through the existing engine. OA's checked WAV-F32 encoder is ported directly;
  semantic encode/save remains an explicit blocking readback boundary.
- The donor's 17 Audio operations enter through the shared schema and typed
  Audio semantic values, lowering to 24 generated kernel identities. Signal,
  filtering, reverb, STFT, Mel, and MFCC oracles pass on the named Vulkan
  device.
- `AudioCapture` and `AudioPlayer` use CPAL device streams and bounded
  lock-free SPSC rings; incremental playback decode remains private to the
  player. `AudioEncoder` preserves explicit packet flush/close semantics.
- CPU Keccak-f[1600], SHAKE-128/256, KMAC-256, typed 32-byte hashes, and
  arbitrary-leaf Merkle trees/proofs directly port OA's algorithms. FIPS 202
  and SP 800-185 known-answer tests plus incremental and malformed-state tests
  gate the surface. KMAC sponge and temporary encoding storage are securely
  erased through `core::memory`.
- `SecureBuffer` preserves the donor's non-owning erase-on-drop contract with
  observable best-effort Linux page locking. CPU ML-DSA-65 owns typed keys and
  signatures; secret state is non-serializing and zeroized by the dependency.
- Schema-owned U8 Matrix kernels port batch SHAKE-128/256, Keccak-f[1600], and
  power-of-two Merkle reduction. Multi-level Merkle is one semantic operation,
  and deferred SHAKE output can feed it without host observation.

Audio device sessions remain Experimental pending cross-backend hardware
qualification. Compressed streaming encode and low-latency effects remain
Planned. Device-side ML-DSA remains Planned; the donor experiments were
incomplete and were not exposed by its public signing API.

The dependency-ordered device cryptography plan is owned by the
[vkPQC roadmap](vkPqcRoadmap.md). Its first possible Experimental checkpoint is
public-data ML-DSA-65 batch verification. Full ML-KEM and ML-DSA secret-bearing
operations require the separate
[GPU secret execution and observability](../../cryptography/oaGpuSecretSecurity.md)
contract; ordinary device-local memory, short lifetimes, and erasure do not
prevent capture by the owning process or a privileged debugger. Generic
cuPQC-style BigInt, arbitrary-field NTT, Poseidon2, and broad Merkle facilities
remain optional later projects rather than dependencies of the PK slice.

## Stage 6 — Multi-device local execution

Prove explicit transfer between two local devices before adding placement
automation. Admit transport paths in evidence order:

1. same logical device or device-group path with queried peer capabilities;
2. compatible external memory plus explicit external synchronization;
3. bounded host staging correctness path.

Remote transport is not part of this stage.

## Stage 7 — Image and vision pipeline

The initial semantic value is Experimental: `Image` now composes one dense
`Matrix`, validates NCHW/NHWC/CHW/HWC/HW rank and channel contracts, and
exposes immutable zero-copy borrowing plus consuming extraction. The former
zero-sized `Image`, `Video`, and `Presenter` TODO APIs and the TODO-backed
`vision::resize`/`normalize` functions were removed rather than treated as
capabilities.

The complete donor geometric family is Experimental under `image`: resize,
crop, flip, rotate, pad, center crop, remap, affine warp, and perspective warp.
Its schema-owned FP32 kernels preserve NCHW/CHW Image layout and format, keep
coordinate and transform inputs as Matrix-kind graph values, declare exclusive
physical output writes, and pass odd-shape independent host oracles—including
all five border modes—on the recorded Intel/Mesa device. The engine now enables
available Int64/Int8/8-bit-storage features, derives per-artifact requirements
from SPIR-V, omits unsupported pipelines, and rejects their use during
executable-graph construction. The prior shader-module VUID is absent from
fresh core and synchronization-validation ML runs. Image still requires its own
fresh validation qualification; this roadmap entry is not that evidence.

The complete donor 20-operation pixel family is Experimental under `image`:
pointwise thresholds and intensity transforms, grayscale, explicit semantic
channel reorder, blending/compositing/erase, color twist, and explicit-seed
Philox noise. Independent scalar and image-layout oracles pass on the same
Intel/Mesa device. The noise artifacts share the documented `shaderInt64`
validation prerequisite; hardware execution alone is not a clean-validation
claim.

The 20-operation filter family and per-channel normalization are Experimental,
completing all 50 tensor-native donor `FnImage` operations. Shared physical
filter lowering remains internal to 20 distinct schema contracts. Exact
identity/zero/fixed-derivative and normalization oracles pass on the same
Intel/Mesa device; composite morphology is a correctness reference path with no
throughput claim.

The donor still-image boundary is also Experimental under `image`: JPEG, PNG,
WebP, BMP, and TGA memory/file decoding uploads normalized FP32 NCHW Images;
encoding and file saving perform explicit blocking readback. JPEG/WebP quality,
path inference, capability queries, and the packed RGBA8 render-session sink
are checked. Color conversion, fused resize-normalize, and segmentation overlay
preserve semantic Image/Matrix graph kinds and pass hardware oracles.
Texture saving is Render-owned and uses the admitted packed RGBA8 Texture
value plus explicit Matrix readback.

The complete donor `FnDetection` surface is Experimental in `vision`: pairwise
IoU, deterministic class-aware/agnostic NMS, classification confusion,
binary-mask counts, three-stage detection AP/mAP, and segmentation confusion
plus metrics. Schema-owned kernels preserve GPU-resident outputs, async
readiness, semantic decomposition provenance, explicit atomic U32 collision
ownership, and replay-safe accumulator clearing. Donor vectors, odd dispatches,
ignored labels, deterministic tie breaking, invalid contracts, and repeated
plan submission pass the local hardware suite. It shares the same
clean-validation blocker above and is not yet a cross-device qualification
claim. Tracking and future typed detection values remain later Vision work.

Continue the stage by clearing that shared validation prerequisite, then extend
metadata only alongside its consuming operation. Preserve extent, format,
layout, color, readiness, and alias semantics. Reuse the same engine, schema,
graph, event, and shader systems. Image codecs and transformations belong to
`image`; detection, tracking, segmentation, and other interpretation belong to
`vision`. Do not reproduce the donor C++ Vision umbrella as the Rust owner of
Image or Video.

## Stage 8 — Stateful media and presentation

The `VideoFrame` value path is Experimental. It retains one packed Image, one
packed RGBA8 Texture, shared immutable planar YUV420 host bytes, or a private
native decoded-image slot, validates each backing's shape, and carries checked
presentation timing plus source matrix/range metadata. The donor
`FnVideo::fromTexture` bridge shares Texture storage and producer readiness
without a copy or Image semantic erasure. Native frames expose only
backend-neutral format, coded extent, producer completion, and a same-Engine
consumer-completion contract; raw Vulkan handles remain private and decoded
planes are not misrepresented as a dense Matrix.

The first packet-source session is Experimental. `VideoDemuxer` builds a
bounded, validated index for one unfragmented MP4 video track; reads payloads
incrementally, normalizes AVC/HEVC access units to Annex-B, prepends codec
configuration after open/seek, and defines explicit EOS/seek/close behavior.
H.264, H.265, AV1, and VP9 donor fixtures pass open/read/seek/close tests. A
matching packet-sink session is Experimental. `VideoMuxer` streams H.264/H.265
samples into an extended-size MP4 `mdat`, records bounded sample metadata,
selects 32- or 64-bit chunk offsets, and finalizes AVC/HEVC configuration plus
an optional native PCM-S16 track only at its explicit `finalize` boundary.
`close` and Drop never manufacture a movie trailer. Synthetic multi-sample and
PCM-track files reopen through `VideoDemuxer`; single donor AVC High and HEVC
Main streams also reopen and decode through an independent FFmpeg check.
Encoded packets retain distinct microsecond presentation and decode timestamps;
the muxer emits signed version-one composition offsets when order differs, so
the donor streams preserve B-frame timing instead of flattening it. A separate
selected-device query reports physical Vulkan Video queue and codec
extension support. Engine construction enables the advertised queue/extension
chain. Exact typed profile/format queries, decode-family command submission and
pool-correct retirement, plus private H.264/H.265/AV1/VP9 session creation and
memory binding now pass on the local Intel/Mesa device. Parsed donor H.264
SPS/PPS and HEVC VPS/SPS/PPS records also create and destroy dependent Khronos
standard-video session-parameter objects on that device. The HEVC record path
retains and lowers baseline profile-tier-level, DPB, scaling, short-/long-term
references, PCM, VUI/HRD, and tile geometry; unsupported HEVC extension syntax
fails closed. Exact-profile native output and DPB image allocation now passes
for every advertised typed decoder. Coincident profiles use separate one-layer
slot images when advertised and otherwise use a layered image; distinct-only
profiles use separate layered output and DPB images. Allocation fails
closed on unsupported disjoint plane binding. The H.264/H.265 recorders extract
their one VCL NAL from each donor access unit, emit the three-byte byte-stream
prefix used by the Khronos/FFmpeg Vulkan paths, and uploads it into a
profile-chained `VIDEO_DECODE_SRC` buffer whose initialized range satisfies the
queried size and offset alignments. Source buffers use dedicated memory so each
replacement remains bound at offset zero; this avoids the local Mesa recorder's
invalid CPU mapping of a non-zero VMA suballocation. Progressive donor H.264
and H.265 commands bind session parameters, reconstructed DPB slots, output
resources, planned resets, image/buffer barriers, decode-family submission,
and timeline retirement. The selected
queue family's result-status support is reported independently; an
exact-profile status query encloses the operation and returns `COMPLETE` on the
local Intel/Mesa device after event completion. An explicit decode-family
release and matching compute-family acquire then copy the NV12 planes to
host-visible storage; planar normalization matches all 1,382,400 bytes of each
independently decoded FFmpeg first frame through fixed SHAKE-256 oracles. H.264 VUI
and optional HRD syntax now parse into typed backend-neutral values and lower
into the standard-video VUI table; incompatible dual HRD tables fail closed.
Sequence/PPS scaling-list syntax likewise retains its presence/default masks
and 4x4/8x8 values, including SPS-resolved 4:4:4 list counts, and lowers without
reordering into `StdVideoH264ScalingLists`. HEVC SPS/PPS scaling-list syntax now
resolves default and predicted matrices, DC coefficients, and diagonal scans
into owned raster-order 4x4/8x8/16x16/32x32 values and now lowers them into the
live HEVC parameter object. HEVC SPS short-term reference sets
now retain direct and predictor masks plus resolved delta-POC order, fixing the
old skip walk's lost predicted-set cardinality; long-term SPS references retain
their POC-LSB and current-picture flag. HEVC slice parsing now consumes the
inline-versus-SPS RPS selector even when the SPS contains no sets, fixing the
donor stream's shifted second-picture reference syntax. A transactional private
DPB planner derives wrap-aware POC, reset/retention/recycling decisions, and
exact current-before/current-after reference slots for every donor HEVC sample.
The reusable Vulkan recorder now consumes those plans, binds active slot POCs
and current reference lists, orders prior decode writes before DPB reads, and
submits all 60 reordered donor pictures with `COMPLETE` result status on the
recorded Intel/Mesa device. Qualification then performs an explicit
decode-family release, compute-family acquire/copy/release, and next-decode
acquire for every output slot. The corresponding H.264 planner/recorder handles
progressive POC-type-zero streams, sliding-window references, and MMCO 1/5/6;
MMCO 2–4 and multi-slice pictures fail closed. Presentation ordering for both
60-picture donors matches independently decoded FFmpeg planar YUV420 frames
byte-for-byte.

Those qualified AVC/HEVC paths now run through `oa::video::VideoDecoder` and
its root identity alias. Creation binds exact demux metadata to the Engine-owned
session; synchronous `decode` materializes host-retained planar YUV420 frames
with packet timing and parsed VUI color metadata, then emits them in bounded
presentation order using the SPS reorder depth. Explicit `flush` returns the
delayed tail and requires a new random-access packet, `close` releases the idle
session, and Drop performs no submission or wait. Public API tests match all 60
decoder-ordered frames per codec to FFmpeg, then prove flush/seek/keyframe
reuse. Decoder-session availability is true only for enabled,
status-query-capable progressive H.264 Baseline/Main/High, H.265 Main, AV1
Main without film grain, and VP9 Profile 0, all 8-bit 4:2:0 rows.

Synchronous `decode_native` follows that same codec and display-order path
without the decode-to-host copy. A returned frame retains its qualified native
image set and slot. Live clones and pending consumer events exclude the slot
from DPB recycling; the consumer event must belong to the same Engine and
follow producer readiness. Retirement keeps storage alive through completion
even after frame, decoder, and original Engine handles are released. Complete
60-picture H.264, H.265, AV1, and VP9 native-output tests prove retention and recycling on
the recorded Intel/Mesa device. Decoder-owned retained-slot readback now
performs the explicit decode/compute/decode ownership round trip and matches all
60 display-order frames per codec byte-for-byte against FFmpeg. The qualified
device uses one queue family; the recorded split-family barriers remain
unqualified. Public asynchronous delivery, visible crop and plane-stride
metadata, and typed Render/ML native consumers remain planned.

The first composed `oa::video::VideoPlayer` now owns one demuxer and one public
decoder rather than adding another codec path. Open returns with the first
display-order frame ready; explicit advance, checked wall-clock pacing,
play/pause, looping/non-looping EOS, timestamp seek through a preceding
keyframe, reset, flush, counters, and close are shipped for local admitted
streams. Bounded shared-backing presentation history adds exact backward and
signed frame stepping; cache misses deterministically replay through a bounded
preceding window. A validated container timestamp index also admits absolute
display-frame seek without confusing decode order for display order. The
60-frame AVC donor proves monotonic presentation order, EOS, seek/reset reuse,
pacing, close, zero-decode cache hits, byte-identical replay after eviction,
and exact frame-30 selection after EOS. Audio synchronization, network
reconnect, RGBA conversion pools, and native frame consumption remain planned
Media/Render integration work; the player continues to request host-planar
decoder output.

Exact H.264 High and H.265 Main encode capability and format discovery is now
public and backend-neutral. The query reports common encode limits,
rate-control/feedback support, typed codec-specific level/slice/reference/QP
and HEVC tile/block limits, plus recognized and unknown input/DPB formats. Both
donor profiles pass on Intel Iris Xe/Mesa 26.2.2. This is the evidence seam for
session allocation; encoder-session availability remains false until an actual
create/encode/flush/close path and independent bitstream oracle pass.

The next decoder checkpoint is asynchronous native display delivery, followed
by typed Render/ML native-plane consumers and conversion. AV1 now has bounded
borrowed OBU parsing, raw/IVF access-unit picture inventory, and an owned
backend-neutral sequence-header record covering operating points, timing,
profile, extent, coding tools, component/chroma/color configuration, and film
grain. The complete 60-packet donor locks Main 8-bit 4:2:0 1280-by-720
metadata, while a synthetic reduced still-picture header locks the
specification's implicit operating-point-ID branch rather than the donor's
misaligned read. That record now lowers to the Khronos AV1 standard-video color,
timing, flags, extent, and tool fields; exact-profile validation rejects
mismatches before Vulkan, and the local Intel/Mesa device accepts the resulting
session-parameter object. Backend-neutral AV1 frame parsing now carries explicit
eight-slot reference state through every donor coded/show-existing picture and
retains the standard-video submission fields through quantization,
segmentation, loop filter, CDEF, restoration, transform, skip, and admitted
motion/grain syntax. Checked tile parsing produces access-unit-relative byte
ranges for every donor combined or separate tile group. Frame IDs,
decoder-model removal timing, short-reference derivation, non-uniform tiles,
non-identity global motion, and applied film grain remain fail-closed. AV1 frame
fields now lower to owned standard-video picture tables; quantizer U/V
difference, feature/update masks, restoration metadata, and Q16 identity global
motion have direct tests. Transactional DPB planning maps eight logical
references onto bounded physical slots, deduplicates active bindings, resolves
show-existing frames without decode allocation, respects unavailable native
leases, and leaves state unchanged on failure. The AV1 recorder now binds
deduplicated active references plus an inactive begin-coding reconstruction
association, submits exact frame-header/tile ranges, queries result status, and
performs the existing explicit decode/readback ownership round trip. The first
donor keyframe completes on Intel Iris Xe/Mesa 26.2.2 and its complete YUV420
readback matches FFmpeg through a fixed SHAKE-256 oracle. The public decoder
now assembles complete pictures transactionally, decodes hidden pictures,
resolves show-existing frames from the physical DPB, and exposes both
host-planar and retained-native outputs. All 60 displayed donor frames match
FFmpeg byte-for-byte in both modes; the native proof retains an early frame
across later DPB recycling. Per-picture size override, super-resolution,
distinct render size and the broader syntax listed above remain planned.
VP9 now has exact `vpcC` Profile 0 admission, complete raw/IVF and superframe
parsing, retained loop-filter/segmentation/reference state, transactional
logical-to-physical DPB planning, and Khronos standard-video picture lowering.
The released Ash loader remains authoritative while a revision-pinned official
Ash binding contributes only the missing Vulkan-Headers-1.4.350 VP9 C-ABI
records through raw `pNext` chains. Complete 60-frame host-planar and
retained-native runs return `COMPLETE` on Intel Iris Xe/Mesa 26.2.2 and match independent
FFmpeg YUV420 output byte-for-byte. Higher VP9 profiles, 10/12-bit output,
dynamic coded/render extents, and broader container syntax remain planned.
The muxer now supplies the container sink needed by a future
encoder and recorder, but it does not make either session shipped.

The H.264 expansion now has a transactional progressive POC-type-zero DPB
planner consumed by the reusable Vulkan recorder and public decoder. It plans
and decodes every reordered donor High-profile sample within the SPS and
16-slot bounds. Long-term MMCO 2–4, interlaced content, and multi-slice pictures
remain fail-closed.

Add one session at a time after its state machine, borrowed-engine lifetime,
external synchronization, and explicit close/drain behavior are specified.
`audio` and `video` remain direct public domains. The sibling `media` module is
introduced only for cross-track sources, clocks, transport, and synchronized
playback. Render owns Texture and Presenter adapters for Image, VideoFrame, or
scene output. Vulkan Video, platform codecs, sound libraries, and WSI are
implementation backends, not the public media model.

Principal values and lifecycle-bearing sessions may be explicitly re-exported
from `lib.rs` as identity aliases. Operations remain module-only. This permits
both `oa::AudioPlayer` and `oa::audio::AudioPlayer` without two
implementations; callers may locally alias `oa::audio` as `oaa`.

The first Render resource is Experimental: `Texture` retains exact packed
RGBA8 U8 storage with checked host upload/readback and the Image-codec file
sink. It deliberately claims no native sampled-image or render-target support.
Schema-owned Texture conversion/clear/blit, native backing, a headless Renderer
target ring, graphics/Scene, and Presenter follow in that dependency order.

## Deferred until their dependencies exist

- generalized training sessions beyond the Stage 5 vertical slice;
- generalized autograd;
- distributed execution and collectives;
- remote workers and satellites;
- audio device graphs;
- broad rendering and UI;
- release performance comparisons;
- Python packaging beyond the first schema parity proof.

## Rejected migration patterns

- translating the C++ directory tree or class hierarchy line by line;
- maintaining two execution owners or a default global owner;
- publishing TODO-backed APIs to reserve names;
- handwritten registries that duplicate schemas or shader reflection;
- build scripts that rewrite source or ignore generator failure;
- expanding domain breadth before the first vertical slice is proven.
