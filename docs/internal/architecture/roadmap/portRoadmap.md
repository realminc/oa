# OA Rust Port Roadmap

**Status:** Planned

**Updated:** 2026-09-09

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
This is not generalized autograd or an NLP-suite completion. The canonical
Char-RNN and Char-Transformer rows pass their exact 300-step corpus, loss,
accuracy, and fixed-prompt generation gates. Recurrent streaming state, RNG
replay, dynamic specialization, built-in callback policies, and the remaining
NLP matrix remain Planned. The donor-backed `TrainingLoop` connects eager and
captured completion to epoch/work accounting, loss metrics, ordered callback
control, and exact GPU timestamps. Its prepare/record path now owns automatic
second-step capture, cached replay, preparation-node rejection, and explicit
safe recapture. Plan rejection preserves and eagerly submits the source step,
then disables capture until recapture is requested. Successful automatic
capture pre-records the reusable untimed command transactionally, so its first
replay is a cache hit. Schema-owned replay roles now reject host-stepped AdamW
and require exactly one graph-state advance before all replay updates. Successful
programs expose ordered semantic-validation-through-command-recording stage
evidence and the donor-compatible `oa.training_compilation.v2` JSON report;
captured semantic and executable state are also available through normalized
`oa.semantic_graph.v2` and handle-free `oa.execution_graph.v3` reports. Philox
replay transformation remains Planned.
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
proof, and explicit source-path fallback. Training capture retains the proven source lowering. Current
non-mutating ML schemas also emit compatibility contracts; their
dispatches preserve scalar semantic attributes and a captured Transformer
forward reaches DNN analysis without compatibility compute nodes. Reached tape
nodes attach forward outputs to generation-checked contiguous backward ranges,
and DNN analysis can distinguish their training operations. AdamW remains
explicitly outside semantic capture until versioned mutation is ported.
Remaining work must establish:

- exact OA schema contract identities beyond the Matrix compatibility seed;
- versioned semantic mutation for optimizer updates;
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
current tape admits Linear, Embedding, LayerNorm, GELU, causal attention,
equal-shape FP32 addition, stacked Elman Rnn, and zero-copy reshape chains
terminating in cross-entropy, and the optimizer updates
FP32 values out of place. Recursive Module traversal now binds a composed
character model to that optimizer exactly once per parameter. LayerNorm uses a
two-pass last-dimension variance calculation and has a complete parameter/input
adjoint checked through an embedding predecessor. Complete this stage with:

1. port exact OA operation identities over the connected Matrix compatibility
   seed, attach ML semantic contracts, and qualify DNN partitions for physical lowering;
2. extend semantic bindings to all domains and port stable-resource frames,
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
7. graph-resident AdamW step state, stable gradient/input slots, and fixed-shape
   training-program replay are Experimental; add RNG state, dynamic
   specialization, and multi-buffered input staging;
8. extend the Experimental donor-backed `TrainingLoop` lifecycle from exact
   eager/captured completion, epoch/work accounting, loss metrics, ordered
   callback control, GPU timing, automatic prepare/record capture, and safe
   recapture, and source-preserving capture fallback to checkpoint, evaluation,
   scheduling, and `TrainingSession` control using the same Rust-native traits
   and borrowing;
9. expand beyond the Experimental Char RNN and Char Transformer tutorials
   across Byte/BPE and the remaining GRU, Transformer, MoE, and Mamba matrix.

The source-by-source authority and current gap are recorded in
[OA ML port inventory](../../porting/oaMlPortInventory.md). No new ML shader is
admitted without checking that ledger and the live OA donor first.

The complete NLP suite and performance comparison are release gates, not
implementation-fragment tests.

## Independent host-domain checkpoint — Audio codecs and Crypto (Experimental)

Two bounded host-facing slices reuse the existing Matrix/runtime foundation
without changing ML lowering or the generated operation surface:

- `Audio` composes a non-empty planar FP32 `[channels, samples]` Matrix with a
  non-zero sample rate and checked speaker layout. Synchronous WAV/PCM, FLAC,
  and MP3 decode uses a private, feature-bounded Symphonia adapter and uploads
  through the existing engine. OA's checked WAV-F32 encoder is ported directly;
  semantic encode/save remains an explicit blocking readback boundary.
- CPU Keccak-f[1600], SHAKE-128/256, KMAC-256, typed 32-byte hashes, and
  arbitrary-leaf Merkle trees/proofs directly port OA's algorithms. FIPS 202
  and SP 800-185 known-answer tests plus incremental and malformed-state tests
  gate the surface. KMAC sponge and temporary encoding storage are securely
  erased through `core::memory`.

Audio DSP operations are not handwritten around the Matrix schema. Port their
OA operation records and Slang routes as later complete vertical slices.
Capture, playback, streaming encode, and low-latency effects remain explicit
session work. Vulkan batch hash operations and ML-DSA remain Planned until
their dispatch, dependency, secret-data, and qualification contracts land.

## Stage 6 — Multi-device local execution

Prove explicit transfer between two local devices before adding placement
automation. Admit transport paths in evidence order:

1. same logical device or device-group path with queried peer capabilities;
2. compatible external memory plus explicit external synchronization;
3. bounded host staging correctness path.

Remote transport is not part of this stage.

## Stage 7 — Image and vision pipeline

Add `Image` storage views and metadata, then one complete upload → resize →
readback slice. Preserve extent, format, layout, color, readiness, and alias
semantics. Reuse the same engine, schema, graph, event, and shader systems.

## Stage 8 — Stateful media and presentation

Add one session at a time after its state machine, borrowed-engine lifetime,
external synchronization, and explicit close/drain behavior are specified.
Vulkan Video and WSI are implementation backends, not the public media model.

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
