# OA Rust Agent Contract

This file is the mandatory repository-specific entry point for automated
contributors. Shared engineering policy remains owned by the linked rules in
`template/.cursor/rules`; do not copy those rules into this repository.

## Read first

Before changing architecture or public behavior, read:

1. `docs/internal/architecture/oaArchitecture.md` — canonical Rust target.
2. `docs/internal/architecture/roadmap/portRoadmap.md` — dependency order and
   current implementation stage.
3. `docs/internal/porting/oaCompatibility.md` when translating an OA C++
   concept.
4. `docs/internal/architecture/oaSourceStructure.md` and
   `docs/internal/porting/oaCppToRust.md` when adding or moving a public module,
   type, operation, session, or re-export.
5. The relevant linked `.cursor/rules/*.mdc`, subsystem document, source, and
   tests.

The C++ repository is evidence for behavior and hard-won constraints, not a
source-layout or implementation template. If the Rust architecture, C++
architecture, a plan, and live code disagree, report the conflict.

Language-neutral OA implementations are donor source, not merely inspiration.
Before adding an operation, Slang kernel, graph pass, optimizer path, or
training facility, audit the corresponding C++ schema, source, tests, and
documentation. Preserve proven algorithms and shader bodies by default;
restrict Rust-specific work to ownership, API, ABI, metadata, and mechanical
module adaptation. A new implementation requires a recorded reason why the OA
implementation cannot be ported or a correctness-gated measurement proving the
replacement. See `docs/internal/porting/oaMlPortInventory.md` for the active ML
ledger.

## Repository-specific invariants

- `Engine` is the sole local owner of Vulkan instances, logical devices,
  memory, queues, kernels, scheduling, and profiling. Do not add a public
  `Runtime` owner or second execution facade.
- `core` is the public foundation module for semantic values, checked metadata,
  shared error/result contracts, and other backend-neutral primitives. `lib.rs`
  explicitly re-exports the admitted common vocabulary at the crate root.
- Every public type has one owning module and implementation. `lib.rs` may
  explicitly re-export admitted principal values and session types at the root
  so `oa::AudioPlayer` and `oa::audio::AudioPlayer` are the same item.
  Stateless operations remain module functions such as `oa::matrix::add` and
  are never duplicated at the root. Short names such as `oaa` or `oaml` are
  caller-local `use ... as ...` aliases, not additional OA crates or modules.
- Raw Vulkan, allocator, OS, and third-party errors remain private to their
  adapters and are translated into the backend-neutral `core::Error` contract.
- Values preserve semantics when storage is shared. Operations are stateless;
  stateful external and iterative processes are sessions that borrow an engine.
- Semantic operations remain separate from executable Vulkan work.
- Eager domain operations return semantic values and do not require a public
  submit/wait ceremony. Explicit engine submission remains available for
  capture, orchestration, and profiling and returns an exact `Event`. Host
  observation may flush and wait; `try_*` observation never waits. `Drop` never
  submits, waits, drains, reads back, or finalizes a session.
- One operation schema owns every mechanically derivable Rust, Python,
  validation, autograd, registry, documentation, and test surface.
- Every ported kernel and algorithm records its OA donor path and adaptation
  class. Do not independently rewrite a working OA shader to satisfy a Rust
  call site.
- Kernel selection is private lowering policy and remains vendor-neutral at the
  public API.
- Port behavior through complete vertical slices. Do not translate the C++
  repository line by line or create a parallel framework.

## Current stage

The repository remains Experimental. No GPU operation is Shipped. The
one-device runtime, schema-generated out-of-place `f32` elementwise family,
`i32` Matrix-add dtype proof, and FP32 `matrix::mat_mul_nt` with a generated
64×64×16 tiled kernel and runnable SDK oracle are implemented checkpoints.
Backend-neutral `core::memory` provides checked ordinary and explicit one-way
streaming copy, ordinary and fixed-work equality, and secure erasure. The
x86-64 implementation retains OA's qualified small-copy and AVX2/AVX-512
streaming policies; Vulkan host uploads use that explicit streaming contract.
This is an Experimental correctness checkpoint, not a performance claim.
Broader integer operations, broadcasting, in-place mutations, GEMM routing and
specialized variants, low-precision storage, and broader domains remain
incomplete. Engine-owned structured logging, a queried indexed device/driver
startup banner, and explicit whole-plan Vulkan device timing are Experimental.
The fresh-process MatMul recording runner and
six-shape suite are Experimental; accepted baselines, cross-implementation
comparison, calibrated host/device clocks, structured runtime metrics, and
accepted regression policy remain Planned. Execution-plan diagnostics now expose
normalized graph, barrier, recording, cache-hit, submission, rebinding, and
fallback evidence plus logical/physical resource counts without exposing Vulkan
or kernel routing policy. Training programs additionally expose ordered
compilation-stage evidence; schema-owned replay roles reject host-stepped AdamW
and frozen RNG, require one optimizer-state advance before replay updates, and
pair every replay RNG use with one graph-resident counter advance. Plans and
training programs expose deterministic handle-free `oa.semantic_graph.v2`,
`oa.execution_graph.v3`, and `oa.training_compilation.v2` evidence reports.
The private executable-graph foundation snapshots resolved compute dispatches,
records multiple nodes in one primary command buffer, and derives per-buffer
RAW, WAR, and WAW barriers. A private engine-owned execution session batches
eager work until blocking observation or `Engine::checkpoint`; `try_read`
neither submits nor waits. Isolated capture and immutable engine-associated
plan replay are Experimental. Untimed replay caches one simultaneously
submittable command; read-only Matrix inputs can be rebound under exact semantic,
ownership, and no-alias validation, invalidating that cache. Timed replay uses
an independently owned query pair per submission. Mutable output slots, general
semantic value identity, calibrated clocks, and non-compute graph nodes remain
Planned.
Planned APIs and scaffold modules are not capability claims.

The host-domain Audio and Cryptography slices are Experimental. `Audio` composes
checked planar FP32 Matrix storage with sample-rate and channel-layout
semantics; private Symphonia WAV/FLAC/MP3 decode, OA-derived WAV-F32
encode/save, schema-owned Audio DSP, typed semantic dispatch, and CPAL-backed
capture/player sessions are implemented checkpoints. CPU Keccak-f[1600],
SHAKE-128/256, KMAC-256, typed hashes, and arbitrary-leaf Merkle proofs pass
donor KAT and property tests. Borrowed secure memory, CPU ML-DSA-65, U8 Matrix,
and schema-owned Vulkan batch SHAKE/Keccak/Merkle are implemented checkpoints.
Cross-backend Audio device qualification, compressed streaming encode, and
device-side ML-DSA remain Planned.

The complete 50-operation donor tensor-native Image surface is Experimental. `oa::image` has
schema-owned resize, crop, flip, rotate, pad, center-crop, remap, affine-warp,
and perspective-warp FP32 NCHW/CHW kernels, preserves Image layout,
format, and semantic graph identity, and passes independent odd-shape host
oracles on the recorded Intel/Mesa device. Clean Vulkan qualification remains
blocked by the pre-existing Philox/Dropout `shaderInt64` feature mismatch.
Typed color/resize-normalize/segmentation compositions and JPEG/PNG/WebP/BMP/TGA
one-shot codecs also pass their focused tests; additional layouts and image
autograd remain Planned.

The donor `FnDetection` surface is Experimental in `oa::vision`: schema-owned
pairwise IoU, deterministic NMS, classification confusion, binary-mask counts,
three-stage detection AP/mAP, and replay-safe segmentation metrics pass donor
vectors and local hardware oracles. Atomic U32 accumulators have explicit
collision ownership and schema-owned clear stages for repeatable plan replay.
Clean Vulkan qualification shares the pre-existing Philox/Dropout
`shaderInt64` blocker; tracking, typed detection values, and cross-device
qualification remain Planned.

The first ML training seed is Experimental. U32 class/token-index Matrix
storage, FP32 Linear, Embedding, LayerNorm, causal multi-head attention, GELU,
differentiable residual addition, TransformerBlock, and stacked Elman Rnn, zero-copy
differentiable reshape, stable mean cross-entropy, a thread-affine consumed
GradientTape, finite-difference checked parameter gradients, deterministic
repeated-index scatter-add, complete LayerNorm adjoints and BPTT, and
out-of-place AdamW updates pass the serial local hardware contract. Object-safe
`Module` composition adds
constructor-owned recursive parameter/buffer/child registration, deterministic
dotted traversal, duplicate-identity rejection, train/eval propagation, and a
composed character-model optimizer proof. The canonical Char-Transformer row
matches the C++ topology, 22-tensor and 10,875-scalar parameter contract, final
loss/accuracy regime, and exact greedy continuation. The standalone canonical
semantic graph data model and its core operation-contract vocabulary are now a
direct donor-backed structural port. Matrix schemas generate truthful OARS
compatibility contracts, Matrix dispatches record typed semantic values and
metadata-view lineage, and captured plans validate direct executable ownership.
These compatibility hashes are not presented as OA donor hashes. The private
donor-backed DNN analyzer consumes that graph, uses generated compatibility
roles, and applies exact donor-qualified FP32 inference QKV projection+bias and
Linear+Linear+SwiGLU gate/up replacements while preserving many-to-one semantic
provenance and eliminated-intermediate lifetime evidence.
Unqualified and training candidates retain source execution with explicit
fallback evidence.
Current ML kernels record generated contracts and semantic attributes.
Schema-owned Philox uniform/normal and inverted Dropout preserve explicit
64-bit seeds; capture selects replay variants with private device counters, and
the Dropout adjoint regenerates the exact forward mask. A Rust-native Dropout
module provides recursive train/eval behavior. Private
in-place AdamW state/parameter/moment writes produce fresh semantic SSA values
aliasing their prior versions while retaining the same physical storage;
captured optimizer nodes therefore have schema-owned provenance. Reached
`GradientTape` nodes now attach their forward
outputs and contiguous backward operation ranges to the captured semantic
graph. Additional physical OaDna/DNN providers, shared/serialized RNG streams, broader transient allocation policy,
live training sessions, and most of the OA ML
operation/module/shader catalog remain unported. A donor-backed `TrainingLoop`
now owns exact eager/captured completion, epoch/work accounting, loss metrics,
ordered callback control, device timing, automatic prepare/record capture,
requested recapture, source-preserving whole-program capture fallback, and
transactional command pre-recording before the first replay, plus conservative
allocation-ordinal stable frames after eager warm-up. Progress, summary, CSV,
validation, checkpoint/restore-best, early-stop, phase, and learning-rate-schedule
callbacks are connected. The eager iterator accepts the object-safe optimizer
contract; SGD, Adam, AdamW, and Muon use donor-backed GPU updates, while Adam,
AdamW, exact no-momentum SGD, and Muon persistence share the native `.oam`
codec. Generalized autograd, recurrent streaming state, live training control,
captured device-state Muon, and the remainder of the NLP suite remain Planned. The canonical
Char-RNN and Char-Transformer rows complete their exact 300-step
corpus/sampler/model workloads, reach the C++ loss and accuracy gates, and
reproduce their fixed-prompt greedy text.
Its current wall time is not performance parity.

## Required baseline

Run the narrowest relevant proof followed by:

```bash
python3 -m unittest discover -s test/py -v
python3 tools/gen/fn/generate.py --check
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

Use the linked Rust, Vulkan, kernel, validation, documentation, Git, and rule
contracts for their additional scoped gates. Preserve unrelated user work.
