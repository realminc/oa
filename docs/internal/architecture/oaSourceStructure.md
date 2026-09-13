# OA Rust source and module structure

**Status:** Architecture reference

**Updated:** 2026-09-12

**Authority:** [OA Rust Architecture](oaArchitecture.md)

**Donor references:** OA C++ `docs/internal/core/oaSourceStructure.md`,
`docs/internal/core/oaMatrix.md`, and `docs/internal/core/oaFnMatrix.md`

This document applies the canonical ownership model to Rust files and public
paths. It replaces the donor's include-tree rules; it does not copy the C++
directory hierarchy.

## Public topology

```text
oa
├── core       foundational values, metadata, errors, and host primitives
├── runtime    Engine, Event, plans, and execution evidence
├── matrix     stateless numerical operations
├── image      still-image codecs and transformations
├── audio      audio values, codecs, DSP, capture, and output sessions
├── video      video-frame values, codecs, containers, and capture sessions
├── media      cross-track source, clock, transport, and playback sessions
├── vision     detection, tracking, segmentation, and interpretation
├── render     Texture, Mesh, Material, Scene, Renderer, and Presenter
├── ml         ML values, modules, operations, and training sessions
├── ui         focused interaction and diagnostic presentation when admitted
├── plot       retained diagnostic figures and artists when admitted
└── cryptography  hashing, secure memory, signatures, and PQC operations
```

This is a responsibility graph, not a claim that every listed module ships.
The compatibility ledger and subsystem documents own current status.

`audio`, `video`, and `image` remain direct public domains. `media` composes
them when a source contains synchronized tracks or a shared transport clock;
it is not their superclass or required namespace prefix. `vision` consumes
Image and VideoFrame values. `render`, UI, and Plot consume typed values or
Texture views without taking ownership of their source domains.

## Dependency direction

```text
core values
  ├─> matrix/image/audio/video operations
  ├─> vision interpretation
  └─> render resources

audio + video
  └─> media track coordination

Image + VideoFrame
  ├─> vision
  └─> Texture -> render -> presentation/UI/Plot

runtime services
  <─ borrowed by every device-backed value or session
```

`Engine` remains the sole execution owner. A domain session borrows it; a
module must not create a second runtime, allocator, scheduler, or presentation
owner. Cycles are resolved through backend-neutral values, explicit adapter
traits, or private lowering descriptions rather than upward imports into Core.

## Compute implementation map

The Matrix, graph, DNN, shader, and Vulkan files are successive stages of one
operation route. They are not peer implementations and must not call around one
another:

```text
operation schema
  -> generated function in the owning operation module and OperationContract
  -> Matrix validation and output-value construction
  -> SemanticGraph operation/value SSA
  -> private DNN/OaDna analysis and semantic lowering
  -> ExecutableGraph nodes and semantic provenance
  -> generated KernelId and embedded ShaderArtifact
  -> runtime::vk pipeline lookup, command recording, and submission
```

The concrete file ownership is:

| Path | Sole responsibility |
| --- | --- |
| `core/matrix.rs` | The `Matrix` semantic value, checked metadata, storage handle, and observation boundary. It owns no stateless math operation. |
| `matrix.rs` and `matrix/` | The public `oa::matrix` facade for Core numerical operations, their validation, output construction, autograd attachment, and first lowering request. Schema-owned families use names such as `elemwise.gen.rs`, `reduce.gen.rs`, and `blas.gen.rs`. |
| `image.rs` and `image/` | The public still-image facade. `geometric`, `pixel`, `filter`, and `color` own typed transforms; `codec.rs` plus `codec/` own one-shot host codecs without moving Image storage or semantics into Vision. |
| `video.rs` and `video/` | The public Video facade. `frame.rs` owns packed-frame semantics, `nal.rs` bounded elementary-stream operations, `capability.rs` backend-neutral device evidence, `demux.rs` the bounded MP4 packet-source session, and `mux.rs` the streaming MP4 sink session. |
| `runtime/vk/video.rs` and `runtime/vk/video/` | The private Vulkan Video session and shared image/bitstream/synchronization implementation. Codec children such as `video/av1.rs` own only codec-specific StdVideo lowering and transactional DPB plans; they do not own sessions, queues, submission, or public parsing. |
| `vision.rs` and `vision/` | The public interpretation facade. Families such as `detection` consume semantic Matrix, Image, or VideoFrame values but do not own those value types, codecs, or transforms. |
| `cryptography.rs` and `cryptography/` | The public `oa::cryptography` facade. General CPU primitives remain at the facade root, schema-owned device hashing lives in `hash`, secret-bearing ML-DSA lives in `pqc`, and `SecureBuffer` remains a host-only value. |
| `cryptography/hash.rs` and `cryptography/hash/lowering.rs` | Public batch-hash signatures and their private semantic-to-executable lowering. Multi-level Merkle reduction remains one semantic operation. |
| `ml/matrix.rs` and `ml/matrix/` | The public `oa::ml::matrix` facade for the ML-owned C++ `FnMatrix` extension. Operation-family wrappers live here; this directory owns neither loss nor optimizer lowering. |
| `ml/loss.rs` and `ml/loss/` | The public `oa::ml::loss` operation facade corresponding to C++ `FnLoss`; generated loss families remain adjacent to this owner. |
| `ml/policy.rs` | The public `oa::ml::policy` operation facade and `PolicyResult`. Composite policy functions reuse Core Matrix operations inside one private runtime lowering transaction; they own no policy-only kernels or graph. |
| `ml/environment.rs` and `ml/environment/session.rs` | Checked environment values and graph-native transforms plus the native `Environment` behavior and `EnvironmentExecution` lifecycle. The session borrows the sole Engine recorder as a movable transaction; it owns no second graph, queue, allocator, or scheduler. |
| `ml/collector.rs` | Borrowed same-device categorical rollout collection. One complete horizon becomes one Environment transaction and exact Event without an implicit wait. |
| `ml/evaluation.rs` | One-shot deterministic policy evaluation and its explicit telemetry boundary: one horizon, one submission/wait, and three compact rollout readbacks. |
| `ml/actor_critic.rs` | The environment-neutral discrete Actor-Critic behavior contract and default seeded two-tower categorical MLP. It is a registered Module composition, not an RL runtime or operation facade. |
| `ml/lowering.rs` and `ml/lowering/` | Transitional private lowering partitioned into `matrix`, `loss`, `optim`, and shared semantic-dispatch mechanics. Each owner is deleted as schema-generated family implementations land; this is not a public facade or kernel registry. |
| `ml/autograd.rs` | The public autograd facade. It exports `GradientTape`, not concrete gradient nodes or backward-kernel helpers. |
| `ml/autograd/tape.rs` | Thread-local tape selection, saved-node traversal, version preflight, gradient accumulation, and semantic forward/backward provenance. It records work but never submits or waits. |
| `ml/autograd/matrix.rs` and `ml/autograd/loss.rs` | Private forward-operation attachment facades grouped by semantic owner; they do not own public operation signatures. |
| `ml/autograd/matrix/*.gen.rs` and `ml/autograd/loss/*.gen.rs` | Schema-generated attachments when saved Matrix fields and node construction are mechanical. |
| `ml/autograd/matrix/{linear,embedding,norm,recurrent,attention}.rs` | Explicit family attachments for compound saved state; the schema still owns their exhaustive `manual` policy and family placement. |
| `ml/autograd/node.rs` | The current compact private saved-value record. As coverage grows, family records split below `ml/autograd/matrix/` and `ml/autograd/loss/`; no public node catalog is introduced. |
| `ml/optimizer.rs` and `ml/optim.rs` | `optimizer.rs` owns object-safe optimizer policy plus stateful SGD/Adam/AdamW/Muon state. The public `optim` operation facade owns stateless optimizer-adjacent transformations such as global gradient-norm clipping. Optimizer dispatches do not become Matrix-Core operations. |
| `ml/training.rs` and `ml/training/` | Public training umbrella over the iterator, immutable captured program, callbacks, schedules, and algorithm coordinators. These files share one lifecycle and do not create a second graph or runtime owner. |
| `ml/training/rollout.rs` | Alternating Collect/Update coordinator over one retained RolloutBuffer and the ordinary ItTraining lifecycle. |
| `ml/training/ppo.rs` | Complete caller-driven categorical PPO collection and full-batch update coordinator over ActorCritic and ItRolloutTraining. |
| `ml/training/dqn.rs` | Environment-neutral DQN update coordinator over caller-owned Modules, Optimizer, and ReplayBuffer. It composes `ItTraining`; it owns no second trainer base, graph, allocator, or queue. |
| `ml/training/sac.rs` | Fixed-temperature SAC coordinator with separate actor and twin-critic optimizer lifecycles over one replay and Engine owner. |
| `ml/training/target.rs` | Private preflighted exact named-parameter synchronization shared by off-policy trainers. |
| `ml/training/schedule.rs` | Pure donor-backed step/metric-to-learning-rate policies. Policy mutation passes through the callback context's `Optimizer` behavior contract; concrete state remains in its optimizer owner. |
| `ml/training/callbacks.rs` and `ml/training/callbacks/` | Built-in presentation and control policies grouped below one callback facade. CSV, validation, checkpoint, phase, early-stop, and schedule families consume completed snapshots and may request cooperative stop, exclude external wall time, persist exact optimizer state, or update exposed optimizer policy. |
| `ml/checkpoint.rs` and `ml/checkpoint/` | Translation between a registered live Module plus sealed `CheckpointOptimizer` owner and the native model-file representation, with best/latest path and rotation policy. It owns host observation and engine upload boundaries, not wire parsing. |
| `ml/model_file.rs` | Private native `.oam` wire codec and integrity authority. A future public `oa::ml::ModelFile` facade may expose this semantic artifact without renaming it Archive or duplicating the codec under `io`. External format translators remain separate operations. |
| `core/operation/generated.rs` | Stable backend-neutral operation contracts generated from the operation schemas. |
| `runtime/semantic_graph.rs` | Backend-neutral values, views, operation SSA, mutation versions, attributes, and autograd provenance. It contains no selected kernel identity or Vulkan handle. |
| `runtime/session.rs` and the private engine lowering scope | Transactional composition of multiple retained executable graphs beneath one generated semantic contract. Nested scopes defer to the outer owner; abandoned scopes roll back only their new work. |
| `runtime/dnn.rs` and `runtime/dnn/` | Private donor-compatible OaDna/DNN analysis and graph replacement. It consumes semantic operations and may replace their source executable nodes; it is not the public `ml` API and owns no second graph. |
| `runtime/executable_graph.rs` | Concrete executable nodes, hazards, resource lifetimes, and the many-to-many link back to semantic operations. Its current nodes execute on Vulkan, but their orchestration remains engine-owned rather than a second Vulkan runtime. |
| `runtime/shader.rs` and `runtime/shader/` | Target-independent generated kernel identity, embedded artifact bytes, reflection, and shader metadata. A generated kernel row owns its optional semantic-operation contract mapping. |
| `runtime/vk/` | Vulkan instance/device/queue/buffer/descriptor/pipeline/command implementation. It consumes executable nodes and shader artifacts without deciding public operation semantics. |
| `src/slang/` | Slang modules and kernel source. Shader bodies do not own Rust API validation or graph policy. |

Within `src/slang`, physical directories name semantic owners rather than
backend mechanics:

| Path | Shader responsibility |
| --- | --- |
| `core/` | Entry-point-free metadata and reusable support modules. Activation and storage formulas live in `core/math`; deterministic random primitives live in `core/rng`. |
| `matrix/<family>/` | Core Matrix operation kernels grouped as `elemwise`, `reduce`, `rng`, and `blas`. |
| `ml/nn/<family>/` | Neural-network entry points grouped by operation family, including `activation`, `attention`, `embedding`, `layer_norm`, `linear`, `rms_norm`, `rnn`, `rope`, `swiglu`, and `vq`. |
| `ml/loss/<family>/` and `ml/optim/<family>/` | Loss and optimizer kernels are grouped by semantic family. Forward/backward loss files retain the loss name (`<loss>/<loss>_forward.slang` and `<loss>/<loss>_backward.slang`); generic `forward.slang` and `backward.slang` basenames are forbidden. Optimizer families currently include `sgd`, `adam`, `adamw`, `muon`, and `grad_clip`, with descriptive basenames retained inside each owner. |
| `<domain>/<family>/` | Image, Vision, Audio, Cryptography, and later admitted domains use the same semantic-family rule. |

A reusable formula module is not an operation kernel. For example,
`core/math/activations.slang` owns scalar `sigmoid`, `silu`, and derivative
formulas, while `ml/nn/activation/*.slang` owns dispatchable forward and
backward operations. Every shipping entry-point source is schema-referenced;
empty directory markers and speculative shader stubs are forbidden.

A semantic-capable `KernelId` and its `OperationContract` are generated from the
same schema row. High-level code must not supply a free-form executable
operation name or independently pair a kernel with a contract. Lowering-only
DNN kernels deliberately have no one-to-one semantic contract; their executable
nodes carry the complete set of source semantic operation IDs instead.

`runtime/shader` therefore does not move below `runtime/vk`: Slang source,
reflection, embedded artifacts, and stable kernel identity are target-neutral
authorities even though Vulkan is the only admitted consumer today. Vulkan
pipeline and command objects remain below `runtime/vk`. Likewise,
`runtime/executable_graph` stays distinct from `runtime/semantic_graph`: one
semantic operation may decompose into several executable nodes, and one fused
node may own several semantic operations.

`fn`, `fns`, `fn_matrix`, and `kernels` are not Rust ownership directories.
The C++ `Fn*` prefix communicates a stateless namespace; the corresponding Rust
module already communicates that role. Checked-in generated Rust is named for
its operation family with the `.gen.rs` suffix and lives beside its facade.
Slang bodies remain under `src/slang`, while generated kernel identity and
artifacts remain under `runtime/shader`.

Autograd is paired by schema ownership, not by placing tape state in operation
files. A differentiable operation row owns its forward function, backward
operation identity, attachment policy, saved state, and tests. Generated
forward/backward wrappers live with the operation facade; generated private
attachments live with the corresponding autograd family. `core/autograd.rs`
is only a backend-neutral observer bridge from foundational Matrix operations
to an active ML tape, preserving the dependency direction from Core to ML.

## File rules

- `lib.rs` is the curated crate facade. It uses explicit re-exports and does
  not expose implementation modules by wildcard. Principal values and session
  types may have admitted root identity aliases; free operations do not.
- A top-level `<domain>.rs` file is that domain's public facade and may also own
  a compact implementation. A `<domain>/` directory holds cohesive private or
  public submodules once the facade would mix responsibilities.
- Files are named for domain concepts: `clip.rs`, `frame.rs`, `texture.rs`,
  `capture.rs`, or `player.rs`. Generic names such as `value.rs`, `types.rs`,
  `common.rs`, and `utils.rs` require a narrower owning concept.
- One-shot transformations use verbs such as `decode.rs`, `encode.rs`, or
  public functions such as `decode_file`. Stateful protocol owners use nouns
  such as `decoder.rs`, `encoder.rs`, and `player.rs` only when they own an
  actual session state machine.
- Public types use normal Rust PascalCase; functions, modules, and files use
  snake_case. C++ prefixes, member suffixes, and parameter-direction prefixes
  are not retained.
- Generated checked-in files use the repository's `.gen.rs` convention and
  remain schema-owned. Build-time generation writes below `OUT_DIR`.
- External weight containers are private adapters below `ml/weights`; explicit
  model translators own source-name/shape policy and emit the single native
  `.oam` codec. They are not placed in a generic `io` or `archive` namespace.

File placement and public path need not be identical. A private implementation
module may be explicitly re-exported through its domain facade. A principal
type may then be explicitly re-exported by `lib.rs`: `oa::AudioPlayer` and
`oa::audio::AudioPlayer` are the same definition. This preserves freedom to
split files without creating a second implementation or operation route.

## Value and operation pairing

The C++ `Fn*` split becomes an idiomatic Rust type/module pairing:

```text
C++                                      Rust
oa::Matrix                               oa::Matrix
oa::FnMatrix::matMulNt (Core)            oa::matrix::mat_mul_nt
oa::FnMatrix::gelu (ML extension)        oa::ml::matrix::gelu

oa::Image                                oa::Image
oa::FnImage::resize                      oa::image::resize

oa::Audio                                oa::Audio
oa::FnAudio::normalize                   oa::audio::normalize

oa::VideoFrame                           oa::VideoFrame
oa::FnVideo::<operation>                 oa::video::<operation>

oa::FnDetection::boxIou                  oa::vision::box_iou
```

C++ reopens the same `oa::FnMatrix` namespace from Core and ML headers. Rust
does not emulate namespace reopening: it preserves subsystem ownership through
`oa::matrix` and `oa::ml::matrix`. Loss, policy, flow, environment, image,
audio, video, vision, render, and cryptography operation families follow the same
rule and enter their owning lowercase domain module.

The semantic value and operation module have different jobs even when their
English names match. Constructors and convenience methods may forward to the
same schema-owned operation, but they never introduce another implementation.

## Root identity aliases and local abbreviations

The owning domain remains visible while the crate root preserves concise OA
type spelling:

```rust,ignore
use oa::{AudioPlayer, audio};

let first = AudioPlayer::open(&engine, config)?;
let second = audio::AudioPlayer::open(&engine, config)?;
```

Principal semantic values and lifecycle-bearing sessions are eligible for an
explicit root re-export. Supporting configuration, packet, descriptor, and
implementation types remain in their owning module unless repeated use proves
that a root alias improves the common surface. Stateless functions are always
called through their operation module.

Applications may choose short local aliases:

```rust,ignore
use oa::{audio as oaa, core as oac, ml as oaml, vision as oacv};

let player = oaa::AudioPlayer::open(&engine, config)?;
```

These aliases are ordinary Rust imports. OA does not create abbreviated public
modules, packages, features, or crates. Python may use equivalent local module
aliases while binding root and owning-module names to the same admitted class
object.

## Current Audio example

```text
src/rs/audio.rs              curated facade
src/rs/audio/clip.rs         finite planar Audio and channel metadata
src/rs/audio/codec.rs        private one-shot codec family facade
src/rs/audio/codec/decode.rs synchronous one-shot WAV/FLAC/MP3 decode
src/rs/audio/codec/encode.rs synchronous one-shot WAV-F32 encode/save
src/rs/audio/signal.rs       stateless signal operations and validation
src/rs/audio/transform.rs    stateless feature transforms and validation
src/rs/audio/lowering.rs     private semantic-to-physical dispatch assembly
src/rs/audio/encoder.rs      stateful PCM-S16 packet encoder
src/rs/audio/player.rs       incremental decode and output session
src/rs/audio/capture.rs      real-time input session
```

The signal and transform families enter through the shared operation schema;
their handwritten files own domain validation and lowering assembly rather
than a second Matrix-forwarding API. `audio.rs` explicitly re-exports admitted
functions at `oa::audio::*`. There is no public decoder session because the
one-shot codec operations own no lifecycle; incremental decoding remains
private to `AudioPlayer`. Do not add a redundant `audio/audio.rs` layer or
expose private family and codec names as additional public routes.

`AudioChannelLayout` remains supporting metadata at
`oa::audio::AudioChannelLayout`. The principal `Audio` value and the
lifecycle-bearing `AudioCapture`, `AudioEncoder`, and `AudioPlayer` sessions
receive root identity aliases; their configuration and packet types do not.

## Current Video example

```text
src/rs/video.rs              curated value and operation facade
src/rs/video/av1.rs          bounded OBU inventory and semantic sequence-header parsing
src/rs/video/av1/frame.rs    frame-header, reference-state, and tile-range parsing
src/rs/video/frame.rs        retained Image/Texture/host/native VideoFrame backing and metadata
src/rs/video/texture.rs      stateless Texture-to-VideoFrame adaptation
src/rs/video/nal.rs          CPU-only Annex-B split, emit, and parameter sets
src/rs/video/capability.rs   queried hardware/session capability separation
src/rs/video/decoder.rs      stateful H.264/H.265 hardware decode session
src/rs/video/demux.rs        bounded seekable MP4 packet-source session
src/rs/video/mux.rs          streaming H.264/H.265 MP4 sink and packet contract
src/rs/video/player.rs       composed local demux/decode/playback session
```

`oa::VideoFrame` is the root identity alias of `oa::video::VideoFrame`.
`NalUnit`, packet/container/time-base values, `VideoFrameTiming`, and color
descriptors remain supporting types in `oa::video`. The donor's stateless
`FnVideo` NAL utilities are free functions on that same module; they are not
duplicated at the root and do not create a Rust `FnVideo` facade. The
lifecycle-bearing `VideoDecoder`, `VideoDemuxer`, `VideoMuxer`, and
`VideoPlayer` receive root identity aliases; no forwarding implementation is
introduced. Their configuration, container, packet, codec, statistics, and
audio-track descriptors remain supporting types under `oa::video`.

`vision.rs` curates the admitted donor `FnDetection` operations from
`vision/detection.rs`. Their generated schema, executable lowering, and
independent oracles live with Vision; Image transforms and Video sessions do
not move underneath that module.

## Current ML training example

```text
src/rs/ml/training.rs             curated training facade
src/rs/ml/training/iterator.rs    mutable ItTraining lifecycle and metrics
src/rs/ml/training/program.rs     immutable captured TrainingProgram
src/rs/ml/training/callbacks.rs   built-in callback facade and presentation
src/rs/ml/training/callbacks/     CSV, validation, phase, and policy families
src/rs/ml/training/schedule.rs    pure learning-rate schedule policies
src/rs/ml/training/rollout.rs     shared Collect/Update lifecycle
src/rs/ml/training/ppo.rs         categorical PPO coordinator
src/rs/ml/training/dqn.rs         DQN replay and target-update coordinator
src/rs/ml/training/sac.rs         paired SAC actor and critic coordinator
src/rs/ml/training/target.rs      private exact target synchronization
```

This mirrors `ml/nn.rs` plus `ml/nn/`: the facade defines one subsystem and
curates its public surface, while the directory separates responsibilities.
`ItTraining` and `TrainingProgram` are not one type. The iterator owns mutable
step, epoch, callback, and timing state; the program owns one compiled replay
contract that may be retained by the iterator. `oa::ml::ItTraining` is an
identity re-export of `oa::ml::training::ItTraining`, not a forwarding type.

## SDK-owned Rust workloads

Concrete tutorial workloads remain below `sdk/rs` even when they need native
Rust types or GPU kernels:

```text
sdk/rs/mod.rs                              crate-connected SDK facade
sdk/rs/ml/rl/cart_pole.rs                  native vectorized CartPole session
sdk/rs/ml/rl/lunar_lander.rs               versioned Lunar workload facade
sdk/rs/ml/rl/lunar_lander/environment.rs   scalar episode/reward oracle
sdk/rs/ml/rl/lunar_lander/terrain.rs       checked deterministic terrain oracle
sdk/rs/ml/rl/lunar_lander/physics.rs       scalar dynamics/observation oracle
sdk/rs/ml/rl/lunar_lander/vector.rs        native vector Environment session
sdk/rs/ml/rl/lunar_lander/training.rs      task policy curriculum and evaluation evidence
sdk/rs/ml/alm.rs                           ALM SDK facade and temporal tokenizer
sdk/rs/ml/alm/prior.rs                     causal motion-token prior
sdk/rs/ml/alm/model.rs                     product ownership/persistence tree
sdk/rs/ml/alm/clip.rs                      native frozen CLIP text tower
sdk/rs/ml/alm/tokenizer.rs                 explicit canonical CLIP byte-BPE asset parser
sdk/rs/tutorials/ml/rl/lunar_lander_ppo.rs runnable teacher/raw-PPO workflow
sdk/rs/slang/ml/rl/cart_pole/reset.slang   schema-owned reset kernel
sdk/rs/slang/ml/rl/cart_pole/step.slang    schema-owned dynamics kernel
sdk/rs/slang/ml/rl/lunar_lander/reset.slang schema-owned Lunar reset kernel
sdk/rs/slang/ml/rl/lunar_lander/step.slang schema-owned Lunar dynamics kernel
```

The facade is available as `oa::sdk` so examples and tests consume one type
identity, while reusable `oa::ml` remains environment-neutral. SDK workloads
borrow the same Engine, record the same semantic and executable graphs, and use
the same generated shader registry. They do not own a second runtime, expose
raw Vulkan, or move task-specific policy into the ML library.

## Migration rule

Before moving a donor file or public name:

1. classify each contract as a value, operation, or session;
2. identify its storage and execution owner;
3. preserve semantic identity and failure behavior;
4. translate C++ namespaces into Rust modules, not classes or wrapper structs;
5. keep codec, Vulkan, OS, and third-party implementation types private;
6. prove one vertical slice before publishing a broad facade;
7. delete or supersede the old canonical route after verification.

The detailed spelling table is [C++ to Rust API translation](../porting/oaCppToRust.md).
