# OA Rust documentation

The Rust implementation has an Experimental one-device compute foundation,
schema-generated Matrix operations, and differentiable FP32 Linear, U32
Embedding, stacked Elman Rnn, reshape, cross-entropy, and AdamW training slices.
Recursive Rust-native Module ownership supplies deterministic parameter and
buffer traversal for composed models. Fixed-shape `TrainingProgram` capture
replays the complete Char-Transformer forward/backward/AdamW graph through one
cached Vulkan command with stable input and optimizer storage. The donor-backed
`ItTraining` now connects eager and captured steps to exact completion,
epoch/work accounting, loss metrics, ordered callbacks, GPU timestamps, and
automatic fixed-shape warm-up/capture/replay.
Capture now validates schema-classified optimizer replay state before source
commit and exposes ordered compilation-stage evidence without leaking kernel or
Vulkan policy.
The first Image value slice now validates dense Matrix rank, axis layout, and
logical channel format while preserving one storage owner. Image operations,
native video planes, codecs, and rendering sessions remain Planned rather than
placeholder APIs. The first packed `VideoFrame` path now retains one checked
Image together with presentation timing and source color metadata.
Documentation distinguishes target contracts from verified behavior;
the presence of a source module or shader does not imply a shipped capability.

## Start here

- [OA Rust Architecture](internal/architecture/oaArchitecture.md) — canonical
  ownership, execution, module, value, and generation contracts.
- [Rust Source Structure](internal/architecture/oaSourceStructure.md) — module
  ownership, facade/re-export rules, domain dependencies, and file naming.
- [Rust Port Roadmap](internal/architecture/roadmap/portRoadmap.md) — dependency
  order and acceptance gates.
- [OA Compatibility Ledger](internal/porting/oaCompatibility.md) — concepts
  preserved, redesigned, deferred, or rejected from the C++ implementation.
- [C++ to Rust API Translation](internal/porting/oaCppToRust.md) — `Fn*` to
  lowercase modules, root identity aliases, naming, and session mapping.
- [Values and Storage](internal/core/oaValues.md) — Buffer backing, Matrix,
  Image, Audio, VideoFrame, Texture, and checked zero-copy relationships.
- [Media Boundary](internal/media/oaMedia.md) — Audio/Video sibling domains,
  shared transport and clock coordination, sinks, and session lifecycle.
- [Vulkan Linear Math](internal/vlm/oaVlm.md) — packed host spatial values,
  fixed conventions, failure behavior, and current verification.
- [Audio](internal/audio/oaAudio.md) — planar FP32 value semantics,
  schema-owned DSP, private codecs, and capture/player/encoder sessions.
- [Image](internal/image/oaImage.md) — dense Matrix composition, checked layout
  and channel semantics, and the boundary before image operations.
- [Video](internal/video/oaVideo.md) — packed frame retention, timing and color
  metadata, Annex-B operations, and the native-plane/session admission order.
- [Vision](internal/vision/oaVision.md) — direct `FnDetection` translation,
  interpretation ownership, and the first operation acceptance gate.
- [Cryptography](internal/cryptography/oaCryptography.md) — CPU primitives,
  secure memory, ML-DSA-65, Vulkan batch hashing, secret-data boundaries, and
  current verification.
- [vkPQC roadmap](internal/architecture/roadmap/vkPqcRoadmap.md) — cuPQC scope,
  ML-DSA/ML-KEM dependency order, source ownership, and acceptance gates.
- [GPU secret execution](internal/cryptography/oaGpuSecretSecurity.md) — capture,
  debugger, protected-memory, confidential-compute, entropy, and erasure threat
  model for future secret-bearing device operations.
- [ML Documents](internal/ml/README.md) — active Rust contracts and explicit
  migration/defer decisions for OA's larger ML documentation set.
- [ML Foundation](internal/ml/oaMl.md) — current differentiable training slices,
  ownership, failure behavior, and remaining NLP prerequisites.
- [Native Model Files](internal/ml/oaModelFile.md) — `.oam` persistence,
  integrity, checkpoint mapping, and OA C++ interoperability.
- [Elman RNN](internal/ml/oaRnn.md) — GPU whole-sequence scan, complete BPTT,
  limitations, and numerical evidence.
- [NLP Tutorial Suite](internal/ml/oaNlpSuite.md) — canonical 300-step Char-RNN
  and Char-Transformer training gates
  workload and current correctness/performance evidence.
- [Reinforcement Learning](internal/ml/oaRl.md) — checked environment value
  contracts and the dependency order before RL execution and algorithms.
- [Compute Architecture](internal/compute/oaCompute.md) — current executable
  path, ownership, synchronization, and graph boundary.
- [Executable Graph](internal/compute/oaExecutableGraph.md) — owned dispatch
  snapshots, resource hazards, and current batching boundary.
- [Compute Kernel System](internal/compute/oaComputeKernel.md) — schemas,
  attributes, dtype tokens, storage helpers, shader ABI, and validation.
- [CUDA Rust Assessment](internal/compute/oaCudaRust.md) — dated research on
  NVIDIA's SIMT/tile Rust tracks and the write-domain/preflight ideas adopted
  into OARS planning.
- [Performance Evidence](internal/performance/oaPerformance.md) — benchmark,
  timing, profiling, and claim protocol.
- [Memory Comparison](internal/performance/oaMemoryComparison.md) — matched
  Rust std/OARS evidence, C++ porting references, and current admission status.
- [VLM Comparison](internal/performance/oaVlmComparison.md) — matched
  OA C++/GLM/OARS workload, oracle, and current optimization evidence.
- [Runtime Logging](internal/runtime/oaLog.md) — engine-owned sinks, component
  vocabulary, macro routing, and failure-bearing lifecycle.
- [Execution Session](internal/runtime/oaExecutionSession.md) — private eager
  recording, automatic submission boundaries, and result readiness.
- [Execution Plan](internal/runtime/oaExecutionPlan.md) — isolated capture,
  immutable replay, engine identity, and completion behavior.
- [Device Timing](internal/runtime/oaDeviceTiming.md) — explicitly instrumented
  plan replay, event readback, timestamp wrap, and capability behavior.
- [Initial Rust Architecture](internal/research/initialRustArchitecture.md) —
  original design notebook retained as Research, not current authority.

## Status vocabulary

- **Canonical:** current cross-module target contract.
- **Shipped:** implemented and verified by named evidence.
- **Experimental:** implemented but unstable or incompletely qualified.
- **Planned:** accepted direction with a roadmap dependency and acceptance gate.
- **Research:** alternatives and evidence without an implementation promise.

Subsystem documentation is added when its roadmap dependency becomes active.
The C++ OA documentation remains historical and behavioral evidence; it is not
copied wholesale into this repository.
