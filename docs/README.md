# OA Rust documentation

The Rust implementation has an Experimental one-device compute foundation,
schema-generated Matrix operations, and differentiable FP32 Linear, U32
Embedding, stacked Elman Rnn, reshape, cross-entropy, and AdamW training slices.
Recursive Rust-native Module ownership supplies deterministic parameter and
buffer traversal for composed models. Fixed-shape `TrainingProgram` capture
replays the complete Char-Transformer forward/backward/AdamW graph through one
cached Vulkan command with stable input and optimizer storage. The donor-backed
`TrainingLoop` now connects eager and captured steps to exact completion,
epoch/work accounting, loss metrics, ordered callbacks, GPU timestamps, and
automatic fixed-shape warm-up/capture/replay.
Capture now validates schema-classified optimizer replay state before source
commit and exposes ordered compilation-stage evidence without leaking kernel or
Vulkan policy.
Documentation distinguishes target contracts from verified behavior;
the presence of a source module or shader does not imply a shipped capability.

## Start here

- [OA Rust Architecture](internal/architecture/oaArchitecture.md) — canonical
  ownership, execution, module, value, and generation contracts.
- [Rust Port Roadmap](internal/architecture/roadmap/portRoadmap.md) — dependency
  order and acceptance gates.
- [OA Compatibility Ledger](internal/porting/oaCompatibility.md) — concepts
  preserved, redesigned, deferred, or rejected from the C++ implementation.
- [Vulkan Linear Math](internal/vlm/oaVlm.md) — packed host spatial values,
  fixed conventions, failure behavior, and current verification.
- [Audio](internal/audio/oaAudio.md) — planar FP32 value semantics, private
  codec boundary, WAV-F32 sink, and current verification limits.
- [Crypto](internal/crypto/oaCrypto.md) — CPU Keccak/SHAKE/KMAC, typed hashes,
  Merkle proofs, secret-data boundary, and current verification.
- [ML Documents](internal/ml/README.md) — active Rust contracts and explicit
  migration/defer decisions for OA's larger ML documentation set.
- [ML Foundation](internal/ml/oaMl.md) — current differentiable training slices,
  ownership, failure behavior, and remaining NLP prerequisites.
- [Elman RNN](internal/ml/oaRnn.md) — GPU whole-sequence scan, complete BPTT,
  limitations, and numerical evidence.
- [NLP Tutorial Suite](internal/ml/oaNlpSuite.md) — canonical 300-step Char-RNN
  and Char-Transformer training gates
  workload and current correctness/performance evidence.
- [Compute Architecture](internal/compute/oaCompute.md) — current executable
  path, ownership, synchronization, and graph boundary.
- [Executable Graph](internal/compute/oaExecutableGraph.md) — owned dispatch
  snapshots, resource hazards, and current batching boundary.
- [Compute Kernel System](internal/compute/oaComputeKernel.md) — schemas,
  attributes, dtype tokens, storage helpers, shader ABI, and validation.
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
