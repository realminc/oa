# OA Rust documentation

The Rust implementation has an Experimental one-device compute foundation,
schema-generated `f32` elementwise slice, and an `i32` Matrix-add dtype proof.
Documentation distinguishes target contracts from verified behavior;
the presence of a source module or shader does not imply a shipped capability.

## Start here

- [OA Rust Architecture](internal/architecture/oaArchitecture.md) — canonical
  ownership, execution, module, value, and generation contracts.
- [Rust Port Roadmap](internal/architecture/roadmap/portRoadmap.md) — dependency
  order and acceptance gates.
- [OA Compatibility Ledger](internal/porting/oaCompatibility.md) — concepts
  preserved, redesigned, deferred, or rejected from the C++ implementation.
- [Compute Architecture](internal/compute/oaCompute.md) — current executable
  path, ownership, synchronization, and graph boundary.
- [Executable Graph](internal/compute/oaExecutableGraph.md) — owned dispatch
  snapshots, resource hazards, and current batching boundary.
- [Compute Kernel System](internal/compute/oaComputeKernel.md) — schemas,
  attributes, dtype tokens, storage helpers, shader ABI, and validation.
- [Performance Evidence](internal/performance/oaPerformance.md) — benchmark,
  timing, profiling, and claim protocol.
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
