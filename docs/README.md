# OA Rust documentation

The Rust implementation is at the architecture and repository-foundation
stage. Documentation distinguishes target contracts from verified behavior;
the presence of a source module or shader does not imply a shipped capability.

## Start here

- [OA Rust Architecture](internal/architecture/oaArchitecture.md) — canonical
  ownership, execution, module, value, and generation contracts.
- [Rust Port Roadmap](internal/architecture/roadmap/portRoadmap.md) — dependency
  order and acceptance gates.
- [OA Compatibility Ledger](internal/porting/oaCompatibility.md) — concepts
  preserved, redesigned, deferred, or rejected from the C++ implementation.
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
