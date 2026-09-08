# OA Rust Execution Session

**Status:** Experimental private runtime

**Updated:** 2026-09-07

Each `Engine` owns one private `ExecutionSession`. It is the mutable eager
recording boundary; it does not own or wrap the Vulkan device, queue, allocator,
pipelines, retirement service, or logger.

## Recording

A non-empty domain operation validates and lowers into a borrowed
`ComputeDispatch`. The session transactionally resolves that dispatch into an
owned executable-graph snapshot, retains every referenced buffer, and marks
written storage as recorded. Lowering returns the semantic output immediately.

Successive operations can consume recorded outputs because their graph nodes
reference the same retained storage. No host wait or Vulkan submission occurs
during ordinary operation construction. Empty operations record nothing and
produce immediately ready empty storage.

## Submission boundaries

Two boundaries submit pending eager work:

- blocking host observation flushes only when the observed storage is still
  recorded, then waits for its exact submitted event;
- `Engine::checkpoint` submits the entire pending batch and returns that exact
  event, or records an empty checkpoint when no eager work is pending.

Submission joins pending snapshots into one hazard-planned executable graph,
records one primary command buffer, advances the engine timeline once, and
transfers command and resource lifetime to asynchronous retirement. Every
written storage in the batch retains a clone of the resulting event.

`try_read` treats recorded work and incomplete submitted work as `NotReady`. It
never flushes, submits, or waits. Dropping a matrix, event, session, or engine
never submits or waits. A value that outlives the public `Engine` retains the
same private engine state, allowing a later blocking observation to flush its
producer without creating another runtime owner.

## Failure behavior

Dispatch resolution fails transactionally before the session changes.
Preflight, command recording, submission, epoch exhaustion, and retirement
failures are returned at the flush/checkpoint boundary. Once a pending batch
has been removed for a failed submission, each written storage retains a
persistent production-failed readiness state; later observation cannot expose
its unproved contents as ready.

`Drop` only releases host-owned pending graphs and buffers. It does not attempt
fallible recovery or surprise execution.

## Tests

Private graph-planner invariants remain inline under `#[cfg(test)]`. Public
integration tests under `tests/` prove that chained operations need no manual
submission, `try_read` remains non-blocking before submission,
`Engine::checkpoint` submits a pending batch, results survive public-engine
drop, and hardware output matches independent host values. Core,
synchronization, and GPU-assisted validation are separate hardware gates.

## Capture transfer

Capture temporarily marks the engine session as isolated and requires no
pending eager work. Nested capture, blocking observation, and submission are
rejected while the scope is active. Success transfers the executable graph and
written-storage readiness bindings into an engine-associated immutable
`ExecutionPlan`; error or unwind aborts captured work and restores eager
recording without submission.

The remaining reusable-execution dependency is semantic operation/value
identity plus compiled command caching and stable mutable input slots.
