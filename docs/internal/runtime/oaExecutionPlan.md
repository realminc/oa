# OA Rust Execution Plan

**Status:** Experimental

**Updated:** 2026-09-08

`ExecutionPlan` is engine-associated executable work produced by
`Engine::capture`. Its operation structure is immutable; admitted read-only
Matrix inputs may be rebound without exposing Vulkan resources, kernel routing,
barriers, or descriptor policy.

## Capture

```rust
let (plan, output) = engine.capture(|| {
    let product = matrix::mat_mul_nt(&input, &weight)?;
    matrix::add(&product, &bias)
})?;
```

Capture records ordinary domain operations through an isolated private
execution session. It never records a Vulkan command buffer, submits, or waits.
The current baseline requires the eager session to be empty before capture and
rejects nested and empty captures. A closure error aborts the isolated graph,
marks escaped captured outputs failed, and restores normal eager recording.
The same restoration occurs during unwinding; `Drop` does not execute work.

Capture outputs retain a captured readiness state. Non-blocking observation
returns `NotReady`; blocking observation and use by another operation fail with
`FailedPrecondition` until the plan is submitted. This prevents an unsubmitted
captured producer from silently entering an eager or second captured graph.

## Submission and replay

```rust
let event = engine.submit(&plan)?;
event.wait()?;
let values = output.read_f32()?;
```

Only the originating engine accepts a plan. Foreign-engine submission returns
`InvalidArgument` without changing the plan or its outputs. Pending eager work
on the originating engine is submitted first; the existing timeline dependency
orders plan work after it.

The first untimed submission preflights and records the retained executable
graph into a `SIMULTANEOUS_USE` primary command buffer. Unchanged submissions
reuse that recording and may be queued while an earlier replay is pending.
Captured output storage retains the newest exact timeline `Event`. Retirement
references keep a cached command and every bound buffer alive when the plan is
dropped; final completion returns the command to its originating pool without
waiting in `Drop`.

## Stable Matrix input bindings

```rust
let graph_id = plan.diagnostics().graph_id();
plan.bind_matrix_input(&captured_input, &next_input)?;
let event = engine.submit(&plan)?;
```

The captured Matrix remains the stable slot identity after repeated rebinding.
The replacement must belong to the same engine, match shape and dtype exactly,
and cannot alias another plan resource. Only graph resources whose every access
is read-only are admitted. Rebinding never submits or waits. Because bindless
descriptor indices are part of the recorded push payload, a changed binding
invalidates the cached command and the next untimed submission records it once;
later unchanged submissions reuse it. This avoids updating a descriptor still
referenced by pending work.

`ExecutionPlan::diagnostics` reports deterministic executable graph identity,
node and barrier counts, admitted input bindings, command recordings, cache
hits, successful submissions, rebindings, and fallbacks. Equivalent captures
normalize physical buffer identities; the graph identity includes operation
labels, stable kernel IDs, exact embedded SPIR-V bytes, logical alias structure,
push values, and workgroups. It is diagnostic identity, not a serialized ABI.
The fallback count is zero because the current runtime fails closed and has no
fallback route.

## Timed replay

```rust
let event = engine.submit_timed(&plan)?;
let duration = event.device_duration()?;
```

Timed submission instruments only that replay with one timestamp pair around
the complete executable graph. `device_duration` waits for the exact event;
`try_device_duration` returns `None` without waiting while it is incomplete.
Both reject an event produced by ordinary untimed submission.

Each timed replay owns a fresh query pool, so several replays may be queued
without reusing active queries. The event and retirement path retain the pool
until its commands complete. Duration conversion uses the selected compute
queue's timestamp valid-bit width and device timestamp period. It does not
measure host recording, submission, wait, or readback and does not correlate
the device clock to a host clock.

## Ownership and limitations

The plan retains the originating private engine handle, current input bindings,
captured outputs, kernel identities, push values, workgroups, and cached command
ownership. It is deliberately neither `Clone` nor a mutable graph editor.

The current baseline does not yet provide mutable output bindings, general
semantic value identities, graph serialization, shape polymorphism, concurrent
host-thread submission, or cross-device submission. Timed replay deliberately
records a fresh instrumented command and does not use the untimed cache. Those
are separate capabilities with their own contracts and tests.

## Acceptance evidence

External integration tests must prove successful capture, no submission before
replay, exact output readiness, dependent-node correctness, repeated replay,
foreign-engine rejection, dirty/empty/nested capture rejection, closure-error
restoration, and persistent failure for escaped aborted outputs. Hardware paths
run separately under core, synchronization, and GPU-assisted validation. Cache
tests additionally queue the same recording twice, rebind an input used by
multiple nodes, prove unchanged graph identity and exact output, reject alias,
shape, output, unrelated, and foreign-engine bindings, and drop the plan before
the final cached submissions retire.
