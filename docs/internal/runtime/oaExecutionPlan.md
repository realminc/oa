# OA Rust Execution Plan

**Status:** Experimental

**Updated:** 2026-09-08

`ExecutionPlan` is immutable engine-associated executable work produced by
`Engine::capture`. It is not a public graph builder and does not expose Vulkan
resources, kernel routing, barriers, or descriptor policy.

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

Each submission preflights and records the retained executable graph into a
fresh primary command buffer, submits asynchronously, and returns its exact
timeline `Event`. Captured output storage retains that event. Repeated
submission is legal and replaces output readiness with the newest replay event.
Dropping the plan or event never waits.

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

The plan retains the originating private engine handle, exact captured buffers,
kernel identities, push values, workgroups, and output readiness bindings. It
is deliberately neither `Clone` nor a mutable graph API.

The current baseline does not yet provide mutable stable input slots, semantic
operation/value identities, graph serialization, command-buffer caching,
shape polymorphism, concurrent replay, or cross-device submission. Those are
separate capabilities with their own contracts and tests.

## Acceptance evidence

External integration tests must prove successful capture, no submission before
replay, exact output readiness, dependent-node correctness, repeated replay,
foreign-engine rejection, dirty/empty/nested capture rejection, closure-error
restoration, and persistent failure for escaped aborted outputs. Hardware paths
run separately under core, synchronization, and GPU-assisted validation.
