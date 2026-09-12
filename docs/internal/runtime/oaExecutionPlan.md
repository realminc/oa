# OA Rust Execution Plan

**Status:** Experimental

**Updated:** 2026-09-09

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

The Engine's bindless storage heap is sized from queried update-after-bind
limits and capped at 262,144 buffers on integrated GPUs or 1,048,576 on
discrete GPUs, matching the donor's conservative device classes. If a driver
cannot allocate its advertised request, creation halves the request down to at
most 65,536 rather than failing immediately. This replaces the early
1,024-buffer prototype ceiling, which could not represent a complete 500-step
policy-evaluation graph. Descriptor slots remain retirement-owned and are never
recycled while submitted work can reference them.

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

`is_complete` queries the latest plan replay without waiting, and `wait`
completes that replay plus host retirement even when the submission's returned
`Event` was dropped. Completion of the latest same-queue timeline point proves
all earlier plan replays. Consuming `reset` performs that explicit wait and then
releases the plan; it is the Rust ownership adaptation of resetting a mutable
empty C++ plan shell. Ordinary destruction remains non-blocking.

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
The fallback count includes observable DNN provider and transient-alias
materialization fallbacks. DNN analysis runs before alias materialization: an
exact donor-qualified inference QKV or gate/up partition may replace three
source nodes with one multi-owner node; gate/up replacement additionally proves
that both eliminated projection intermediates have no external consumer or
unaccounted storage owner. Every rejected candidate retains its original source
interval and reason. Training-program capture disables these inference-only
replacements. The plan retains its original distinct storage when alias-arena
allocation or replacement hazard planning fails; the reason remains available
through `ExecutionPlan::alias_materialization_fallback_reason`.
Memory diagnostics also distinguish logical captured resources from distinct
physical post-alias buffers and report planned nontrivial alias groups plus
their potential logical-byte savings. These counts feed the training compiler
report; they are not allocator-residency claims.

## Executable evidence

`debug_report_json` emits the donor-compatible `oa.execution_graph.v3`
diagnostic shape. It contains normalized resources in first-access order,
inclusive lifetimes, executable nodes and their semantic owners, generated
kernel and dtype names, dispatch groups, access effects, RAW/WAR/WAW barriers,
exact current stage/access masks, cache-compilation state, and the most recent
submission timeline value. Kernel names and dtype vocabulary are generated
from the operation schema rather than formatted from Rust enum names.

The report is deliberately handle-free: it omits Vulkan handles, device
addresses, bindless descriptor indices, and push-constant payloads. Hashes are
diagnostic identities rather than a persistence ABI. The current backend has
one compute queue, direct kernel selection, no indirect dispatch, and no host
node in an executable plan. The bindless descriptor set belongs to the engine,
not the graph. The corresponding graph-owned descriptor, queue/fallback, and
barrier counts are therefore explicit zeros rather than inferred capabilities.

OARS alias arenas use one physical `VkBuffer` identity for all logical members.
Consequently the executable report exposes that arena as one resource and
leaves `alias_groups` empty; logical group membership and potential versus
materialized savings remain in `ExecutionPlan::diagnostics` and the training
compilation report. This differs mechanically from OA C++ alias views without
weakening the semantic-resource or synchronization evidence.

## Stable transient arenas

When capture runs inside a sealed stable-resource frame, the plan snapshots a
deterministic semantic value-to-resource table plus each resource's byte size
and inclusive first/last executable-node access. Semantic external values,
replay inputs, and explicitly observed outputs are never candidates. Rust
`Storage` owner counts must also be fully explained by the capture transaction;
an escaped Matrix or view therefore rejects that resource without guessing.

Qualified non-overlapping intervals are assigned by the donor's stable greedy
interval-coloring rule. Groups with at least two members are rebound
transactionally to one host-visible Vulkan storage-buffer arena sized to the
largest member. OA C++ currently creates distinct aliasing `VkBuffer` views over
one VMA allocation; OARS uses one private `VkBuffer` identity for the whole
group. This is a deliberate Rust-runtime adaptation: raw handles are not part of
the semantic or public contract, it consumes fewer bindless descriptors, and
the existing buffer-hazard planner can see every arena transition directly.

After rebinding, barriers are recomputed before publication. For the current
single compute queue, a previous member's last compute-shader storage access is
the producer and the next member's first compute-shader storage access is the
consumer. The barrier uses the exact read/write storage access masks; buffers
have no image layout, and queue-family ownership transfer is not required. The
recorded command retains the arena through timeline retirement. Replaced source
allocations bypass the general reuse pool and are destroyed after all proved
capture owners are released.

`alias_materialized_count` and `materialized_alias_savings` report logical
resource bytes replaced by arenas. They do not claim VMA block-byte or resident
memory reduction. Allocation or barrier-planning failure preserves the original
distinct-storage graph, records one fallback and its reason, and publishes an
otherwise valid plan. Semantic analysis, ownership qualification, and capture
contract failures remain errors rather than optimization fallbacks.

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

The current baseline does not yet provide mutable output bindings, persisted
graph serialization, shape polymorphism, concurrent host-thread submission, or
cross-device submission. Transient arenas are currently limited to the
host-visible Matrix placement used by the Vulkan baseline. Timed replay
deliberately records a fresh instrumented command and does not use the untimed
cache. Those are separate capabilities with their own contracts and tests.

## Acceptance evidence

External integration tests must prove successful capture, no submission before
replay, exact output readiness, dependent-node correctness, repeated replay,
foreign-engine rejection, dirty/empty/nested capture rejection, closure-error
restoration, and persistent failure for escaped aborted outputs. Hardware paths
run separately under core, synchronization, and GPU-assisted validation. Cache
tests additionally queue the same recording twice, rebind an input used by
multiple nodes, prove unchanged graph identity and exact output, reject alias,
shape, output, unrelated, and foreign-engine bindings, and drop the plan before
the final cached submissions retire. Alias tests prove inclusive lifetime
coloring, escaped-owner rejection, a shared graph arena, the added arena hazard,
reported logical savings, exact repeated replay, and separate clean core and
synchronization-validation runs. Lifecycle tests also drop the returned event,
wait through the plan, query completion, consume `reset`, and prove an
independently retained output remains valid.
Report tests additionally parse the JSON, prove deterministic normalized
resource and hazard identities, match the two-node RAW barrier to the recorded
compute/storage scopes, observe pre/post-submission lifecycle state, and reject
backend-private descriptor, address, and push-payload fields.
