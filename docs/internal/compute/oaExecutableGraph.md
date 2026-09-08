# OA Rust Executable Graph

**Status:** Experimental private foundation

**Updated:** 2026-09-08

The executable graph is OA's private owned snapshot between target-independent
`ComputeDispatch` lowering and Vulkan command encoding. It is retained by the
public immutable `ExecutionPlan`, but is not itself a public graph builder,
semantic graph, or autograd tape.

## Implemented contract

Graph construction resolves every borrowed storage binding against the
submitting engine and copies everything needed after lowering returns:

- stable operation and selected kernel identity;
- retained Vulkan buffer owners and declared read/write access;
- copied semantic push constants;
- direct workgroup dimensions.

Construction is transactional: an empty graph, foreign storage, or missing
buffer fails without recording or submission. Command preflight resolves every
pipeline, checks all three workgroup dimensions against the selected device,
prepends retained buffers' bindless indices, and validates the resulting push
payload against reflected shader ABI before beginning a command buffer.

## Buffer hazard planning

The planner retains the last relevant access for every live buffer across
unrelated nodes. Duplicate aliases in one node are merged into one access
state. It emits a buffer dependency only for a real conflict:

| Previous access | Current access | Dependency |
|---|---|---|
| write | read | RAW |
| read | write | WAR |
| write | write | WAW |
| read | read | none |

Several conflicts at the same node boundary share one `DependencyInfo`; each
resource still receives its own `BufferMemoryBarrier2`. For the current
compute-only, one-queue graph, both synchronization stages are
`COMPUTE_SHADER`, access masks come from the exact storage read/write
declarations, barriers cover the complete buffer, and queue-family indices are
ignored because no ownership transfer occurs.

This rule is deliberately narrow. Transfer, indirect, render, video, image
layout, subrange, and cross-queue dependencies must add their own node/access
vocabulary and synchronization proof instead of reusing a generic
compute-to-compute barrier.

## Current execution boundary

The eager lowerer creates one owned snapshot per non-empty operation and records
it into the engine's private execution session. Blocking observation or
`Engine::checkpoint` joins pending snapshots and encodes their dependent nodes
in one primary command buffer. The recorded command retains all referenced
buffers through exact timeline retirement.

Inter-submission ordering remains the current serialized timeline chain. Graph
state ends at submission; the timeline wait owns the next submission's memory
edge. Host observation waits for the exact producing event before mapped
readback.

## Testing boundary

Pure hazard and alias invariants live beside the private planner under
`#[cfg(test)]`; they are absent from normal library builds. Consumer-visible
API contracts and normal hardware behavior live under `test/rs/runtime/`, including
automatic observation flush and explicit checkpoint submission. The private
multi-node recorder additionally retains an ignored unit-level hardware oracle
until public capture makes that exact boundary externally constructible.

## Capture and replay

An isolated capture transfers the graph and its written-storage readiness
bindings into a public immutable `ExecutionPlan`. Submission validates engine
identity, flushes earlier eager work, records the retained graph, submits
without waiting, and attaches the exact event to every captured output. The
same plan can be replayed repeatedly.

An untimed plan records its primary command buffer once and shares it across
unchanged submissions with `SIMULTANEOUS_USE`. Exact retirement ownership keeps
that command and its buffers alive through every pending replay. A validated
read-only Matrix input rebind preserves stable captured-slot identity, replaces
every occurrence, recomputes hazards, and invalidates the cached command. Alias
introduction is rejected, so normalized graph structure and its diagnostic ID
remain stable. Mutable output bindings and general semantic value identity
remain incomplete. Vulkan timestamp queries attach to exact executable regions
and submission events rather than timing operation construction; timed replay
uses a fresh instrumented command rather than the untimed cache.

`Engine::submit_timed` allocates one independent two-query timestamp pool for a
single replay. Command recording resets it, writes at top-of-pipe immediately
before the graph and bottom-of-pipe immediately after it, and retains it through
the exact event and command retirement. Ordinary eager and plan submission stay
uninstrumented.
