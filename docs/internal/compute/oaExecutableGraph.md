# OA Rust Executable Graph

**Status:** Experimental private foundation; broader node and scheduling model is Planned

**Updated:** 2026-09-10

The executable graph is the private retained layer between backend-independent
semantic compilation and Vulkan command encoding. A public immutable
`ExecutionPlan` owns it, but it is not a public graph builder, the semantic
graph, an autograd tape, or a DNN-provider API.

## Implemented node contract

Each current node retains:

- the semantic operation owners and selected generated kernel identity;
- engine-validated buffer owners and `Read`, `Write`, or `ReadWrite` access;
- copied typed push constants;
- direct workgroup dimensions.

Construction is transactional. Empty graphs, foreign storage, invalid resource
ownership, unavailable pipelines, reflected ABI mismatch and out-of-range
dispatch dimensions fail before submission. Recorded commands retain all
referenced resources until their exact timeline epoch retires.

The Vulkan recorder additionally resolves each node into a private
`PreparedDispatch` before command-buffer recording begins. Current preflight
proves exact pipeline presence, legal nonzero direct group counts, descriptor
index and typed push encoding, and reflected push size. The target prepared
value also carries capability, binding-range, alias, access, workspace, and
physical write-domain evidence. Failure to prepare any node aborts the complete
recording transaction.

The target node vocabulary additionally needs transfer/copy/fill, direct and
indirect compute, image processing, render, video, presentation, external
semaphore and host-boundary nodes. Each kind carries its own access, layout,
queue and capability contract; it cannot inherit a generic compute-buffer
barrier by convenience.

## Current hazard plan

The planner retains the last relevant access for every aliased buffer across
unrelated nodes. Bindings that alias in one node merge into one access state.

| Prior | Current | Dependency |
|---|---|---|
| write/read-write | read/read-write | RAW or combined |
| read/read-write | write/read-write | WAR or combined |
| write/read-write | write/read-write | WAW or combined |
| read | read | none |

Current nodes are storage-buffer compute on one queue. Barriers therefore use
`COMPUTE_SHADER` stages, exact shader-storage read/write access masks, full
logical buffer ranges and ignored queue-family indices. Several conflicts at
one boundary share a `DependencyInfo`, while each resource gets its own
`BufferMemoryBarrier2`.

Physical disjoint-write metadata is Planned and initially proves race freedom
inside one dispatch. It does not automatically narrow an inter-node barrier.
Subrange synchronization requires an independently validated producer and
consumer range, alias closure, stage/access scope, and lifetime proof; until
then the full logical buffer remains the conservative synchronization domain.

Inter-submit ordering currently serializes on the previous timeline value at
`ALL_COMMANDS`. The same engine timeline now admits commands from the selected
Vulkan Video decode family, and every recorded command carries the pool to
which retirement must return it. This is queue-lifetime plumbing, not semantic
graph scheduling: transfers, images, subranges, indirect arguments, external
resources and cross-queue data use still require producer/consumer-specific
stages, accesses, layouts, ownership transfer and lifetime edges.

## Eager batching and observation

The private engine-owned `ExecutionSession` accumulates non-empty eager graph
snapshots. Blocking host observation or `Engine::checkpoint` joins pending
nodes into one hazard-planned primary command buffer and submits it. A
zero-element operation produces an immediately ready empty value without
flushing unrelated work.

`Matrix::read` flushes the producing batch and waits for its exact event.
`try_read` never flushes or waits and reports `NotReady`. A recording or
submission failure is retained by affected outputs. `Drop` performs no hidden
submission or synchronization.

## Capture, compilation and replay

Capture is isolated, rejects pending eager work, nesting, failure and empty
capture, and never submits or waits. A successful immutable plan retains its
outputs and originating engine identity.

Untimed replay caches one command buffer marked `SIMULTANEOUS_USE`. Stable
read-only Matrix inputs may be rebound only under exact dtype, shape, ownership
and no-alias validation; rebinding recomputes hazards and invalidates the cache.
Exact-type uploads can instead overwrite a stable input after prior completion
without changing its descriptor or command recording. Timed replay records a
fresh command because it owns an independent timestamp query pair.

Captured training additionally uses stable read/write parameter, gradient and
optimizer-state slots, graph-resident AdamW step state, and replay RNG counter
advances. General mutable output rebinding and persisted executable-plan
serialization remain Planned.

Compilation must eventually be an explicit transaction:

```text
validate semantic graph
  -> functionalize mutation/aliases
  -> form legal OaDna partitions
  -> choose precision, placement and kernel candidates
  -> plan virtual values, transient memory and workspace
  -> schedule queues and synchronization
  -> preflight every pipeline, binding and dispatch
  -> record reusable executable variants
  -> publish immutable plan
```

Failure leaves the source graph and prior plan intact. Source-preserving eager
fallback is explicit and reported; it is never a silent CPU fallback.

## GPU-driven execution

OARS already avoids repeated host recording for an unchanged captured plan.
Additional GPU-side control should be capability-gated and justified by an
end-to-end profile:

- indirect dispatch handles device-produced workgroup counts, with bounds and
  producer synchronization proven without host mapping;
- `VK_EXT_device_generated_commands` can let device-written command streams
  drive supported compute commands, but requires queried layouts, preprocess
  memory, buffer-device-address rules and explicit synchronization;
- `VK_AMDX_shader_enqueue` exposes execution-graph behavior only on supporting
  AMD devices and cannot define the portable baseline;
- a persistent scheduler kernel can consume a device work queue on ordinary
  Vulkan compute, but may sacrifice occupancy, fairness, debuggability and
  watchdog safety.

GPU-driven nodes do not replace the semantic graph. The host still validates
and compiles an allowed execution envelope; device data chooses only among the
bounded work admitted by that plan.

## Diagnostics and profiling

`ExecutionPlan::debug_report_json` emits handle-free
`oa.execution_graph.v3` evidence derived from the same graph and barrier plan.
It reports normalized identity, nodes, semantic owners, resources, lifetimes,
accesses, barriers, cache/rebind/submission state and fallback counters. It is a
diagnostic diff surface, not a stable cache or interchange ABI.

`Engine::submit_timed` resets and writes one Vulkan timestamp pair around the
complete graph. Its event owns the wrap-corrected device duration. This excludes
host planning, recording, queue submission, waiting and readback and is not a
calibrated host/device timeline.

Target profiling adds per-phase and selected-node timestamps, calibrated clock
correlation where available, pipeline/candidate identity, queue occupancy
evidence and exact unexpected-fallback counters. Instrumented and normal plans
remain distinct.

## Reuse and memory

The current exact-size retired-buffer pool and graph-lifetime transient arenas
reuse allocations only after proven completion/liveness edges. Target planning
must jointly decide:

- virtual values eliminated by fusion;
- first/last executable use and observable semantic lifetime;
- alignment, memory type, usage and alias compatibility;
- workspace lifetime for tuned providers;
- stable external/captured slots versus transient storage;
- queue ownership and retirement epoch.

Memory savings are invalid if an externally visible intermediate, backward
value, imported resource or asynchronous consumer remains live.

## Acceptance gates

- pure graph tests for RAW/WAR/WAW, alias merge, non-conflicts and deterministic
  reports;
- transaction/failure tests for capture, compile, preflight and cache
  invalidation;
- hardware oracles for multi-node eager and replay paths;
- separate core and synchronization validation;
- GPU-assisted validation for shader bounds and lifetime checkpoints;
- exact producer/consumer/resource/stage/access/layout/queue/lifetime proof for
  every new synchronization rule;
- fresh-process measurement before claiming that a graph feature reduces
  overhead.

## Primary references

- [Vulkan synchronization chapter](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization)
- [Vulkan device-generated commands](https://registry.khronos.org/vulkan/specs/latest/pdf/vkspec.pdf)
- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [`VK_AMDX_shader_enqueue` proposal](https://github.khronos.org/Vulkan-Site/features/latest/features/proposals/VK_AMDX_shader_enqueue.html)
- [OARS CUDA Rust assessment](oaCudaRust.md)
