# OA Rust ML Foundation

**Status:** Experimental

**Updated:** 2026-09-09

**Architecture:** [OA Rust Architecture](../architecture/oaArchitecture.md)

## Current checkpoint

The first differentiable vertical slice is implemented for one-device Vulkan:

```text
FP32 input [B, I]
  -> Linear(weight [O, I], bias [O])
  -> FP32 logits [B, O]
  -> mean cross-entropy(U32 targets [B])
  -> scalar FP32 loss
  -> GradientTape backward
  -> parameter gradients
  -> in-place AdamW parameter/moment update
```

The next admitted chain is also implemented:

```text
U32 token IDs [...]
  -> Embedding(weight [V, D]) -> F32 [..., D]
  -> zero-copy reshape [N, D]
  -> Linear -> cross-entropy
  -> reshape adjoint -> deterministic embedding scatter-add
```

The recurrent primitive path now extends that chain through a stacked Elman
RNN with a fused whole-sequence scan and complete BPTT.

The Transformer primitive chain is implemented as a complete training slice:

```text
F32 [..., D]
  -> LayerNorm(weight [D], bias [D], epsilon)
  -> F32 [..., D]
  -> complete input/weight/bias adjoint
```

Forward normalization uses a two-pass last-dimension variance calculation.
The module starts with unit weight and zero bias, registers both parameters,
and accepts arbitrary nonempty input rank as long as the final dimension is
`D`. Tanh-approximate GELU, differentiable equal-shape residual addition, and
multi-head causal scaled dot-product attention now compose with four Linear
projections into a pre-normalized `TransformerBlock`.

The object-safe `Module` trait and constructor-owned `ModuleRegistry` provide
direct and recursive parameter/buffer traversal, registration-derived dotted
paths, duplicate-identity rejection, train/eval propagation, and scoped
evaluation. A composed `Embedding -> Rnn -> Linear` module tree drives one
complete backward and AdamW step without manually concatenating leaf parameter
arrays. Registration-addressed parameter and AdamW checkpoint serialization is
Experimental; the Char-Transformer gate proves a fresh-owner roundtrip.

The complete NLP consumers are the canonical Char-RNN and Char-Transformer
tutorials: exact C++ corpus and sampler, `[64, 16]` batches, widths 32/64, 300
AdamW steps, all-position accuracy, and fixed-prompt greedy generation. See
[OARS NLP tutorial suite](oaNlpSuite.md).

`Matrix` remains the only numerical value type. `Parameter` is a stable
trainable handle around a Matrix value and its accumulated gradient; it is not
a second tensor type. `Engine` remains the sole execution owner. ML operations
record through the same eager session, executable graph, hazard planner,
bindless descriptor heap, and completion contract as Matrix operations.

The normalized `tools/gen/fn/schema/ml_training.json` record owns semantic
contracts, differentiation relationships, stable kernel identities, reflected
push layouts, and oracle requirements for this slice. The Slang sources remain
first-class implementations under `src/slang/ml` and the generated embedded
artifact registry is private runtime policy.

## Reverse-mode ownership

`GradientTape` is a thread-affine recording scope. Operations record semantic
relationships into the innermost active tape; backward closes and consumes that
tape, validates saved parameter mutation versions before recording gradient
work, and accumulates gradients on stable Parameter handles. Backward records
GPU operations but does not submit or wait.

This checkpoint admits a cross-entropy root and chains composed from Linear,
Embedding, LayerNorm, Gelu, causal attention, FP32 addition, Rnn, and reshape.
It does not claim generalized autograd.
Broadcasting, reductions, other recurrent families, arbitrary roots, higher
derivatives remain Planned. `TrainingProgram` captures the currently admitted
fixed-shape forward/backward/AdamW chain without making autograd itself a
general graph compiler.

The Experimental donor-backed `TrainingLoop` is the single ordinary iteration
lifecycle above eager execution and `TrainingProgram`. It owns exact step/epoch
accounting and completion while borrowing the engine, AdamW optimizer, metrics,
and callbacks. Callback hooks receive immutable snapshots and return explicit
continue/stop decisions or failures; they do not become another graph owner.
Its automatic `step(prepare, record)` path performs one eager warm-up, captures
the second fixed-shape step, prepares fresh stable inputs on every later step,
and skips graph authoring during replay. Explicit recapture discards the old
program only between completed steps. Whole-program compilation rejection
preserves and eagerly submits the exact source recording, disables further
capture attempts, and exposes its cumulative count and latest reason. Explicit
recapture re-enables compilation without erasing that evidence.
Successful automatic capture records the reusable untimed command before the
source session is committed, making the first replay a cache hit and preserving
the source graph if command recording fails. Replay-safety validation also runs
before commit: generated kernel metadata distinguishes safe work, host-stepped
optimizer updates, optimizer-state advance, and optimizer-state update. The
validator requires exactly one advance before every replay update and rejects
host-stepped AdamW. Each successful program exposes ordered immutable
compilation-stage records; automatic capture reports command recording as
applied, while explicit lazy capture reports it as not run.

`compilation_debug_report_json` emits the donor's deterministic
`oa.training_compilation.v2` report: ordered stages, DNN partition analysis,
semantic-to-executable lowering counts, and per-resource logical memory
evidence. Names use complete JSON escaping. Applied, inherited, and explicit
fallback partition counts now reflect physical lowering rather than analysis
alone, and every fallback carries a stable reason. Automatic training capture
retains the proven source lowering: its recognized multi-operation candidates
are explicit fallbacks instead of inference-only replacements.
`semantic_debug_report_json` exposes the same captured program through
`oa.semantic_graph.v2`, including values, views, typed attributes, accesses,
mutation/alias edges, control dependencies, and autograd ranges without backend
handles or kernel policy.
`debug_report_json` exposes the lowered captured work through
`oa.execution_graph.v3`: normalized physical resources, generated kernel and
dtype identities, semantic owners, dispatches, exact planned hazards, and the
latest replay timeline. It omits Vulkan handles, addresses, descriptor indices,
and push payloads. Together the three reports preserve the donor's semantic,
compiler, and executable evidence split without making DNN or Vulkan policy
public.

The first physical OaDna slices mechanically port OA's bounded FP32 QKV
projection+bias replacement. It admits exactly three inference Linear+bias
operations sharing a `[1024, 32]` activation with three `[32, 32]` weights and
`[32]` biases, then replaces their three executable dispatches with the donor
multi-output shader while retaining all three semantic owners. Any shape,
layout, dtype, alias, training, or provenance mismatch keeps the original
executable interval and is reported as a fallback. The second slice ports the
public two-input `swiglu(gate, up)` operation and its complete adjoint, then
recognizes two Linear+bias projections followed by SwiGLU. At the donor's exact
`M=1024, N=64, K=32` inference shape, it replaces all three nodes with
`GateUpSwigluBias`; an externally retained or externally consumed gate/up
intermediate rejects replacement. These exact qualifications are Experimental
providers, not general QKV or gated-FFN claims.

## Optimizer behavior

AdamW owns parameter update policy and FP32 first/second moment values. A step
records one in-place read/write dispatch for each parameter with a gradient and
advances the stable Parameter's mutation version. The parameter and both moment
buffers retain their identities; a previously captured read-only plan therefore
observes completed optimizer updates without rebinding. Existing Matrix handles
returned by `Parameter::data` name that stable storage and observe later steps.
General public Matrix mutation remains unavailable. `zero_grad` discards
accumulated gradient handles and performs no submission or wait. Construction
rejects duplicate stable parameter handles, including inputs not obtained from
a module registry.

The current scalar defaults are beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, and
weight decay `0.01`. Captured AdamW uses one six-word U32 state buffer for the
exact replay step and five bit-preserved FP32 scalar settings; one graph-head
dispatch increments the step before every parameter update. Mixed precision,
FP32 master weights, learning-rate schedules, fused multi-parameter updates,
and checkpoint persistence-format stability remain Planned.

## Fixed-shape training replay

`TrainingProgram::capture` records one forward, backward, and AdamW step without
executing it. All parameters must receive a gradient. The resulting program
retains stable parameter, moment, gradient, intermediate, loss, and optimizer
state storage and replays one cached Vulkan command buffer. It verifies the
originating optimizer and every stable resource identity before submission;
ordinary `zero_grad`, eager optimizer steps, checkpoint restore, or a second
capture make an older program fail with `FailedPrecondition` instead of silently
using detached state. The scalar loss is an explicit observed captured output;
semantic inputs and outputs retain deterministic physical-resource bindings and
executable lifetime ranges without exposing Vulkan handles. Capture does not
advance the optimizer's completed-step counter or parameter versions; the first
successful replay does.

`is_complete` and `wait` observe the latest submitted replay without requiring
the caller to retain its returned event. Consuming `reset` waits and releases
the captured program; this replaces the donor's reusable empty shell because
Rust capture constructs a new owned program value. `Drop` remains non-blocking.

`upload_input` writes a new exact-type, exact-count host batch into an
unrebound read-only input slot. It waits for the preceding replay before
overwriting that host-visible buffer, so the command and bindless descriptor
remain unchanged. Qualified disjoint transient lifetimes may share private
arenas with an observable distinct-storage fallback. Multi-buffered input
staging, dynamic shapes, RNG replay state, and broader transient allocator
policy remain Planned.

The current schemas contain no replay-safe Philox kernel family. Random values
must therefore be prepared outside the captured executable program until the
donor state-advance/uniform/normal transformation is ported; host seed values
must never be frozen into reusable command recording.

## Failure and observation

- Linear requires nonempty FP32 feature dimensions, `[B, I]` input,
  `[O, I]` weight, and `[O]` bias on one engine.
- Cross-entropy requires nonempty FP32 logits `[N, C]` and U32 targets `[N]`.
- Embedding requires FP32 weight `[V, D]` and same-engine U32 indices; it
  preserves the full index shape and appends `D`.
- Rnn requires nonempty F32 input `[B, S, I]`, returns `[B, S, H]`, and
  currently admits zero-state biased layers with `H <= 1024`.
- LayerNorm requires nonempty F32 input `[..., D]`, same-engine F32 weight and
  bias `[D]`, and a finite positive epsilon. It preserves the complete shape.
- Reshape is a zero-copy storage view with a distinct semantic value identity.
- An out-of-range target produces NaN without reading beyond the logits row.
- An out-of-range embedding index produces a NaN row without reading beyond
  the table.
- Reusing a consumed tape or changing a saved Parameter before backward fails.
- Host loss/gradient/parameter reads use the normal Matrix observation boundary
  and therefore submit and wait for the exact producing eager batch.
- There is no CPU execution fallback.

## Evidence and remaining gates

The external Rust ML test compares Linear, LayerNorm, and stable cross-entropy
forward results against independent host oracles, compares Linear, LayerNorm,
and repeated-index Embedding gradients against central finite differences,
checks LayerNorm input-gradient propagation through an Embedding predecessor,
checks the reshape adjoint, checks the first in-place AdamW update against an
independent host implementation, proves a captured reader observes the stable
updated parameter storage without rebinding, verifies material loss reduction,
and covers invalid shape/dtype/ownership and out-of-range index behavior. The module-tree proof
also verifies deterministic dotted traversal, persistent buffer metadata,
recursive mode propagation, duplicate ownership rejection, complete gradient
reachability, and one optimizer update over the composed character model.
The Release NLP acceptance run additionally fixes the complete 300-step loss,
accuracy, and generated-text gates; it is not yet a performance-parity or
checkpoint-roundtrip claim.

Run the hardware checkpoint serially:

```bash
cargo test --test ml --all-features -- --ignored --test-threads=1
```

Core validation, synchronization validation, GPU-assisted validation, broader
odd/boundary/poison coverage, and fixed-clock performance qualification remain
unverified. This implementation is Experimental.
