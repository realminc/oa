# OA Rust ML Foundation

**Status:** Experimental

**Updated:** 2026-09-10

**Architecture:** [OA Rust Architecture](../architecture/oaArchitecture.md)

## Current checkpoint

The first differentiable vertical slice is implemented for one-device Vulkan:

```text
FP32 input [B, I]
  -> Linear(weight [O, I], bias [O])
  -> FP32 logits [B, O]
  -> mean cross-entropy(U32/non-negative I32 targets [B])
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

The donor VQ-VAE bottleneck is connected through `ml::matrix::{vq_assign,
vq_lookup,vq_ema_update,detach}` and `ml::nn::{VectorQuantizer,
ResidualVectorQuantizer}`. Assignment preserves squared-L2 lower-index tie
breaking and emits I32 tokens. The module applies one zero-copy stop-gradient
view for the straight-through estimator, retains codebook/EMA state as
persistent non-gradient buffers, and functionally advances that state through
the semantic graph. Highest-norm deterministic seeding is an explicit host
completion boundary. See [Vector quantization](oaVq.md).

The donor GRU sequence path is also connected. Each stacked layer hoists its
input projection into one existing batched Linear operation and records one
whole-sequence recurrent scan. Reverse mode records one BPTT scan plus the
existing Linear parameter adjoint; neither direction submits once per timestep.

The first Mamba-3 slice connects the grouped SISO selective-state recurrence
and the donor short backward as generated semantic operations. Forward remains
one physical dispatch; backward transactionally owns its six private stages
and returns all eleven adjoints through the ordinary `GradientTape`. An
independent CPU recurrence, complete finite differences, and grouped-versus-
expanded-head equivalence pass on Intel Vulkan. The module, recurrent step,
MIMO, long-sequence backward, and tutorial remain pending. See
[Mamba-3](oaMamba3.md).

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

The same block now has construction-time conditioned dense and MoE variants.
They immutably register a zero-initialized `C -> 6D` adaptive projection and
preserve the donor AdaLN-Zero scale, shift, and residual-gate order. The block
therefore begins as an exact identity on the residual stream while its
adaptive gates receive gradients. This intentionally replaces the donor's
post-construction `enableAdaptiveConditioning` mutation so one Rust module
keeps one fixed structural registry.

Those primitives now drive the first complete generative-model consumer.
`FlowTimeEmbedding` retains the donor's GPU sinusoidal path and nonpersistent
frequency buffer. `FlowTransformer` composes dense or dropless-MoE blocks with
bidirectional attention, optional padding masks, mutable sequence geometry,
AdaLN-Zero conditioning, and a final LayerNorm. `FlowDenoiser` adds trainable
positions, input/output projections, optional per-sample condition dropout,
and classifier-free guidance without introducing another execution owner.
The Rust construction path is deterministic by explicit seed; its registered
position parameter is bridged into the active tape before ordinary Matrix
composition so it receives a real gradient. See [Flow models](oaFlow.md).
The sibling `ml::flow` operation module owns linear flow matching, explicit
Euler integration, and broadcast-masked MSE with schema-generated identities,
single-pass forward providers, and complete reverse paths.

The attention operation has both the standard materialized-probability route
and the explicit causal Flash provider under one semantic identity. Flash
retains row-wise FP32 log-sum-exp and records deterministic Q and K/V adjoints;
the packed Transformer module remains on the standard route until provider
selection policy and comparative performance evidence are admitted.

The object-safe `Module` trait and constructor-owned `ModuleRegistry` provide
direct and recursive parameter/buffer traversal, registration-derived dotted
paths, duplicate-identity rejection, train/eval propagation, and scoped
evaluation. A composed `Embedding -> Rnn -> Linear` module tree drives one
complete backward and AdamW step without manually concatenating leaf parameter
arrays. Registration-addressed persistence now writes and reads the native OA
`.oam` v3 model-file format: parameters map to Weights, persistent registered
buffers map to State, Adam/AdamW moments are flattened in registration order,
no-momentum SGD uses the donor's empty state arrays, and training step/rate map
to Progress. The integrity test proves atomic overwrite,
fresh-owner restoration, v1/v2 read compatibility, payload-corruption rejection,
and bidirectional parsing with the OA C++ `modelctl`. See
[OARS model files](oaModelFile.md).

The complete NLP consumers are the canonical Char-RNN and Char-Transformer
tutorials: exact C++ corpus and sampler, `[64, 16]` batches, widths 32/64, 300
AdamW steps, all-position accuracy, and fixed-prompt greedy generation. See
[OARS NLP tutorial suite](oaNlpSuite.md).

The reinforcement-learning foundation admits its value contract, first GPU
algorithm primitive, and fixed-capacity rollout session. `EnvironmentSpace`, `EnvironmentSpec`, and `EnvironmentTransition`
preserve checked observation, action, reward, termination, and truncation
metadata without adding another execution owner or synchronizing Matrix data.
Its public graph-native normalization, action-scaling, and reward-clipping
operations compose existing differentiable Core kernels under generated
Environment semantic identities;
`ml::advantage::{normalize,gae}` mechanically ports OA's differentiable
whole-rollout standardization and reverse-time estimator with exact
termination/truncation semantics. `RolloutBuffer` records fused time-major
append, allocation-free GAE finalization, and device-side validity reset while
retaining all storage across collection cycles. `ItRolloutTraining` composes
the donor Collect/Update schedule and existing `ItTraining` lifecycle,
including transactional capture rejection and host-cursor rollback before an
update. `ReplayBuffer` adds
preallocated circular off-policy append and deterministic seeded device-side
sampling with exact sampled indices. Core `matrix::sample_logits` now owns
greedy, dense, TopK, and nucleus categorical selection with replay-safe Philox
state, while `matrix::gather_last_dim` and its deterministic adjoint provide
the policy/DQN selection primitive. Categorical and tanh-normal policy result
compositions are connected with one semantic operation each. The object-safe
`ActorCritic` contract and seeded default categorical two-tower Module feed a
complete caller-driven `PpoTrainer`; its clipped-policy forward/backward,
policy/value/entropy total-loss composition, GAE, full-batch update, metrics,
and rejected-collection rollback are connected. DQN selected-action,
detached Bellman-target, Smooth-L1 composition,
and Q-only reverse mode are connected as one semantic operation. The DQN
trainer now composes ReplayBuffer, target-network deep copies, modules,
autograd, Optimizer, and `ItTraining` without introducing another execution or
training owner. SAC twin-critic and actor objectives likewise preserve detached
targets, entropy regularization, and differentiable minimum-Q selection under
their own semantic identities. Its trainer composes independently observable
actor and critic `ItTraining` units, continuous replay, target inference, and
exact twin-target synchronization. Native Environment transaction ownership,
categorical rollout collection, and deterministic evaluation telemetry are
connected through the same Engine. Native CartPole and its donor-sized PPO
learning/checkpoint gate now run through this path; atomic trainer-progress
checkpoint/resume remains Planned. See the
[reinforcement-learning foundation](oaRl.md).

`Matrix` remains the only numerical value type. `Parameter` is a stable
trainable handle around a Matrix value and its accumulated gradient; it is not
a second tensor type. `Engine` remains the sole execution owner. ML operations
record through the same eager session, executable graph, hazard planner,
bindless descriptor heap, and completion contract as Matrix operations.

The normalized `tools/gen/fn/schema/ml_training.json` record owns semantic
contracts, differentiation relationships, stable kernel identities, reflected
push layouts, and oracle requirements for ML-owned operations in this slice.
Core `matrix_reduce.json` separately owns Softmax, LogSoftmax, and their
adjoints. Slang sources remain first-class implementations under their semantic domain in `src/slang`;
the generated embedded artifact registry lives at
`runtime/shader/registry.gen.rs` and remains private runtime policy.

## Reverse-mode ownership

`GradientTape` is a thread-affine recording scope. Operations record semantic
relationships into the innermost active tape; backward closes and consumes that
tape, validates saved parameter mutation versions before recording gradient
work, and accumulates gradients on stable Parameter handles. Backward records
GPU operations but does not submit or wait.

This checkpoint admits scalar roots from Smooth L1, MSE, L1, BCE,
cross-entropy, and masked cross-entropy, and chains composed from Linear,
Embedding, LayerNorm, BatchNorm2d, Conv1d, ConvTranspose1d, Conv2d,
ConvTranspose2d, RMSNorm,
the donor activation family (SiLU, ReLU, tanh, sigmoid,
Leaky ReLU, ELU, Mish, Softplus, and GELU), SwiGLU, standard scaled dot-product
attention with causal and optional additive masking, axis-aware
Softmax, LogSoftmax, rank-three BMM NN/NT/TN, NCHW AvgPool2d, MaxPool2d,
AdaptiveAvgPool2d, and Upsample, FP32 addition, Rnn, Gru, and reshape. Each activation owns a schema row, a paired
backward row, a mechanically adapted donor Slang kernel, generated attachment
metadata, and a private reverse-mode node. Leaky ReLU deliberately saves the
forward input because the donor backward shader branches on input sign; this
corrects the older donor schema annotation that named the output.
It does not claim generalized autograd.
Broadcasting, reductions, remaining recurrent families, arbitrary roots, higher
derivatives remain Planned. `TrainingProgram` captures the currently admitted
fixed-shape forward/backward/AdamW chain without making autograd itself a
general graph compiler.

The Experimental donor-backed `ItTraining` is the single ordinary iteration
lifecycle above eager execution and `TrainingProgram`. It owns exact step/epoch
accounting and completion while borrowing the engine, optimizer, metrics,
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

The public `training.rs` umbrella owns private `training/iterator.rs`,
`training/program.rs`, `training/callbacks.rs`, callback-family, and
`training/schedule.rs` and `training/session.rs` modules, parallel to the
existing `nn.rs`/`nn/` split.
`ItTraining` is mutable lifecycle state;
`TrainingProgram` remains a distinct immutable executable. Built-in
`ProgressBar`, `TrainingSummary`, `EarlyStopping`, `LearningRateScheduler`,
`CsvLogger`, `Validation`, `Checkpoint`, and `PhaseSchedule` callbacks expose donor reporting,
evaluation, and control behavior without becoming graph or synchronization
owners. Validation publishes one shared completed result and excludes evaluator
wall time from training throughput. Phase transitions are validated against the
iterator epoch map and borrow its existing optimizer. AdamW mirrors scheduled
rates into graph-resident replay state after completion. Checkpoint policy uses
the same native codec and a sealed persistence capability rather than a second
AdamW-only save path.

The connected `TrainingSession` is a cloneable thread-safe control and
observation handle attached to the one `ItTraining` owner. It retains bounded
commands, revisioned results, and completed-step snapshots while engine-thread
handlers remain borrowed by the iterator. Commands are applied only before a
new body begins. Pause/resume/stop, checkpoint/evaluation/rebuild handlers,
typed live parameters, automatic-program recapture requests, named metrics,
blocking paused wakeup, and explicit Completed/Failed publication therefore do
not create a second scheduler or make Vulkan-backed training thread-mobile.

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

`Optimizer` is the object-safe policy contract shared by eager training and
callbacks. `NoOpOptimizer` supports externally authored updates without taking
another parameter owner. FP32 `Sgd` owns its selected parameter handles and an
optional zero-initialized momentum matrix per parameter; its ordinary and
momentum kernels mechanically preserve the donor shader order. Both routes are
schema-owned and explicitly classified as host-stepped, so current
whole-program capture rejects them instead of replaying stale host state. Their
independent Vulkan oracle passes on Intel Iris Xe with Mesa 26.2.2 and Vulkan
1.4.354. No-momentum SGD supports exact native checkpoint restoration; a live
momentum state fails closed because the donor v3 SGD payload has no momentum
array.

FP32 `Adam` likewise owns zero-initialized first and second moments and
preserves the donor moment, bias-correction, and parameter-update order. It
shares eager training and callback policy through `Optimizer`; captured replay
remains unclaimed. Its independent first-step bias-correction oracle and native
checkpoint round trip pass on the same Intel Vulkan device.

`ml::optim::clip_grad_norm` mechanically ports OA C++ `FnOptim::clipGradNorm`
as one asynchronous variadic semantic operation. It skips empty gradients,
reduces the combined FP32 L2 norm across at most sixteen same-engine matrices,
then scales every admitted gradient in place without a host read or wait. The
fixed 18-descriptor push ABI and parameter-buffer threshold preserve the donor
fix for short gradient collections. Capture records aligned input/output
aliases for every gradient and maps the operation to its reduction and scaling
nodes; scratch storage remains a private lowering detail.

FP32 `Muon` owns one momentum matrix per parameter. Rank-two parameters lower
one semantic optimizer operation into donor Nesterov, Frobenius normalization,
five-step Newton-Schulz orthogonalization, orientation handling, Moonshot
scaling, and decoupled-weight-decay dispatches. Other ranks use the donor fused
vector route; OA does not silently split those parameters into AdamW. Independent
GPU tests cover the matrix, vector, tall/transposed, and both-dimensions-large
routes against the donor algorithm. Native `.oam` restoration preserves its one
moment plus scalar/step state and produces the same following update from a
fresh owner. Captured device-state Muon remains Planned.

AdamW owns parameter update policy and FP32 first/second moment values. Its
shader math mechanically preserves OA C++'s decoupled-weight-decay and moment
update order. A step groups complete sets of four gradients into the donor's
four-parameter physical kernel and uses the scalar donor kernel for the tail.
Each grouped node retains four schema-owned AdamW operations in the semantic
graph rather than inventing a fused public operation. The parameter and both moment
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
dispatch atomically increments the step before every parameter update. The
canonical 22-parameter Transformer therefore lowers optimizer updates to five
four-way nodes plus two scalar tails rather than 22 scalar nodes. Mixed
precision, FP32 master weights, and captured Muon remain Planned.
Native `.oam` structural compatibility is implemented;
public general-purpose `ModelFile`, model translators, quantized restoration,
serialized replay RNG/descriptors remain Planned.

## Metric observation

`oa::ml::metric::scalar_loss` and `oa::ml::metric::accuracy` are deliberately
blocking host observations, matching the donor `FnMetric` role. Categorical
argmax and counting remain on the owning device through the Core
`oa::matrix::categorical_accuracy_count` operation; only one U32 count is read
back. The Matrix schema also owns the masked counter used by future sequence
metrics. U8, U32, and I32 labels are admitted, equal maxima select the first
class, and the output remains a `[1]` U32 Matrix until a caller explicitly
observes it. The current completed-step `LossMetric` remains independent of
prediction/label metrics because the iterator snapshot does not yet carry
those values.

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
staging, dynamic shapes, and broader transient allocator policy remain Planned.

Schema-owned Philox uniform/normal operations preserve the complete 64-bit seed
and distribution parameters. Eager calls use the effective recorded seed.
Capture replaces each frozen-seed physical kernel with a replay variant reading
one graph-resident U32 counter, followed by exactly one counter-advance node.
Training-program validation rejects frozen RNG and unmatched replay/advance
pairs. Inverted Dropout uses the same transformation in forward and backward,
so both regenerate the same mask for a step and advance together on later
replays. The current state is private to each captured operation; shared random
streams, explicit offsets beyond the current ABI, and serialized RNG state
remain Planned.

## Failure and observation

- Linear requires nonempty FP32 feature dimensions, rank-at-least-two
  `[..., I]` input, `[O, I]` weight, and `[O]` bias on one engine. Lowering
  flattens the leading extents into GEMM rows and restores them as `[..., O]`.
- Cross-entropy requires nonempty FP32 logits `[N, C]` and U32 or non-negative
  I32 targets `[N]`. Negative and out-of-range indices produce NaN without an
  out-of-bounds access. Donor UInt8 targets and BF16 logits remain gated on
  those scalar types entering the Rust value/storage contract.
- Masked cross-entropy additionally requires an FP32 mask `[N]` on the same
  engine and a caller-supplied `valid_count` in `1..=N`. Zero mask values
  exclude their rows and produce exact-zero logits adjoints. OARS preserves
  the donor contract by not reading the mask back to recompute that denominator.
- Smooth L1, MSE, L1, and BCE require nonempty, equal-shape FP32 prediction and
  target matrices on one engine and return the mean as a scalar Matrix. Their
  reverse routes differentiate only prediction. Smooth L1 selects the donor's
  fused one-workgroup scalar-mean candidate; MSE, L1, and BCE retain explicit
  per-element loss, Sum, and Scale dispatches. BCE preserves the donor's
  `[1e-7, 1 - 1e-7]` prediction clamp.
- Embedding requires FP32 weight `[V, D]` and same-engine U32 indices; it
  preserves the full index shape and appends `D`.
- Rnn requires nonempty F32 input `[B, S, I]`, returns `[B, S, H]`, and admits
  zero-state biased or bias-free layers with `H <= 1024`. Its input projection
  uses canonical Linear lowering before the donor-backed recurrent scan.
- LayerNorm requires nonempty F32 input `[..., D]`, same-engine F32 weight and
  bias `[D]`, and a finite positive epsilon. It preserves the complete shape.
- BatchNorm2d requires nonempty F32 NCHW input and same-engine F32 channel
  vectors `[C]`. Training computes centered population statistics and advances
  persistent running mean/variance with finite momentum in `[0, 1]`; evaluation
  reads that state without mutation. Both paths have complete input and affine
  adjoints, and evaluation treats running statistics as fixed.
- Conv2d requires nonempty F32 NCHW input, square grouped OIHW weight, channel
  bias, nonzero stride/groups, and input/output channels divisible by groups.
  Its spatial extent uses floor geometry over symmetric padding. One structured
  reverse operation produces deterministic input, weight, and bias adjoints.
- Conv1d requires nonempty F32 NCL input, OIK weight, channel bias, and nonzero
  stride/dilation. Forward lowers through im2col, tiled MatMulNt, broadcast bias,
  and batched transpose; padding and dilation use the donor floor geometry.
  Its current deterministic explicit adjoints are correct but not yet the
  donor's newer GEMM-composed wide-training route.
- ConvTranspose1d requires nonempty F32 NCL input, IOK weight, and nonzero
  stride/dilation. It is bias-free. Its output length is
  `(input_length - 1) * stride - 2 * padding + effective_kernel`; padding that
  removes the complete output is rejected. Forward is the Conv1d data adjoint,
  while backward reuses canonical Conv1d GEMM and parameter-adjoint work.
- ConvTranspose2d requires nonempty F32 NCHW input, square IOKK weight,
  output-channel bias, and nonzero stride. Each output extent is
  `(input_extent - 1) * stride - 2 * padding + kernel`. Forward preserves the
  donor data-adjoint plus bias lowering; its structured backward produces the
  input, weight, and bias adjoints without float atomics. The build-time SPIR-V
  and reflected-ABI gates pass; live GPU oracle execution is still pending.
- AvgPool2d requires nonempty F32 NCHW input, nonzero square kernel and stride,
  and a kernel that fits the symmetrically padded input. Its output uses floor
  spatial geometry, excludes padding from the averaging divisor, and preserves
  batch and channel dimensions. The input adjoint uses the identical geometry.
- MaxPool2d uses the same geometry, exposes pooled F32 values plus exact U32
  flat-input argmax indices, and sends each output adjoint to its saved winner.
  Equal maxima select the first valid input in row-major window order.
- AdaptiveAvgPool2d requires nonzero output height and width and computes exact
  independent floor/ceiling bins. Rectangular and non-divisible geometry is
  supported; forward and adjoint reject arithmetic outside the U32 shader ABI.
- Upsample requires nonempty F32 NCHW input and a positive integer scale.
  Nearest uses exact integer source ownership; bilinear uses the donor
  align-corners-false half-pixel convention. Both adjoints are deterministic
  gather kernels.
- Reshape is a zero-copy storage view with a distinct semantic value identity.
- An out-of-range target produces NaN without reading beyond the logits row.
- An out-of-range embedding index produces a NaN row without reading beyond
  the table.
- Reusing a consumed tape or changing a saved Parameter before backward fails.
- Host loss/metric/gradient/parameter reads use the normal Matrix observation
  boundary and therefore submit and wait for the exact producing eager batch.
- There is no CPU execution fallback.

## Evidence and remaining gates

The external Rust ML test compares Linear, LayerNorm, BatchNorm2d, Conv1d,
ConvTranspose1d, grouped Conv2d, stable cross-entropy,
masked cross-entropy, Smooth L1, MSE, L1, and BCE forward results against
independent host oracles,
compares the four core-loss prediction adjoints plus Linear, LayerNorm,
BatchNorm2d, Conv1d, ConvTranspose1d, Conv2d, and
repeated-index Embedding gradients against central finite differences,
checks LayerNorm input-gradient propagation through an Embedding predecessor,
checks the reshape adjoint, checks the first in-place AdamW update against an
independent host implementation, proves a captured reader observes the stable
updated parameter storage without rebinding, verifies material loss reduction,
and covers invalid shape/dtype/ownership and out-of-range index behavior. The module-tree proof
also verifies deterministic dotted traversal, persistent buffer metadata,
recursive mode propagation, duplicate ownership rejection, complete gradient
reachability, and one optimizer update over the composed character model.
The Release NLP acceptance run additionally fixes the complete 300-step loss,
accuracy, generated-text, and fresh-owner checkpoint gates; it is not yet a
performance-parity claim.

Run the hardware checkpoint serially:

```bash
cargo test --test ml --all-features -- --ignored --test-threads=1
```

Core validation, synchronization validation, GPU-assisted validation, broader
odd/boundary/poison coverage, and fixed-clock performance qualification remain
unverified. This implementation is Experimental.
