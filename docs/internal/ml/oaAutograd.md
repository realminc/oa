# OARS reverse-mode differentiation

**Status:** Experimental narrow implementation; generalized coverage Planned

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md) and
[OA Rust source and module structure](../architecture/oaSourceStructure.md)

**Donor reference:** OA C++ `docs/internal/ml/oaAutograd.md`

## Public contract

`oa::ml::GradientTape` is a thread-affine recording scope. Ordinary operations
run eagerly against their Matrix engine and record semantic edges into the
innermost active tape. `backward(&loss)` closes and consumes the tape, validates
saved versions, and records ordinary backward operations. It does not submit or
wait; Matrix observation, training orchestration, or an explicit Engine boundary
owns completion.

Concrete nodes, attachment functions, and backward-kernel dispatch are private.
Rust does not port the donor's concrete `Grad*` classes into public types. A
failed backward returns `oa::Error`; there is no logging-only compatibility
overload.

The current root must be a scalar F32 result produced by an admitted loss.
Supported paths are composed from:

- `ml::nn::Linear`;
- `ml::nn::Embedding` with U32 indices;
- `ml::nn::LayerNorm` over the final dimension;
- training and evaluation `ml::nn::BatchNorm2d` over NCHW channels;
- grouped NCHW `ml::matrix::conv_2d` and trainable `ml::nn::Conv2d`;
- NCL `ml::matrix::conv_1d` and trainable `ml::nn::Conv1d`, including dilation;
- bias-free NCL `ml::matrix::conv_transpose_1d` and trainable
  `ml::nn::ConvTranspose1d`;
- NCHW `ml::matrix::conv_transpose_2d` and trainable
  `ml::nn::ConvTranspose2d`;
- `ml::nn::RmsNorm` and `ml::matrix::rope` over packed rotary heads;
- donor-backed `ml::matrix::{silu, relu, tanh, sigmoid, leaky_relu, elu,
  mish, softplus, gelu, swiglu}` activations;
- rank-three `ml::matrix::{bmm, bmm_nt, bmm_tn}` batched products;
- `ml::matrix::scaled_dot_product_attention`, including causal and optional
  additive-mask execution, its explicit causal Flash provider, and its packed
  causal compatibility composition;
- equal-shape FP32 `matrix::add` residual paths;
- axis-aware FP32 `matrix::{softmax, log_softmax}` using their saved forward
  outputs;
- NCHW FP32 `ml::matrix::{avg_pool_2d, max_pool_2d}` with saved pooling
  geometry and exact U32 max-pool indices;
- rectangular FP32 `ml::matrix::adaptive_avg_pool_2d` with independent output
  extents;
- integer-scale NCHW `ml::matrix::upsample_2d` in nearest or bilinear mode;
- stacked `ml::nn::Rnn` whole-sequence scans;
- `ml::nn::GruCell` explicit-state steps and stacked `ml::nn::Gru` batched
  input projections plus whole-sequence scans;
- zero-copy `Matrix::reshape` / `matrix::reshape` views;
- `ml::loss::{smooth_l1, mse, l1, bce, cross_entropy,
  masked_cross_entropy}`; the four matching-shape losses detach their target,
  while both cross-entropy variants differentiate only their logits;
- gradient addition at converging paths.

This is not a generalized autograd claim. Broadcasting, other reductions, arbitrary
roots, detached/no-grad scopes, higher derivatives, remaining recurrent families,
and the remaining Matrix/ML catalog are Planned.

## Operation and node pairing

The donor generator already treats autograd as part of an operation schema row:
the row selects an attachment policy, saved state, concrete private node, and
backward operations. OARS preserves that ownership while using Rust modules:

```text
operation schema row
  |-> public forward/backward wrappers in the owning operation family
  |-> backend-neutral OperationContract and differentiation identity
  |-> private autograd attachment in the matching autograd family
  `-> forward/backward oracle, mutation, and graph-provenance tests
```

The physical implementation is:

| Path | Responsibility |
| --- | --- |
| `ml/autograd.rs` | Public facade exporting only `GradientTape`. |
| `ml/autograd/tape.rs` | Tape lifecycle, reverse traversal, version preflight, accumulation, and semantic provenance. |
| `ml/autograd/matrix.rs` | Private ML Matrix attachment facade. |
| `ml/autograd/matrix/activation.gen.rs` | Schema-generated activation attachments whose original inputs, saved forward values, and typed scalar state are mechanically derivable. |
| `ml/autograd/matrix/{linear,embedding,norm,conv,pool,upsample,position,recurrent,attention}.rs` | Family-owned attachments whose compound saved state or node construction remains explicit. |
| `ml/autograd/loss.rs` | Private loss attachment facade. |
| `ml/autograd/loss/core.gen.rs` | Schema-generated Smooth L1, MSE, L1, BCE, cross-entropy, and masked-cross-entropy attachments. |
| `ml/autograd/node.rs` | Compact private saved-value record for the currently admitted catalog. |
| `core/autograd.rs` | Private observer bridge for Core Matrix operations; no ML policy. |
| `ml/lowering/` | Transitional Matrix, loss, and optimizer lowering families, deleted independently as operation-family generation lands. |

Every currently differentiable forward row records an autograd mode and family.
`generated` rows emit attachments from three distinct schema-owned categories:
original input identities used as gradient destinations, Matrix values retained
for the adjoint, and typed scalar state such as Leaky ReLU or ELU `alpha`.
The admitted activation family, SwiGLU, and core loss family use this path.
`manual` rows explicitly
name the family that owns compound saved state or node construction; Linear,
Embedding, LayerNorm, BatchNorm2d, Conv1d, ConvTranspose1d, Conv2d,
ConvTranspose2d, RMSNorm,
AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Upsample, RoPE, BMM, recurrent, and
attention currently use that path. Core
Softmax and LogSoftmax attach through the private `core/autograd.rs` observer
and preserve the same schema-owned semantic operation and backward range. Manual
does not mean unowned: the schema is the exhaustive attachment-policy catalog,
and validation rejects missing, misplaced, or malformed policies.

Public wrappers remain under `ml::matrix` or `ml::loss`. Slang sources,
`KernelId`, and Vulkan pipelines are separate lowering authorities and never
become autograd nodes. Direct edits to `.gen.rs` attachment files are
forbidden; new mechanically derivable attachments must be generated from the
same normalized operation row.

## Graph contract

Each differentiable forward operation records:

- ordered input identities;
- only the values and typed scalar state needed by its adjoint;
- the forward output identity;
- saved parameter versions where mutation is possible;
- a nonzero sequence number for deterministic reverse traversal.

Original inputs and saved values are intentionally not conflated. An adjoint
may derive from the forward output while still needing the original input's
semantic identity as the accumulation destination. Generated attachment
functions therefore accept both categories, de-duplicate a Matrix present in
both, and store scalar derivative parameters explicitly. Schema validation
rejects missing input identity, malformed scalar types, duplicate scalar names,
and scalar/Matrix name collisions.

During execution capture, each reached node attaches its forward output and
tape sequence to the semantic graph. After the node records its adjoint work,
the attachment receives the contiguous semantic-operation range emitted by that
backward step. Operations in that range retain the forward operation identity
and tape sequence. Kernel names remain executable lowering detail.

Capture or submission abort advances the semantic recording generation, so a
retained numerical tape cannot amend a later graph with stale identifiers.
Metadata-only reshape nodes preserve value lineage but own no independent
semantic operation.

## Mutation and saved values

Backward preflights every saved Parameter version before recording any
gradient operation. An optimizer update between forward and backward therefore
returns `FailedPrecondition` without leaving a partially recorded adjoint.

AdamW is the first private in-place writer. It records explicit read/write
access over stable Parameter storage, advances the saved-value version after
successful recording, and emits fresh semantic identities that alias the same
parameter/moment/state storage. General public shared-storage mutation remains
unadmitted.

Matrix clones preserve semantic identity. Reshape creates a new semantic value
identity while sharing storage and readiness. Future mapped-write or external
memory surfaces must participate in the same version contract before they may
mutate values saved for backward.

## Determinism and accumulation

Leaf gradients accumulate on stable Parameter handles. Intermediate gradients
are keyed by semantic value identity and combined with `matrix::add` in reverse
tape order. Deferred or fused accumulation is legal only if it preserves the
same result, provenance, and failure boundary.

LayerNorm backward produces input, affine-weight, and affine-bias adjoints.
BatchNorm2d saves the per-channel mean and population variance used by its
forward call. Training backward removes the channel means of both the scaled
output gradient and its normalized correlation; evaluation backward uses the
fixed-statistics affine derivative. Both modes produce input, weight, and bias
adjoints, while persistent running statistics remain detached state.
Conv2d saves its input, grouped OIHW weight value, stride, padding, groups, and
optional module parameter identities. Its structured backward emits input,
weight, and bias adjoints as one semantic operation with two executable
dispatches. Each output scalar has one invocation and gathers contributions in
a fixed order, so grouped and depthwise accumulation requires no float atomics.
Conv1d saves its input, OIK weight value, stride, padding, dilation, and optional
module parameter identities. Its input and parameter adjoints use the donor's
explicit deterministic kernels and appear as one structured semantic backward
operation. The forward remains the canonical donor im2col/GEMM composition;
the newer compositional GEMM backward is not yet the active Rust route.
ConvTranspose1d saves its input, IOK weight value, stride, padding, dilation,
and optional module parameter identity. Its structured backward records four
physical nodes as one semantic operation: Conv1d im2col, tiled MatMulNt,
batched transpose, and the existing deterministic weight adjoint. The unused
normal-convolution bias adjoint is private temporary storage and never becomes
a parameter or semantic output.
ConvTranspose2d saves its input, IOKK weight, stride, padding, and optional
weight/bias parameter identities. Its structured backward emits deterministic
input, weight, and bias adjoints through two physical dispatches. The reverse
node and finite-difference oracle are present, but the current hardware run did
not pass the device-enumeration boundary and is not yet runtime evidence.
GRU records its recurrence-free input projection through the ordinary Linear
node. Its recurrent node saves projected gates, previous hidden states,
recurrent weight/bias values, optional Parameter identities, and bias mode.
Backward emits one whole-sequence BPTT dispatch producing input-gate and
hidden-gate adjoints, then reuses Linear parameter backward across all `B*S`
rows for recurrent weight and bias gradients. The input-gate adjoint flows
through reshape into the earlier Linear node, which owns input, input-weight,
and input-bias gradients. Parameter versions are preflighted before either
backward dispatch is recorded.
GRU cell reverse mode preserves the caller-owned hidden identity and combines
its direct update-gate adjoint with the recurrent Linear data adjoint. The
pointwise, Linear-data, Linear-parameter, and addition dispatches remain one
structured semantic backward operation.
Embedding backward assigns one invocation to each table scalar and gathers
matching token contributions in input order, giving deterministic repeated-index
accumulation without float atomics. This is a correctness baseline, not a
performance qualification.

Activation adjoints preserve the donor formulas and saved-state choices. ReLU,
tanh, sigmoid, ELU, and Softplus use the forward result where that is sufficient;
SiLU, GELU, Mish, and Leaky ReLU retain the input. The Leaky ReLU choice follows
the donor shader rather than its stale output-saved schema comment.

Softmax preserves the selected signed `dim` attribute and saves its forward
output. Its adjoint applies `s * (dout - sum(dout * s))` independently to each
selected-axis slice; it does not recompute Softmax or flatten non-final axes.
LogSoftmax uses the same axis contract and saves its forward output; its
adjoint applies `dout - exp(output) * sum(dout)` per slice.

AvgPool2d saves the input identity plus its square kernel, stride, and padding.
Its deterministic input adjoint assigns one invocation to each input element
and gathers every overlapping output contribution. Border windows divide by
their number of valid input elements, exactly as forward does, so padding does
not dilute edge values or their gradients.

MaxPool2d additionally saves its U32 flat-input argmax Matrix. The adjoint uses
the same one-invocation-per-input ownership and gathers gradients from every
overlapping output whose saved argmax equals that input. This avoids float
atomics and preserves deterministic accumulation order.

AdaptiveAvgPool2d saves its requested output height and width. Forward derives
each spatial bin with floor/ceiling integer boundaries; backward assigns one
invocation per input and gathers every bin containing it with the same divisor.

Upsample saves the positive integer scale and interpolation mode. Nearest
backward gathers the exact scale-by-scale output block owned by an input pixel.
Bilinear backward gathers every half-pixel sample weight that references the
input pixel, including duplicated clamped edge coordinates, and is therefore a
deterministic equivalent of the donor's atomic scatter.

## Acceptance

Every differentiable slice requires:

- an independent forward oracle and analytic or central finite-difference
  adjoint;
- forward/backward shape and dtype rejection;
- odd, zero, repeated-index, and branch-accumulation cases where meaningful;
- saved-version rejection before any partial backward recording;
- repeated backward and zero-gradient lifecycle checks;
- capture/replay equivalence and deterministic traversal where claimed;
- an end-to-end loss gate for its admitted model path.

The current hardware suite is:

```bash
cargo test --all-features --test ml -- --ignored --test-threads=1
```

The captured-training oracle additionally requires every reached attachment to
be completed and at least one backward operation to carry forward provenance.
Core validation, synchronization validation, GPU-assisted validation, broader
branch accumulation, and schema-generated attachment coverage remain required
before promotion.
