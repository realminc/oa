# OARS neural-network modules

**Status:** Experimental recursive module ownership and native persistence

**Updated:** 2026-09-10

## Current implementation

`oa::ml::nn::Linear`, `oa::ml::nn::Embedding`, `oa::ml::nn::LayerNorm`,
`oa::ml::nn::BatchNorm2d`,
`oa::ml::nn::{Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d}`, and
`oa::ml::nn::{RnnCell, Rnn, GruCell, Gru}` are state-owning Rust types implementing the object-safe
`oa::ml::Module` trait.
Their `forward` methods validate semantic inputs, call private domain lowering,
and attach tape metadata when a trainable parameter participates. They do not
own an engine, queue, command buffer, kernel registry, or submission path.

`Parameter` is a stable `Rc<RefCell<_>>` handle around one F32 Matrix value, its
optional accumulated gradient, its `requires_grad` policy, and a mutation
version. It is deliberately not another tensor/value type. OARS types are
currently thread-affine, matching the engine.

`Linear` owns weight `[O, I]` and an optional bias `[O]`, accepts input
`[..., I]` with rank at least two, and preserves every leading dimension. A
bias-free Linear retains only a private physical zero vector for the current
GEMM ABI; it registers and exposes no fake trainable bias. `Embedding` owns weight `[V, D]`,
accepts arbitrary-shape U32 indices, and returns `indices_shape + [D]` rather
than OA C++'s flattened gather result. The richer shape is an intentional Rust
API improvement; callers can obtain `[N, D]` through the zero-copy differentiable
reshape view when required.

`Rnn` owns weights `[H, I]` and `[H, H]` per layer plus optional paired biases
`[H]`, accepts `[B, S, I]`, and returns `[B, S, H]`. Each layer lowers through
one canonical Linear input projection and one whole-sequence recurrent scan;
complete BPTT similarly reuses the Linear adjoints. Bias-free layers retain
private immutable zero values for lowering without registering fake parameters.
The module does not submit once per timestep. See [OARS Elman RNN](oaRnn.md).
`RnnCell` exposes the same one-layer parameter layout with explicit `[B, I]`
input and caller-owned `[B, H]` hidden state. Its ordinary `Module::forward`
convenience starts from a newly allocated zero state.

`Gru` owns weights `[3H, I]` and `[3H, H]` per layer plus optional paired
biases `[3H]`, accepts `[B, S, I]`, and returns `[B, S, H]`. It preserves the
donor reset/update/candidate ordering and zero initial state. Each layer lowers
to one batched Linear input projection and one whole-sequence recurrent scan;
backward uses one BPTT scan and the existing Linear parameter-adjoint route.
Bias-free construction retains private immutable zero values for lowering but
does not register or train fake bias parameters. See [OARS GRU](oaGru.md).
`GruCell` exposes the same one-layer parameters while accepting explicit
`[B,I]` input and caller-owned `[B,H]` hidden state through `step`. Its ordinary
`Module::forward` convenience starts from a newly allocated zero state.

`LayerNorm` owns affine weight and bias vectors `[D]`, normalizes the final
dimension of any nonempty F32 input, and preserves its shape. Its forward and
complete adjoint are schema-owned operations; the generic baseline favors a
stable two-pass variance calculation over the subtractive variance formula.

`BatchNorm2d` owns affine weight/bias parameters and persistent running
mean/variance buffers `[C]`. Training computes centered per-channel population
variance over N, H, and W, normalizes the batch, and records a functional
momentum update that replaces both registered state handles without a host
read. Evaluation consumes the current running state without mutating it. The
training and inference input adjoints are distinct; inference treats running
statistics as fixed state. This deliberately corrects the donor backward path,
which accepted but did not honor its training-mode flag.

`Conv2d` owns grouped OIHW weight
`[output_channels, input_channels / groups, kernel, kernel]` and channel bias.
Its public geometry preserves the donor stride, symmetric padding, and group
contract while rejecting zero dimensions and non-divisible channel counts.
Forward is one semantic operation; reverse mode records one structured semantic
adjoint lowered into deterministic input- and parameter-gradient kernels. The
module initializer is seeded and uses a fan-in-scaled symmetric uniform weight
distribution with zero bias.

`Conv1d` owns OIK weight `[output_channels, input_channels, kernel]` and
channel bias. Its NCL forward preserves the donor's im2col, tiled MatMulNt,
bias-broadcast, and batched-transpose lowering rather than reviving the retired
scalar forward kernel. Stride, symmetric padding, and dilation are explicit.
The current structured backward uses OA's retained deterministic data and
parameter adjoint kernels; replacing those with the donor's newer compositional
GEMM adjoint remains a performance-parity gate for wide training workloads.

`ConvTranspose1d` is the bias-free adjoint of Conv1d. It owns IOK weight
`[input_channels, output_channels, kernel]` and preserves the donor module's
stride, symmetric padding, and fixed unit dilation. Its public Matrix operation
also admits explicit dilation. Forward reuses the Conv1d data-adjoint math;
backward reuses the ordinary Conv1d im2col/tiled-GEMM route for its input
adjoint and the existing deterministic parameter adjoint for its weight. It
therefore adds no second convolution algorithm or bias parameter.

`ConvTranspose2d` owns IOKK weight
`[input_channels, output_channels, kernel, kernel]` and output-channel bias.
Its NCHW forward preserves the donor's Conv2d-data-adjoint plus channel-bias
lowering. Its structured backward uses ordinary Conv2d math for the input
adjoint and the donor deterministic transposed-convolution parameter reduction.
Signed intermediate coordinates replace the donor shader's unsigned-underflow
expressions. The implementation, reflected ABI, and independent tests are
connected; live GPU oracle execution remains pending after the current Vulkan
loader returned `ERROR_INITIALIZATION_FAILED` during device enumeration.

`RmsNorm` owns one weight vector `[D]`, normalizes the final dimension of any
nonempty F32 input, and has a complete input/weight adjoint. `Ffn` is the donor
SwiGLU feed-forward module: it registers `norm`, `gate`, `up`, and `down`
children, applies `RMSNorm -> (gate, up) -> SwiGLU -> down`, and adds the input
residual. It introduces no second operation or lowering route. Its Vulkan
contract test proves the seven donor parameter paths and gradients through the
entire composition.

`Rope` is a parameterless adapter over the schema-owned rotary-position
operation. It preserves the donor's split-half Llama pairing, exact theta-base
and position-offset semantics, and transpose adjoint while rejecting odd head
dimensions that would leave donor output elements unwritten.

`Swiglu` preserves the donor MLP rather than duplicating `Ffn`: it owns direct
`gate_weight`, `up_weight`, and `down_weight` parameters, optionally owns the
three matching biases, flattens and restores arbitrary leading dimensions, and
has no residual or normalization. Biasless mode uses private immutable zero
values for the existing Linear lowering but does not expose or train fake bias
parameters. Both parameter layouts complete reverse-mode traversal on Vulkan.

`Relu`, `Gelu`, and `Silu` are parameterless donor-compatible operation
adapters. They still implement `Module` so callers can place them in ordinary
module trees, but their registries are empty and their `forward` methods route
to the same `ml::matrix` operation authority used by free-function callers.
The remaining admitted activation functions stay stateless operations until a
donor NN module contract requires a named adapter.

`Softmax` and `LogSoftmax` are parameterless donor reduction adapters. Their
defaults select the last dimension and an explicit signed dimension preserves
the C++ axis contract. The module and free-function paths use
`matrix::{softmax, log_softmax}`; both saved-output adjoints return through the
Core Matrix schema rather than ML-owned duplicate kernels.

`AvgPool2d` and `MaxPool2d` are parameterless donor NCHW pooling adapters. Their
one-argument constructors use a non-overlapping square window; `with_options`
selects kernel size, stride, and symmetric padding. Average-pool padding
positions contribute to neither the sum nor its divisor. Max-pool returns an
explicit U32 argmax Matrix from the free operation, selects the first equal
maximum in window order, and retains those indices for its deterministic
adjoint. Both module registries remain empty.

`AdaptiveAvgPool2d` accepts a square output size through `new` or independent
height and width through `with_output_size`. It uses exact floor/ceiling bin
boundaries per axis, including rectangular and non-divisible input geometry.
This is a recorded correction: the donor-generated class accepts one extent,
its older tests call two, and its implementation derives width behavior from
height through a square fixed-window approximation.

`Upsample` is the parameterless adapter for integer-scale NCHW interpolation.
Its default is donor-compatible bilinear interpolation; `with_mode` selects
nearest or align-corners-false half-pixel bilinear behavior. Both physical
candidates retain one `ml::matrix::upsample_2d` semantic identity. Their input
adjoints use deterministic gather ownership; the bilinear path therefore does
not depend on floating-point atomics or their scheduling order.

`Identity` preserves the input's semantic Matrix identity. `Flatten` creates a
differentiable zero-copy reshape view over an inclusive PyTorch-style dimension
range and retains the donor `(1, -1)` default. Invalid resolved dimensions fail
with `InvalidArgument` rather than a process assertion. `Sequential` owns child
modules through its one `ModuleRegistry`, assigns deterministic `layer_N` names
or accepts an explicit name, and forwards in registration order. It does not
keep a second child list or execution graph.

`MultiHeadAttention` owns four registered Linear projections and applies scaled
dot-product attention to packed `[B*S,D]` values. Its explicit
`AttentionMode` selects causal or bidirectional visibility, and its
`AttentionBackend` selects Auto, Standard, or the fail-closed causal Flash
provider. Auto retains Standard because current Intel evidence does not qualify
Flash. Arbitrary additive masks use the Standard route and explicit Flash
rejects them. Configured attention dropout applies only in training mode. It
composes BMM-NT, scaled/masked Softmax, replay-safe Matrix Dropout, and BMM;
immutable causal/zero masks are prepared once per encountered batch geometry
and cached instead of adding a mask dispatch to every replay. Runtime sequence
length may change without rebuilding projection weights. `TransformerBlock`
owns the two pre-normalization modules, attention, and two-layer GELU FFN; both
residual additions participate in reverse-mode traversal. Conditioned dense
and MoE constructors additionally own a zero-initialized
`adaptive_modulation` Linear child from `C` to `6D`. Their AdaLN-Zero forward
repeats each `[B,6D]` modulation row across the sequence, applies the donor's
attention and feed-forward scale/shift/gate order, and starts as an exact
identity residual path. Rust construction replaces the donor's mutable
`enableAdaptiveConditioning`: the complete child registry is fixed before the
block is returned. Core `matrix::repeat_interleave` supplies the differentiable
rank-one through rank-four expansion; its adjoint corrects the donor's
non-leading-axis coordinate decomposition by reducing with input dimensions
and expanded-output strides. The generic causal
attention route supports any positive head count dividing `D` and is proven
with a two-head forward and Q/K/V finite-difference case. Module-policy tests
prove Standard/Flash forward equivalence, bidirectional visibility, additive
masking, bias-free projection ownership, AdaLN-Zero dense/MoE identity and
adaptive-parameter gradients, and TransformerBlock policy propagation, masked
forward, and sequence-length changes. Captured train/eval
and reverse evidence additionally proves that
attention Dropout advances through replay and regenerates its exact adjoint
mask.

`Transformer` is the Rust spelling of donor `NnTransformer`. It owns token and
position embeddings, an arbitrary positive stack of registered `block_N`
children, final normalization, and vocabulary projection. The canonical
character tutorial is now only a fixed-configuration adapter over this module;
it no longer owns a second Transformer implementation.

## Recursive ownership contract

A module owns one `ModuleRegistry` and registers during construction:

- direct trainable parameters;
- named non-trainable buffers;
- named child modules.

Registration takes `&mut ModuleRegistry`; after a module is shared through
`Rc`, its structure is fixed. Child modules are owned `Rc<dyn Module>` handles,
while parameters remain stable `Rc<RefCell<_>>` handles. The registry does not
borrow objects that can outlive their owners and does not own an engine.

Local names are nonempty ASCII identifiers without dots. Parameters, buffers,
and children share one local namespace. Dotted recursive paths such as
`recurrent.layer0.weight_ih` are derived from the tree rather than supplied by
callers. Registration rejects duplicate local parameter, buffer, and child
identity. Recursive traversal also rejects a child, parameter, or buffer
identity reached through two independently constructed branches, so an
optimizer cannot silently update one handle twice and persistence cannot see
ambiguous aliases. `AdamW` independently rejects duplicate handles at its own
boundary.

`parameters` and `named_parameters` expose only direct parameters.
`all_parameters`, `all_named_parameters`, and `all_named_buffers` traverse in
deterministic depth-first registration order. `num_parameters` counts trainable
scalar values with checked arithmetic. `train`, `eval`, and the RAII
`scoped_eval` guard propagate through the same child tree. Buffers retain an
explicit persistence flag. Native `.oam` checkpoints serialize persistent
buffers into State and exclude non-persistent buffers. Named buffer records
retain stable interior handles so a validated restore updates the registry's
live value without rebuilding the module tree.

The Vulkan integration proof builds an owned
`Embedding -> Rnn -> Linear` character model, observes seven unique parameter
paths and one persistent buffer, propagates evaluation mode, runs complete
BPTT, verifies every parameter receives a gradient, and performs one AdamW
step. The model-file proof additionally restores that persistent buffer while
leaving a non-persistent buffer unchanged. It also exercises ambiguous-ownership
rejection.

The C++ proposal to prefix every layer `Nn*` is not carried over. Rust's
`oa::ml::nn` module already distinguishes stateful layers, so `nn::Linear` and
`nn::Embedding` are canonical spellings.

## Adding a layer

1. Decide whether it owns state or is a stateless semantic operation.
2. Add any new operation contract, shader ABI, artifact, autograd relation, and
   oracle to the owning schema before exposing the module.
3. Keep backend selection and submission below the public layer.
4. Prove forward values, gradients, parameter traversal, an optimizer update,
   odd shapes, invalid dtypes/ownership, and reuse behavior.
5. Extend the native `.oam` evidence for every newly admitted parameter or
   persistent-state representation.

Fusion remains private lowering policy. A fused route must preserve the same
public values and adjoints and must have its own correctness and performance
evidence.
