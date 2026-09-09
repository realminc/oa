# OARS neural-network modules

**Status:** Experimental recursive module ownership; persistence Planned

**Updated:** 2026-09-08

## Current implementation

`oa::ml::nn::Linear`, `oa::ml::nn::Embedding`, `oa::ml::nn::LayerNorm`, and
`oa::ml::nn::Rnn` are state-owning Rust types implementing the object-safe
`oa::ml::Module` trait.
Their `forward` methods validate semantic inputs, call private domain lowering,
and attach tape metadata when a trainable parameter participates. They do not
own an engine, queue, command buffer, kernel registry, or submission path.

`Parameter` is a stable `Rc<RefCell<_>>` handle around one F32 Matrix value, its
optional accumulated gradient, its `requires_grad` policy, and a mutation
version. It is deliberately not another tensor/value type. OARS types are
currently thread-affine, matching the engine.

`Linear` owns weight `[O, I]` and bias `[O]`. `Embedding` owns weight `[V, D]`,
accepts arbitrary-shape U32 indices, and returns `indices_shape + [D]` rather
than OA C++'s flattened gather result. The richer shape is an intentional Rust
API improvement; callers can obtain `[N, D]` through the zero-copy differentiable
reshape view when required.

`Rnn` owns four parameters per layer, accepts `[B, S, I]`, and returns
`[B, S, H]`. Its whole-sequence scan and complete BPTT remain device operations;
the module does not submit once per timestep. See [OARS Elman RNN](oaRnn.md).

`LayerNorm` owns affine weight and bias vectors `[D]`, normalizes the final
dimension of any nonempty F32 input, and preserves its shape. Its forward and
complete adjoint are schema-owned operations; the generic baseline favors a
stable two-pass variance calculation over the subtractive variance formula.

`MultiHeadAttention` owns four registered Linear projections and applies causal
scaled dot-product attention to packed `[B*S,D]` values. `TransformerBlock`
owns the two pre-normalization modules, attention, and two-layer GELU FFN; both
residual additions participate in reverse-mode traversal. The generic causal
attention baseline supports any positive head count dividing `D` and is proven
with a two-head forward and Q/K/V finite-difference case.

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
explicit persistence flag, but serialization is not implemented yet.

The Vulkan integration proof builds an owned
`Embedding -> Rnn -> Linear` character model, observes seven unique parameter
paths and one persistent buffer, propagates evaluation mode, runs complete
BPTT, verifies every parameter receives a gradient, and performs one AdamW
step. It also exercises ambiguous-ownership rejection.

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
5. Add persistence evidence once the checkpoint format and roundtrip contract
   exist.

Fusion remains private lowering policy. A fused route must preserve the same
public values and adjoints and must have its own correctness and performance
evidence.
