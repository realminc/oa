# OARS reverse-mode differentiation

**Status:** Experimental narrow implementation; generalized engine Planned

**Updated:** 2026-09-08

## Public contract

`oa::ml::GradientTape` is a thread-affine recording scope. Ordinary operations
run eagerly against their Matrix engine and record semantic edges into the
innermost active tape. `backward(&loss)` closes and consumes the tape, validates
saved parameter versions, and records adjoint operations. It does not submit or
wait; Matrix observation and explicit engine boundaries retain that authority.
Backward rejects another live tape on the same thread so adjoint construction
cannot leak semantic records into a nested or outer recording scope.

The current root must be the scalar F32 result of mean cross-entropy. Supported
paths are composed from:

- `ml::nn::Linear`;
- `ml::nn::Embedding` with U32 indices;
- `ml::nn::LayerNorm` over the final dimension;
- tanh-approximate `ml::gelu`;
- multi-head causal scaled dot-product attention;
- equal-shape FP32 Matrix addition for Transformer residual paths;
- stacked `ml::nn::Rnn` whole-sequence scans;
- zero-copy `Matrix::reshape` / `matrix::reshape` views;
- `ml::loss::cross_entropy`;
- gradient addition for converging paths.

This is not a generalized autograd claim. Broadcasting, reductions, arbitrary
roots, detached/no-grad scopes, higher derivatives, other recurrent families,
and the remaining Matrix/ML operation catalog are Planned.

## Identity and ownership

A Matrix clone preserves one semantic value identity. A reshape creates a new
semantic identity while sharing storage and readiness state. Core Matrix code
publishes only a private backend-neutral reshape record; it does not depend on
ML tape policy or Vulkan.

Concrete tape nodes are private. A node saves only the Matrix values, stable
`Parameter` handles, shapes, and versions required by its adjoint. Parameters
own the single live accumulated gradient. The engine remains sole owner of
allocation, command recording, queues, synchronization, and submission.

During execution capture, each reached differentiable node attaches its
schema-owned forward output and nonzero tape sequence to the semantic graph.
After that node emits its adjoint work, the attachment is completed with the
contiguous semantic-operation range produced by the backward step. Operations
in that range carry the owning forward operation and tape sequence. Submission
or capture abort advances the recording generation, so a retained numerical
tape cannot amend a later graph with stale operation identifiers. Metadata-only
reshape nodes preserve their value lineage but own no independent operation.

## Mutation safety

Every saved parameter captures its version. Backward validates every saved
version before recording any gradient work, so an optimizer update between
forward and backward fails with `FailedPrecondition` rather than constructing a
partially stale backward pass. AdamW is the first private in-place writer: it
records explicit read/write access over stable Parameter storage and advances
the version immediately after successful recording. General shared-storage
mutation versions do not exist because OARS has not admitted public in-place
Matrix mutation.

## Determinism and accumulation

LayerNorm backward recomputes row statistics from the saved input and produces
input, affine-weight, and affine-bias adjoints. Its input adjoint is proven
through an Embedding predecessor, rather than only inspecting leaf parameters.

Embedding backward assigns one invocation to each table scalar and gathers all
matching token contributions in input order. Repeated indices therefore use a
deterministic scatter-add result without float atomics. This is the correctness
baseline, not a performance qualification; routed atomic, sorted, or segmented
implementations may replace it only behind the same semantic contract.

## Acceptance

Every differentiable slice requires an independent forward oracle, central
finite differences or an analytic adjoint, repeated-index and odd-shape cases,
saved-version rejection, optimizer-update proof, and an end-to-end loss gate.
The current hardware suite is:

```bash
cargo test --all-features --test ml -- --ignored --test-threads=1
```

The captured training oracle additionally requires every tape attachment to be
expanded and at least one backward operation to carry forward provenance. Core
validation, synchronization validation, GPU-assisted validation, and broader
branch accumulation remain required before promotion.
