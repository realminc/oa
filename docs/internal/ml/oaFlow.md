# OA Rust Flow Models

**Status:** Experimental

**Updated:** 2026-09-11

**Architecture:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Donor:** OA C++ `FlowTimeEmbedding`, `FlowTransformer`, `FlowDenoiser`, and
`test/cpp/ml/generative/testFlow.cpp`

## Admitted model stack

The Rust port exposes one registered model family under `oa::ml::nn`:

```text
FlowTimeEmbedding
  F32 [B] or [B,1]
  -> sin/cos frequencies on the owning Engine
  -> F32 [B,D]

FlowTransformer
  F32 [B,S,D] or [B*S,D]
  -> bidirectional dense or dropless-MoE Transformer blocks
  -> optional binary key mask [B,S] or [B,S,1]
  -> optional AdaLN-Zero condition [B,D]
  -> final LayerNorm
  -> shape-preserving output

FlowDenoiser
  F32 sample [B,S,I]
  -> input projection + learned position [S,D]
  -> time embedding + optional projected condition
  -> FlowTransformer
  -> output projection
  -> F32 [B,S,I]
```

`Engine` remains the sole execution, storage, queue, and synchronization owner.
The model modules contain registered child modules, parameters, and buffers;
they do not submit or wait. All forward paths record through existing Matrix
and ML operations.

## Donor-preserving decisions

- Time frequencies are `scale * exp(-log(max_period) * i / (D/2))`, stored as
  one nonpersistent `[1,D/2]` buffer. The batch path uses Mul, Sin, Cos, and
  Concat on device.
- Every Flow Transformer block is bidirectional. `0/0` expert geometry selects
  a dense GELU FFN; `0 < K <= E` selects the shared dropless MoE.
- A token mask becomes the donor additive key mask `(mask - 1) * 1e4`, repeated
  over heads and query positions.
- Adaptive conditioning uses the exact zero-initialized six-part scale, shift,
  and residual-gate path already owned by `TransformerBlock`.
- Conditioning dropout is per sample and does not use inverted scaling. Rust
  derives the exact binary keep mask from the existing replay-safe inverted
  Dropout operation, then cancels its scale. Time conditioning is never
  dropped.
- Classifier-free guidance enters a bounded evaluation scope and computes
  `unconditional + (conditional - unconditional) * guidance_scale`.
- The learned position Matrix is one registered `Parameter`. Forward records
  it as a tape leaf before normal broadcast addition; there is no duplicate
  parameter or denoiser-specific gradient implementation.

## Rust adaptations and current boundary

Constructors accept an explicit `Engine` and seed. Invalid donor contracts
return `oa::Error` instead of terminating through a C++ assertion. Optional
empty donor Matrices become `Option<&Matrix>`. Flow model inputs are currently
F32; the donor time embedding's implicit cast from other scalar types remains
unported and is not silently emulated.

The donor `oa::FnFlow` surface is connected idiomatically as `oa::ml::flow`:

- `linear_match` returns state and velocity from one semantic operation and one
  single-pass provider. It accepts scalar, batch-vector, or ordinarily
  broadcastable F32 time;
- `euler_step` advances state through one semantic operation and one provider;
  and
- `masked_mse` mechanically adapts the donor fused workgroup reduction and
  exact prediction adjoint, including its broadcast mask and all-padding zero.

OA C++ lowers the first two through generic Matrix compositions inside an
`OpLoweringScope`. OARS now has the equivalent private transactional lowering
scope, first exercised by categorical policy. Linear Match and Euler retain
their already-qualified single-pass providers as explicit behavior-equivalent
replacements: they remove intermediates while retaining independent host and
reverse-mode oracles and one public semantic identity. Masked MSE retains the
donor shader math. All three are schema-owned; they are functions, not module
methods.

The image/motion generative tutorials remain Planned.

## Evidence

The ignored hardware suite in `test/rs/ml/test_flow.rs` covers:

- the exact independent CPU time-frequency oracle;
- dense and MoE construction, shape preservation, and bidirectional mode;
- padding-key invariance for both FFN families;
- complete reverse mode through every registered denoiser parameter, including
  learned position;
- exact linear-match/Euler forward reconstruction and endpoint adjoints;
- masked-MSE value, all-padding, broadcast, and parameter-adjoint oracles;
- one semantic graph identity for each public Flow operation;
- CFG identities at guidance scales zero and one; and
- invalid geometry, expert routing, time shape, dropout, and guidance inputs.

The current hardware qualification is Intel Iris Xe, Mesa 26.2.2, Vulkan
1.4.354. Run it serially:

```bash
cargo test --all-features --test ml flow -- --ignored --test-threads=1
```
