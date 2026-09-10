# OARS scaled dot-product attention

**Status:** Experimental connected standard and Flash donor port

**Updated:** 2026-09-10

**Authority:** [OA Rust ML foundation](oaMl.md)

**Donor references:** OA C++ `fnMatrixAttention.cpp`,
`fnMatrixFlashAttention.cpp`, `mlFnMatrixAttention.toml`, and the attention,
Softmax, and BMM Slang families

## Public semantic route

`oa::ml::matrix::scaled_dot_product_attention` accepts Q, K, and V as
same-engine F32 `[BH,S,Dh]` matrices. An optional additive mask has shape
`[BH*S,S]`; causal masking is an independent Boolean contract. The operation
returns `[BH,S,Dh]` context and privately retains the `[BH,S,S]` probabilities
needed by reverse mode.

The packed compatibility function accepts `[B*S,D]` values and is only
composition:

```text
split_heads(Q, K, V)
  -> scaled_dot_product_attention(BH,S,Dh; optional mask; causal)
  -> merge_heads(context)
```

There is no second packed-attention semantic kernel. `split_heads` preserves
the donor `[B*S,D] -> [B*H,S,D/H]` permutation, and `merge_heads` is its exact
inverse. The one-head case is a differentiable zero-copy reshape view.

`oa::ml::matrix::flash_attention_causal` is the donor compatibility spelling
for explicitly requesting the fused provider. It records the same SDPA semantic
operation rather than publishing another operation identity.

## Standard physical lowering

One standard SDPA semantic operation lowers to three executable nodes:

```text
bmm_nt(Q, K) -> scaled/masked Softmax -> bmm(P, V)
```

The Softmax node owns the SDPA semantic contract, while the complete three-node
range is recorded as its physical provenance. Absence of the optional mask is
preserved as an absent semantic input rather than represented by a fake Matrix.
The narrow-row Softmax provider is selected when `S <= 32`; broader rows use
the generic donor schedule. Both implement implicit causal masking and optional
additive masking without exposing provider choice through the public API.

## Reverse mode

The saved probability matrix and existing batched products define the adjoint:

```text
dP = bmm_nt(dO, V)
dScores = softmax_scaled_masked_backward(P, dP, scale)
dQ = bmm(dScores, K)
dK = bmm_tn(dScores, Q)
dV = bmm_tn(P, dO)
```

The additive mask is detached. Head-transform adjoints use the inverse
transform. Backward records ordinary semantic operations and does not submit or
wait.

The Flash provider instead retains one FP32 log-sum-exp per query row. Its
adjoint recomputes probabilities and uses two deterministic dispatches: one
workgroup per query row owns `dQ`, and one workgroup per key row owns `dK` and
`dV`. No floating-point atomic accumulation is required.

## Evidence and limits

Independent host oracles cover causal and additive-mask forward execution,
generic and narrow Softmax schedules, head permutation/round-trip behavior, and
invalid geometry. Finite differences cover Q, K, and V. Capture evidence proves
one SDPA semantic operation over BMM-NT, Softmax, and BMM executable nodes and
preserves optional-mask presence. Differential tests prove the Flash forward
against the standard causal route and its Q/K/V adjoints against finite
differences, including executable and semantic provenance. These tests pass on
Intel Iris Xe with Mesa 26.2.2 and Vulkan 1.4.354.

`nn::MultiHeadAttention` owns the four Linear projections and exposes the donor
`AttentionMode::{Causal,Bidirectional}` and
`AttentionBackend::{Auto,Standard,Flash}` policy. `Auto` remains Standard until
canonical device-specific evidence qualifies another provider. Explicit Flash
is fail-closed unless visibility is causal, storage is F32, and `S <= 1024`;
arbitrary additive masks always use Standard or fail when Flash was explicitly
requested. `TransformerBlock` forwards its visibility policy to this same
registered attention child, exposes the same masked forward, and can change
runtime sequence length without rebuilding its projection or FFN parameters.
Cached attention masks are keyed by both batch and current sequence geometry.

When configured dropout is nonzero and the module is training, the Standard
module path expands into BMM-NT, scaled/masked Softmax, Matrix Dropout, and BMM.
The existing schema-owned replay transformation advances the Philox stream and
the adjoint regenerates the exact forward mask. Causal and all-zero masks are
immutable geometry values cached once per batch size; they do not add a kernel
dispatch to each captured replay. Evaluation returns to the ordinary Standard
SDPA route. Flash rejects a module configured with nonzero dropout even while
that module is in evaluation mode, matching the donor's explicit eligibility
contract.

Only F32 contiguous storage is admitted. The Flash provider is causal-only and
accepts no additive mask. Cross-attention with unequal sequence lengths,
broadcast masks, mixed precision, qualified automatic provider selection, and
further architecture-specific providers remain Planned.

The checked-in Transformer component profiler compares both providers under the
same captured `[64,16,32]` workload. One exploratory Intel Xe run with five
warmups and fifteen timed replays measured 0.058 ms standard versus 0.236 ms
Flash forward, and 0.230 ms versus 0.704 ms forward plus backward. This is not a
release performance claim: it is single-process evidence that Flash must not be
selected automatically on this device. `Auto` and the packed Transformer module
therefore continue to use the standard route until the canonical fresh-process
benchmark matrix proves a device policy.
