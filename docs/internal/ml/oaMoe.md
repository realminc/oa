# OARS Mixture of Experts

**Status:** Experimental connected sparse core, dense oracle, and balancing objectives

**Updated:** 2026-09-11

**Donor authority:** OA C++ `docs/internal/ml/oaMoe.md`,
`source/cpp/lib/oa/ml/nn/moe/moe.cpp`, `mlFnMatrixMoe.toml`, and the MoE
system/gradient tests

## Admitted contract

`oa::ml::nn::Moe` is a pre-normalized sparse top-k feed-forward module with a
residual connection. Its current forward is:

```text
input [T,D]
  -> RMSNorm
  -> router Linear [D,E]
  -> unbiased Softmax probabilities
  -> optional persistent routing bias on selection logits only
  -> deterministic TopK
  -> selected-probability normalization [T,K]
  -> stable dropless expert-major plan
  -> gather [T*K,D]
  -> grouped Linear [E,2H,D]
  -> SiLU(first H) * second H
  -> grouped Linear [E,D,H]
  -> direct gated combine [T,D]
  -> zero or more always-on shared SwiGLU expert deltas
  -> residual add
```

The module owns the RMSNorm and router as registered children. Routed experts
are four direct persistent parameters in the donor checkpoint layout:

- `expert_gate_up_weight [E,2H,D]`;
- `expert_gate_up_bias [E,2H]`;
- `expert_down_weight [E,D,H]`;
- `expert_down_bias [E,D]`.

No per-expert child modules or per-forward concatenation graph exists. The
constructor clamps routes per token into `1..=E`, matching OA C++, and rejects
more than the current route planner's 256-expert limit.

Sparse grouped execution is the default. `set_sparse_execution(false)` selects
the donor's opt-in mathematical oracle: it constructs a dense selected gate,
evaluates every expert independently from a Slice view of the same four stacked
parameters, weights each expert result, and sums the deltas. The Rust lowering
uses an equivalent per-expert accumulation instead of materializing donor
`[T,E,D]` Concat storage. It is a differential reference, not a performance
fallback or a second parameter owner.

`with_seed_and_shared_experts` optionally registers canonical `Swiglu`
children as `shared_expert_N`. Each child consumes the same normalized input
and contributes its complete ungated delta, preserving the donor DeepSeekMoE
specialization anchor without creating a second expert implementation.

The module also owns persistent non-trainable `routing_bias [1,E]`. A positive
`set_balance_rate` enables it for the TopK decision only; gate magnitudes still
come from the unbiased router Softmax. `update_routing_bias` records the donor
GPU update from the latest selection mask and neither submits nor reads back.
Overloaded experts move down by gamma, underloaded experts move up by gamma,
and exactly balanced experts do not change. The default rate is zero, so newly
constructed modules retain the earlier selection behavior.

An independent opt-in `set_aux_loss_alpha` policy records the donor
Switch/GShard objective
`alpha * E / K * sum(mean(mask, 0) * mean(probabilities, 0))`. The hard mask
remains detached, while the unbiased probability mean differentiates through
Softmax and the router. `aux_loss()` is an OARS FP32 scalar (`[]`) so callers
can add it directly to an ordinary task-loss scalar before backward. The
coefficient defaults to zero; a disabled forward resets the result to a stable
zero scalar and cannot expose a stale prior loss.

`set_router_z_loss_beta` independently enables the donor router objective
`beta * mean(logsumexp(logits)^2)`. OARS preserves the numerically stable donor
identity `logits[:,0] - log_softmax(logits)[:,0]`; it does not substitute the
overflow-prone `log(sum(exp(logits)))` form. When both balancing losses are
enabled, `aux_loss()` is their ordinary Matrix sum.

## Execution and differentiation

Routing indices and the expert plan are device-resident detached values. Gate
magnitudes remain differentiable through unbiased Softmax probabilities. Stable
packing, grouped projections, concatenated SiLU-multiply, and direct combine
all record ordinary semantic operations through the one Engine-owned session.
Forward and backward do not submit, wait, inspect offsets, or build route maps
on the CPU.

Core `matrix::{add, mul}` own multidirectional broadcast lowering. They retain
their original exact-shape candidates and select schema-owned rank-eight F32
broadcast candidates only when shapes differ; Add additionally admits wrapping
I32. Their reverse paths reduce each adjoint over its broadcast axes before
returning it to the saved input shape. Core `scale` also has its donor scalar
adjoint. The MoE routing-bias value and hard selection mask are gradient-free,
while both gate and Switch-loss router gradients remain valid.

The tape accepts any recorded FP32 scalar root and seeds it with one. Loss
nodes consume their upstream scalar, which makes `task_loss + aux_loss()` one
ordinary differentiable graph rather than a MoE-specific backward route.
The z-loss path also ports Core rank-one through rank-four FP32 `matrix::slice`
and its donor MatrixCopyRegion adjoint. `matrix::sub`, Slice, LogSoftmax, Mul,
Sum, Scale, and Reshape therefore form one normal reverse graph.

The dense oracle records the four stacked parameter values as tape leaves.
Slice adjoints from every expert accumulate by stable value identity before one
gradient is committed to each registered Parameter. Core MatMulNt now carries
its donor reverse contract; OARS composes the existing BMM NN/TN providers for
its two rank-two adjoints. Reciprocal carries its donor derivative and combines
with broadcast Mul for selected-gate normalization.

Grouped Linear backward is one semantic operation lowered into two executable
nodes: a tiled input adjoint and a fused expert weight/bias adjoint. The latter
stages each output-gradient tile once and lets only the first input tile write
each expert bias row. Bias-free `grouped_gemm_m` shares the variable-row tiled
geometry while using its donor weight-only adjoint. Empty experts therefore
write exact zero parameter gradients without atomics.

`route_stats` is the explicit telemetry boundary. It records two axis
reductions, then reads only the resulting `[E]` load and probability vectors.
Normalized load entropy, peak-load ratio, and dead-expert count are computed
on the host from those reduced values; the `[T,E]` matrices are never copied.

## Evidence

The Intel Iris Xe Vulkan gate covers:

- deterministic expert-major planning and exact route maps;
- selected route-weight forward and finite-difference backward;
- gather/combine forward, closed-form reverse, and invalid contracts;
- grouped Linear forward across expert-boundary tiles;
- exact grouped `dX`, `dW`, and `dB` against a closed-form host oracle;
- bias-free grouped GEMM forward and exact `dX`/`dW`, including an empty
  expert, against an independent closed-form host oracle;
- full sparse-module output and selection mask against an independent CPU MoE;
- end-to-end finite differences for input, RMSNorm weight, router weight/bias,
  and all four stacked expert parameters;
- exact shared-expert composition against a separately constructed canonical
  SwiGLU path, recursive parameter registration, and nonzero shared-expert
  gradient with top-1 routed execution;
- exact reduced route statistics for a tied-router collapse case: load
  fractions, mean probabilities, normalized entropy, peak ratio, and dead
  count;
- donor-equivalent device-resident routing-bias mutation and persistent-buffer
  registration;
- a tied-router two-forward gate proving that bias changes only expert
  selection while stored gate probabilities remain exactly uniform;
- exact combined Switch/z-loss normalization, router-bias adjoint, and
  disabled-loss reset;
- sparse/dense output parity and all four stacked expert-gradient comparisons
  on one parameter owner;
- closed-form MatMulNt and reciprocal reverse-mode values;
- general rank-aligned F32 Add/Mul and wrapping-I32 Add broadcasting, plus
  reverse tests proving sum-to-input-shape adjoints and composed scalar roots.

This proves the admitted small-shape FP32 correctness contract. It is not a
grouped-GEMM performance or cross-vendor qualification claim.

## Remaining OA parity

The following donor features are still Planned and are not represented by fake
methods or fallback behavior:

- Byte/BPE NLP rows, OA C++ checkpoint conversion, and OaDna grouped-provider
  integration. The Char-MoE Transformer block, captured 300-step training loop,
  generation, and native `.oam` round trip are connected;
- broader dtype and tuned irregular grouped-GEMM providers.

These enter in donor dependency order. Host telemetry may synchronize only at
an explicit metric boundary and may read reduced `E`-scalar values, never the
token/expert route matrix.
