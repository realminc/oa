# OARS Elman RNN

**Status:** Experimental donor-backed sequence scan and explicit-state cell

**Updated:** 2026-09-10

**Donor references:** OA C++ `ml/nn/rnn/rnn.cpp`,
`ml/fnmatrix/nn/fnMatrixRnn.cpp`, `rnnScan.slang`, `rnnScanBwd.slang`, and
`mlFnMatrixRecurrent.toml`

## Contract

`oa::ml::nn::Rnn` is a batch-first, stacked Elman recurrent module:

```text
gates_i[b,t] = input[b,t] @ weight_ih^T + bias_ih
h[b,t] = tanh(gates_i[b,t] + h[b,t-1] @ weight_hh^T + bias_hh)
```

Input has shape `[B, S, I]`; output has shape `[B, S, H]`. Every layer starts
from a zero hidden state and feeds its complete output sequence into the next
layer. The public constructor preserves the existing biased default;
`with_seed_and_bias(..., false, ...)` and `from_weights` admit the donor's
bias-free contract. Parameter order is layer order, then `weight_ih`,
`weight_hh`, and the two biases when present. The current scan admits
`1 <= H <= 1024` because one workgroup retains the hidden vector in shared
memory.

## Execution ownership

The Rust module now follows the donor decomposition rather than owning a second
combined RNN implementation:

1. reshape `[B, S, I]` to `[B*S, I]` without copying storage;
2. run the canonical `linear` operation once for the complete input projection;
3. reshape its result to `[B, S, H]` without copying storage;
4. run one `rnn_scan` workgroup per batch element across all timesteps.

The scan owns only the recurrent dependency. Hidden state remains in
group-shared memory while time advances, so the host does not submit per-step
slices, matrix multiplications, or pointwise operations. Bias presence is a
structural forward attribute; freezing a bias never removes its numerical
contribution.

Stable operation IDs 29 and 30 now name `rnn_scan` and `rnn_scan_backward`;
IDs 315 and 316 name `rnn_cell` and `rnn_cell_backward`.
Their shader metadata, semantic contracts, registry entries, and autograd
pairing come from `tools/gen/fn/schema/ml_training.json`. The old monolithic
`rnn` and `rnn_backward` Rust kernels and lowering path are removed.

## Backward decomposition

`rnn_scan_backward` performs the reverse-time recurrence once per batch:

```text
delta_t = (d_output_t + d_hidden_from_t_plus_1) * (1 - h_t * h_t)
d_gates_i_t = delta_t
d_gates_h_t = delta_t
d_hidden_from_t = delta_t @ weight_hh
```

The semantic backward operation lowers to the donor scan adjoint followed by
the existing Linear parameter adjoint over saved previous-hidden rows. The
ordinary input projection's existing Linear node then computes the original
input, `weight_ih`, and `bias_ih` gradients. This preserves one operation
authority for Linear validation and gradient math instead of duplicating it in
RNN.

The tape saves the projected gates, previous-hidden sequence, recurrent weight,
optional recurrent bias, and relevant parameter versions. It reconnects the
projected-gate gradient through the two reshape views to the Linear node and
then to preceding modules by semantic value identity.

## Evidence

Generator tests pin IDs 29/30 and 315/316, manual recurrent autograd ownership, donor
provenance, and regeneration idempotence. Build-time Slang compilation,
SPIR-V reflection, and the complete Rust ML test binary compile with the new
route. The sequence hardware oracle passes on Intel Iris Xe with Mesa 26.2.2
and Vulkan 1.4.354:

```text
U32 tokens -> Embedding -> Rnn -> reshape -> Linear -> cross-entropy
           <- embedding gradient <- complete BPTT <- parameter gradients
```

It checks forward values against an independent host recurrence and checks all
four biased parameter gradients plus the embedding gradient against central
finite differences. The stacked test also passes bias-free registration,
shape, finite output, parameter order, hidden-size rejection, and input-width
rejection. Focused cell tests pass on the same device and additionally cover a nonzero caller-owned hidden
state, zero-state construction, all four parameter adjoints by finite
difference, frozen-bias semantics, bias-free registration, and invalid input
width.

Run the focused gate with:

```bash
cargo test --all-features --test ml training::character_rnn_forward_and_complete_bptt_match_independent_oracles -- --ignored --exact
cargo test --all-features --test ml training::stacked_rnn_preserves_batch_sequence_and_parameter_order -- --ignored --exact
cargo test --all-features --test ml rnn::rnn_cell_parameter_gradients_match_finite_differences -- --ignored --exact
```

Returning the final state from a sequence, stateful streaming sessions,
bidirectionality, dropout, packed variable-length sequences, and truncated
BPTT remain Planned. They must not be simulated through hidden mutable module
state.
