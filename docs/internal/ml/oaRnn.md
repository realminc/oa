# OARS Elman RNN

**Status:** Experimental forward and complete BPTT baseline

**Updated:** 2026-09-08

## Contract

`oa::ml::nn::Rnn` is a batch-first, stacked Elman recurrent module:

```text
h[b,t] = tanh(
    input[b,t] @ weight_ih^T + bias_ih
  + h[b,t-1] @ weight_hh^T + bias_hh
)
```

Input has shape `[B, S, I]`; output has shape `[B, S, H]`. Each layer starts
from zero hidden state and feeds its complete output sequence into the next
layer. Parameter order is layer order, then `weight_ih`, `weight_hh`,
`bias_ih`, and `bias_hh`. The current public baseline always has biases and
admits `1 <= H <= 1024` because the scan stores one hidden vector in shared
memory.

## Execution

Forward records one whole-sequence Vulkan scan per layer, with one workgroup per
batch sequence. Hidden recurrence stays on the GPU; there are no per-timestep
host submissions, slices, or temporary Matrix values.

Backward records one deterministic full-BPTT dispatch per layer. The baseline
assigns the complete batch to one workgroup, walks time in reverse, and owns
every input and parameter-gradient element without float atomics. It produces
input, input-weight, recurrent-weight, and both bias gradients. This serialized
parameter accumulation is a correctness baseline, not a performance claim.
Future parallel, segmented, or fused implementations must preserve the same
result and remain private kernel routing choices.

## Autograd and ownership

The RNN owns four stable `Parameter` handles per layer but no engine or
submission state. Tape nodes save the input, output, previous hidden sequence,
the two weight values, and all four parameter versions. Backward preflights
versions before recording any adjoint. The input gradient reconnects to earlier
Embedding, reshape, or recurrent nodes by Matrix semantic identity.

Stateful streaming `step`, caller-provided initial/final hidden state,
bidirectionality, dropout, bias-free layers, packed variable-length sequences,
and truncated BPTT are Planned. They must not be simulated through hidden
mutable module state.

## Evidence

The Rust hardware test covers an actual character-model chain:

```text
U32 tokens -> Embedding -> Rnn -> reshape -> Linear -> cross-entropy
           <- embedding gradient <- complete BPTT <- parameter gradients
```

Forward values match an independent host recurrence. Embedding and all four
recurrent parameter gradients match central finite differences, including a
repeated token. A separate two-layer test verifies shape, finite values,
parameter order, and hidden-limit/input-width rejection.

Run:

```bash
cargo test --all-features --test ml -- --ignored --test-threads=1
```

Synchronization validation, GPU-assisted validation, odd/large shape packs,
performance qualification, long-sequence stability, and tutorial-level
convergence remain open.
