# OARS gated recurrent unit

**Status:** Experimental implementation; hardware qualification pending

**Updated:** 2026-09-10

**Donor references:** OA C++ `Gru`, `GruCell`, recurrent operation schemas,
`gruScan.slang`, `gruScanBwd.slang`, and their numerical-gradient tests

## Contract

`oa::ml::nn::Gru` is a batch-first stacked recurrent module. Input uses
`[batch, sequence, input]`; output uses `[batch, sequence, hidden]`. Every
layer starts from a zero hidden state and owns weights `weight_ih [3H, I]` and
`weight_hh [3H, H]`. Biased construction additionally owns `bias_ih [3H]` and
`bias_hh [3H]`; bias-free construction registers only the two weights.
`oa::ml::nn::GruCell` owns the same one-layer contract and exposes `step` for
caller-supplied `[B,H]` state plus `zero_state` for explicit initialization.

Gate order and update equations match OA C++:

```text
r = sigmoid(i_r + h_r)
z = sigmoid(i_z + h_z)
n = tanh(i_n + r * h_n)
h_next = (1 - z) * n + z * h_previous
```

The current groupshared implementation requires `1 <= H <= 1024`. Inputs and
parameters are same-engine F32 values. Invalid rank, dimensions, dtype, engine
ownership, unpaired biases, or gate geometry return `oa::Error` before graph
recording.

## Lowering and autograd

Each layer reshapes `[B,S,I]` to `[B*S,I]` and uses the existing Linear
operation once to produce all input gates. One `gru_scan` dispatch then loops
over the sequence with hidden state resident in groupshared memory. It writes
both `[B,S,H]` output and the previous-hidden saved state required by reverse
mode. There is no host timestep loop and no per-timestep submission.

Reverse mode records `gru_scan_backward` as one semantic operation with two
physical nodes. The first is the donor BPTT scan and produces input-gate and
hidden-gate adjoints. The second reuses the existing Linear parameter-adjoint
kernel over all `B*S` rows for recurrent weight and bias gradients. The
input-gate adjoint traverses the earlier reshape and Linear node, which owns the
original input projection gradients.

`gru_cell_backward` is likewise one semantic operation. Its four physical
stages are the donor gate pointwise adjoint, the existing Linear recurrent-data
adjoint, the existing Linear recurrent-parameter adjoint, and deterministic
addition of direct and recurrent hidden gradients.

## Evidence and limits

The checked-in acceptance pack contains independent multi-batch scan and
nonzero-hidden cell forward oracles, central finite differences for all four
parameter groups on both paths, stacked/bias-free/frozen-bias registration
checks, and invalid-input validation. Schema
generation, Slang compilation, SPIR-V validation, and reflected ABI checks run
without a device. The current sandbox returns `ERROR_INITIALIZATION_FAILED`
during Vulkan physical-device enumeration, so those hardware tests remain
ignored and this document makes no runtime or performance claim yet.

Returning the final hidden state from the sequence module, caller-supplied
sequence initial state, bidirectionality, dropout between layers, packed
sequences, broader dtypes, and routed specialized kernels remain Planned. They
must extend this operation family rather than introduce a timestep-driven
second implementation.
