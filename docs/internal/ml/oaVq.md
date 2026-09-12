# OARS vector quantization

**Status:** Experimental connected donor port

**Updated:** 2026-09-11

**Donor authority:** OA C++ `fnMatrixVq.cpp`, `vq.cpp`, `vqAssign.slang`,
`vqEmaUpdate.slang`, and `testVqAssign.cpp`

## Public contract

`oa::ml::matrix` owns the stateless primitives:

- `detach` creates a zero-copy Matrix view with a fresh semantic identity and
  no reverse-mode edge;
- `vq_assign` maps FP32 latent rows `[N,D]` to the nearest FP32 codebook rows
  `[K,D]`, producing I32 indices `[N]` and gathered FP32 values `[N,D]`;
- `vq_lookup` decodes I32 indices through a codebook without attaching a
  codebook gradient;
- `vq_ema_update` produces fresh `embed_sum`, `cluster_size`, and `codebook`
  Matrix values from the previous state.

`oa::ml::nn::VectorQuantizer` owns one persistent codebook and its two EMA
buffers. They are registered buffers, never gradient parameters. Its
`quantize` method returns the donor straight-through value
`z_e + detach(z_q - z_e)` and `beta * mean((z_e - z_q)^2)`. The hard assignment
and codebook path remain detached; the encoder receives the identity adjoint
plus the commitment-loss adjoint.

`ResidualVectorQuantizer` registers independently owned child quantizers as
`level0`, `level1`, and so on. It assigns each running residual, sums selected
codes, applies one straight-through estimator to the sum, and retains each
level's residual and token indices for its EMA transition.

## Numerical and state parity

Assignment scans code rows in ascending order and updates only on a strictly
smaller squared L2 distance. Equal-distance ties therefore choose the lower
index. EMA preserves the donor update order:

```text
N_k = decay * N_k + (1 - decay) * count_k
m_k = decay * m_k + (1 - decay) * sum(z_e assigned to k)
e_k = m_k / max(N_k, epsilon)
```

Optional normalization rescales only each codebook row to unit RMS; `embed_sum`
retains raw EMA units. A code below `dead_threshold` is copied from the latent
row selected by the donor's exact wrapping-u32 `(code, step)` hash, and its EMA
population becomes one.

OARS represents this update as fresh semantic SSA outputs and then replaces the
values behind the same registered `NamedBuffer` handles. This differs from the
donor's physical in-place writes but preserves module state identity,
checkpoint traversal, equations, ordering, and deferred execution.

## Synchronization and seeding

Quantization, lookup, commitment loss, residual composition, and EMA update only
record device work. They do not submit or wait.

`seed` is intentionally exceptional: like OA C++, it is an initialization
completion boundary. Matrix square, axis sum, TopK, and VQ lookup rank and
select the highest-L2-norm rows entirely on the GPU with lower-index tie
breaking. The result initializes both the codebook and `embed_sum`, while
`cluster_size` starts at ones. The module installs those values into its
persistent buffer handles and waits for one exact checkpoint. Residual VQ
repeats this greedily on the residual left by each already seeded level.

## Evidence and remaining work

Hardware-backed Intel Vulkan tests cover exact assignment and I32 tokens,
strict tie behavior, quantized values, commitment loss, straight-through
gradient identity through an Embedding parameter, EMA equations, exact
dead-code revival, unit-RMS normalization, highest-norm seeding, lookup,
residual decode equivalence, and recursive persistent-buffer ownership.

The donor VQ tutorial/model consumer, broader odd-size randomized differential
pack, performance qualification, and persistence of the non-buffer EMA step
counter remain pending. A model-file round trip preserves all three registered
state buffers but is not yet an exact mid-training resume claim because revival
depends on that step counter.
