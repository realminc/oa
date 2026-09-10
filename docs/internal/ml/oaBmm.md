# OARS batched matrix multiplication

**Status:** Experimental connected donor port

**Updated:** 2026-09-10

**Authority:** [OA Rust ML foundation](oaMl.md)

**Donor references:** OA C++ `mlFnMatrixAttention.toml`, `fnMatrixBmm.cpp`,
and `ops/bmm{,Nt,Tn}{,Tiled16}.slang`

## Semantic operations

`oa::ml::matrix` owns three same-engine F32 operations:

```text
bmm     [N, M, K] @ [N, K, P]   -> [N, M, P]
bmm_nt  [N, M, K] @ [N, P, K]^T -> [N, M, P]
bmm_tn  [N, K, M]^T @ [N, K, P] -> [N, M, P]
```

Inputs must be nonempty rank-three matrices with equal batch and compatible
inner extents. Shape and shader-ABI arithmetic is checked before allocation.
The functions record work through the owning `Engine`; they do not submit or
wait.

## Private physical routing

Each semantic operation owns a generic kernel and a lowering-only tiled-16
provider. OARS preserves the donor's exact route: tiled FP32 execution is used
when `N <= 65535` and `M`, `K`, and `P` are all at least eight; every other
admitted shape uses the generic one-output-per-invocation kernel. Tiled providers
remain private and retain their semantic operation as graph provenance.

The schema records one-element exclusive writes for generic kernels and
exclusive 16-by-16 workgroup tiles for tiled kernels. Stable IDs 317 through
322 are OARS compatibility identities; earlier committed OARS operations
already occupy the donor-local ID range.

## Reverse mode

Reverse mode composes the same family rather than adding duplicate derivative
kernels:

```text
C = bmm(A, B)     dA = bmm_nt(dC, B)  dB = bmm_tn(A, dC)
C = bmm_nt(A, B)  dA = bmm(dC, B)     dB = bmm_tn(dC, A)
C = bmm_tn(A, B)  dA = bmm_nt(B, dC)  dB = bmm(A, dC)
```

These nodes participate in the ordinary `GradientTape` lifecycle and preserve
semantic value identities. Kernel selection is not exposed through autograd.

## Evidence and limits

Independent host oracles cover all three layouts on generic and odd tiled
shapes. Capture evidence checks the selected physical kernel and semantic
owner. Finite differences cover both operands of every reverse formula, and
invalid rank, batch, inner-dimension, and dtype contracts fail closed. The
hardware suite passes on Intel Iris Xe with Mesa 26.2.2 and Vulkan 1.4.354.

Only F32 is admitted. Broadcasting, empty batches, strided storage, mixed
precision, vendor libraries, and tuned architecture-specific providers remain
unimplemented. The standard donor attention route now composes BMM-NT and BMM
around scaled/masked Softmax; BMM does not itself own attention semantics. See
[scaled dot-product attention](oaAttention.md).
