# OARS Mamba-3

**Status:** Experimental connected SISO primitive

**Updated:** 2026-09-12

**Donor authority:** OA C++ `fnMatrixMamba3.cpp`, `autogradMamba3.h`,
`mamba3SisoFwd.slang`, the short-backward shader family, and
`TestMamba3.SisoGroupedQkMatchesExpandedHeadsShort`

## Admitted Rust surface

`oa::ml::matrix::mamba3_siso` owns the stateless selective-state recurrence.
Its explicit `SsmConfig` fixes the tensor geometry and optional z-gate and D
skip. The SISO boundary requires rank one and rejects MIMO output
normalization rather than silently ignoring either field.

The eleven operands preserve the donor layouts:

```text
C, B                 [batch, length, groups, state]
x, z, output         [batch, length, heads, head_dim]
A-times-dt, dt, trap [batch, length, heads]
angle                [batch, length, rope_angles]
C bias, B bias       [heads, state]
D                    [heads]
```

Zero groups means one Q/K group per head. Otherwise the head count must be
divisible by the group count, and each head reads the state group selected by
that quotient. All values are same-Engine FP32 matrices. The forward records
one generated semantic operation and one physical dispatch without host
observation.

`mamba3_siso_backward` returns all eleven explicit adjoints. `GradientTape`
records the forward as one node and routes those adjoints to every watched
predecessor by semantic value identity. The public operation therefore does
not expose the physical backward stages or create a second autograd path.

## Donor recurrence and physical lowering

The forward mechanically preserves cumulative rotary angle, interleaved
rotation of B/C state channels, exponential decay, trapezoidal shifted gate,
selective outer-product state update, D skip, and optional SiLU z gate. OARS
retains the donor stable semantic identities 342 for forward and 341 for
backward.

The short backward is one semantic operation lowered transactionally to six
physical kernels: prepare, saved-state reconstruction, reverse scan,
postprocessing, batch/head reduction, and grouped-Q/K reduction. Temporary
matrices remain private runtime values. Physical IDs 411 through 415 are
lowering-only identities; callers and autograd see only the backward semantic
operation.

The donor P16 reverse shader used subgroup arithmetic. The OARS adaptation
uses a deterministic 16-lane groupshared reduction, avoiding an implicit
subgroup-size capability requirement while retaining the same mathematical
reduction. This is a portability adaptation, not a new recurrence.

Every tensor and private temporary must fit the shader's U32 indexing ABI.
The current routes additionally admit:

- forward: `head_dim <= 128`, `state <= 128`, `rope_angles <= 64`, and
  `2 * rope_angles <= state`;
- backward: the donor short route only, with `length <= 16`,
  `head_dim <= 16`, `state <= 32`, and `rope_angles <= 8`.

Unsupported long or wide backward shapes return `MissingCapability` before
recording any physical work. Zero batch, sequence, head, group, head width, or
state extents are invalid for this stateful recurrence.

## Evidence

On Intel Iris Xe TGL GT2, Mesa 26.2.2, Vulkan 1.4.354:

- forward matches an independent CPU implementation of the complete
  recurrence on a deterministic `[2,6,2,3,4,2]` geometry;
- the short backward matches central finite differences for every element in
  all eleven gradient families on an odd-state geometry;
- grouped Q/K forward matches explicitly expanded per-head Q/K, and grouped
  B/C adjoints match deterministic sums of the expanded per-head adjoints;
- all seven Slang modules compile and pass `spirv-val --target-env vulkan1.3`.

The normal repository generator, Rust test, Clippy, strict rustdoc, formatting,
and generated-idempotence gates remain required at each checkpoint.

## Remaining donor surface

This slice does not yet admit the generic/chunked long-sequence backward,
recurrent step and cache, preprocessing projections, gated output RMSNorm,
the parameter-owning `nn::Mamba3` module, shared-state MIMO forward/backward/
step, or the NLP tutorial. Those must reuse this semantic primitive and the
existing Module, autograd, execution-plan, iterator, callback, and model-file
owners. They must not introduce a parallel Mamba runtime or copy the donor's
historical performance claims into Rust.
