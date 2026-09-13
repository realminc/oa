# OARS Mamba-3

**Status:** Experimental complete FP32 SISO/MIMO module, training, and recurrent inference

**Updated:** 2026-09-12

**Donor authority:** OA C++ `fnMatrixMamba3.cpp`, `autogradMamba3.h`,
`mamba3Preprocess*.slang`, the complete SISO forward/backward/step shader
families, the MIMO forward/backward/step shader family, and `testMamba3.cpp`

## Admitted Rust surface

`oa::ml::matrix::mamba3_preprocess` splits a packed `[rows, projected_width]`
FP32 projection into x, z, B, C, dt, A-times-dt, trap, and angle values. Its
explicit `Mamba3PreprocessConfig` owns every behavior-changing geometry and
numerical bound. B and C are RMS-normalized independently per
`num_groups * mimo_rank` state row; dt uses clamped stable softplus and
A-times-dt uses the donor heavy-tail transform. The forward is one physical
dispatch and one eight-output semantic operation.

`mamba3_preprocess_backward` returns packed-projection and dt-bias adjoints as
one semantic operation. OARS mechanically adapts the donor backward into a
row-local fused stage followed by an increasing-row dt-bias reduction. This
removes the donor's contended floating-point CAS accumulation without changing
the formula. Crate-global identities 416 through 418 are used because donor-
local identity 204 is already the OARS Cryptography Merkle-copy identity.

One private tape node owns all eight output identities. Reverse traversal
collects every reached output adjoint, creates exact zero adjoints for outputs
that do not lead to the scalar root, and records the fused backward once. It
does not reproduce the donor's eight counter-coupled nodes or eight backward
dispatches.

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

`oa::ml::nn::Mamba3` owns the donor parameter set and exact projection →
preprocess → SISO or MIMO scan → optional gated RMSNorm → output-projection path. Parameter
names remain `in_proj`, `dt_bias`, `B_bias`, `C_bias`, `D`, `out_proj`, and
optional `mimo_x`, `mimo_z`, `mimo_o`, and `norm_weight`, so native `.oam`
checkpoints have one stable module identity instead of a compatibility wrapper.
Rank one selects SISO. Ranks two through eight preserve the donor shared-state
MIMO geometry and differentiate all fifteen operands through one tape node.

`Mamba3::new_state` creates a zeroed `Mamba3State` for one module instance and
batch size. `Mamba3::step` advances exactly one `[B,1,D]` token. The cache is an
explicit caller-owned value containing SSM, cumulative-angle, previous-key,
and previous-value storage; it is not hidden mutable module state. A state from
another module instance and recurrent execution inside an active gradient tape
fail before recording. The low-level `mamba3_siso_step` operation mutates those
four buffers, exposes four corresponding semantic alias outputs, and therefore
advances their SSA versions while retaining storage.

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

For `length >= 64`, `head_dim/state <= 32`, `rope_angles <= 8`, and no more
than 256 MiB of saved state history, lowering selects the donor's chunk-16
pipeline. Prepare, forward summary/prefix, independent-chunk gate recovery,
reverse summary/prefix, reverse scan, finalization, and one OARS deterministic
completion reduction form a single nine-dispatch semantic transaction. The
completion stage replaces the donor's four nested general-purpose reductions;
it does not change the recurrence or public graph.

All remaining admitted shapes use the donor bounded-memory generic reverse.
It stores chunk boundaries and recomputes one 32-token chunk at a time, then
uses one deterministic OARS completion reduction. Slang 2026.5 crashes while
optimizing this donor kernel when reflection is enabled, so its schema row
explicitly selects `-O0` and the exact subgroup ballot/arithmetic capabilities.
That compiler workaround is local to this artifact and does not change other
shader optimization.

The donor P16 reverse shader used subgroup arithmetic. The OARS adaptation
uses a deterministic 16-lane groupshared reduction, avoiding an implicit
subgroup-size capability requirement while retaining the same mathematical
reduction. This is a portability adaptation, not a new recurrence.

Every tensor and private temporary must fit the shader's U32 indexing ABI.
The current routes additionally admit:

- forward: `head_dim <= 128`, `state <= 128`, `rope_angles <= 64`, and
  `2 * rope_angles <= state`;
- backward: short, chunked, or bounded generic lowering over the full forward
  geometry; the router applies the narrower limits above only to select a
  faster physical provider.
- MIMO: `mimo_rank` in `1..=8`, `head_dim * state <= 4096`, and the same
  forward geometry bounds.

Zero batch, sequence, head, group, head width, or state extents are invalid for
this stateful recurrence. The current dense Matrix dtype is FP32; BF16 islands
remain outside the admitted Rust dtype vocabulary rather than silently
falling back or changing precision.

## Evidence

On Intel Iris Xe TGL GT2, Mesa 26.2.2, Vulkan 1.4.354:

- forward matches an independent CPU implementation of the complete
  recurrence on a deterministic `[2,6,2,3,4,2]` geometry;
- the short backward matches central finite differences for every element in
  all eleven gradient families on an odd-state geometry;
- the bounded generic backward matches every element in all eleven gradient
  families at sequence length 17;
- the optimized chunked backward matches every element in all eleven gradient
  families at its sequence-length-64 routing boundary;
- grouped Q/K forward matches explicitly expanded per-head Q/K, and grouped
  B/C adjoints match deterministic sums of the expanded per-head adjoints;
- all ten Slang modules compile and pass `spirv-val --target-env vulkan1.3`;
- fused preprocess forward matches an independent host implementation;
- packed-projection and dt-bias adjoints match central finite differences for
  every element of a small odd-state geometry;
- all eight preprocess output branches merge through one tape node, while a
  single-output path proves unused outputs contribute exact zero adjoints.
- donor gated RMSNorm matches an independent forward oracle and complete
  finite differences for input, gate, weight, and optional bias;
- the parameter-owning SISO module produces finite gradients for all 172
  scalars in a small complete-block test;
- six recurrent primitive steps match the full grouped-SISO scan, and five
  complete module steps including gated RMSNorm and both projections match
  full-sequence module output;
- MIMO forward matches an independent host recurrence, MIMO step matches its
  full scan, and all fifteen adjoint families match central finite differences;
- the rank-two parameter-owning module registers 356 scalars, produces finite
  gradients for every parameter, and matches full-sequence and recurrent-step
  execution;
- the canonical 10,915-parameter Char Mamba-3 model completes 300 AdamW steps
  at learning rate 0.003 with initial loss 3.303064, evaluation loss 0.190578,
  final-batch accuracy 0.929688, and exact greedy continuation `to be that is
  the question whether tis nobler in the mind to suffer the slings and ar`;
- the release SDK tutorial completed the same gate on the recorded device in
  2.46 seconds wall time, reporting 8.19 ms/step wall and 7.482 ms/step mean GPU
  time. These are one-run functional diagnostics, not canonical benchmark
  claims.

The normal repository generator, Rust test, Clippy, strict rustdoc, formatting,
and generated-idempotence gates remain required at each checkpoint.

Engine creation now queries and enables available core `shaderInt64` plus
Vulkan 1.2 `shaderInt8`, `storageBuffer8BitAccess`, and
`uniformAndStorageBuffer8BitAccess` features. It derives each embedded
artifact's requirements from its SPIR-V `OpCapability` instructions, creates
only pipelines admitted by the selected device's features and subgroup
properties, and rejects an unavailable kernel while building the executable
graph, before semantic or output-readiness state changes. Unknown capability
values fail artifact validation rather than bypassing admission. A fresh
core-validation SISO run and a separate
synchronization-validation chunked-backward run complete without a validation
message on the recorded Intel device. A GPU-assisted SISO forward run also
completes without an OA VUID or shader-memory report; the layer emits only its
own limit-adjustment and disabled ray/mesh-validator warnings. This is still
not universal device qualification.

## Remaining qualification surface

The complete donor FP32 SISO/MIMO operation and module topology is connected.
Remaining work is broader dtype support, the Android-specific bounded global-
scratch provider, canonical fixed-clock OA C++ differentials, and broader
cross-device validation. These additions must reuse the existing Module,
autograd, execution-plan, iterator, callback, and model-file owners; they must
not introduce a parallel Mamba runtime or copy historical performance claims.
