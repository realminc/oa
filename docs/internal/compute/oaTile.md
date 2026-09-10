# OaTile — Generated Cross-Vendor Kernel Construction

**Status:** Planned architecture; current FP32 matmul is one Experimental fixed tile

**Updated:** 2026-09-10

OaTile is the private construction system for a bounded family of linear
algebra and bandwidth kernels. It separates reusable algorithm pieces from
device-qualified combinations so OARS can improve performance without copying
thousands of opaque per-GPU kernels or betting on one universal schedule.

## Portability contract

OaTile is portable at the semantic and construction level, not in the claim
that one tile is fast everywhere. Every family has:

- one precise mathematical/storage contract;
- a portable bounds-checked baseline;
- a finite set of generated candidates;
- queried capability predicates;
- independent oracle and validation fixtures;
- device-pack qualification and tuning evidence.

Kernel selection remains private. Vendor names, architecture codes, tile sizes
and tuning knobs do not appear in public Matrix/Image/Vision APIs.

An OaTile logical tile also defines an ownership domain. A mutable output or
workspace partition belongs to exactly one invocation, subgroup, workgroup, or
logical tile unless the candidate declares an atomic or ordered-reduction
policy. The logical grid should derive from checked partition extent and tail
policy rather than from an unrelated unchecked host calculation. This contract
is generated candidate metadata and diagnostic evidence; it is not a Rust
borrow exposed by the public API.

NVIDIA's `cutile-rs` is useful research evidence for this design because it
derives grid geometry and exclusive mutable ownership from host-side tensor
partitioning. OaTile does not adopt its CUDA runtime, AST capture, CUDA Tile IR,
or public programming surface. OARS continues to lower its private construction
model into explicit Slang/SPIR-V candidates for Vulkan; see the
[CUDA Rust assessment](oaCudaRust.md).

## Component model

```text
Operation/partition contract
  -> operand layouts and movement
  -> compute atom
  -> main loop
  -> work scheduler
  -> epilogue tree
  -> tail/bounds strategy
  -> complete candidate
  -> Slang specialization/module
  -> reflected SPIR-V artifact
```

### Compute atoms

- scalar/vector arithmetic baseline;
- subgroup arithmetic where subgroup size/operations are queried;
- cooperative-matrix atom for exact supported types, scopes and dimensions;
- packed dot/dequantize atom for an admitted encoding.

### Operand movement

- direct coalesced global access;
- shared-memory staging with bank/layout policy;
- transposed or swizzled tiles;
- asynchronous/cooperative loading only under an exact supported mechanism;
- image sampling/storage paths with explicit format and layout semantics.

### Main loops and schedulers

- output-stationary tiled loops;
- small-M/N specialization;
- split-K with explicit reduction;
- Stream-K-like static or dynamic tile assignment;
- grouped/batched descriptor traversal;
- persistent work queues for narrowly qualified workloads;
- 1D/2D bandwidth traversal for conversion/normalization microfusions.

### Epilogue tree

The generated tree may contain store, scale, beta/source accumulation, bias,
residual, activation, clamp, quantize/dequantize, dtype/layout conversion and
auxiliary output. A candidate advertises exactly which trees it implements.
Arbitrary user shader injection is outside the semantic contract.

## Device profiles and packs

A capability profile records relevant facts, not marketing families:

- subgroup sizes and controls;
- workgroup/invocation and shared-memory limits;
- cooperative-matrix properties;
- scalar/vector storage and arithmetic features;
- descriptor and buffer/image limits;
- memory heaps, cache-relevant observations and integrated/discrete topology;
- timestamp support, driver/conformance and stable device identity.

The portable pack targets the repository's admitted Vulkan baseline. Optional
packs group candidates qualified on devices with similar facts: mobile
bandwidth/thermal, integrated unified-memory, desktop discrete, and accelerator
classes. A pack can use vendor/device matching for qualification but always
rechecks capabilities and never changes semantics.

## Variant budget

Do not generate the Cartesian product. A candidate enters a pack only when:

1. it represents a materially different schedule or capability;
2. a tracked workload shape needs it;
3. it passes the full oracle/validation matrix;
4. it wins or fills a correctness/capability gap under canonical measurement;
5. its artifact and maintenance cost fit the pack budget.

Retire a losing Experimental candidate instead of carrying it indefinitely.
Shipped stable IDs are reserved and never reused.

The initial useful set should be closer to tens of candidates per important
family than thousands per architecture. Expand from observed gaps: small-M,
tail-heavy, wave-underfilled, bandwidth-bound, reduction-bound, low-precision
and fused preprocessing.

## Selection interface

OaTile generates candidates; it does not own public routing. A provider such as
OaBlasLt supplies the full problem, receives legal candidates, and returns an
immutable plan through the common selector/tuner. Selection keys include every
tile, atom, scheduler, epilogue, specialization and artifact hash.

## First implementation slices

1. Describe the existing FP32 64x64x16 matmul as one generated OaTile
   candidate without changing behavior.
2. Add one smaller portable tile and a small-M candidate; prove exact routing
   diagnostics.
3. Add deterministic split-K and a workspace-free fallback.
4. Add strided batch/grouped descriptors.
5. Add a bandwidth family for Image convert + normalize, then resize +
   normalize after image sampling semantics land.
6. Add cooperative-matrix atoms one dtype/device profile at a time.
7. Add Stream-K-like scheduling only after tile-wave profiles show a gap.

## Validation

Candidate generation must prove uniqueness, bounded count and deterministic
output. Candidate validation also rejects missing or contradictory writable
partition/collision policy. Kernel tests cover zero/one/odd/tail/large practical
shapes, alignment, strides, deliberate partition overlap, aliasing, workspace
limits, out-of-bounds poison regions, numeric special values and deterministic
mode. Reflection and `spirv-val` gate every artifact. GPU-assisted validation
and device-specific profilers qualify real hardware.

Performance admission requires an equivalent portable baseline, exact route and
fallback identity, at least seven fresh-process samples, median/spread and raw
artifacts. A vendor profiler explains a result; it does not replace that gate.

## Donor and licensing rule

OA donor Slang algorithms are ported with provenance. ROCm source is research
evidence for architecture, scheduling and tests, not automatically copied code.
Before adapting a kernel or generator, record its license, exact source path,
algorithmic versus literal adaptation and independent verification plan.

## Primary references

- [hipBLASLt Stream-K](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.0/how-to/how-to-use-streamk.html)
- [Slang SPIR-V target support](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html)
- [Vulkan cooperative matrix properties](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#spirvenv-capabilities-table-CooperativeMatrixKHR)
- [NVlabs cutile-rs](https://github.com/NVlabs/cutile-rs)
