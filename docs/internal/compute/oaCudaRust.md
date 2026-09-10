# CUDA Rust Assessment for OARS

**Status:** Research; the first OARS write-ownership slice is Experimental

**Updated:** 2026-09-10

This note records what OARS should and should not take from NVIDIA's initial
CUDA Rust projects. It is an architecture input, not an OARS CUDA-backend plan
or a capability claim. The canonical compute boundary remains
[OA Rust Compute Architecture](oaCompute.md), and shipping OARS kernels remain
Slang compiled to SPIR-V for Vulkan.

## Audited projects

NVIDIA announced two complementary Rust programming levels on 2026-09-08:

| Project | Programming level | Compilation and execution | Safety mechanism relevant to OARS |
|---|---|---|---|
| `cuda-oxide` | explicit SIMT threads, blocks, shared memory and synchronization | a custom `rustc` backend lowers Rust MIR through Pliron/LLVM to PTX; generated host launchers use the CUDA runtime | `DisjointSlice` expresses per-thread exclusive writes; a launch contract validates geometry and device limits before producing a `PreparedLaunch` |
| `cutile-rs` | logical tensor tiles, with physical thread and memory-layout mapping delegated to the compiler | a captured Rust AST is JIT-compiled through CUDA Tile IR to a cubin | mutable tensors are partitioned into non-overlapping tiles and ownership remains live across lazy launch and completion |

Both projects are early-stage NVIDIA research software. `cuda-oxide` is alpha
and currently depends on a pinned nightly Rust toolchain; `cutile-rs` uses
stable Rust but still describes incomplete features and expected API breakage.
Both target CUDA/NVIDIA hardware rather than Vulkan/SPIR-V. Their repositories
and requirements are volatile, so this dated audit must be repeated before any
integration decision.

## What the work validates

The projects independently reinforce several existing OARS decisions:

- semantic operations should not expose block dimensions, vendor routes, or
  shader identity;
- physical tile and SIMT schedules are different lowering levels, not different
  public mathematical operations;
- raw GPU command encoding is an unsafe boundary that should consume a checked
  launch description rather than revalidate fragments opportunistically;
- immutable inputs may be shared, while every non-atomic mutable destination
  needs an exclusive write domain;
- lazy work must retain inputs, outputs, pipelines, and execution ownership
  until exact completion;
- compile-time specialization is private artifact identity, not a runtime
  policy argument in a public operation.

OARS already implements the central structural pieces: `Engine` is the sole
runtime owner, semantic and executable graphs are separate, values retain
pending work, captured plans replay immutable command recordings, and private
`PreparedDispatch` preflight sits immediately above `vkCmdDispatch`.

## Experimental adoption 1: access and write-domain contracts

The strongest transferable idea is not a Rust device language. It is making
the ownership of writable GPU address ranges explicit and mechanically
checkable.

OARS currently records whole-value semantic `Read`, `Write`, and `ReadWrite`
effects and plans full-buffer Vulkan hazards. That proves inter-dispatch
ordering for the current storage-buffer path, but does not describe why two
invocations inside one dispatch cannot race. The target contract separates:

| Level | Owner | Required fact |
|---|---|---|
| Semantic access | operation schema | which logical value or admitted view is read, written, or read-written, including legal input/output aliases |
| Physical write domain | kernel candidate metadata | how invocation, subgroup, workgroup, or tile indices map to writable element/byte ranges |
| Collision policy | kernel candidate metadata | exclusive, atomic, ordered reduction, intentionally racy and therefore rejected, or another explicitly proved policy |
| Inter-node synchronization | executable planner | producer/consumer ranges, stages, accesses, layouts, queues, ownership transfer, lifetime, and completion |

A future candidate record should be able to express concepts equivalent to:

```text
logical_domain: rows
write_partition: exclusive_per_workgroup
partition_extent: output_row_width
tail_policy: bounds_checked
workspace_partition: exclusive_per_workgroup
```

This spelling is illustrative, not a frozen schema. The invariant is stable:
every non-atomic write is either uniquely owned or rejected. A reduction or
atomic route records its actual collision and numeric-order policy instead of
pretending to be disjoint.

This metadata should generate host admission checks, deterministic diagnostic
evidence, and zero/one/odd/overlap/alias/poison tests. It does not replace
shader review or GPU-assisted validation, and a declared partition is not by
itself a proof that the shader computes the declared address.

The connected OARS slice covers Core Softmax, Sum, and Scale candidates plus ML
LayerNorm, RMSNorm, and core-loss candidates. Their schema rows now own exact
writable binding ordinals, logical domains, invocation/workgroup partitions,
partition extents, exclusive collision policy, checked-tail policy, and
workspace partition. Generation rejects missing, unknown, duplicate, or
contradictory facts. Generated artifact metadata reaches executable diagnostics
and the private Vulkan dispatch preflight, which rejects a missing binding, a
declared read-only destination, or any writable binding omitted by the
candidate.

This slice is Experimental and deliberately incomplete. Other candidates emit
`physical_write: null`; they receive no write-ownership claim. The current
metadata does not yet encode byte-range formulas, admitted aliases,
capability/specialization predicates, atomic writes, or ordered reductions.
Shader inspection, differential tests, and Vulkan validation remain required
to establish that code follows the declared mapping.

For vkPQC, the initial one-workgroup-per-independent-operation schedule should
use an exclusive output and workspace partition. That prevents accidental
cross-operation races. It does not prove constant-time behavior or protect
secret storage from capture.

## Experimental adoption 2: prepared-dispatch evidence

The Experimental Vulkan recorder already constructs a private
`PreparedDispatch`. Pipeline creation validates shader workgroup limits and
reflected ABI; dispatch preflight resolves the exact pipeline, validates all
three workgroup counts, encodes descriptor indices and typed push constants,
and checks the final reflected push size. Only that prepared value reaches the
unsafe command-encoding function.

The first write-ownership slice adds the selected artifact's physical-write
contract and exact host access agreement to `PreparedDispatch`. The target
prepared value still needs to bind together:

- semantic operation and selected candidate/artifact identity;
- queried capability and specialization predicates;
- exact binding ownership, access modes, ranges, and alias policy;
- physical write-domain or collision policy;
- workspace allocation and lifetime;
- direct or indirect dispatch-domain proof;
- reflected descriptor, push-data, and workgroup ABI.

Preparation remains private and transactional. A failed proof publishes no
recording and cannot fall back silently. Raw dispatch construction remains an
unsafe internal diagnostic/testing escape hatch, not a domain-lowering API.

## OaTile relationship

`cutile-rs` supports the direction of [OaTile](oaTile.md), but does not define
its implementation. OaTile remains a generated cross-vendor construction
system whose logical tiles lower to explicit Slang candidates. It can borrow
the principle that tile partitioning determines both output ownership and
logical grid geometry while still choosing different physical subgroup,
workgroup, shared-memory, and tail schedules for each Vulkan device class.

The tile level is appropriate for GEMM, reductions, conversion, normalization,
and other regular data-parallel families. An explicit SIMT candidate remains
necessary where algorithms require precise lane exchange, shared-memory
placement, barriers, variable loops, bit packing, or rejection sampling. Initial
ML-KEM and ML-DSA kernels therefore remain explicit Slang/SIMT work rather than
being forced through a public tile DSL.

## Planned inspection surface

The `cargo oxide inspect`, compilation-pipeline, sanitizer, debugger, and test
commands show the value of making the physical result easy to audit. OARS
should add one repository-owned inspection command or report that resolves:

```text
semantic operation and contract
  -> selected provider, family, candidate, and fallback reason
  -> queried capabilities and specialization values
  -> logical domain and physical workgroup/write partition
  -> reflected descriptor and push ABI
  -> SPIR-V content hash, validation result, and optional disassembly
  -> plan node, barriers, resource ranges, and retained completion
```

This is a diagnostic and evidence surface, not a stable executable-plan ABI.
It should extend the existing semantic/executable reports and build/reflection
checks rather than create a second registry.

## Rejected changes

- **No CUDA Rust shipping backend now.** PTX and CUDA Tile IR cannot implement
  the Vulkan portability contract and would create another runtime, artifact,
  synchronization, and qualification path.
- **No device language split.** Shipping kernels remain Slang. Rust host and
  Slang device surfaces derive their mechanical ABI from the operation schema
  and reflection rather than duplicating it manually in one Rust source file.
- **No public tile/block knobs.** Workgroup, subgroup, tile, and candidate
  selection remains private lowering policy.
- **No vendor terminology in semantic schemas.** `DisjointSlice`, CUDA blocks,
  cubins, PTX, and CUDA Tile IR are donor concepts, not OARS public identities.
- **No confidentiality inference.** Borrow checking, partition ownership, and
  race freedom do not stop the owning process, a privileged debugger, capture
  tools, driver instrumentation, spills, or physical attacks from observing
  secrets. The governing contract remains
  [GPU secret execution and observability](../cryptography/oaGpuSecretSecurity.md).

An NVIDIA-only experimental oracle or benchmark provider may be reconsidered
only for a named operation when it supplies evidence unavailable from the OA
donor and CPU/reference paths. It must remain outside the canonical Vulkan
route and cannot introduce a second public execution owner.

## Adoption order and proof

1. Extend candidate/schema design with access range, write-domain, collision,
   tail, and workspace-partition vocabulary.
2. Add generator rejection tests for incomplete or contradictory contracts.
3. Carry those facts into semantic and executable diagnostics without changing
   public domain APIs.
4. Extend prepared-dispatch preflight and add transactional failure tests.
5. Add shader oracles and overlap/alias/poison tests for one simple row- or
   tile-partitioned kernel.
6. Run separate core, synchronization, and GPU-assisted Vulkan validation on
   named hardware.
7. Admit subrange synchronization or allocation reuse only after the physical
   range and lifetime proofs are independently tested.
8. Add the unified kernel-inspection report and capture its exact compiler,
   SPIR-V tools, device, driver, and build provenance.

## Primary references

- [NVIDIA: Introducing CUDA Rust—Two Tracks for Writing GPU Kernels](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
- [NVlabs cuda-oxide repository](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide book](https://nvlabs.github.io/cuda-oxide/)
- [NVlabs cutile-rs repository](https://github.com/NVlabs/cutile-rs)
- [cuTile Rust documentation](https://nvlabs.github.io/cutile-rs/main/)
- [CUDA Tile IR documentation](https://docs.nvidia.com/cuda/tile-ir/latest/)
- [Vulkan compute pipelines](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#pipelines-compute)
