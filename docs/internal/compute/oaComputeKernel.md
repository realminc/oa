# OA Rust Compute Kernel System

**Status:** Canonical target; generated elementwise, Softmax, and one FP32 matmul family are Experimental

**Updated:** 2026-09-10

Applications invoke semantic operations. They never choose a shader filename,
vendor, tile or route. OARS compiles Slang to immutable embedded SPIR-V,
validates its reflected ABI, constructs engine-owned Vulkan pipelines, and
selects physical kernels privately.

## Current path

```text
tools/gen/fn/schema/*.json
  -> tools/gen/fn/generate.py
     -> generated Rust APIs/contracts/KernelId registry/tests
     -> generated one-entry Slang modules
  -> build.rs
     -> slangc + reflection checks
     -> spirv-val --target-env vulkan1.3
     -> SPIR-V under OUT_DIR
  -> include_bytes! embedded artifact
  -> exact runtime reflection check
  -> engine-owned ComputePipeline
  -> ComputeDispatch
```

A normal Cargo build never rewrites checked-in source. Missing tools, generated
drift, compilation errors, invalid SPIR-V, reflection disagreement or pipeline
creation errors fail closed. Every module exports one compute entry point named
`main`; stable identity comes from generated metadata and artifact content, not
that module-local symbol.

The current registry chooses an exact kernel for each operation/dtype. The
FP32 `matrix::mat_mul_nt` route is one 64x64x16 output/K tile using 256 threads.
Axis-aware FP32 `matrix::softmax` assigns one 256-thread workgroup to each
outer/inner slice and uses the donor's max-subtracted two-reduction schedule;
its saved-output adjoint uses the same slice geometry.
No general candidate set, cost model, tuning cache, device pack or portable
fallback ladder exists yet.

## Four distinct identities

| Identity | Purpose |
|---|---|
| Semantic contract | user-visible meaning, validation, effects and numeric policy |
| Kernel family | reusable algorithm/schedule such as tiled GEMM or resize-normalize |
| Candidate | complete family parameters, capabilities, workspace and numeric behavior |
| Artifact | exact SPIR-V bytes, entry point, reflected ABI, specialization and compiler provenance |

Stable kernel IDs are non-zero, collision-checked and never inferred from
names. Before shipping, OARS needs a durable allocation ledger. A tuning cache
stores full candidate/artifact identity, not a semantic operation ID or a
vendor-library solution ordinal.

## Candidate schema

A future generated candidate row must be sufficient to reject an illegal route
without creating a pipeline:

```text
Candidate {
  stable_id, family_id, artifact_id,
  input/output encodings and layouts,
  alignment and stride predicates,
  workgroup and logical tile geometry,
  subgroup/cooperative-matrix requirements,
  shared-memory and workspace formulas,
  supported epilogue tree,
  compute/accumulator/output precision,
  determinism and alias policy,
  supported shape range and tail strategy,
  device-pack provenance and qualification state
}
```

Generated presence means “eligible for validation,” never “fast,” “supported
everywhere,” or “selected.” The runtime filters by the full problem and queried
capabilities, the heuristic ranks legal candidates, and tuning may replace the
ranking only after correctness passes.

## Portable lattice, not variant explosion

OARS should build a curated lattice from orthogonal components:

- atom: scalar, vector, subgroup or cooperative-matrix computation;
- operand movement: direct/global, shared-memory staged, transposed or packed;
- main loop: output-stationary tiled, split-K, Stream-K-like or persistent;
- scheduler: one tile/workgroup, grouped/batched, work stealing or fixed grid;
- epilogue: store, scale, bias, activation, residual, clamp, convert or auxiliary
  output;
- tail path: predication, scalar cleanup or padded input contract.

Only combinations tied to a device class and workload need become artifacts.
The default portable pack stays intentionally small. Qualified packs can add
cooperative-matrix shapes, subgroup widths, local-memory layouts or mobile
bandwidth schedules without exposing vendor names through the semantic API.

High-value initial families are:

1. portable FP32 tiled GEMM with odd tails;
2. small-M GEMM;
3. strided/grouped batch GEMM;
4. split-K and Stream-K-like GEMM with deterministic and non-deterministic
   reductions kept distinct;
5. fused GEMM epilogues;
6. bandwidth microfusions for image/video/audio conversion and normalization;
7. reductions with explicit accumulator and order policy.

See [OaTile](oaTile.md) for the construction model.

## Slang's role

Slang should own reusable shader modules, interfaces/generics, compile-time
specialization and portable source-level algorithms. Reflection validates the
actual target ABI. SPIR-V specialization constants may reduce source
duplication where drivers genuinely specialize them, but every value remains
part of pipeline identity.

Slang is not a performance portability oracle. One source may compile on many
vendors while needing distinct schedules or artifacts to perform well. OARS
therefore measures candidates per capability/device profile and preserves the
winning exact artifact.

## ABI

The current global descriptor ABI is set 0, binding 0, a runtime array of
storage buffers. Push constants begin with descriptor indices in binding order
and then typed `u32`/`f32` scalars. Reflection validates names, types, offsets,
range, descriptor shape, workgroup size, entry point and OA attributes.

The target heap may add storage images, sampled images and samplers. A kernel
declares only what it uses. Descriptor capacity, allocation size, logical
element count, push range and each dispatch dimension are independent limits.
Image kernels additionally own format features, layouts, subresource ranges and
sampling/color contracts.

## Launch and write-domain proof

The Experimental Vulkan path already uses a private `PreparedDispatch` between
preflight and unsafe command encoding. Pipeline creation validates reflected
workgroup and push ABI against device limits; dispatch preparation resolves the
exact pipeline, validates all three group counts, encodes resource indices and
typed push values, and rejects a final push-size mismatch before recording.
For Core reductions/Scale and ML LayerNorm/RMSNorm/core losses, it also verifies
that every schema-classified destination exists with write access and that no
additional writable binding escapes the candidate contract. The same generated
evidence is emitted by execution-plan diagnostics; other candidates remain
visibly unclassified rather than inheriting an implicit claim.

The target candidate contract additionally states how physical invocations,
subgroups, workgroups, or logical tiles own every writable range. A normal
non-atomic output must have an exclusive partition. Colliding writes require an
explicit atomic or reduction policy, including storage type and numeric order;
the admitted U32 accumulator vocabulary uses `shared_atomic_contributors` with
`atomic_u32`. An unclassified collision is illegal. Semantic input/output alias permission
remains schema-owned, while the candidate owns its physical index-to-range
mapping and tail behavior.

Write extent distinguishes a single element from bounded vectorized ownership.
`up_to_two_elements` and `up_to_four_elements` describe adjacent outputs owned
by one counter-based RNG invocation; bounds-checked tails remain mandatory.
They do not relax the exclusive-partition rule or imply atomic collision
handling.

Prepared dispatch evidence now joins selected artifact identity, binding access
modes, the classified write-domain policy, dispatch geometry, and reflected ABI
for that initial slice. It must still gain queried capabilities, byte-range
formulas, aliases, and workspace lifetimes before the target contract is
complete. Only the prepared value reaches the unsafe Vulkan command encoder.
This is orthogonal to inter-node synchronization: exclusive ownership proves
intra-dispatch race freedom, while the executable graph still proves producer
and consumer visibility, ordering, queue ownership, lifetime, and reuse.

The design input and non-goals are recorded in the
[CUDA Rust assessment](oaCudaRust.md).

## Dtype vocabulary

Metadata uses lowercase physical tokens such as `f32`, `i32`, `f16` and
`bf16`. Storage dtype does not imply compute or accumulator precision. Packed
Q4/Q8 formats have their own encoded-value descriptors and per-binding
contracts.

Before FP16/BF16 admission, reconcile the C++ donor's differing BF16 rounding
helpers and unsafe adjacent-half packed stores. Test round-to-nearest-even,
NaN/Inf, odd tails, padded final words, alignment and concurrent paired stores.
Native 8/16-bit access is used only under exact queried Vulkan/SPIR-V support.

## Pipeline lifecycle

Current engine construction eagerly creates generated pipelines. The target
may lazily create or preload them, but it must:

- key every behavior-affecting specialization and layout fact;
- serialize an implementation-owned cache only with driver/device/build
  provenance and Vulkan compatibility checks;
- retain pipelines and layouts through command retirement;
- report cold creation separately from steady replay;
- never hide a compile/create failure behind CPU execution.

## Adding a candidate

1. Port or define the semantic contract and independent oracle.
2. Record donor path and adaptation class.
3. Reuse audited common storage/math modules.
4. Add complete candidate metadata and a stable identity at its schema owner.
5. Regenerate and prove immediate regeneration is empty.
6. Compile, reflect and validate the exact SPIR-V.
7. Test capability rejection, zero/odd/tail, bounds, alias, reuse and poison
   behavior.
8. Run Vulkan validation profiles on named hardware.
9. Measure with exact route/fallback evidence before changing default policy.

## Source map

- schemas: `tools/gen/fn/schema/`
- generator: `tools/gen/fn/generate.py`
- Core support modules: `src/slang/core/` (`math`, `rng`, and metadata)
- Matrix kernels: `src/slang/matrix/` (`elemwise`, `reduce`, `rng`, and `blas`)
- ML kernels: `src/slang/ml/` (`nn/<family>`, `loss/<family>`, and
  `optim/<family>`)
- build/reflection gate: `build.rs`
- generated registry: `src/rs/runtime/shader/registry.gen.rs`
- Vulkan pipelines/descriptors: `src/rs/runtime/vk/`
- executable description: `src/rs/runtime/dispatch.rs`

Loss-family directories do not shorten entry-point filenames to
`forward.slang` or `backward.slang`. They retain self-identifying basenames:
`<loss>_forward.slang` and `<loss>_backward.slang`. This keeps compiler,
validation, profiling, and diagnostic paths unambiguous when several loss
families are present in the same build.

Optimizer kernels follow the same family ownership rule:
`optim/sgd/sgd_momentum.slang`, `optim/adamw/adamw_graph.slang`, and
`optim/muon/muon_apply.slang`. Multi-pass operations remain self-identifying,
for example `optim/grad_clip/clip_grad_norm_reduce.slang` and
`optim/grad_clip/clip_grad_norm_scale.slang`. A flat optimizer directory is
not a stable source layout as the operation set grows.

Physical layout follows semantic ownership. A module such as
`core/math/activations.slang` exports reusable scalar formulas and has no entry
point. Files below `ml/nn/activation/` are independently dispatchable
activation operations and each owns one `main` entry point. The shared formula
module and the operation family are therefore complementary, not competing
implementations. Empty catch-all `common`, `math`, `types`, or `matrix` shader
stubs are not retained as future placeholders.

## Primary references

- [Slang reflection API](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/09-reflection.html)
- [Slang SPIR-V specialization constants](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html)
- [Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/pdf/vkspec.pdf)
- [NVIDIA CUDA Rust announcement](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
