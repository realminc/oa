# OA Rust Compute Kernel System

**Status:** Canonical contract; generated elementwise and FP32 `mat_mul_nt` are Experimental

**Updated:** 2026-09-08

OA turns Slang entry points into immutable embedded SPIR-V artifacts and private
Vulkan pipelines. Applications call semantic operations; they do not compile
shaders, register providers, select source files, or choose vendor kernels.

## Authority and end-to-end path

The current generated path is:

```text
tools/gen/fn/schema/matrix_elemwise.json + matrix_blas.json
  -> tools/gen/fn/generate.py
     -> generated Rust operation functions
     -> generated stable KernelId/artifact metadata
     -> generated one-entry Slang modules
     -> generated external oracle tests
  -> build.rs
     -> Slang compilation and reflection
     -> reflected ABI and OA-attribute validation
     -> spirv-val --target-env vulkan1.3
     -> SPIR-V under OUT_DIR
  -> include_bytes! compile-time embedding in the private artifact registry
  -> fail-closed push-block reflection from the exact embedded SPIR-V
  -> device-owned ComputePipeline
  -> generic ComputeDispatch recording
```

The operation schema owns semantic meaning. Shader attributes own physical
kernel facts. Reflection proves the compiled interface. The runtime owns
selection and execution. None of these sources substitutes for another.

A normal Cargo build never rewrites checked-in files. It fails when generated
sources drift from the schema, an expected tool is missing, compilation fails,
reflection differs, or SPIR-V validation fails.

`OUT_DIR` is only the Cargo build boundary between `slangc` and
`include_bytes!`. No runtime shader path exists: every schema-owned module is
part of the library binary. Engine construction currently creates every
generated pipeline eagerly. Push-constant byte size is not repeated in the
schema or generated registry; pipeline layout and dispatch validation reflect
it from the exact embedded module, while build-time Slang reflection separately
checks field names, types, offsets, attributes, and workgroup geometry.

## Entry-point convention

Every compiled shader module exports exactly one stage entry point named
`main`. Entry-point names are module-local implementation symbols, not global
kernel identity, so operation schemas do not store or deduplicate them. A
source containing multiple dispatchable kernels is split into one module per
kernel. Helper functions may retain descriptive Slang names because they are
not stage entry points.

The fixed name preserves the established C++ OA pipeline convention and keeps
build compilation, reflection, and Vulkan pipeline creation on one invariant.
Debugging and registry identity use the source/artifact name, stable kernel ID,
OA attributes, and compiled module hash instead of overloading the function
name.

## Stable identity

Every admitted physical kernel module has a generated non-zero stable ID and an
exact embedded artifact. IDs are never inferred from filenames or shader-name
substrings. Once an ID reaches a Shipped checkpoint, it is never reused for a
different physical contract; retired ordinals remain reserved.

The current IDs are local Experimental `u16` values checked for collisions
across the elementwise and BLAS schemas. Before either schema owner ships, OA
must add a durable owner-prefix or equivalent collision-free allocation ledger.
Physical GEMM variants may
need a separate exact variant identity rather than pretending every tile is a
new semantic operation.

Artifact identity ultimately includes the compiled module hash, fixed `main` entry point,
descriptor/push ABI, workgroup geometry, required capabilities, specialization
values, and build provenance. A successful name lookup proves identity, not
semantic legality or numerical correctness.

## OA Slang attributes

OA attributes are exported once from the `attributes` module in
`src/slang/common/attributes.slang`; kernels use `import attributes` rather
than textual preprocessor inclusion:

| Attribute | Meaning |
|---|---|
| `kernel_name` | stable physical operation label |
| `domain` | semantic owner such as `matrix`, `vision`, or `ml` |
| `variant` | physical implementation family such as `generic` |
| `requires` | one required device capability |
| `dtype` | normalized dense storage dtype |
| `layout` | required physical layout when one exists |
| `status` | `experimental`, `stable`, or `deprecated` |
| `tune_family` | private measurement/tuning family |

Slang's `[shader("compute")]` and `[numthreads(...)]` remain language/compiler
attributes rather than OA metadata.

Attributes do not encode full operation semantics, autograd rules, shape
inference, routing preference, benchmark results, or public API availability.
Those facts belong to the operation schema, planner, evidence artifacts, and
status documentation.

## Dtype vocabulary

Dense dtype identifiers use lowercase scalar tokens consistently in operation
schemas, shader attributes, artifact/cache keys, and diagnostics:

| Rust semantic spelling | Metadata token | Physical meaning |
|---|---|---|
| `DType::F16` when admitted | `f16` | IEEE binary16 storage |
| `DType::Bf16` when admitted | `bf16` | bfloat16 storage |
| `DType::F32` | `f32` | IEEE binary32 storage |
| `DType::I32` | `i32` | 32-bit signed integer storage |
| `DType::F64` when admitted | `f64` | IEEE binary64 storage |
| other integer variants when admitted | `i8`, `u8`, `i16`, ... | dense integer storage of that width |

`[dtype("f32")]` is therefore canonical. `fp32` is a legacy metadata spelling,
not a Rust alignment benefit. BF16 remains `bf16`: it is the established name
of the format even though Rust has no built-in `bf16` primitive. PascalCase is
used only for the Rust enum variant.

The dtype attribute describes storage for that physical kernel. It does not
also mean accumulator precision. A future BF16 GEMM may have `bf16` inputs,
`f32` accumulation, and either `bf16` or `f32` output; those are separate typed
facts in its plan and metadata. Mixed-storage kernels require per-binding dtype
contracts rather than one ambiguous dispatch-wide label.

Q4/Q8 and imported quantized formats are packed encodings with scales, block
sizes, and logical shapes. They are semantic encoded values, not dense dtypes.

## Shared storage modules

Element-index addressing and storage conversion are solved once in shared
Slang modules. Current generated elementwise and matmul shaders import
`src/slang/common/storage.slang` and call the dtype-matched `load_f32`/
`store_f32` or `load_i32`/`store_i32` helpers; they never regenerate or
open-code those functions.

When low-precision storage is ported, preserve the lessons already proved in
C++ OA and revalidate them in this compiler/runtime:

- `f32` uses four bytes per logical element and bit-preserving load/store;
- BF16 conversion defines round-to-nearest-even and NaN/Inf handling;
- packed 16-bit stores need atomic CAS when adjacent invocations may update
  opposite halves of one 32-bit word, unless dispatch ownership proves a whole
  word has one writer;
- packed reads need a padded final word when the physical access reads 32 bits;
- fixed-`f32` optimizer state and gradient atomics stay explicitly `f32` even
  inside a low-precision operation;
- native 8/16-bit accesses are used only when the exact Vulkan/SPIR-V capability
  and alignment contract is queried and enabled.

Do not copy every historical OA helper blindly. The generic OA `DTYPE`
specialization path, fixed-dtype modules, and mixed-storage helpers represent
different contracts. Choose one based on the operation schema and include all
specialization values in pipeline/cache identity. In particular, a plain
packed-word load/modify/store is not safe merely because it existed in an old
helper.

The port audit found two C++ sources that must be reconciled before BF16 is
admitted: the fixed BF16 helper rounds FP32 to BF16 with round-to-nearest-even,
while the generic `DTYPE` helper stores the upper 16 bits without that rounding.
The older fixed FP16 helper also uses a plain packed-word load/modify/store,
which can lose an adjacent invocation's half-word update. OARS therefore has
not copied either behavior as an assumed contract. The BF16/FP16 vertical slice
must select and test the rounding rule, special values, odd tails, final-word
padding, and concurrent paired stores before adding its `DType` variant or
kernel route.

## Bindless and push ABI

The current Experimental ABI declares one runtime array:

| Set | Binding | Resource |
|---:|---:|---|
| 0 | 0 | storage buffers |

The target global heap reserves additional bindings for storage images,
sampled images, and samplers when those value paths are implemented. A shader
declares only arrays it uses.

Push constants place descriptor indices first in buffer-binding order, followed
by scalar fields. Domain lowerers provide only the scalar payload; the generic
runtime recorder prepends every buffer's bindless index automatically. Current
shader layouts are:

- binary: left index, right index, output index, element count;
- unary: input index, output index, element count;
- unary-scalar: input index, output index, element count, `f32` scalar;
- matmul NT: left index, right index, output index, M, N, K.

Every field is four bytes. Reflection validates names, scalar types, offsets,
sizes, total range, descriptor set/binding, runtime-array shape, workgroup size,
the fixed `main` entry point, and OA attributes. Rust records typed
`U32` and `F32` payload values and rejects a combined bindless-header plus
payload byte count that differs from the artifact ABI.

Descriptor capacity, buffer byte size, logical element count, push range, and
the three dispatch workgroup limits are independent queried constraints.

Thread workgroup geometry and logical output-tile geometry are separate. The
current FP32 matmul uses 256 threads to produce a 64×64 output tile while
walking K in blocks of 16. Generated artifact metadata owns both geometries so
host dispatch never infers logical coverage from `[numthreads]`.

## Pipeline and selection policy

The current device eagerly creates one pipeline per generated matrix
artifact. All pipelines share the engine-owned descriptor layout and set;
pipeline layouts remain artifact-specific because push ranges can differ.

Future lazy creation or preload must resolve the same exact key. A pipeline key
contains every behavior-affecting specialization value and layout fact. Kernel
selection filters candidates by dtype, layout, shape, alignment, subgroup,
extensions, memory/workspace, and numeric policy before Vulkan creation or
recording. Route overrides remain private to tests, benchmarks, and diagnostics.

## Adding or porting a kernel

1. Define or extend the semantic operation contract and independent oracle.
2. Reuse the shared storage/math module; port a proven helper only after
   auditing its access, conversion, capability, and concurrency contract.
3. Add physical kernel metadata, stable identity, requirements, dtype/layout,
   workgroup geometry, and numeric policy at their owning schema.
4. Generate all derived surfaces and prove immediate regeneration is empty.
5. Compile every module's `main`, validate reflection, and run `spirv-val`.
6. Test zero, odd/tail, minimum, practical boundary, invalid, alias, reuse, and
   poison cases where meaningful.
7. Run core, synchronization, and applicable GPU-assisted validation separately
   on a named device.
8. Measure only after correctness, workload equivalence, and route/fallback
   identity are established.

Registry presence and pipeline creation are necessary but never sufficient
proof that an operation is correct or Shipped.

## Source map

- operation authority: `tools/gen/fn/schema/`
- operation generator: `tools/gen/fn/generate.py`
- shared Slang modules: `src/slang/common/`
- generated one-entry Slang modules: `src/slang/matrix/elemwise/` and
  `src/slang/matrix/blas/`
- build compilation/reflection gate: `build.rs`
- generated private artifact identity: `src/rs/runtime/shader/generated.rs`
- artifact validation: `src/rs/runtime/shader.rs`
- Vulkan descriptors and pipelines: `src/rs/runtime/vk/`
- generic executable description: `src/rs/runtime/dispatch.rs`
