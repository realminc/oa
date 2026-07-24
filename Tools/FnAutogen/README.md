# OaFnAutogen

Schema-driven code generator for the `OaFn*` API. Stdlib Python only — no
pip deps, runs anywhere with Python 3.11+.

The schemas are the source of truth; this document and
`Tools/FnAutogen/oafnautogen.py` define the public generation workflow.

## Status

**Context API generator** — 169 ops across Core, Ml, Vision, Audio, Ui, and
Crypto schemas under `Tools/FnAutogen/Schema/`.

Each op emits:

  - `.gen.h` declaration fragments for the unified context-based API
  - `.gen.cpp` implementations for auto bodies
  - optional `.gen.slang` shaders when `forward_op` is present
  - optional autograd class fragments when `[ops.autograd]` is differentiable
  - optional GTest scaffolding when a CPU reference can be derived
  - optional fixed-kernel registry rows, stable ID constants, and parallel
    CMake name/source lists when `[ops.kernel]` owns the kernel identity

Schemas with `body = "manual_context"` keep validation, graph recording and
dispatch handwritten. When an op supplies exact `api_params`, the generator
owns its public declarations and can emit the mechanical global-engine
forwarder. A manual schema without an exact generated artifact emits no
placeholder `.gen.h` or `.gen.cpp`; stale generated files are removed during
regeneration. Manual-only autograd categories and generated-source groups with
no emitted `.cpp` likewise produce no placeholder header or empty CMake
manifest.

Manual lowerings can still make differentiation schema-owned. Setting
`autograd.attach = "standard"` emits a mechanical gradient-node attachment;
`"broadcast_binary"` selects the ordinary node for equal shapes and the
broadcast-reducing node otherwise. Optional `input_ranks` encode structured
rank guards without embedding arbitrary C++ in TOML. Optional `state` entries
copy typed non-matrix API parameters such as epsilon into named gradient-node
members, keeping forward and backward configuration identical. These private
helpers are emitted to `AutogradAttach.gen.h` and included only by their
lowering units.

`kind = "session_command"` describes a stateful command recorded into an
owner's explicit session. It emits semantic operation metadata without
inventing a public stateless `OaFn*` declaration. Its lowering remains private
to the owning session. Such a schema must set
`contract.value_validation = "session_command"`, `shape_rule = "explicit"`,
and the frozen `dtype_rule = "match_input"` sentinel. The sentinel does not
claim that heterogeneous session values share one dtype: the private command
must validate exact shapes and dtypes before recording. Ordinary operations
are rejected if they attempt to use this escape hatch. The frozen descriptor
still permits at most eight ordered semantic attributes.

Schemas with `body = "cpp_expr"` generate C++ bodies over the public `OaFn*`
surface instead of dispatching a dedicated kernel. Use `cpp_expr` for one-line
ops, or `cpp_body` for small multiline bodies that need a shared temporary. A
schema may instead use `manual_context` when lowering requires several
executable nodes; `MlFnLoss.toml` uses that route for `CrossEntropy` while the
schema still owns every mechanically derivable surface.

## Usage

```bash
# Generate all schemas to Tools/FnAutogen/Output/
python3 Tools/FnAutogen/oafnautogen.py

# Dry run — print what would be written
python3 Tools/FnAutogen/oafnautogen.py --dry-run

# Write generated files into Source/ and Test/
python3 Tools/FnAutogen/oafnautogen.py --live

# Custom schema and output dir
python3 Tools/FnAutogen/oafnautogen.py \
    --schema  Tools/FnAutogen/Schema/Core/CoreFnMatrixElemwise.toml \
    --registry Source/Public/Oa/Core/KernelRegistry.h \
    --out     Tools/FnAutogen/Output
```

`--schema` is an isolated preview mode and cannot be combined with `--live`:
the operation registry, Python bindings, autograd attachment, documentation,
and source manifests require the complete schema set. Generated files are
written only when their bytes change, so an unchanged full regeneration
preserves timestamps and does not invalidate build outputs.

## Layout

  - `Schema/<Domain>/*.toml` — source of truth, one file per category
  - `oafnautogen.py` — CLI entrypoint and code emitters
  - `oafnautogen_lib/` — config, schema validation, layout inference
  - `Output/` — generated files, gitignored, drop-in replacements for the
    hand-written counterparts in `Source/`
  - `README.md` — this file

## Validation

Before writing anything the generator:

  1. Parses `Source/Public/Oa/Core/KernelRegistry.h` for kernel name/ID rows,
     then merges schema-owned `[ops.kernel]` rows while rejecting duplicate
     names, IDs, or sources.
  2. Checks every `kernel_forward` is either in the hand-written registry or
     owned by that operation's validated `[ops.kernel]` block.
  3. Checks PascalCase, duplicate op names, body/kind compatibility,
     scalar-param completeness, output shape/dtype vocabulary, dispatch
     workgroup modes, exact public API parameters, and autograd formula
     vocabulary.

`python3 Tools/check_autogen_drift.py` additionally rejects stale FnAutogen
files which are no longer emitted. A clean gate therefore proves both that
generated content matches and that dead placeholder artifacts have not
survived a schema change.

Adding a new op = add a `[[ops]]` entry. If it owns a new fixed kernel, add one
`[ops.kernel]` block with its stable prefix/local ID, category, origin, and
shader source. The generator writes the install-safe public registry row and ID
constant plus the private CMake source pair. Existing shared kernels remain
valid `kernel_forward` references without a new ownership block.

## Adding a new op

  1. Add `[[ops]]` to the appropriate schema TOML.
  2. For a new fixed kernel, add `[ops.kernel]` to that operation; do not patch
     generated registry or ID files.
  3. Run the generator.
  4. (Round-trip phase) Diff `Output/...` against the hand-written file.
     Fix schema or template until the diff is intentional.
  5. (Cut-over phase) Delete hand-written `.h`/`.cpp`/`.slang` and replace
     with the `.gen.*` outputs (or wire CMake to consume `Output/` directly).

## Round-trip diff against hand-written code

```bash
# Compare a generated shader against the hand-written one
diff -u Source/Private/Oa/Core/Shader/Compute/Flat/Add.slang \
        Tools/FnAutogen/Output/Private/Oa/Core/Shader/Compute/Flat/Add.gen.slang

# Compare the elemwise .cpp
diff -u Source/Private/Oa/Core/FnMatrix/Elemwise/FnMatrixElemwise.cpp \
        Tools/FnAutogen/Output/Private/Oa/Core/FnMatrix/Elemwise/FnMatrixElemwise.gen.cpp
```

Differences should be either intentional schema/template changes or bugs to fix
before live regeneration.

## Good Next Targets

  - Add reusable body schemas for loss functions, reductions, optimizer steps,
    image/audio shape transforms, and GEMM/fusion routing.
  - Move the remaining C++/Slang/test emitters from `oafnautogen.py` into
    `oafnautogen_lib/` modules without changing generated output.
  - Generate Python binding metadata from the same schema once the C++ surface
    is stable.
