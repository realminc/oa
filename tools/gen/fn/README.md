# Operation generation

Schema-driven code generator for the `oa::Fn*` API. Stdlib Python only — no
pip deps, runs anywhere with Python 3.11+.

The schemas are the source of truth; this document and
`tools/gen/fn/generate.py` define the public generation workflow.

## Status

**Context API generator** — operations across Core, Ml, Vision, Audio, Ui, and
Crypto schemas under `tools/gen/fn/schema/`. The live count is derived from
the schema set; do not hardcode it here because migration removes duplicate and
phantom rows while adding previously untracked public contracts.

Each op emits:

  - `.gen.h` declaration fragments for the unified context-based API
  - `.gen.cpp` implementations for auto bodies
  - optional `.gen.slang` shaders when `forward_op` is present
  - optional autograd class fragments when `[ops.autograd]` is differentiable
  - optional Python registrations when `[ops.python]` is present
  - optional GTest scaffolding when a CPU reference can be derived
  - optional fixed-kernel registry rows, stable ID constants, and parallel
    CMake name/source lists when `[ops.kernel]` owns the kernel identity
  - optional private `oa::Dnn` operation-role and epilogue rows when `[ops.dnn]`
    classifies the exact semantic name and contract hash

Schemas with `body = "manual_session"` keep validation, graph recording and
dispatch handwritten. When an op supplies exact `api_params`, the generator
owns its public declarations and can emit the mechanical active-context
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
helpers are emitted as an owning-domain
`autograd/autogradAttach.gen.h`/`.gen.cpp` pair. The header keeps only the
disabled-autograd eligibility check inline; the generated implementation owns
concrete-node construction so lowering units do not need the complete private
node catalog merely to invoke a schema attachment. Generated gradient classes
likewise live under the schema domain and are collected by that domain's
generator-owned `autograd.gen.h` manifest. `autograd.attach = "manual"` records
a multi-output or otherwise non-mechanical attachment as schema-owned while
requiring explicit implementation provenance; it does not generate a duplicate
helper.

`[ops.dnn]` is private planning metadata, not a public API or executable-route
promise. It may classify a schema-owned semantic operation by one validated
role and, for GEMM operations, an optional epilogue. An epilogue conditional on
an input must name an optional contract input. The complete schema set emits
one exact-name plus contract-hash table for automatic `oa::SemanticGraph`
capture; unknown or mismatched operations remain portable.

`autograd.node_header` may override the owner-specific private gradient-node
catalog. A schema whose emitted source group belongs to one family may set
top-level `autograd_node_header` once; an operation-level `node_header` remains
the higher-priority override. Every Core and ML gradient policy must select an
exact family header; generation fails instead of recreating a domain-wide node
catalog. Core never includes ML merely to attach a foundational adjoint. Copy,
for example, selects
`oa/core/autograd/matrix/autogradShape.h` rather than importing the complete
Core catalog into its generated implementation. Hand-written index lowerings
similarly compose `autogradIndex.h` with shape only when they also attach a
reshape node. The shared manual-attachment implementation applies the same
schema defaults and includes the sorted union of its selected node headers; it
does not silently replace narrow ownership with the complete domain catalog.

`[ops.python]` owns the public camelCase function and keyword-argument names and
documentation from the same exact `api_params` as C++. Non-matrix return values
use `[ops.python_result]` to declare the owned C++ result type and exposed
members. The generator emits one binding fragment per domain, including typed
C++ defaults such as `oa::StftConfig{}` for Python signatures. Each fragment is
included by the binding for its declared C++ namespace, so Core and ML extend
one native `oa.FnMatrix` module instead of registering flat domain functions.
The generator also emits `source/py/oa/_schemaSurface.py`, which owns the
matching public namespace and compatibility-domain exposure inventory. Public
and stable non-void operation rows are not allowed to omit either surface.

Schema `namespace` values remain fully qualified semantic identities such as
`oa::FnAudio`; registry names, Python exposure, documentation, and contract
hashes all derive from that identity. Generated public declaration fragments
are namespace-neutral because their stable umbrella opens `namespace oa` and
then `namespace FnAudio`. C++ source spells every `Fn*` namespace with those
two explicit blocks; compressed declarations such as `namespace oa::FnAudio`
are rejected by the schema-coverage gate. Never edit a `.gen.h` or `.gen.cpp`
file to change namespace spelling.

`kind = "session_command"` describes a stateful command recorded into an
owner's explicit session. It requires `[ops.session]` metadata naming the
owner, visible state transition, effects, and completion boundary; it is not
counted as a stateless semantic operation. Its lowering remains private to the
owning session. A command may additionally carry an internal operation
contract when it emits schema-owned device work, as the vector RL commands do.
Such an internal contract sets
`contract.value_validation = "session_command"`, `shape_rule = "explicit"`,
and the frozen `dtype_rule = "match_input"` sentinel. The sentinel does not
claim that heterogeneous session values share one dtype: the private command
must validate exact shapes and dtypes before recording. Ordinary operations
are rejected if they attempt to use this escape hatch. The frozen descriptor
still permits at most eight ordered semantic attributes.

`surface = "host_utility"` classifies pure CPU helpers such as Annex-B parsing.
They use `body = "cpu_util"`, cannot name a device kernel or semantic operation
contract, and remain ordinary host APIs rather than fake device operations.
`surface = "public_cpp_operation"` is a real semantic device operation in the
installed C++ API which is deliberately outside the curated Python package
surface. It requires the same exact signature and semantic contract as
`public_operation`; the distinction only suppresses Python registration and
must not be used to hide an otherwise admitted Python operation.

Schemas with `body = "cpp_expr"` generate C++ bodies over the public `oa::Fn*`
surface instead of dispatching a dedicated kernel. Use `cpp_expr` for one-line
ops, or `cpp_body` for small multiline bodies that need a shared temporary. A
schema may instead use `manual_session` when lowering requires several
executable nodes; `mlFnLoss.toml` uses that route for `crossEntropy` while the
schema still owns every mechanically derivable surface.

## Usage

```bash
# Generate all schemas to ignored build/gen/fn previews
python3 tools/gen/fn/generate.py

# Dry run — print what would be written
python3 tools/gen/fn/generate.py --dry-run

# Write generated files into source/ and test/
python3 tools/gen/fn/generate.py --live
python3 tools/gen/fn/checkSchemaCoverage.py
python3 tools/gen/fn/checkPublicOperationCoverage.py \
    --compile-commands build/release/compile_commands.json

# Custom schema and output dir
python3 tools/gen/fn/generate.py \
    --schema  tools/gen/fn/schema/core/coreFnMatrixElemwise.toml \
    --registry source/cpp/lib/oa/runtime/kernelRegistry.h \
    --out     build/gen/fn
```

`--schema` is an isolated preview mode and cannot be combined with `--live`:
the operation registry, Python bindings, autograd attachment, documentation,
and source manifests require the complete schema set. Generated files are
written only when their bytes change, so an unchanged full regeneration
preserves timestamps and does not invalidate build outputs.

## Layout

  - `schema/<domain>/*.toml` — source of truth, one file per category
  - `generate.py` — CLI entry point and emitters
  - `config.py`, `schema.py`, `layout.py` — validation and output ownership
  - `testGenerate.py` — generator contract tests
  - `README.md` — this file

## Validation

Before writing anything the generator:

  1. Parses the generated standalone and Tile fragments composed by
     `source/cpp/lib/oa/runtime/kernelRegistry.h`, then merges schema-owned
     `[ops.kernel]` rows while rejecting cross-authority duplicate names, IDs,
     or sources.
  2. Checks every `kernel_forward` is either in another generated authority or
     owned by that operation's validated `[ops.kernel]` block.
  3. Checks camelCase, duplicate op names, body/kind compatibility,
     scalar-param completeness, output shape/dtype vocabulary, dispatch
     workgroup modes, exact public API parameters, and autograd formula
     vocabulary.

`python3 tools/gen/checkDrift.py` regenerates into a temporary tree and rejects
drift, missing files, generator-owned files which are no longer emitted, and
non-idempotent timestamp rewrites. A clean gate proves the live tree matches the
schema authority and that dead placeholder artifacts have not survived.

`python3 tools/gen/fn/checkSchemaCoverage.py` is the schema-row depth gate. It
requires every contract identity to be referenced by a live lowering, every
public/stable non-void operation to own a Python registration and generated
namespace exposure, and every reverse-differentiable contract to own an
autograd policy. It also rejects compressed `namespace oa::Fn*` declarations
across live C++ source, SDK, extension, and test trees. Its numerator and
denominator are derived from the complete schema inventory on every run.

That denominator cannot detect a handwritten public function which has no
schema row. `checkPublicOperationCoverage.py` is the independent breadth
gate: after CMake configuration, it uses the configured Clang command to derive
unique function identities from every public header which opens an `oa::Fn*`
namespace, then compares those identities with the schema set. Schema-only
kernel, internal-operation, and session-command identities are reported
separately. Both gates must pass before public schema breadth is complete.

Adding a new op = add a `[[ops]]` entry. If it owns a new fixed kernel, add one
`[ops.kernel]` block with its stable prefix/local ID, category, origin, and
shader source. The generator writes the private runtime registry row, the
install-safe public typed ID constant, and the private CMake source pair.
Existing shared kernels remain valid `kernel_forward` references without a new
ownership block. Public consumers use `oa/runtime/computeKernel.h`; the fixed
table itself is not SDK surface.

## Adding a new op

  1. Add `[[ops]]` to the appropriate schema TOML.
  2. For a new fixed kernel, add `[ops.kernel]` to that operation; do not patch
     generated registry or ID files.
  3. Run the preview generator and inspect the diff.
  4. Run `python3 tools/gen/checkDrift.py`; it must report the intended
     live drift and no unexplained stale output.
  5. Regenerate with `--live`, then rerun schema coverage, public-operation
     coverage, generation drift, and the owning tests.

## Round-trip diff against hand-written code

```bash
# Compare a generated shader against the hand-written one
diff -u source/cpp/lib/oa/core/shader/compute/flat/add.slang \
        build/gen/fn/cpp/lib/oa/core/shader/compute/flat/add.gen.slang

# Compare the elemwise .cpp
diff -u source/cpp/lib/oa/core/fnmatrix/elemwise/fnMatrixElemwise.cpp \
        build/gen/fn/cpp/lib/oa/core/fnmatrix/elemwise/fnMatrixElemwise.gen.cpp
```

Differences should be either intentional schema/template changes or bugs to fix
before live regeneration.

## Good Next Targets

  - Continue splitting large emitters by output contract without creating a
    second CLI or generation authority.
  - Generate more shape/dtype inference and independent oracle scaffolding
    where the current schema explicitly selects a manual implementation.
  - Keep every newly admitted public operation vertical: C++, Python,
    validation, semantic lowering, autograd where applicable, docs, and tests
    must enter through one schema change.
