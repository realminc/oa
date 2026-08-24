# OA source generation

`tools/gen` is the single home for maintained source generators. A generator
owns a schema or an asset transformation; it is not a second source tree and it
does not keep checked-in preview copies.

## Layout

```text
tools/gen/
  generate.py        coordinated schema generation
  checkDrift.py       checked-in artifact and idempotence gate
  io.py               write-if-changed support
  fn/                 operation, binding, autograd and operation-kernel schema
  nn/                 neural-network module schema
  type/               public and private type schema
  kernel/             standalone fixed-kernel authority
  tile/               generated GEMM/kernel lattice
  example/            paired beginner SDK workflows and inventory
  python/             post-build Python stub generation
  font/               deterministic font-atlas generation
  animation/          offline animation asset conversion
```

Directories are lowercase namespaces. Maintained Python files, functions,
parameters and locals use camelCase. Python classes use PascalCase. Schema and
output filenames follow the language or public API they generate.

## Commands

```bash
# Preview every schema generator below ignored build/gen/ directories.
python3 tools/gen/generate.py

# Select one or more authorities.
python3 tools/gen/generate.py --fn --example

# Validate without writing.
python3 tools/gen/generate.py --dry-run

# Update checked-in generated artifacts.
python3 tools/gen/generate.py --live

# Prove checked-in artifacts equal a fresh, idempotent generation.
python3 tools/gen/checkDrift.py
```

`type`, `fn`, `nn`, `kernel`, `tile`, and `example` participate in the
coordinated gate.
The Python stub generator is post-build because it introspects the compiled
extension. Font and animation generators require explicit source assets and
are invoked independently.

## Output ownership

- `source/cpp/include/**/gen/` contains install-safe private mechanical
  fragments when a stable public header must include generated declarations.
- `source/cpp/lib/**/gen/` contains private runtime registries, manifests and
  metadata.
- `source/cpp/lib/**/gemm/gen/` contains generated GEMM route metadata.
- `cmake/gen/` contains generated build inventories only.
- `source/py/`, `test/`, `sdk/{cpp,py}/examples`, `sdk/examples.json`, and
  generated documentation contain the corresponding schema-derived surfaces.
- `build/gen/<authority>/` contains disposable preview output and is ignored.

A generated ownership banner names its generator and schema. Edit the schema or
generator, never a generated artifact. A second generation with unchanged
inputs must preserve bytes and timestamps.

## Adding a generator

Add a new authority only when no existing schema can own the output. It must
provide preview and live modes, write only changed bytes, identify every output
with an ownership banner, remove stale owned outputs, include independent
validation tests, and participate in `checkDrift.py`. Do not add another
checked-in `output/` mirror.
