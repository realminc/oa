# OA KernelAutogen

`KernelAutogen` owns fixed shader entry points that are neither one-to-one
operation kernels nor generated OaTile GEMM variants.

The three kernel authorities are deliberately non-overlapping:

- `tools/fnAutogen/schema/**.toml [ops.kernel]` owns operation and kernel-only
  semantic rows;
- `tools/oaTile/schema/*.toml` owns GEMM compilation, IDs, capabilities, and
  routes;
- `tools/kernelAutogen/schema.toml` owns every remaining fixed shader's source,
  compilation profile, stable packed ID, category, stage, and retired-ID ledger.

The generated files are composed by `KernelRegistry.h` and CMake. The configured
`spirv_shader_list.txt` is then derived from the exact compilation outputs and
is the sole embedding manifest. Compiled content hashes remain build outputs;
they cannot truthfully be source-schema constants.

## Commands

```bash
python3 tools/kernelAutogen/kernelautogen.py
python3 tools/kernelAutogen/kernelautogen.py --live
python3 tools/kernelAutogen/test_kernelautogen.py
python3 tools/check_autogen_drift.py
python3 tools/audit_kernels.py --build-dir build/release --strict
```

The generator writes only when bytes change. It rejects missing sources,
duplicate names or packed IDs, invalid stages/profiles/conditions, invalid
reservation ranges, and active IDs inside the reservation ledger. The strict
configured-build audit performs the final cross-authority collision, coverage,
compiled-output, production-liveness, and typed-ID proof.

Add a row here only when no operation schema or OaTile schema owns it. Retiring
a row means deleting the active entry and reserving its exact local ordinal;
never renumber or compact shipped IDs.
