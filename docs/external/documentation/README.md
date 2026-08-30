# Writing public OA documentation

**Status:** Current public API, example and tutorial contract

**Updated:** 2026-08-18

## One source for each surface

| Surface | Mechanical authority | Published result |
|---|---|---|
| C++ values, types and inheritance | `source/cpp/include/oa` parsed through `tools/documentation/oaApiSurface.cpp` | compiler-derived C++ reference snapshot |
| Python names and signatures | nanobind registrations, generated stubs and checked `oa.__all__` | Python reference snapshot |
| Schema operations | operation schemas and generators | matching C++, Python, validation and reference surfaces |
| Tutorial excerpts | checked source between `OA_DOC_BEGIN` and `OA_DOC_END` markers | developer-site code examples |
| Tutorial media | approved files under `sdk/asset/docs` with provenance | offline SDK and developer-site presentation assets |
| API examples | checked SDK source markers plus `tools/gen/example/schema/examples.toml` publication metadata | generated inventory and API-linked source excerpts |

Never hand-copy a signature into generated reference data. Committed generated
files are reviewable snapshots for standalone site builds, not a second source.
If extraction is wrong, fix the source comment, schema or generator.

## API reference quality

The Maya and Vulkan references are the quality baseline: a developer should be
able to understand ownership, call the API correctly and find a working example
without reverse-engineering implementation source.

Document each intended public type or operation with:

1. a one-sentence purpose;
2. ownership and lifetime, including borrowed engines and sessions;
3. native types, inheritance and overloads;
4. parameter direction and meaning, including shapes, dtype, layout and units;
5. return value, error/status and empty/zero-size behavior;
6. submission and completion behavior (`oa::Event`, blocking, or host-only);
7. required capability and unavailable behavior;
8. a minimal example and links to larger tutorials where useful.

The current compiler extractor preserves pointers, lvalue/rvalue references,
defaults, public inheritance and comments. The source link remains authoritative
for method `const`, ref and `noexcept` qualifiers that clang-doc JSON does not
expose. Python documentation represents the actual imported binding and may
intentionally differ in ownership syntax. The language selector must choose the
corresponding source; it must never rename a Python signature and call it C++.

## Examples and tutorials

`sdk/{cpp,py}/examples` and `sdk/{cpp,py}/tutorials` are maintained public
source material with different scope:

| Property | SDK examples | SDK tutorials |
|---|---|---|
| Goal | smallest complete use of one API concept | teach a workflow or cross-module system |
| Typical size | one short source file | multiple steps, assets and supporting prose |
| Languages | paired where the API is bound | paired unless a capability is language-specific |
| Proof | builds/runs and checks its result | end-to-end correctness or smoke gate |
| Publication | linked from the API symbol | dedicated tutorial page with expected output |

Scratch experiments, benchmarks and regression tests do not belong in
`examples/`. Use an untracked scratch directory, `test/`, or `benchmark/`.

A published tutorial contains:

- a clear goal, prerequisites and expected output;
- supported capabilities and exact commands;
- C++ and Python source paths;
- marker-derived excerpts from those files;
- source assets with rights/provenance and generated output under
  `sdk/asset/docs/<topic>`;
- ownership, synchronization and failure behavior when relevant;
- an automated correctness or smoke gate;
- links to the API reference and the next useful tutorial.

Use the same semantic workload in paired languages. Language-specific setup may
differ, but copied constants, assets and expected results must agree. Do not put
an untested code block in Markdown when it can be extracted from compiled or
executed source.

## Publication gate

After changing public code or documentation:

1. build and run the narrow C++ and Python examples/tutorials involved;
2. regenerate C++ and Python reference snapshots and tutorial data;
3. regenerate `tools/documentation/generateExamples.py` and run its drift check;
4. run `python3 tools/documentation/checkExternalDocs.py`;
5. build the developer site and inspect changed pages at desktop and narrow widths.

From the sibling developer-site checkout:

```bash
python3 scripts/test_generate_cpp_api.py
npm run docs:generate
npm run docs:check
npm run build
```

Standalone deployment consumes committed generated snapshots because its CI
does not clone the private development tree.
