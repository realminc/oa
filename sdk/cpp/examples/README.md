# OA examples

**Status:** Public example contract; existing ML inventory is under migration

`sdk/{cpp,py}/examples` contains the smallest maintained, executable uses of individual OA
API concepts. Examples sit beside the API reference: they answer “what is the
least code needed to do this correctly?”

Larger workflows, cross-module applications and teaching narratives belong in
`sdk/{cpp,py}/tutorials`. Correctness and regression coverage belongs in `test/`; performance
claims belong in `benchmark/`. Temporary experiments stay untracked and are not
published as examples.

## Acceptance contract

Each new example must:

- use the current public API and its normal ownership model;
- have one narrow purpose and no unrelated framework scaffolding;
- check its result and return failure when the operation fails;
- be registered in CMake/CTest or the applicable Python test profile;
- build and run in the configuration documented beside it;
- link to its API entry and use a paired language version where that public
  surface exists in both C++ and Python.

An example is not a scratch test. It remains versioned and must be updated when
its public API changes.

`tools/gen/example/schema/examples.toml` is the source of truth for the
publication index. Generated beginner workflows emit paired C++ and Python
source, build/test registration, and `sdk/examples.json` together. Each entry
owns one stable ID, canonical module, paired source paths, build profile,
language-specific API symbols and runtime capability. C++ and Python symbol
inventories resolve through the same generated reference catalog; they are not
assumed to use identical lifecycle helpers. The checked source between matching
`OA_DOC_BEGIN` and `OA_DOC_END` markers is the only code published by the site.

Generate and verify the public example snapshot with:

```bash
python3 tools/documentation/generateExamples.py
python3 tools/documentation/generateExamples.py --check
```

The generator rejects missing pairs, duplicate IDs, unsafe paths, unregistered
C++ targets, untested Python files and marker drift.

## Examples versus tutorials

| Property | Example | Tutorial |
|---|---|---|
| Scope | one API concept | complete workflow or subsystem |
| Context | minimal | prerequisites, design explanation and next steps |
| Assets | only when required by the call | curated inputs and expected outputs |
| Verification | direct result assertion | end-to-end correctness/smoke gate |
| Documentation | API-linked summary | dedicated generated page |

The existing `sdk/cpp/examples/ml` files predate this policy and include test-like and
experimental programs. Treat that directory as cleanup debt: convert useful
single-concept programs to this contract, move regression coverage to `test/cpp/`,
and remove abandoned experiments. Do not copy its historical structure as the
template for new work.

## Maintained examples

| Concept | C++ | Python | Gate |
|---|---|---|---|
| Device matrix addition | `cpp/core/matrix.cpp` | `py/core/matrix.py` | `ExampleCoreMatrix`; GPU Python profile |
| Audio file processing | `cpp/audio/audio.cpp` | `py/audio/audio.py` | `ExampleAudio`; GPU Python profile |
| Image resize | `cpp/vision/resize.cpp` | `py/vision/resize.py` | `ExampleVisionResize`; GPU Python profile |
| Batch SHAKE-256 | `cpp/crypto/shake256.cpp` | `py/crypto/shake256.py` | `ExampleCryptoShake256`; crypto-GPU Python profile |
| Retained line plot | `cpp/plot/line.cpp` | `py/plot/line.py` | `ExamplePlotLine`; GPU Python profile |

## Related paths

- `sdk/{cpp,py}/tutorials/` — maintained paired end-to-end learning material
- `test/{cpp,py}/` — correctness and regression gates
- `benchmark/` — controlled performance measurement
- [Writing public OA documentation](../docs/external/documentation/README.md)
- [OA developer documentation](https://dev.realm.software/)
