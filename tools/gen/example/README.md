# Paired SDK example generation

`examples.toml` owns the stable example inventory and the semantic workflows
for generated beginner examples. A workflow lists typed operations and checks
once; the generator emits idiomatic C++ and Python sources, CMake registration,
Python test registration, and `sdk/examples.json`.

C++ emitters include explicit engine ownership and completion. Python emitters
use the process-scoped lazy runtime and never expose engine initialization or
shutdown in beginner code. Existing handwritten examples can remain indexed
with `generated = false` while they migrate to the workflow vocabulary.

Run a preview, focused tests, or an intentional live update with:

```bash
python3 tools/gen/example/generate.py
python3 tools/gen/example/testGenerate.py
python3 tools/gen/example/generate.py --live
```

Generated files carry an ownership marker and unchanged generation preserves
their timestamps. `python3 tools/gen/checkDrift.py` verifies the checked tree.
