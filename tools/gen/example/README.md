# Paired SDK example inventory

Beginner examples are ordinary, source-owned C++ and Python programs under
`sdk/*/examples`. They remain directly reviewable and runnable because tutorial
control flow is not a mechanically derivable API surface.

`examples.toml` owns only stable publication metadata: pair identity, paths,
build target, execution profile, API backlinks, checked assets, and captured
evidence. The inventory tool validates both sources and their documentation
markers, then generates CMake registration, Python test registration, and
`sdk/examples.json`. It never writes example source.

Run a preview, focused tests, or an intentional live update with:

```bash
python3 tools/gen/example/generate.py
python3 tools/gen/example/testGenerate.py
python3 tools/gen/example/generate.py --live
```

Generated inventory files carry an ownership marker and unchanged generation
preserves their timestamps. `python3 tools/gen/checkDrift.py` verifies them;
the example programs themselves remain outside generator ownership.
