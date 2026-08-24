# Tile generation

`generate.py` turns the bounded TOML manifests in `schema/` into the CMake,
fixed-kernel and runtime matmul metadata consumed by OA. `Base.toml` owns the
non-templated correctness and CoopVec routes. Raw tiled GEMM and the fused
epilogue families refer to one reviewed, software-pipelined `GemmTiled.slang`
arithmetic template; epilogue selection is compile-time.

```bash
python3 tools/gen/tile/generate.py --live
python3 tools/gen/tile/testGenerate.py
python3 tools/gen/checkDrift.py
```

Never hand-edit a generated Tile artifact. Add a new record only when its stable
kernel ID is reserved, its geometry passes the generator constraints, and a
real workload justifies compiling and measuring it.
