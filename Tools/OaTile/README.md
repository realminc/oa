# OaTile generator

`oatile.py` turns the bounded TOML lattice in `Schema/` into the CMake,
fixed-kernel and runtime matmul metadata consumed by OA. Raw GEMM and the
Bias/Bias+ReLU/Bias+GELU families all refer to one reviewed, software-pipelined
`GemmTiled.slang` arithmetic template; epilogue selection is compile-time.

```bash
python3 Tools/OaTile/oatile.py --live
python3 Tools/OaTile/test_oatile.py
python3 Tools/check_autogen_drift.py
```

Never hand-edit an `OaTile*.gen.*` file. Add a new record only when its stable
kernel ID is reserved, its geometry passes the generator constraints, and a
real workload justifies compiling and measuring it.
