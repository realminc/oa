# OaNnAutogen

Schema-driven generator for `OaModule` layer wrappers.

This generator owns mechanical NN layer boilerplate. It intentionally does not
own `OaModule`, optimizer behavior, runtime dispatch, or graph policy.

Generated layer files are split one layer per file:

```text
Source/Private/Oa/Ml/Nn/Relu.gen.h
Source/Private/Oa/Ml/Nn/Relu.gen.cpp
```

Layer names use PascalCase without acronym spelling:

```text
Relu, Gelu, Silu, Softmax, RmsNorm
```

Run:

```bash
python3 Tools/NnAutogen/oannautogen.py --live
```

The VS Code task `Ml - OaFnAutoGen` runs both `OaFnAutogen` and `OaNnAutogen`.
