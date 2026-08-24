# OA Python test profiles

Python tests mirror OA's domain hierarchy under `test/py`. Production
bindings and the installable package remain under `source/py`.

CTest registers only profiles whose build capabilities are present:

- `TestPythonHost`: host-only, non-crypto contracts;
- `TestPythonGpu`: the non-crypto host baseline plus Vulkan contracts;
- `TestPythonCryptoHost`: crypto host contracts and canonical stub drift;
- `TestPythonCryptoGpu`: crypto host plus Vulkan hashing contracts;
- `TestPythonGymnasium`: the external adapter contract, only when Gymnasium is
  importable by the configured interpreter.

Every registered profile passes `--oa-forbid-skips`. A selected test therefore
fails if its capability is broken; unavailable optional packs are omitted at
configure time instead of producing a runtime skip.

Build and run the registered profiles with:

```bash
cmake --build Build/PythonEditable --target _oa
ctest --test-dir Build/PythonEditable --output-on-failure -L python
```

For a direct source-build run:

```bash
OA_PYTHON_BUILD_DIR=Build/PythonEditable \
  Build/PythonEnv/bin/python -m pytest \
  --oa-profile=gpu --oa-forbid-skips test/py
```
