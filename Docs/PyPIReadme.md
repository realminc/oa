# oapython

`oapython` is the Python distribution for
**[OA](https://github.com/realminc/oa)**, a cross-vendor Vulkan 1.4 GPU compute,
machine-learning, vision, audio, and cryptography framework. Install `oapython`; import
it as `oa`.

> **0.7 development preview.** The bindings execute real OA C++ objects and Vulkan
> kernels, but the Python API and binary ABI are not frozen.

## Install

```bash
python -m pip install oapython
```

The current wheel targets Linux x86-64, CPython 3.12, and glibc 2.39+. The host must
provide a Vulkan loader and a working vendor ICD.

## First computation

```python
import oa

assert oa.runtime.OaInitComputeEngine()

x = oa.core.Rand([64, 32])
layer = oa.ml.OaLinear(32, 64)
with oa.Context():
    y = layer.Forward(x)
```

## Modules

| Namespace | Surface |
|---|---|
| `oa.core` | matrices, shapes, dtypes, factories, and tensor operations |
| `oa.runtime` | engine lifecycle and compute contexts |
| `oa.ml` | modules, autograd, losses, optimizers, training, and metrics |
| `oa.vision` | image operations, codecs, video, and capture |
| `oa.audio` | decode/encode and GPU audio processing |
| `oa.crypto` | host cryptography and GPU public-data hashing |

Python uses the same compute graph and kernels as native C++; it is not a NumPy or CPU
fallback implementation. End-to-end MNIST and NLP training examples live under
[`Tutorial/Py`](https://github.com/realminc/oa/tree/main/Tutorial/Py).

## Links

- [Source and issue tracker](https://github.com/realminc/oa)
- [Documentation](https://dev.realm.software/)
- [License](https://github.com/realminc/oa/blob/main/LICENSE)

OA is source-available under the Business Source License 1.1 and converts to Apache-2.0
on the Change Date stated in the license.
