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
from oa import *

a = OaFnMatrix.Ones([2, 3])
b = OaFnMatrix.Full([2, 3], 2.0)
c = (a + b) * 0.5
print(OaFnMatrix.CopyToHost(c))
```

Import is host-only. The first device-backed request creates the native OA host;
beginner code does not initialize an engine manually.

## Public surface

| Owner | Surface |
|---|---|
| `OaMatrix`, `OaFnMatrix` | matrices, shapes, dtypes, factories, operators, and numerical operations |
| `OaModule`, `OaFnLoss`, `OaFnAutograd` | modules, autograd, losses, optimizers, training, and metrics |
| `OaImage`, `OaFnImage`, `OaImageDecoder`, `OaImageEncoder` | image operations, still-image codecs, video, and capture |
| `OaAudio`, `OaFnAudio`, `OaAudioDecoder`, `OaAudioEncoder` | semantic audio decode, processing, and encode |
| `OaFnHash` and Crypto values | host cryptography and GPU public-data hashing when included in the wheel |

Python uses the same compute graph and kernels as native C++; it is not a NumPy or CPU
fallback implementation. Public names retain their C++ PascalCase spelling so a Python
prototype translates directly to the C++ API. Lowercase domain modules remain
compatibility aliases, not the canonical tutorial vocabulary.

Core, ML, Audio, Vision, end-to-end MNIST, and the complete 16-entry NLP suite live under
[`Tutorial/Py`](https://github.com/realminc/oa/tree/main/Tutorial/Py).
The current desktop matrix and preview-to-preview performance history are published in
the [NLP benchmark](https://github.com/realminc/oa/blob/main/Docs/Benchmarks/OaNlpSuite.md).

## Links

- [Source and issue tracker](https://github.com/realminc/oa)
- [Documentation](https://dev.realm.software/)
- [License](https://github.com/realminc/oa/blob/main/LICENSE)

OA is source-available under the Business Source License 1.1 and converts to Apache-2.0
on the Change Date stated in the license.
