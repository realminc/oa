# oapython

`oapython` is the Python distribution for
**[OA](https://github.com/realminc/oa)**, a Vulkan 1.4 GPU-first compute,
machine-learning, Vision, Audio, Ui, Plot, Media, and cryptography framework. Install
`oapython`; import it as `oa`.

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

a = FnMatrix.ones([2, 3])
b = FnMatrix.full([2, 3], 2.0)
c = (a + b) * 0.5
print(FnMatrix.copyToHost(c))
```

Import is host-only. The first device-backed request creates the native OA host;
beginner code does not initialize an engine manually.

## Public surface

| Owner | Surface |
|---|---|
| `Matrix`, `FnMatrix` | matrices, shapes, dtypes, factories, operators, and numerical operations |
| `QuantMatrix`, `Quantization` | semantic Q4/Q8 inference weights, pack/dequantize, fused matmul, and `.oam` persistence |
| `Module`, `FnLoss`, `FnAutograd` | modules, autograd, losses, optimizers, training, and metrics |
| `Image`, `FnImage` | image operations and still-image codecs; Vision/Media sessions own video and capture |
| `Audio`, `FnAudio` | semantic audio decode/processing operations, GPU saturation/Biquad/SOS filters, and stateful packetization |
| `plot` | retained figures, Line/Scatter/Bar/Histogram/Heatmap/Image artists, themes, and deterministic image output |
| `FnHash` and Crypto values | host cryptography and GPU public-data hashing when included in the wheel |

Python uses the same compute graph and kernels as native C++; it is not a NumPy or CPU
fallback implementation. Classes use PascalCase and operations use camelCase in both
languages, so a Python prototype translates directly to the C++ API. Lowercase domain
modules expose the same objects as the root package.

GPU Audio operations use the same public values in Python and C++:

```python
from oa import BiquadCoefficients, FnAudio

audio = FnAudio.decodeFile("input.wav")
section = BiquadCoefficients()
section.b0, section.b1, section.b2 = 0.25, 0.5, 0.25
filtered = FnAudio.sosFilter(audio, [section])
mastered = FnAudio.saturate(filtered, 6.0, 0.5)
FnAudio.saveWavF32("output.wav", mastered)
```

Core, Ml, Audio, Vision, Plot, end-to-end MNIST, and the canonical 15-workload NLP suite live under
[`sdk/py/tutorials`](https://github.com/realminc/oa/tree/main/sdk/py/tutorials).
The current desktop matrix and preview-to-preview performance history are published in
the [NLP benchmark](https://github.com/realminc/oa/blob/main/docs/external/benchmarks/oaNlpSuite.md).

## Links

- [Source and issue tracker](https://github.com/realminc/oa)
- [Documentation](https://dev.realm.software/)
- [License](https://github.com/realminc/oa/blob/main/LICENSE)

OA is source-available under the Business Source License 1.1 and converts to Apache-2.0
on the Change Date stated in the license.
