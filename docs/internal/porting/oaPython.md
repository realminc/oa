# OA Rust to Python binding

**Status:** Rust-first preview binding contract

Python mirrors the Rust public ownership graph. It is not a compatibility copy
of the former C++ class namespaces and does not create a second runtime.

| Rust | Python | Meaning |
| --- | --- | --- |
| `oa::Engine` | `oa.Engine` | root identity re-export |
| `oa::runtime::Engine` | `oa.runtime.Engine` | owning-module identity |
| `oa::Matrix` | `oa.Matrix` | root identity re-export |
| `oa::core::Matrix` | `oa.core.Matrix` | owning-module identity |
| `oa::Image` | `oa.Image` | root identity re-export |
| `oa::image::Image` operations | `oa.image` | image codecs and transformations |
| `oa::Audio` | `oa.Audio` | root identity re-export |
| `oa::audio` operations | `oa.audio` | audio codecs, DSP, and features |
| `oa::matrix::ones(&engine, shape)` | `oa.matrix.ones(engine, shape)` | stateless operation |
| `oa::matrix::add(&a, &b)` | `oa.matrix.add(a, b)` | input-derived Engine |
| `oa::ml::matrix::gelu(&x)` | `oa.ml.matrix.gelu(x)` | differentiable ML operation |

Caller abbreviations remain ordinary imports:

```python
import oa.audio as oaa
import oa.core as oac
import oa.matrix as oam
import oa.ml as oaml
```

OA does not publish `oaa`, `oac`, `oam`, or `oaml` alias modules. A domain is
importable only after it contains a real bound capability; an empty Python
namespace is not a feature claim.

The PyPI distribution is named `oapython` because the unrelated `oa`
distribution is owned by another project. Installation and import therefore
have intentionally different names:

```bash
python -m pip install oapython
python -c "import oa; print(oa.__version__)"
```

The extension crate lives below `sdk/py` and links the same Rust `oa` crate.
PyO3 wrappers contain Rust values directly. Matrix, ML, Image, and Audio
operations remain GPU-native; typed reads and `to_list` are explicit host
synchronization and copy boundaries. Encoded media crosses Python as `bytes`.

Version `0.8.0` is the Rust-first continuation of the historical `oapython`
`0.7.x` C++ binding line. Its admitted surface includes runtime events, all
currently bound dense dtypes, the functional Matrix and ML operation spine,
byte-model helpers, Image values/codecs/common transforms, and Audio values/
codecs/DSP/features. Stateful neural-network, training, media, render, and
presentation objects remain Rust-only pending explicit lifetime and callback
contracts.

Python exceptions currently preserve OA's contextual display message. Stable
typed exception categories, generated binding signatures, zero-copy NumPy/
buffer exchange, awaitable events, and remaining domains are follow-up gates.
Python tests execute against the same hardware Vulkan path as Rust.
