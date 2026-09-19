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

The native PyO3 implementation lives below `src/py`. The importable package,
Cargo/Maturin metadata, and tutorials live below `sdk/py`; its
Cargo library target points at the canonical native binding root. PyO3 wrappers
contain Rust values directly. Matrix, ML, Image, and Audio
operations remain GPU-native; typed reads and `to_list` are explicit host
synchronization and copy boundaries. Encoded media crosses Python as `bytes`.

Version `0.8.2` is the Rust-first continuation of the historical `oapython`
`0.7.x` C++ binding line.

**Shipped:** runtime ownership and logging, explicit events, the currently
bound dense Matrix dtypes and operations, Image values/codecs/transforms,
Audio values/codecs/DSP/capture/playback, cryptography, vision metrics, and
the current video demux/decode/mux/playback value and session surface. The
wheel is exercised through the installed `oa` package rather than by importing
the source tree.

**Experimental:** neural-network modules, optimizers, training sessions,
training programs, callbacks, reinforcement-learning trainers, flow matching,
and SDK NLP recipe wrappers are hand-written PyO3 adapters. They are useful
for port validation, but their presence is not evidence of C++ Python API
parity. In particular, SDK recipe wrappers are not automatically part of the
long-term public binding merely because the underlying Rust SDK owns them.

**Planned:** generate the Python binding and type-stub surfaces from the
operation schema, reconcile the experimental surface with the C++ Python API,
and port the remaining presenter/UI, plotting, render, capture, recorder, and
video capability APIs. Each admitted session still requires an explicit
ownership, callback, shutdown, and error contract.

**Rejected:** duplicating every Rust SDK helper as a permanent Python public
type, treating a passing import as parity, or keeping stale declarations in
the type stub.

Python exceptions currently preserve OA's contextual display message. Stable
typed exception categories, generated binding signatures and stubs, zero-copy
NumPy/buffer exchange, awaitable events, and the remaining domains are
follow-up gates. Python tests execute against the same hardware Vulkan path as
Rust.
