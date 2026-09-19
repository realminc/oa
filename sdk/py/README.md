# OA Python SDK

`oapython` is the native Python distribution for the Rust OA runtime. It
installs the `oa` import package and mirrors Rust's public module structure:

```bash
python -m pip install oapython
```

```python
import oa

engine = oa.Engine()
one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
total = oa.matrix.add(one, two)

assert total.shape == [2, 3]
assert total.to_list() == [3.0] * 6

image = oa.Image.from_matrix(oa.matrix.ones(engine, [3, 2, 2]), "chw", "rgb")
flipped = oa.image.flip(image, horizontal=True, vertical=False)
```

Python mirrors Rust module ownership: root value aliases remain concise,
stateless operations live in `oa.matrix`, and the runtime owner is explicit.
Callers may use ordinary local abbreviations such as `import oa.matrix as oam`
or `import oa.core as oac`; OA does not publish duplicate alias modules. Host
observation through `read_f32()` or `to_list()` is the synchronization boundary.

The canonical PyO3 implementation lives in `../../src/py`. This SDK directory
owns the importable package, packaging metadata, and tutorials.

Build from this directory with:

```bash
python -m venv .venv
.venv/bin/pip install maturin patchelf ziglang
.venv/bin/maturin develop
```

Tests are located in the repository `test/py/` directory and run with:

```bash
python -m unittest discover -s test/py -v
```

Build the portable Python 3.10+ Linux wheel with:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/maturin build --release --zig
```

The explicit Zig build prevents a developer workstation's newer glibc symbols
from leaking into a wheel labeled for PyPI.

The `0.8.3` Rust-first preview ships runtime ownership and logging, explicit
events, typed dense matrices, Matrix and ML functional operations, Image
values/codecs/transforms, Audio values/codecs/DSP/capture/playback,
cryptography, vision metrics, and the current video
demux/decode/mux/playback surface.

Neural-network modules, optimizers, training sessions and programs, callbacks,
reinforcement-learning trainers, flow matching, and SDK NLP recipes are
experimental adapters used for port validation. They are not a promise that
every Rust SDK helper is a permanent Python API. Presenter/UI, plotting,
complete render and capture/recorder support, generated type stubs, and final
C++ Python API reconciliation remain planned work.
