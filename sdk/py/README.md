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

The canonical PyO3 implementation lives in `../../src/py`. Root
`../../pyproject.toml` and `../../uv.lock` own packaging and environment
metadata; this SDK directory owns the importable package, PyO3 crate, and
tutorials.

Build from the repository root with:

```bash
python3 tool/build/workflow.py --profile debug --python-only
```

The workflow creates or reuses the root `.venv`, rebuilds the native extension,
then verifies package imports and native/distribution version agreement in a
fresh Python process. Re-run it after changing native exports; an older installed
extension does not gain renamed classes from edits to the Python wrappers.

Tests are located in the repository `test/py/` directory and run with:

```bash
.venv/bin/python -m unittest discover -s test/py -v
```

Build the portable Python 3.10+ Linux wheel with:

```bash
.venv/bin/pip install ziglang
PATH="$PWD/.venv/bin:$PATH" ZIG_COMMAND=python-zig .venv/bin/maturin build --release --zig
```

The explicit Zig build prevents a developer workstation's newer glibc symbols
from leaking into a wheel labeled for PyPI.

The Rust-first preview exposes runtime ownership and logging, explicit
events, typed dense matrices, Matrix and ML functional operations, Image
values/codecs/transforms, Audio values/codecs/DSP/capture/playback,
cryptography, vision metrics, and the current Video surface.

Neural-network modules, optimizers, training sessions and programs, callbacks,
reinforcement-learning trainers, flow matching, and SDK NLP recipes remain
experimental. Python targets compute, ML, media, Plot, and Viewer preview;
low-level Vulkan and UI construction remain Rust-only. Capability admission and
hardware qualification are separate from binding availability.
