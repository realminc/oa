# OA Python SDK

This is the first native Python binding for the Rust OA runtime. It exposes the
small, executable Matrix slice used by the public introduction:

```python
import oa

engine = oa.Engine()
one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
total = oa.matrix.add(one, two)

assert total.shape == [2, 3]
assert total.to_list() == [3.0] * 6
```

Python mirrors Rust module ownership: root value aliases remain concise,
stateless operations live in `oa.matrix`, and the runtime owner is explicit.
Callers may use ordinary local abbreviations such as `import oa.matrix as oam`
or `import oa.core as oac`; OA does not publish duplicate alias modules. Host
observation through `read_f32()` or `to_list()` is the synchronization boundary.

Build and test from this directory with:

```bash
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop
.venv/bin/python -m unittest discover -s test -v
```

This checkpoint intentionally binds dense FP32 creation, addition,
`mat_mul_nt`, shape/dtype inspection, and readback. The wider Rust surface is
added only through generated binding metadata and matching tests.
