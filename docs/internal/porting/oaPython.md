# OA Rust to Python binding

**Status:** Experimental binding contract

Python mirrors the Rust public ownership graph. It is not a compatibility copy
of the former C++ class namespaces and does not create a second runtime.

| Rust | Python | Meaning |
| --- | --- | --- |
| `oa::Engine` | `oa.Engine` | root identity re-export |
| `oa::runtime::Engine` | `oa.runtime.Engine` | owning-module identity |
| `oa::Matrix` | `oa.Matrix` | root identity re-export |
| `oa::core::Matrix` | `oa.core.Matrix` | owning-module identity |
| `oa::matrix::ones(&engine, shape)` | `oa.matrix.ones(engine, shape)` | stateless operation |
| `oa::matrix::add(&a, &b)` | `oa.matrix.add(a, b)` | input-derived Engine |

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

The extension crate lives below `sdk/py` and links the same Rust `oa` crate.
PyO3 wrappers contain Rust values directly. Matrix operations remain GPU-native;
`read_f32` and `to_list` are explicit host synchronization and copy boundaries.
Python exceptions currently preserve OA's contextual display message; stable
typed exception categories, generated signatures, NumPy/buffer exchange,
async events, and the broader domain surface remain follow-up gates.

The first admitted slice is `Engine`, FP32 `Matrix::from_f32`, `matrix::ones`,
`matrix::full`, `matrix::add`, `matrix::mat_mul_nt`, shape/dtype inspection,
operator addition, and FP32 readback. Its Python test executes against the same
hardware Vulkan path as Rust.
