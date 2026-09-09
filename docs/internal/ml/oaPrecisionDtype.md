# OARS precision and storage dtype

**Status:** F32/I32/U32 storage implemented; broader precision policy Planned

**Updated:** 2026-09-08

## Public vocabulary

`DType` describes a Matrix's logical dense storage representation. The admitted
Rust host types are exact and sealed:

| `DType` | Rust element | Current operation coverage |
|---|---|---|
| `F32` | `f32` | Matrix baseline and current ML forward/backward/AdamW |
| `I32` | `i32` | selected generated Matrix operations |
| `U32` | `u32` | exact upload/readback and ML class/token indices |

An enum variant does not imply universal kernel support. Each operation admits
an explicit dtype route and fails with `InvalidArgument` before recording when
the combination is unsupported. No runtime silently changes dtype or precision.

The Rust enum uses `F32`, `I32`, and `U32`: Rust primitive spellings are
lowercase, while enum variants follow UpperCamelCase. Schema and Slang attribute
tokens remain lowercase (`f32`, `i32`, `u32`).

## Current storage ABI

All admitted values currently occupy four bytes per dense element. Typed
`Matrix::from_slice` and `Matrix::read` require the exact sealed Rust element
type and perform no numeric conversion. ML uses F32 parameters/activations and
U32 indices. Embedding checks an index before accessing a weight row and writes
NaN for an out-of-range row; cross-entropy applies the same fail-visible policy
to an out-of-range class target.

## Deferred precision systems

BF16, F16, F64, quantized weights, mixed precision, master weights, and a global
engine precision policy are not implemented. OA C++'s BF16 capability/trust
gates and Q4/Q8 separation remain design evidence, but their shipped status is
not inherited. Each future dtype needs exact storage ABI, capability admission,
pipeline identity, operation coverage, conversions, autograd, persistence, and
numerical evidence. Unsupported requests must remain fail-closed.
