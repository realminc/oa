# OaDna Provider Admission Snapshot

**Status:** Current generated-role audit; not a generated capability claim

**Updated:** 2026-09-09

This table describes the live OARS compatibility roles generated into
`src/rs/runtime/dnn/generated.rs`. The source of truth is the operation schemas
and `tools/gen/fn/generate.py`; this document is an audited status snapshot.
OARS does not yet generate a documentation artifact equivalent to the C++
donor's `oaDnaSupport.gen.md`. Adding one must extend the generator rather than
hand-maintain a `.gen.md` file.

Candidate admission permits recognition only. Concrete execution still depends
on shape, layout, dtype, alias/liveness, training, workspace and capability
predicates.

| Rust semantic operation | Recognized role | Candidate providers | Current replacement status |
|---|---|---|---|
| `matrix::add` | add | residual norm | no residual-norm replacement |
| `matrix::mul` | multiply | gated FFN | participates only in recognized source patterns |
| `matrix::mat_mul_nt` | matmul | BLASLt epilogue, QKV group, gated FFN | no general BLASLt replacement |
| ML Linear | matmul + optional bias | BLASLt epilogue, QKV group, gated FFN | two exact-shape inference providers below |
| ML GELU | GELU | BLASLt epilogue | source execution/inheritance only |
| causal SDPA | attention | attention | source execution/inheritance only |
| ML SwiGLU | gated multiply | gated FFN | exact gate/up fusion below |

## Applied Experimental providers

| Provider | Exact admitted region | Physical kernel |
|---|---|---|
| QKV projection group | three FP32 row-major zero-offset Linear+bias projections sharing `[1024,32]`; each weight `[32,32]`, bias `[32]`, output `[1024,32]`; inference; no invalid aliases | `MlQkvProjectionBiasF32` |
| Gate/up SwiGLU | two FP32 row-major zero-offset Linear+bias projections sharing `[1024,32]`; weights `[64,32]`, biases `[64]`, outputs `[1024,64]`; SwiGLU; inference; internal values not externally live | `MlGateUpSwigluBiasF32` |

Training and all unqualified shapes retain the source graph with an explicit
fallback reason. Recognition does not count as `Applied`.

## Missing schema admissions

The private analyzer has conceptual variants for color conversion,
resize-normalize, residual/RMS normalization and grouped MoE, but the current
generated role table has no image, vision, video, audio, residual-norm or
grouped-GEMM contract rows. Those providers are Planned.

The first cross-domain generator extension should add exact roles for dense
Image color conversion, resize, normalization, layout conversion and dtype
conversion together with generated positive/near-miss fixtures. It must not add
roles for public operations that do not yet exist.

## Generation target

When documentation generation is implemented, the schema should emit:

- semantic Rust path and stable contract identity;
- normalized role;
- candidate providers and required input edges;
- shape/dtype/layout/metadata rule identifiers;
- lowering class and training policy;
- generated source provenance.

Runtime qualification and measured status remain separate columns sourced from
checked evidence, because a schema cannot prove hardware execution.
