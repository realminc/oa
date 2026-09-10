# OARS DNN Compatibility Layer

**Status:** Experimental private subset of OaDna

**Updated:** 2026-09-09

This document describes the live `src/rs/runtime/dnn.rs` implementation. DNN is
not the owner of OARS compute architecture and is not a public API. It is the
first donor-backed compatibility provider inside the broader cross-domain
[OaDna](oaDna.md) compiler.

## Current data model

The private planner translates eligible semantic values and operations into
`DnnValueDesc`, `DnnOpDesc`, ordered partitions and a `DnnPlan`. It retains:

- semantic and storage-root identities;
- shape, positive strides, dtype and byte offset;
- external/virtual state;
- inputs, outputs, generated role/provider mask and training state;
- workspace/determinism/recompute policy;
- saved-for-backward values and explicit lowering state;
- source/captured/recognized/applied/inherited/fallback counters.

Values are currently restricted to non-empty rank 1-4 descriptions. Operations
without a generated exact name/contract-hash match remain portable.

## Current partition vocabulary

- portable;
- BLASLt epilogue;
- QKV projection group;
- gated FFN;
- residual normalization;
- attention;
- grouped MoE;
- vision preprocessing.

This vocabulary is broader than execution. Only the exact QKV and gate/up
replacements in [the provider snapshot](oaDnaSupport.md) produce new physical
nodes. Attention and already-fused single-operation candidates may inherit the
existing dispatch. Residual norm, grouped MoE and vision preprocessing have no
current generated/executable provider.

## Lowering states

| State | Meaning |
|---|---|
| `Analyzed` | partition has been classified but not lowered |
| `Applied` | qualified replacement nodes were installed |
| `Inherited` | source physical execution already satisfies the recognized region |
| `ExplicitFallback` | recognized provider was not qualified; source execution retained with reason |

Portable partitions remain analyzed and are not counted as provider fallbacks.
Any recognized partition left without a terminal lowering state is an internal
error. `unexpected_fallback_count` must remain zero in accepted performance
evidence.

## Required migration seam

Do not grow `DnnOpType` into a second universal operation enum. As Image,
Vision, Video and Audio contracts land:

1. keep semantic identity in the common `SemanticGraph` and operation schema;
2. move generic region/value/provider/plan mechanics into private OaDna-named
   modules;
3. retain a DNN provider module for tensor-specific patterns and policy;
4. add separate cross-domain provider families such as vision preprocessing;
5. preserve report compatibility or version it explicitly;
6. remove the superseded DNN-only canonical route in the same checkpoint.

The migration must preserve current QKV/gate-up tests and source fallback. A
rename alone is not architectural progress.

## Provider rules

- provider roles are generated from the operation schema;
- pattern recognition uses exact contracts and def-use/liveness constraints;
- training replacements preserve saved values or declared recomputation;
- physical lowering retains every semantic owner and eliminated-intermediate
  lifetime evidence;
- provider selection remains private and vendor-neutral;
- execution failure never triggers silent CPU fallback;
- enum presence, recognition and inherited execution are not capability claims.

## Next vertical slice

The next DNN-specific step is not more enum cases. It is a general
`MatmulProblem`/candidate route beneath existing Linear and `mat_mul_nt`, so
BLASLt epilogues and current exact fusions share one plan/caching system. In
parallel, the first non-DNN OaDna slice should generate and execute dense Image
conversion/normalization microfusion roles.
