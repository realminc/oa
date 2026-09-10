# OaDna — Domain-Native Acceleration

**Status:** Canonical cross-domain target; private DNN-named compatibility slice is Experimental

**Updated:** 2026-09-09

OaDna is OARS's private semantic partitioner and physical optimization planner.
“Domain-native” deliberately includes numerical, image, vision, video, audio,
render, crypto and ML work. It is not a DNN-only API, a public graph dialect, a
vendor plugin surface, or a second execution engine.

The current Rust implementation still lives under `runtime::dnn` and uses
`Dnn*` internal names because it is a narrow donor-backed compatibility port.
That vocabulary describes today's implementation, not the target boundary.
Rename or generalize it only as a complete vertical change after non-ML
semantic contracts enter the schema.

## Layer boundary

```text
SemanticGraph
  -> OaDna canonicalization and legal pattern recognition
  -> provider admission from operation schema
  -> shape/layout/dtype/color/training/liveness/capability predicates
  -> candidate plans from OaBlasLt, OaTile or another private provider
  -> cost/measurement selection
  -> executable nodes with complete semantic provenance
  -> source-preserving explicit fallback when no replacement qualifies
```

OaDna never submits, waits, allocates a device, exposes Vulkan handles or owns
an independent cache/runtime. `Engine` owns compilation services and the final
executable graph owns resources and synchronization.

## Current source truth

The Experimental analyzer:

- consumes the donor-backed semantic graph and generated compatibility roles;
- captures rank-1 through rank-4 positive-shape values with positive strides;
- partitions portable work and recognized DNN patterns in semantic order;
- retains graph hash, saved-for-backward values, workspace/determinism policy,
  explicit fallback reasons and applied/inherited/fallback counters;
- inherits already-proven source dispatches for attention and some single-op
  candidates rather than fabricating a replacement;
- applies only two exact FP32 inference replacements:
  - three shared-input Linear+bias projections at `M=1024, K=32, N=32` into
    `MlQkvProjectionBiasF32`;
  - two Linear+bias projections plus SwiGLU at
    `M=1024, N=64, K=32` into `MlGateUpSwigluBiasF32`.

Both applied paths require exact row-major, zero-offset, non-aliasing values and
strict intermediate liveness. Training candidates retain source execution.
`unexpected_fallback_count` distinguishes an internal failure from a normal
unqualified candidate.

The analyzer contains conceptual roles for residual norm, grouped MoE and
vision preprocessing, but the current generated schema admits no image/vision
operation roles and no corresponding physical replacement. Those enum variants
are scaffolding, not capability.

See [the current provider table](oaDnaSupport.md).
The implementation-specific boundary is recorded separately in
[the Rust DNN compatibility layer](oaDnn.md).

## Why cross-domain fusion matters

For many workloads, bandwidth and launch overhead dominate small arithmetic.
The highest-value first step is often a microfusion that avoids writing and
rereading an intermediate, not a heroic compute kernel.

### Vision and media preprocessing

Target semantic chain:

```text
VideoFrame/Image planes
  -> range/matrix/chroma-aware color conversion
  -> crop/resize with declared sampling and edge policy
  -> channel scale/bias normalization
  -> HWC/NHWC to CHW/NCHW conversion
  -> f32/f16/bf16 or admitted encoded output
```

The common `normalize` and conversion operations must be first-class semantic
operations even when the compiler later removes their intermediates. A valid
microfusion preserves coded/visible extent, plane strides, color primaries,
transfer function, YCbCr matrix, full/limited range, chroma location, alpha
policy, interpolation coordinates, rounding, output layout, dtype and frame
readiness. A generic three-channel assumption is not valid for every Image or
VideoFrame.

Initial admitted pattern should be the simpler dense Image chain:

```text
image::convert_color -> image::resize -> vision/image normalize -> layout/dtype convert
```

Qualify two-node combinations first, then the complete chain. Native NV12/P010
multi-plane video is a later provider after the value, external ownership and
synchronization contracts exist.

### Other initial patterns

- Matrix: matmul + bias + activation/residual/conversion;
- ML: QKV groups, gate/up groups, residual + normalization, attention and
  grouped MoE;
- Audio: sample conversion + channel mix + gain/normalization;
- Image/Render: transfer-function conversion + swizzle + pack/unpack;
- data transforms: elementwise chains whose single-use intermediates are not
  externally observable.

These are candidates, not promises. Pattern definitions come from schemas so
Rust APIs, graph identities, provider roles, docs and tests cannot drift.

## Partition legality

A replacement is legal only when all conditions hold:

- every operation contract/hash and typed attribute matches;
- shapes, layouts, strides, offsets, encodings and domain metadata match;
- aliases and mutations preserve semantic SSA behavior;
- eliminated values have no external observer or out-of-partition consumer;
- control/effect ordering is unchanged;
- training saved values or allowed recomputation are explicit;
- numeric and determinism policies are supported;
- device capabilities and workspace limits pass;
- all output identities and readiness edges are reconstructed;
- the provider has an independent oracle for that exact region.

Pattern matching should operate on def-use relationships, not merely adjacent
operation indices. Canonicalization may make equivalent scale/bias/conversion
forms comparable, but may not erase observable numeric distinctions.

## Provider model

A provider is an internal implementation family with:

- an applicability predicate;
- one or more immutable plan candidates and knob domains;
- workspace and persistent-resource requirements;
- numeric/determinism behavior;
- executable lowering;
- oracle/qualification provenance.

Providers should be Rust traits only where multiple real implementations need
the boundary. Start with a closed internal enum and typed plan variants; avoid a
stable dynamic plugin ABI until ownership, panic containment, thread safety,
versioning and unsafe FFI requirements are proven.

Provider failure is explicit:

- `NotApplicable`: normal, try another candidate/source path;
- `UnsupportedCapability`: normal capability-gated rejection;
- `ResourceLimit`: workspace or device limit rejected;
- `CompileFailure`: plan construction failed and must be reported;
- `ExecutionFailure`: never silently retry on CPU or another semantic path;
- `InternalInvariant`: increments unexpected fallback/error evidence.

## Selection and cache

OaDna chooses a partition; each provider supplies candidates. The common
selector filters legality, uses exact cached winners, applies a cold heuristic,
and may bounded-autotune. Plan identity includes the semantic region, metadata,
numeric/training policy, device/driver/capabilities, provider and artifact
versions, workspace and compiler provenance.

Cache hits pin immutable plans. Vendor or local algorithm ordinals alone are
not durable identities. Selection happens during compilation, never on every
replay.

## What is distinct in OARS

cuDNN, hipDNN and oneDNN Graph recognize and compile graph partitions, and CUDA
Graphs can replay and launch graphs. OARS should not claim those capabilities
are absent. Its intended distinction is one semantic graph spanning domain
types and one executable graph spanning kernels, transfers, rendering and media
while retaining domain metadata. That makes a Video-to-Vision-to-ML pipeline a
single optimization/lifetime problem rather than only a tensor subgraph.

That advantage remains architectural until OARS ships cross-domain operations,
microfusions and measurements.

## Acceptance gates

- generated candidate-provider admission with an empty regeneration diff;
- positive and near-miss pattern tests for metadata, liveness, aliasing,
  training and numeric policy;
- source-versus-fused differential oracle across odd and boundary sizes;
- deterministic normalized graph/executable reports retaining all owners;
- exact fallback counters and reasons;
- Vulkan validation on every admitted device class;
- end-to-end bandwidth/launch evidence, including fusion compilation cost and
  cache behavior, before selecting a replacement by default.

## Primary references

- [cuDNN Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)
- [hipDNN architecture](https://rocm.docs.amd.com/projects/hipdnn/en/latest/conceptual/architecture.html)
- [oneDNN Graph fusion patterns](https://uxlfoundation.github.io/oneDNN/graph_fusion_patterns.html)
