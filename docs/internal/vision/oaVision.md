# OA Rust Vision

**Status:** Experimental complete donor `FnDetection` surface

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Translation:** [C++ to Rust API translation](../porting/oaCppToRust.md)

Vision is the interpretation domain for Images and VideoFrames. It owns
detection, tracking, segmentation, evaluation, and related semantic results.
It does not own still-image codecs, image transforms, video codecs, capture,
playback, Texture, or presentation.

## Public mapping

The donor `oa::FnDetection` namespace becomes free functions in `oa::vision`:

```text
oa::FnDetection::boxIou(...)              oa::vision::box_iou(...)
oa::FnDetection::nms(...)                 oa::vision::nms(...)
oa::FnDetection::confusionMatrix(...)     oa::vision::confusion_matrix(...)
oa::FnDetection::binaryMaskCounts(...)    oa::vision::binary_mask_counts(...)
oa::FnDetection::evaluate(...)            oa::vision::evaluate(...)
oa::FnDetection::evaluateSegmentation(...) oa::vision::evaluate_segmentation(...)
```

No Rust `FnDetection` type, nested `vision::video` module, forwarding facade,
or duplicate root function is permitted. Principal Vision values or sessions
may receive explicit root identity aliases only after their implementation and
contract tests are admitted; configurations and result descriptors remain
module-owned by default.

## Shipped checkpoint

The complete public donor `oa::FnDetection` surface is Experimental in
`oa::vision`: pairwise IoU, deterministic NMS, classification confusion,
binary-mask counts, dataset detection evaluation, and segmentation evaluation.
`vision_detection.json` jointly owns their operation contracts, stable kernel
IDs 125–136, Slang/reflection ABI, physical-write declarations, and embedded
artifacts. Rust result structs keep every output device-resident:

| Operation | Output |
| --- | --- |
| `box_iou` | FP32 `[N, M]` |
| `nms` | I32 selected indices and U32 scalar count |
| `confusion_matrix` | U32 `[C, C]` |
| `binary_mask_counts` | U32 `[4]` in TP, FP, FN, TN order |
| `evaluate` | U32 `[T,C,3]`, FP32 `[T,C,4]`, FP32 `[T]`, FP32 `[1]` |
| `evaluate_segmentation` | U32 `[C,C]`, FP32 `[C,4]`, FP32 mean-IoU and accuracy scalars |

NMS is a one-invocation deterministic donor port: descending finite score,
source-index tie breaking, optional class-agnostic suppression, and no host
sort or compaction. Detection evaluation remains three executable stages under
one semantic operation; segmentation evaluation remains three stages under one
semantic operation. Replay-safe clear stages precede U32 atomic accumulators,
so repeated plan submission cannot retain earlier counts.

The operation preserves donor OA's negative-size rule by clamping width and
height to zero. OARS additionally defines a non-finite box to produce zero for
that pair, avoiding backend-dependent NaN comparisons. Rank, extent, dtype,
ownership, and ABI-size failures return `InvalidArgument` or `OutOfRange`
before recording. Value-domain rejection is deliberately not a synchronous API
check: inputs are GPU-resident, and inspecting every coordinate would add a
hidden submission/readback boundary. A future checked `Boxes` value or explicit
validation operation may enforce stronger value invariants without changing
this operation's asynchronous contract.

Image transformations such as resize and normalization enter `oa::image`, even
though their donor source lived beneath the C++ Vision tree. CPU Annex-B NAL
utilities and future codec sessions enter `oa::video`. This preserves semantic
ownership instead of translating the donor include hierarchy literally.

## Evidence and remaining boundary

The hardware contract covers:

- the donor canonical 2-by-2 values and an independent host IoU oracle;
- an odd 3-by-5 dispatch, negative, zero-area, NaN, and infinity behavior;
- asynchronous readiness and unchanged input storage;
- structural rejection for rank, `[N, 4]`, nonempty, FP32, and same-engine rules;
- semantic/executable provenance with no aliases and an exclusive binding-2
  output write guarded by the shader tail check;
- build-time Slang reflection and `spirv-val` validation.
- donor NMS, confusion, binary-mask, detection AP/mAP, and segmentation vectors;
- deterministic score/index ordering, class-aware and class-agnostic NMS;
- exact ignored-label behavior for classification and segmentation;
- generated `atomic_u32` collision ownership and replay-safe clear stages;
- one-to-three semantic/executable provenance and repeated plan submission.

The current hardware evidence is local Intel/Mesa execution. Clean validation
qualification remains blocked by the unrelated pre-existing Matrix
Philox/Dropout `shaderInt64` feature mismatch documented in the Stage 7
roadmap. This is full coverage of the donor `FnDetection` API, not all future
computer vision: tracking, richer typed detection values, additional
segmentation operations, and cross-device qualification remain Planned.
