# OA Rust Quantization and Numeric Representations

**Status:** Planned beyond current dense FP32/I32/U32 checkpoints

**Updated:** 2026-09-09

OARS currently has no general FP16/BF16/quantized execution claim. This
document ports the donor design boundary without copying its Shipped status.

## One semantic model

`DType` describes dense scalar storage. Quantization is an encoded semantic
value because the logical values require a packed payload plus scale/zero-point
planes, block geometry, encoding version, logical shape and numeric policy.

A future Rust shape is conceptually:

```rust,ignore
let weight = matrix::quantize(&dense, Quantization::Q4)?;
let output = matrix::mat_mul_nt_quantized(&input, &weight)?;
let debug = matrix::dequantize(&weight)?;
```

The actual public API is added only with implementation and external-style
tests. The matmul route consumes the encoded weight directly; silently
materializing an FP32 copy would invalidate memory and performance claims.

## Representation boundary

| Representation | Semantic status |
|---|---|
| dense FP32 | Experimental baseline |
| dense I32 | Experimental Matrix-add proof |
| dense U32 | Experimental index/RNG/training-state uses |
| dense FP16/BF16 | Planned and capability-gated |
| dense FP64 | Planned explicit scientific request, fail closed when unavailable |
| OA-native Q4/Q8 | Planned encoded weight values |
| imported GGUF/vendor formats | distinct importer encodings; never reinterpreted as OA-native |

Input storage, scale storage, compute precision, accumulation and output dtype
remain separate facts in OaBlasLt and kernel metadata.

## First native formats

The donor's symmetric group-of-32 Q4/Q8 design is the first candidate to port:

| Policy | Logical range | Payload/32 values | Scale | Total/block |
|---|---:|---:|---:|---:|
| Q4 | `[-7, 7]` | 16 bytes | one FP32 | 20 bytes |
| Q8 | `[-127, 127]` | 32 bytes | one FP32 | 36 bytes |

Porting requires confirming, not assuming, the donor rules: scale from maximum
finite absolute value, zero/non-finite block behavior, round-to-nearest-even,
clamping and zero-padded partial blocks. The Rust value validates exact payload
and scale lengths, block size, logical shape, offsets and encoding version.

These formats are not byte-compatible with GGML/GGUF K-quant families or
vendor FP4/FP6/FP8 schemes. Import is an explicit conversion with provenance.

## Model and mutation contract

Encoded parameters are inference values until a training/update contract is
implemented. A model loader validates checksums, non-overlap, exact ranges,
dtype/encoding combinations and payload sizes before device upload. Mutation of
a dense source invalidates derived packed/prepacked caches by stable identity
and version.

Serialization versioning belongs to the model-format owner, not this compute
document. No OAM v3 donor claim is attributed to OARS until the Rust model
loader and tests exist.

## Device routes

Every encoding starts with a portable unpack/dequantize oracle and a fused
matmul or consuming operation. Faster candidates may use integer dot products,
cooperative matrices or vendor/device packs only under exact queried features.
Alignment, odd K, partial blocks and scale access are part of candidate
legality.

The selector must account for bandwidth saved, extra unpack arithmetic,
workspace, prepack cost and reuse count. A smaller model file is not proof of a
faster kernel or acceptable model quality.

## Admission gate

- exact independent packing/unpacking oracle;
- zeros, non-finite input, invalid scale, ties, clamp boundaries and partial
  blocks;
- deterministic repeated encoding;
- bounds/poison and packed-access concurrency tests;
- fused consuming route with no hidden dense materialization;
- model serialization/import validation;
- fixed-corpus domain-quality evidence;
- named-device Vulkan validation and fresh-process performance evidence.

No additional quantization level enters the public vocabulary until this full
vertical slice exists.
