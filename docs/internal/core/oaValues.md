# OA Rust values and storage

**Status:** Architecture reference; implementation varies by value

**Updated:** 2026-09-09

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Donor references:** OA C++ `docs/internal/core/oaMatrix.md` and
`docs/internal/core/oaImage.md`

OA values preserve meaning above shared physical storage. Buffer, Matrix,
Image, Audio, VideoFrame, and Texture are not aliases and do not form a public
inheritance hierarchy.

## Layers

| Layer | Contract | Current Rust status |
|---|---|---|
| Physical storage | Byte range, placement, allocation identity, readiness, and retained destruction services | Experimental private `runtime::Storage`; public Buffer is Planned |
| Numerical value | Matrix dtype, shape, element count, and eventually strides/offset | Experimental |
| Still image | Dense extent, dtype, axis layout, and logical channel order | Experimental value slice; codecs, transforms, color, alpha, and multi-plane storage Planned |
| Audio clip | Planar FP32 samples plus sample rate and speaker layout | Experimental typed value, DSP, codec, and device-session slice |
| Video frame | Visible extent, timing, color, backing, and readiness | Experimental packed-Image value; native planes and codec sessions Planned |
| Texture | Sampling, storage/render usage, mip state, and admitted backing | Planned with Render |

The physical layer may use buffers, native images, external memory, or several
planes. Public semantic types expose only checked views appropriate to their
contract.

## Matrix is OA's dense numerical value

OA deliberately uses `Matrix` rather than introducing a parallel `Tensor`
type. Matrix is N-dimensional despite the conventional mathematical name. It
owns or shares typed dense numerical storage plus shape metadata and semantic
identity.

That does not make every OA value a Matrix. Audio and admitted dense Images may
compose a Matrix and expose `as_matrix`. Compressed streams, multi-plane video,
native render images, textures, and topology-bearing geometry require other
storage views and metadata. A conversion is zero-copy only when the target
contract is valid over the exact same byte range, format, layout, readiness,
and lifetime.

## Composition contracts

```text
Audio
  samples: Matrix [channels, samples], f32, planar
  sample_rate
  channel_layout

Image
  admitted Matrix view in NCHW/NHWC/CHW/HWC/HW layout
  width, height, optional batch, and logical channel order
  color space, transfer, range, alpha, and multi-plane storage: Planned

VideoFrame
  one or more image planes
  coded and visible extent
  presentation/decode timestamps
  producer readiness and lifetime

Texture
  image or buffer backing
  sampled/storage/render usage
  sampler, mip, format, layout, and readiness contract
```

`as_matrix`, `as_image`, or `as_texture` never means “forget the source
semantics.” It produces a borrowed or retained checked view while the original
semantic value continues to carry its metadata. Lossy or layout-changing
conversion is an explicit operation.

## Ownership and readiness

- Values retain the one engine's internal services needed to keep storage and
  already-produced work alive.
- Copies of cheap value handles may share storage and semantic identity;
  materialization is explicit.
- Views retain their backing allocation and record alias/provenance metadata.
- Host observation completes the exact producer before exposing bytes;
  non-blocking observation uses an explicitly named `try_*` operation.
- Destruction releases ownership only. It never submits, waits, drains, maps,
  encodes, or finalizes.

## Cross-domain consumers

Image is not owned by Vision, Render, UI, or Plot. Audio is not owned by a
player. VideoFrame is not owned by a decoder after its retained lifetime is
established. Texture is a render resource view, not a replacement for Image or
VideoFrame.

This permits the following without semantic collapse:

```text
Image ───────────> vision::detect
Image ───────────> render Texture view ──> Presenter/UI/Plot
VideoFrame ──────> vision operation
VideoFrame ──────> render Texture view
Audio + VideoFrame ──> media clock and track coordination
```

## Acceptance gate for a new value

A public value requires:

- explicit valid metadata combinations and checked-size arithmetic;
- one storage and readiness owner with safe clone/view/drop behavior;
- alias and zero-copy conversion rules;
- no raw Vulkan, allocator, codec, or third-party handle leakage;
- negative tests for rank, extent, dtype, format, layout, and ownership;
- reuse/poison and producer-lifetime tests where device storage is involved;
- an independent oracle or conformance fixture for every converting operation.
