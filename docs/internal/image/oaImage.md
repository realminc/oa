# OA Rust Image

**Status:** Experimental semantic value, complete 50-op donor tensor surface,
composed helpers, and still-image codecs

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Value model:** [Values and Storage](../core/oaValues.md)

**Donor references:** OA C++ `include/oa/core/image.h`,
`lib/oa/core/image.cpp`, and `test/cpp/core/testImage.cpp`

The current Rust slice admits a dense still image as a checked semantic view
over one `Matrix`. It ports the donor's useful rank, axis, and channel
invariants without copying its inheritance or mutable metadata surface.

## Public contract

`oa::Image` and `oa::core::Image` are identity aliases for the same type.
`Image::new` consumes a Matrix and validates its metadata before construction.
`as_matrix` borrows that backing value without a copy; `into_matrix` consumes
the Image and removes only its image semantics.

| `ImageLayout` | Required Matrix shape |
|---|---|
| `Nchw` | `[batch, channels, height, width]` |
| `Nhwc` | `[batch, height, width, channels]` |
| `Chw` | `[channels, height, width]` |
| `Hwc` | `[height, width, channels]` |
| `Hw` | `[height, width]`, implicitly one Gray channel |

`ImageFormat` expresses logical channel order rather than scalar width:
`Gray`, `GrayAlpha`, `Rgb`, `Rgba`, `Bgr`, and `Bgra` require one through four
channels as their names imply. The backing Matrix continues to own `DType`.
Zero extents remain legal when the Matrix can represent them.

The backing Matrix is exposed immutably. OA therefore cannot retain an Image
whose shape has been mutated out of agreement with its layout and format.
Clone preserves Matrix storage and Image semantic identity under the existing
value lifetime contract. Image identity is distinct from the backing Matrix
identity so captured operations retain `OpValueKind::Image` rather than erasing
the public value into a generic numeric buffer.

## Module boundary

The value implementation lives in `core/image.rs`; still-image codecs and
stateless transformations belong to `oa::image`. Vision consumes Images for
interpretation and does not own resize, normalization, codecs, or storage.
Render may later create a checked Texture view without redefining Image.

The complete donor geometric family is admitted under `oa::image`: `resize`,
`crop`, `flip`, `rotate`, `pad`, `center_crop`, `remap`, `warp_affine`, and
`warp_perspective`. The operations accept FP32 NCHW or CHW Images and preserve
layout and format. `resize` accepts non-zero `u32` target width and height and an explicit
`InterpolationMode::{Nearest, Bilinear}`. It preserves layout and format while
producing a newly allocated Image. Nearest uses the donor floor-coordinate
rule. Bilinear uses half-pixel centers with `align_corners = false` and clamped
edge sampling. Other layouts, dtypes, empty source extents, and zero target
extents fail with a checked error; there is no silent identity fallback.

Crop clips only its right and bottom extent, while requiring an in-bounds
origin. Rotation accepts clockwise multiples of 90 degrees. Padding and warps
share `BorderMode::{Constant, Replicate, Reflect, Reflect101, Wrap}`. Remap
consumes an absolute-coordinate Matrix shaped
`[1|batch, 2, output_height, output_width]`; affine and perspective warps
consume inverse `[2,3]` and `[3,3]` transform matrices. Those auxiliary values
remain `OpValueKind::Matrix` in the semantic graph while pixels and results
remain `OpValueKind::Image`.

The operation contracts and 16×16 FP32 kernel variants are owned by
`tools/gen/fn/schema/image.json`. Each variant declares one exclusive,
bounds-checked output-element write per invocation and no workspace. The Rust
lowering lives in `image/geometric.rs`; the public facade remains
`oa::image`, not a C++-style `FnImage` class or `vision` alias.

The complete 20-operation donor pixel family is also admitted: five threshold
forms, range, clamp, invert, brightness/contrast, gamma, solarize, posterize,
Rec.709 grayscale, channel reorder, alpha blend, masked composite, rectangle
erase, 3×4 RGB color twist, Gaussian noise, and salt-and-pepper noise. Rust
channel reorder requires an explicit output `ImageFormat`; this is an
intentional semantic adaptation because preserving the input format after
reordering channels would make the Image metadata false. Seeded noise uses
counter-based Philox directly in the Image kernel and records the seed as an
operation attribute. Vectorized RNG candidates declare bounded two- or
four-element physical ownership with bounds-checked tails.

The complete 20-operation filter family and per-channel normalization complete
the donor's 50 tensor-native `FnImage` operations. Filters cover arbitrary and
separable convolution, fixed derivative/blur/sharpen operations, rectangular
morphology and its composed forms, Gaussian/unsharp, median/bilateral, and
adaptive thresholds. Each public operation retains its own schema identity
while shared parameterized physical kernels remain an internal lowering
detail. Current composite morphology is a direct single-dispatch reference
lowering; it is correctness-oriented and carries no performance claim.

Three semantic compositions preserve typed Image metadata rather than erasing
the result into Matrix: `convert_color`, `resize_normalize`, and
`segmentation_overlay`. The latter accepts an Int32 label Matrix and FP32
palette Matrix while retaining Image kinds for its source and result in the
semantic graph. `resize_normalize` is one physical dispatch rather than an API
forwarding chain.

## Codec boundary

`oa::image` owns the donor still-image codec surface as one-shot synchronous
host operations:

- `decode_file` and `decode_memory` decode JPEG, PNG, WebP, BMP, or TGA and
  upload normalized FP32 NCHW pixels in explicitly requested Gray, Rgb, or
  Rgba format;
- `encode` and `save_file` explicitly read back one FP32 NCHW/CHW Image and
  encode JPEG, PNG, lossy WebP, BMP, or TGA;
- `can_decode` and `can_encode` report the formats compiled into this build;
- `save_rgba_file` is the packed RGBA8 host sink used after a render session
  has performed its own explicit completion and readback.

The implementation uses `image` 0.25.10 with only the five named format
features and `webp` 0.3.1 for quality-controlled libwebp encoding. These
dependencies remain private. JPEG and WebP quality is checked in `[1,100]`;
other formats ignore quality. Codec calls do not become semantic GPU
operations and do not introduce a hidden CPU execution fallback.
`saveTextureFile` is not published before OARS has a Texture value and a
render-owned completion/readback contract.

## Verified evidence

Automated tests cover:

- exact public root/core type identity;
- rank and channel counts for every layout and format;
- all five layouts and a zero-height image over real Vulkan-backed FP32 Matrix
  storage;
- width, height, batch, channels, dtype, format, and layout observation;
- clone and consuming Matrix recovery with exact readback;
- rejection of rank mismatch, channel mismatch, and non-Gray `Hw` metadata.
- odd 3×4 to 5×7 resize over two batches and three channels against independent
  nearest and bilinear host oracles;
- CHW metadata preservation, asynchronous captured replay, and exact Image-kind
  semantic graph identity with all three schema attributes;
- rejection of unsupported dtype/layout, empty source extent, and zero target
  extent;
- exact crop clipping, horizontal/vertical flip, clockwise rotation, and
  centered-crop oracles;
- asymmetric odd padding against independent host oracles for all five border
  modes;
- remap, affine, and perspective identity/coordinate oracles, including mixed
  Image/Matrix semantic graph kinds;
- checked rejection of invalid regions, rotations, non-finite border values,
  coordinate-map shapes, and transform shapes;
- all 12 pointwise pixel formulas against independent scalar oracles;
- exact grayscale, blend, masked composite, rectangle erase, channel reorder,
  and color-matrix oracles;
- seeded-noise identity/offset properties and repeatability for an identical
  explicit seed;
- all 20 filter entry points through identity, zero-response, fixed Sobel,
  Scharr, and Laplacian oracles plus invalid-kernel/parameter rejection;
- exact per-channel normalization with preserved Image metadata;
- exact RGB/BGR color conversion, fused resize-normalize, and segmentation
  overlay results with typed semantic graph inputs;
- all five codec memory round trips through device upload/readback, exact
  lossless-format samples, lossy range/shape contracts, path inference, RGBA8
  host saving, and invalid extension/size rejection;
- deterministic generator output, schema ownership, reflected push ABI, and
  schema-to-kernel physical-write metadata.

The value, full 50-operation tensor surface, composed helpers, and all five
codec paths pass their hardware tests on Intel Iris Xe through Mesa 26.2.2 and
Vulkan 1.4.354. This proves the tested storage and execution path, not general
driver or device qualification. Both resize SPIR-V artifacts declare only
`Shader` and `RuntimeDescriptorArray` capabilities.

Core and synchronization-validation runs reach and pass the resize tests, but
the repository-wide clean-validation gate is blocked before dispatch by eight
pre-existing Matrix Philox/Dropout artifacts that declare `Int64` while the
engine does not enable `shaderInt64`. The exact reported rule is
`VUID-VkShaderModuleCreateInfo-pCode-08740`. No clean-validation or GPU-assisted
claim is made for this checkpoint until that engine-wide mismatch is fixed.

## Remaining Stage 7 work

Clean core, synchronization, and applicable GPU-assisted validation remain a
qualification dependency. Geometric autograd is not part of the current
contract. Transfer functions, component ranges, alpha interpretation, row
strides, multi-plane storage, native Vulkan images, Texture saving, and
geometric autograd remain Planned until their contracts and oracles land.
