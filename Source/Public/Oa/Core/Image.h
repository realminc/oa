#pragma once

// OaImage — Core media wrapper composed over OaMatrix.
//
// Layered on top of OaMatrix so lifetime, bindless buffers, context recording,
// autograd boundaries, and dtype behavior stay where the rest of OA expects
// them. Image metadata (layout + format) is semantic state on top of the raw
// tensor.
//
// Operators such as a + b stay on OaMatrix; OaImage is not subclassed because
// that would either slice metadata or duplicate every operator surface per
// domain wrapper.
//
// Vision/UI/Audio kernels live in their own modules; this header only owns the
// data wrapper + enums (no runtime / Vulkan dependencies). See
// Docs/Rewrite/ThisIsTheKey/OaImage.md for the full design.

#include <Oa/Core/Matrix.h>

// ─────────────────────────────────────────────────────────────────────────────
// Layout (tensor shape interpretation)
// ─────────────────────────────────────────────────────────────────────────────
//
// Layout maps OaMatrix dimensions to image axes:
//   Nchw — 4D [N, C, H, W]
//   Nhwc — 4D [N, H, W, C]
//   Chw  — 3D [C, H, W]
//   Hwc  — 3D [H, W, C]
//   Hw   — 2D [H, W]            (grayscale, channel axis absent)
//
// A layout conversion is metadata-only when the shape already matches and a
// transpose otherwise. Format is independent of layout.

enum class OaImageLayout : OaU8 {
	Nchw,
	Nhwc,
	Chw,
	Hwc,
	Hw,
};

// ─────────────────────────────────────────────────────────────────────────────
// Format (channel meaning)
// ─────────────────────────────────────────────────────────────────────────────
//
// Format describes what each channel *means*, not where it sits in memory.
// Converting between formats (e.g. Rgb → Bgr, Rgb → Gray) is always a real
// kernel.

enum class OaImageFormat : OaU8 {
	Gray,
	GrayAlpha,
	Rgb,
	Rgba,
	Bgr,
	Bgra,
};

// Number of channels implied by a format. Used by OaImage validation to check
// the layout's channel axis (when present) matches the format.
[[nodiscard]] OaI32 OaImageFormatChannels(OaImageFormat InFormat);

// ─────────────────────────────────────────────────────────────────────────────
// OaImage — semantic wrapper around OaMatrix
// ─────────────────────────────────────────────────────────────────────────────

class OaImage {
public:
	OaImage() = default;

	// Build from an existing OaMatrix. Shape rank must match layout:
	//   Nchw / Nhwc  → rank 4
	//   Chw  / Hwc   → rank 3
	//   Hw           → rank 2
	// Channel axis (when the layout has one) must equal OaImageFormatChannels
	// for the given format. Validation is enforced by Validate() and asserted
	// in debug builds; release builds keep the supplied metadata and let the
	// caller catch downstream mismatches.
	OaImage(OaMatrix InData, OaImageLayout InLayout, OaImageFormat InFormat);

	// Round-trip access to the underlying tensor. Same backing buffer, no copy.
	[[nodiscard]] const OaMatrix& AsMatrix() const { return Data_; }
	[[nodiscard]]       OaMatrix& AsMatrix()       { return Data_; }

	// Semantic accessors — read the correct shape axis for the layout.
	// Width / Height return 0 for an empty image.
	// Channels returns OaImageFormatChannels(Format()) for layouts that omit
	// the channel axis (Hw) so callers don't need a layout-case fork.
	[[nodiscard]] OaI32 Width()    const;
	[[nodiscard]] OaI32 Height()   const;
	[[nodiscard]] OaI32 Channels() const;
	[[nodiscard]] OaI32 BatchSize() const;   // 1 for non-batched layouts

	[[nodiscard]] OaImageLayout Layout() const { return Layout_; }
	[[nodiscard]] OaImageFormat Format() const { return Format_; }

	[[nodiscard]] OaScalarType  GetDtype() const { return Data_.GetDtype(); }
	[[nodiscard]] bool          IsEmpty()  const { return Data_.GetShape().Rank == 0; }

	// Returns true when (layout, format, dtype, shape) are internally
	// consistent. Used by the constructor and exposed for callers that want
	// to validate after manual mutations through AsMatrix().
	[[nodiscard]] bool Validate() const;

private:
	OaMatrix      Data_;
	OaImageLayout Layout_ = OaImageLayout::Nchw;
	OaImageFormat Format_ = OaImageFormat::Rgb;
};

// ─────────────────────────────────────────────────────────────────────────────
// OaImageBatch — uniform batch of images
// ─────────────────────────────────────────────────────────────────────────────
//
// Phase 5: Uniform batch support (var-shape batch deferred to later phase).
// Maps to [N, C, H, W] or [N, H, W, C] tensor with same layout, format, and
// dimensions for all images in the batch.

class OaImageBatch {
public:
	OaImageBatch() = default;

	// Build from an existing OaMatrix. Shape must be rank 4:
	//   Nchw → [N, C, H, W]
	//   Nhwc → [N, H, W, C]
	// Channel axis must equal OaImageFormatChannels for the given format.
	OaImageBatch(OaMatrix InData, OaImageLayout InLayout, OaImageFormat InFormat);

	// Access underlying batch tensor
	[[nodiscard]] const OaMatrix& AsMatrix() const { return Data_; }
	[[nodiscard]]       OaMatrix& AsMatrix()       { return Data_; }

	// Batch dimensions
	[[nodiscard]] OaI32 BatchSize() const;
	[[nodiscard]] OaI32 Width()    const;
	[[nodiscard]] OaI32 Height()   const;
	[[nodiscard]] OaI32 Channels() const;

	// Semantic metadata (same for all images in batch)
	[[nodiscard]] OaImageLayout Layout() const { return Layout_; }
	[[nodiscard]] OaImageFormat Format() const { return Format_; }
	[[nodiscard]] OaScalarType  GetDtype() const { return Data_.GetDtype(); }
	[[nodiscard]] bool          IsEmpty()  const { return Data_.GetShape().Rank == 0; }

	// Validation
	[[nodiscard]] bool Validate() const;

private:
	OaMatrix      Data_;
	OaImageLayout Layout_ = OaImageLayout::Nchw;
	OaImageFormat Format_ = OaImageFormat::Rgb;
};
