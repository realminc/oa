#pragma once

// Image — Core media wrapper composed over oa::Matrix.
//
// Layered on top of oa::Matrix so lifetime, bindless buffers, context recording,
// autograd boundaries, and dtype behavior stay where the rest of OA expects
// them. Image metadata (layout + format) is semantic state on top of the raw
// tensor.
//
// Operators such as a + b stay on oa::Matrix; Image is not subclassed because
// that would either slice metadata or duplicate every operator surface per
// domain wrapper.
//
// Vision/UI/Audio kernels live in their own modules; this header only owns the
// data wrapper + enums and has no runtime or vulkan dependency.

#include <oa/core/matrix.h>

namespace oa {

// ─────────────────────────────────────────────────────────────────────────────
// ImageLayout (tensor shape interpretation)
// ─────────────────────────────────────────────────────────────────────────────

enum class ImageLayout : oa::U8 {
	Nchw,
	Nhwc,
	Chw,
	Hwc,
	Hw,
};

// ─────────────────────────────────────────────────────────────────────────────
// ImageFormat (channel meaning)
// ─────────────────────────────────────────────────────────────────────────────

enum class ImageFormat : oa::U8 {
	Gray,
	GrayAlpha,
	Rgb,
	Rgba,
	Bgr,
	Bgra,
};

[[nodiscard]] oa::I32 imageFormatChannels(ImageFormat inFormat);

// ─────────────────────────────────────────────────────────────────────────────
// Image — semantic wrapper around oa::Matrix
// ─────────────────────────────────────────────────────────────────────────────

class Image {
public:
	Image() = default;

	Image(oa::Matrix inData, ImageLayout inLayout, ImageFormat inFormat);

	[[nodiscard]] const oa::Matrix& asMatrix() const { return data_; }
	[[nodiscard]]       oa::Matrix& asMatrix()       { return data_; }

	[[nodiscard]] oa::I32 width()    const;
	[[nodiscard]] oa::I32 height()   const;
	[[nodiscard]] oa::I32 channels() const;
	[[nodiscard]] oa::I32 batchSize() const;

	[[nodiscard]] ImageLayout layout() const { return layout_; }
	[[nodiscard]] ImageFormat format() const { return format_; }

	[[nodiscard]] oa::ScalarType getDtype() const { return data_.getDtype(); }
	[[nodiscard]] bool         isEmpty()  const { return data_.getShape().rank == 0; }

	[[nodiscard]] bool validate() const;

private:
	oa::Matrix  data_;
	ImageLayout layout_ = ImageLayout::Nchw;
	ImageFormat format_ = ImageFormat::Rgb;
};

// ─────────────────────────────────────────────────────────────────────────────
// ImageBatch — uniform batch of images
// ─────────────────────────────────────────────────────────────────────────────

class ImageBatch {
public:
	ImageBatch() = default;

	ImageBatch(oa::Matrix inData, ImageLayout inLayout, ImageFormat inFormat);

	[[nodiscard]] const oa::Matrix& asMatrix() const { return data_; }
	[[nodiscard]]       oa::Matrix& asMatrix()       { return data_; }

	[[nodiscard]] oa::I32 batchSize() const;
	[[nodiscard]] oa::I32 width()     const;
	[[nodiscard]] oa::I32 height()    const;
	[[nodiscard]] oa::I32 channels()  const;

	[[nodiscard]] ImageLayout layout() const { return layout_; }
	[[nodiscard]] ImageFormat format() const { return format_; }
	[[nodiscard]] oa::ScalarType getDtype() const { return data_.getDtype(); }
	[[nodiscard]] bool         isEmpty()  const { return data_.getShape().rank == 0; }

	[[nodiscard]] bool validate() const;

private:
	oa::Matrix  data_;
	ImageLayout layout_ = ImageLayout::Nchw;
	ImageFormat format_ = ImageFormat::Rgb;
};

} // namespace oa
