// oa::Image implementation — accessors + validation for the Core media wrapper.

#include <oa/core/image.h>

#include <oa/core/std/assert.h>

oa::I32 oa::imageFormatChannels(oa::ImageFormat inFormat) {
	switch (inFormat) {
		case oa::ImageFormat::Gray:       return 1;
		case oa::ImageFormat::GrayAlpha:  return 2;
		case oa::ImageFormat::Rgb:        return 3;
		case oa::ImageFormat::Bgr:        return 3;
		case oa::ImageFormat::Rgba:       return 4;
		case oa::ImageFormat::Bgra:       return 4;
	}
	return 0;
}

namespace {

// layout → expected shape rank.
oa::I32 rankFor(oa::ImageLayout inLayout) {
	switch (inLayout) {
		case oa::ImageLayout::Nchw: return 4;
		case oa::ImageLayout::Nhwc: return 4;
		case oa::ImageLayout::Chw:  return 3;
		case oa::ImageLayout::Hwc:  return 3;
		case oa::ImageLayout::Hw:   return 2;
	}
	return 0;
}

// layout → 0-based index of (N, C, H, W) within the shape, or -1 if absent.
struct LayoutAxes {
	oa::I32 n = -1;
	oa::I32 c = -1;
	oa::I32 h = -1;
	oa::I32 w = -1;
};

LayoutAxes axesFor(oa::ImageLayout inLayout) {
	switch (inLayout) {
		case oa::ImageLayout::Nchw: return {.n = 0, .c = 1, .h = 2, .w = 3};
		case oa::ImageLayout::Nhwc: return {.n = 0, .c = 3, .h = 1, .w = 2};
		case oa::ImageLayout::Chw:  return {.n = -1, .c = 0, .h = 1, .w = 2};
		case oa::ImageLayout::Hwc:  return {.n = -1, .c = 2, .h = 0, .w = 1};
		case oa::ImageLayout::Hw:   return {.n = -1, .c = -1, .h = 0, .w = 1};
	}
	return {};
}

oa::I32 dimAt(const oa::Matrix& inData, oa::I32 inAxis) {
	if (inAxis < 0) return 0;
	const oa::MatrixShape shape = inData.getShape();
	if (inAxis >= shape.rank) return 0;
	return static_cast<oa::I32>(shape[inAxis]);
}

} // namespace

oa::Image::Image(oa::Matrix inData, oa::ImageLayout inLayout, oa::ImageFormat inFormat)
	: data_(oa::move(inData))
	, layout_(inLayout)
	, format_(inFormat)
{
	OA_REQUIRE_MSG(validate(), "oa::Image: shape/layout/format mismatch");
}

bool oa::Image::validate() const {
	const oa::MatrixShape shape = data_.getShape();
	if (shape.rank == 0) return true;  // empty image is trivially valid

	const oa::I32 expectedRank = rankFor(layout_);
	if (shape.rank != expectedRank) return false;

	const LayoutAxes axes = axesFor(layout_);
	// A channel-less HW shape represents exactly one gray channel. Allowing a
	// multi-channel format here would make downstream consumers index channels
	// that do not exist in storage.
	if (axes.c < 0 and format_ != oa::ImageFormat::Gray) return false;
	if (axes.c >= 0) {
		const oa::I32 channels = static_cast<oa::I32>(shape[axes.c]);
		if (channels != oa::imageFormatChannels(format_)) return false;
	}
	return true;
}

oa::I32 oa::Image::width() const {
	return dimAt(data_, axesFor(layout_).w);
}

oa::I32 oa::Image::height() const {
	return dimAt(data_, axesFor(layout_).h);
}

oa::I32 oa::Image::channels() const {
	const LayoutAxes axes = axesFor(layout_);
	if (axes.c < 0) {
		// layout omits the channel axis (e.g. Hw); fall back to the format.
		return oa::imageFormatChannels(format_);
	}
	return dimAt(data_, axes.c);
}

oa::I32 oa::Image::batchSize() const {
	const LayoutAxes axes = axesFor(layout_);
	if (axes.n < 0) return 1;
	return dimAt(data_, axes.n);
}

// ─── oa::ImageBatch implementation ─────────────────────────────────────────────

oa::ImageBatch::ImageBatch(oa::Matrix inData, oa::ImageLayout inLayout, oa::ImageFormat inFormat)
	: data_(oa::move(inData))
	, layout_(inLayout)
	, format_(inFormat)
{
	OA_REQUIRE_MSG(validate(), "oa::ImageBatch: shape/layout/format mismatch");
}

bool oa::ImageBatch::validate() const {
	const oa::MatrixShape shape = data_.getShape();
	if (shape.rank == 0) return true;  // empty batch is trivially valid

	// oa::ImageBatch only supports rank 4 (Nchw or Nhwc)
	if (shape.rank != 4) return false;

	// Must be a batched layout (Nchw or Nhwc)
	if (layout_ != oa::ImageLayout::Nchw && layout_ != oa::ImageLayout::Nhwc) return false;

	const LayoutAxes axes = axesFor(layout_);
	if (axes.c >= 0) {
		const oa::I32 channels = static_cast<oa::I32>(shape[axes.c]);
		if (channels != oa::imageFormatChannels(format_)) return false;
	}
	return true;
}

oa::I32 oa::ImageBatch::batchSize() const {
	const LayoutAxes axes = axesFor(layout_);
	return dimAt(data_, axes.n);
}

oa::I32 oa::ImageBatch::width() const {
	return dimAt(data_, axesFor(layout_).w);
}

oa::I32 oa::ImageBatch::height() const {
	return dimAt(data_, axesFor(layout_).h);
}

oa::I32 oa::ImageBatch::channels() const {
	const LayoutAxes axes = axesFor(layout_);
	return dimAt(data_, axes.c);
}
