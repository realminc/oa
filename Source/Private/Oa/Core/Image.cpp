// OaImage implementation — accessors + validation for the Core media wrapper.

#include <Oa/Core/Image.h>

#include <cassert>

OaI32 OaImageFormatChannels(OaImageFormat InFormat) {
	switch (InFormat) {
		case OaImageFormat::Gray:       return 1;
		case OaImageFormat::GrayAlpha:  return 2;
		case OaImageFormat::Rgb:        return 3;
		case OaImageFormat::Bgr:        return 3;
		case OaImageFormat::Rgba:       return 4;
		case OaImageFormat::Bgra:       return 4;
	}
	return 0;
}

namespace {

// Layout → expected shape rank.
OaI32 RankFor(OaImageLayout InLayout) {
	switch (InLayout) {
		case OaImageLayout::Nchw: return 4;
		case OaImageLayout::Nhwc: return 4;
		case OaImageLayout::Chw:  return 3;
		case OaImageLayout::Hwc:  return 3;
		case OaImageLayout::Hw:   return 2;
	}
	return 0;
}

// Layout → 0-based index of (N, C, H, W) within the shape, or -1 if absent.
struct LayoutAxes {
	OaI32 N = -1;
	OaI32 C = -1;
	OaI32 H = -1;
	OaI32 W = -1;
};

LayoutAxes AxesFor(OaImageLayout InLayout) {
	switch (InLayout) {
		case OaImageLayout::Nchw: return {.N = 0, .C = 1, .H = 2, .W = 3};
		case OaImageLayout::Nhwc: return {.N = 0, .C = 3, .H = 1, .W = 2};
		case OaImageLayout::Chw:  return {.N = -1, .C = 0, .H = 1, .W = 2};
		case OaImageLayout::Hwc:  return {.N = -1, .C = 2, .H = 0, .W = 1};
		case OaImageLayout::Hw:   return {.N = -1, .C = -1, .H = 0, .W = 1};
	}
	return {};
}

OaI32 DimAt(const OaMatrix& InData, OaI32 InAxis) {
	if (InAxis < 0) return 0;
	const OaMatrixShape shape = InData.GetShape();
	if (InAxis >= shape.Rank) return 0;
	return static_cast<OaI32>(shape[InAxis]);
}

} // namespace

OaImage::OaImage(OaMatrix InData, OaImageLayout InLayout, OaImageFormat InFormat)
	: Data_(std::move(InData))
	, Layout_(InLayout)
	, Format_(InFormat)
{
	assert(Validate() and "OaImage: shape/layout/format mismatch");
}

bool OaImage::Validate() const {
	const OaMatrixShape shape = Data_.GetShape();
	if (shape.Rank == 0) return true;  // empty image is trivially valid

	const OaI32 expectedRank = RankFor(Layout_);
	if (shape.Rank != expectedRank) return false;

	const LayoutAxes axes = AxesFor(Layout_);
	if (axes.C >= 0) {
		const OaI32 channels = static_cast<OaI32>(shape[axes.C]);
		if (channels != OaImageFormatChannels(Format_)) return false;
	}
	return true;
}

OaI32 OaImage::Width() const {
	return DimAt(Data_, AxesFor(Layout_).W);
}

OaI32 OaImage::Height() const {
	return DimAt(Data_, AxesFor(Layout_).H);
}

OaI32 OaImage::Channels() const {
	const LayoutAxes axes = AxesFor(Layout_);
	if (axes.C < 0) {
		// Layout omits the channel axis (e.g. Hw); fall back to the format.
		return OaImageFormatChannels(Format_);
	}
	return DimAt(Data_, axes.C);
}

OaI32 OaImage::BatchSize() const {
	const LayoutAxes axes = AxesFor(Layout_);
	if (axes.N < 0) return 1;
	return DimAt(Data_, axes.N);
}

// ─── OaImageBatch implementation ─────────────────────────────────────────────

OaImageBatch::OaImageBatch(OaMatrix InData, OaImageLayout InLayout, OaImageFormat InFormat)
	: Data_(std::move(InData))
	, Layout_(InLayout)
	, Format_(InFormat)
{
	assert(Validate() and "OaImageBatch: shape/layout/format mismatch");
}

bool OaImageBatch::Validate() const {
	const OaMatrixShape shape = Data_.GetShape();
	if (shape.Rank == 0) return true;  // empty batch is trivially valid

	// OaImageBatch only supports rank 4 (Nchw or Nhwc)
	if (shape.Rank != 4) return false;

	// Must be a batched layout (Nchw or Nhwc)
	if (Layout_ != OaImageLayout::Nchw && Layout_ != OaImageLayout::Nhwc) return false;

	const LayoutAxes axes = AxesFor(Layout_);
	if (axes.C >= 0) {
		const OaI32 channels = static_cast<OaI32>(shape[axes.C]);
		if (channels != OaImageFormatChannels(Format_)) return false;
	}
	return true;
}

OaI32 OaImageBatch::BatchSize() const {
	const LayoutAxes axes = AxesFor(Layout_);
	return DimAt(Data_, axes.N);
}

OaI32 OaImageBatch::Width() const {
	return DimAt(Data_, AxesFor(Layout_).W);
}

OaI32 OaImageBatch::Height() const {
	return DimAt(Data_, AxesFor(Layout_).H);
}

OaI32 OaImageBatch::Channels() const {
	const LayoutAxes axes = AxesFor(Layout_);
	return DimAt(Data_, axes.C);
}
