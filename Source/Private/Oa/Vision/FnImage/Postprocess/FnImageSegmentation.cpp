#include <Oa/Vision/FnImage.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <limits>

OaMatrix OaFnImage::SegmentationOverlay(
	OaEngine& InRt,
	const OaMatrix& InImage,
	const OaMatrix& InMask,
	const OaMatrix& InPalette,
	OaF32 InAlpha) {
	(void)InRt;
	const bool imageValid = InImage.Rank() == 4
		&& (InImage.Size(1) == 3 || InImage.Size(1) == 4)
		&& InImage.GetDtype() == OaScalarType::Float32;
	const bool maskValid = InMask.GetDtype() == OaScalarType::Int32
		&& ((InMask.Rank() == 3
			&& InMask.Size(0) == InImage.Size(0)
			&& InMask.Size(1) == InImage.Size(2)
			&& InMask.Size(2) == InImage.Size(3))
		|| (InMask.Rank() == 4 && InMask.Size(1) == 1
			&& InMask.Size(0) == InImage.Size(0)
			&& InMask.Size(2) == InImage.Size(2)
			&& InMask.Size(3) == InImage.Size(3)));
	const bool paletteValid = InPalette.Rank() == 2
		&& InPalette.Size(0) > 0 && InPalette.Size(1) == 3
		&& InPalette.GetDtype() == OaScalarType::Float32;
	const OaU64 elements = static_cast<OaU64>(InImage.NumElements());
	if (!imageValid || !maskValid || !paletteValid
		|| !std::isfinite(InAlpha) || InAlpha < 0.0F || InAlpha > 1.0F
		|| elements > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnImage::SegmentationOverlay expects FP32 NCHW RGB/RGBA, matching Int32 labels, FP32 palette [K,3], and alpha in [0,1]");
		return {};
	}
	OaMatrix out = OaFnMatrix::Empty(InImage.GetShape(), OaScalarType::Float32);
	struct Push {
		OaU32 Batch, Channels, Height, Width, Classes;
		OaF32 Alpha;
	} push{
		static_cast<OaU32>(InImage.Size(0)),
		static_cast<OaU32>(InImage.Size(1)),
		static_cast<OaU32>(InImage.Size(2)),
		static_cast<OaU32>(InImage.Size(3)),
		static_cast<OaU32>(InPalette.Size(0)), InAlpha};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageSegmentationOverlay",
		{&InImage, &InMask, &InPalette, &out}, access,
		&push, sizeof(push), (static_cast<OaU32>(elements) + 255U) / 256U);
	return out;
}

OaMatrix OaFnImage::SegmentationOverlay(
	const OaMatrix& InImage,
	const OaMatrix& InMask,
	const OaMatrix& InPalette,
	OaF32 InAlpha) {
	return SegmentationOverlay(
		*OaComputeEngine::GetGlobal(), InImage, InMask, InPalette, InAlpha);
}
