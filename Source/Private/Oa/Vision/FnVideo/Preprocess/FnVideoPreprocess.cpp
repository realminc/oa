// OaFnVideo — VideoPreprocess category.
// Fused decode → resize → normalize for object-detection feeds.
//
// Today this lands as decode-to-BF16 + OaFnImage::Resize + OaFnImage::Normalize
// rather than the single Nv12ResizeNormalizeToBf16 dispatch named in the
// schema. The fused kernel exists (Source/Private/Oa/Vision/Shader/Compute/Conversion/
// Nv12ResizeNormalizeToBf16.slang) but requires per-plane VkImage→buffer
// staging that OaVideoDecoder doesn't expose yet. The composed version is
// correct; the fused version is a future perf win.

#include <Oa/Vision/FnVideo.h>
#include <Oa/Vision/FnImage.h>
#include "Oa/Vision/Video/Decoder/VideoDecoderInternal.h"
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>

namespace OaFnVideo
{

OaResult<OaMatrix> DecodeResizeNormalize(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	OaU32 InWidth,
	OaU32 InHeight,
	const OaNormalizationParams& InNorm)
{
	OaEngine* rt = InSession.GetEngine();
	if (rt == nullptr) {
		return OaStatus::Error("OaFnVideo::DecodeResizeNormalize: decoder session has no engine");
	}

	// Decode at native resolution into [1,3,H,W] BF16. Skip ImageNet
	// normalization here so we can apply the caller-supplied mean/std below.
	auto matResult = OaVideoDecoderInternal::DecodeFrameToBf16(
		InSession,
		InAccessUnit,
		/*InNormalizeImageNet=*/false);
	if (matResult.IsError()) {
		return matResult;
	}
	OaMatrix mat = matResult.GetValue();

	const auto shape = mat.GetShape();
	const bool nativeMatch =
		shape.Rank == 4 &&
		shape[2] == static_cast<OaI64>(InHeight) &&
		shape[3] == static_cast<OaI64>(InWidth);

	if (not nativeMatch) {
		mat = OaFnImage::Resize(*rt, mat, InWidth, InHeight, OaInterpolationMode::Bilinear);
	}

	mat = OaFnImage::Normalize(*rt, mat, InNorm);

	return OaResult<OaMatrix>(mat);
}

} // namespace OaFnVideo
