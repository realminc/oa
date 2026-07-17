// OaFnImage — stateless GPU image transforms and composed image pipelines.
//
// The schema-generated header owns the public tensor-op declarations. Complex
// validation, graph recording, kernels, and multi-stage pipelines remain
// handwritten in Source/Private/Oa/Vision.

#pragma once

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Image.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Vision/Type.h>

// SaveFile only needs the renderer texture by reference. Avoid pulling the UI
// image surface into every Vision transform consumer.
struct OaTexture;

struct OaNormalizationParams {
	OaF32 Mean[3];
	OaF32 Std[3];
};

namespace OaFnImage {
	// Exact declarations for the 50 tensor-native image operations. Every op
	// has an explicit-engine overload and a global-engine convenience overload.
	// Regenerate with: python3 Tools/FnAutogen/oafnautogen.py --live
	#include "../../../Private/Oa/Vision/FnImage/FnImage.gen.h"

	// Handwritten compositions which intentionally are not primitive image ops.
	[[nodiscard]] OaVec<OaMatrix> ResizeBatch(
		OaEngine& InRt,
		const OaVec<OaMatrix>& InImages,
		OaU32 InTargetWidth,
		OaU32 InTargetHeight,
		OaInterpolationMode InMode = OaInterpolationMode::Bilinear
	);

	[[nodiscard]] OaMatrix DecodeAndPreprocess(
		OaEngine& InRt,
		const OaSpan<const OaU8>& InCompressedFrame,
		OaU32 InTargetWidth,
		OaU32 InTargetHeight,
		bool InNormalizeImageNet = true
	);

	// Synchronous file sink for a device-local packed RGBA8 texture. The output
	// codec is inferred from .png, .jpg/.jpeg, .bmp, or .tga.
	[[nodiscard]] OaStatus SaveFile(
		OaComputeEngine& InEngine,
		const OaTexture& InTexture,
		OaStringView InPath
	);

	// Semantic image overloads preserve or update layout/format metadata while
	// reusing the tensor-native GPU kernels above.
	[[nodiscard]] OaImage Resize(
		const OaImage& InImage,
		OaU32 InWidth,
		OaU32 InHeight
	);
	[[nodiscard]] OaImage Normalize(
		const OaImage& InImage,
		const OaNormalizationParams& InParams
	);
	[[nodiscard]] OaImage ConvertColor(
		const OaImage& InImage,
		OaImageFormat InDstFormat
	);
	[[nodiscard]] OaImage ResizeNormalize(
		const OaImage& InImage,
		OaU32 InWidth,
		OaU32 InHeight,
		const OaNormalizationParams& InParams
	);

	// Blend an Int32 semantic label map over an FP32 NCHW RGB/RGBA image.
	// Mask is [N,H,W] or [N,1,H,W], palette is FP32 [K,3], and invalid labels
	// leave the source pixel unchanged.
	[[nodiscard]] OaMatrix SegmentationOverlay(
		OaEngine& InRt,
		const OaMatrix& InImage,
		const OaMatrix& InMask,
		const OaMatrix& InPalette,
		OaF32 InAlpha = 0.5F
	);
	[[nodiscard]] OaMatrix SegmentationOverlay(
		const OaMatrix& InImage,
		const OaMatrix& InMask,
		const OaMatrix& InPalette,
		OaF32 InAlpha = 0.5F
	);
}

// Deprecated compatibility alias. New code uses OaFnImage.
namespace OaFnVision = OaFnImage;
