// oa::FnImage — stateless GPU image transforms and composed image pipelines.
//
// The schema-generated header owns the public tensor-op declarations. Complex
// validation, graph recording, kernels, and multi-stage pipelines remain
// handwritten in source/cpp/lib/oa/vision.

#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/core/image.h>
#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/core/std/path.h>
#include <oa/core/types.h>
#include <oa/runtime/engine.h>
#include <oa/vision/type.h>

// SaveFile only needs the renderer texture by reference. Avoid pulling the UI
// image surface into every Vision transform consumer.
namespace oa { class Texture; }

namespace oa {

struct NormalizationParams {
	oa::F32 mean[3];
	oa::F32 std[3];
};

namespace FnImage {
	// exact declarations for the 50 tensor-native image operations. Every op
	// has an explicit-engine overload and an active-context convenience overload.
	// Regenerate with: python3 tools/gen/fn/generate.py --live
	#include <oa/vision/fnimage/fnImage.gen.h>

	// Stateless still-image codec boundaries. Decode uploads one normalized
	// Float32 NCHW image through the active engine. Encode and saveFile are
	// explicit synchronous host boundaries which complete and read back their
	// semantic image input.
	[[nodiscard]] oa::Result<Image> decodeFile(
		const oa::Path& inPath,
		ImageFormat inFormat = ImageFormat::Rgb
	);
	[[nodiscard]] oa::Result<Image> decodeMemory(
		oa::Span<const oa::U8> inData,
		ImageFormat inFormat = ImageFormat::Rgb
	);
	[[nodiscard]] oa::Result<oa::Vec<oa::U8>> encode(
		const Image& inImage,
		ImageCodec inCodec,
		oa::U32 inQuality = 90U
	);
	[[nodiscard]] oa::Status saveFile(
		const oa::Path& inPath,
		const Image& inImage,
		oa::U32 inQuality = 90U
	);
	[[nodiscard]] bool canDecode(ImageCodec inCodec) noexcept;
	[[nodiscard]] bool canEncode(ImageCodec inCodec) noexcept;

	// Synchronous file sink for a device-local packed RGBA8 texture. pending
	// work in the engine's matching private recorder completes before readback.
	// The output codec is inferred from .png, .jpg/.jpeg, .bmp, or .tga.
	[[nodiscard]] oa::Status saveTextureFile(
		Engine& inEngine,
		const oa::Texture& inTexture,
		oa::StringView inPath
	);
	// Synchronous host sink for one exact packed RGBA8 image. This is the
	// encoder boundary used by image-backed render sessions after their own
	// explicit readback/completion protocol has produced host bytes.
	[[nodiscard]] oa::Status saveRgbaFile(
		oa::Span<const oa::U8> inRgba,
		oa::U32 inWidth,
		oa::U32 inHeight,
		oa::StringView inPath,
		oa::U32 inQuality = 90U
	);

	// Semantic image overloads preserve or update layout/format metadata while
	// reusing the tensor-native GPU kernels above.
	[[nodiscard]] Image resize(
		const Image& inImage,
		oa::U32 inWidth,
		oa::U32 inHeight
	);
	[[nodiscard]] Image normalize(
		const Image& inImage,
		const NormalizationParams& inParams
	);
	[[nodiscard]] Image brightnessContrast(
		const Image& inImage,
		oa::F32 inBrightness = 0.0F,
		oa::F32 inContrast = 1.0F
	);
	[[nodiscard]] Image convertColor(
		const Image& inImage,
		ImageFormat inDstFormat
	);
	[[nodiscard]] Image resizeNormalize(
		const Image& inImage,
		oa::U32 inWidth,
		oa::U32 inHeight,
		const NormalizationParams& inParams
	);

	// Blend an Int32 semantic label map over an FP32 NCHW RGB/RGBA image.
	// mask is [N,H,W] or [N,1,H,W], palette is FP32 [K,3], and invalid labels
	// leave the source pixel unchanged.
	[[nodiscard]] Matrix segmentationOverlay(
		Engine& inRt,
		const Matrix& inImage,
		const Matrix& inMask,
		const Matrix& inPalette,
		oa::F32 inAlpha = 0.5F
	);
	[[nodiscard]] Matrix segmentationOverlay(
		const Matrix& inImage,
		const Matrix& inMask,
		const Matrix& inPalette,
		oa::F32 inAlpha = 0.5F
	);
} // namespace FnImage

} // namespace oa
