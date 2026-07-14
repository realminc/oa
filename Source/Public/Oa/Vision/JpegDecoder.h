// OA Vision — JPEG Decoder
// CPU-side JPEG decode via stb_image, upload to GPU device matrix
// For dataset loading and image preprocessing pipelines

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>

// JPEG decode result: decoded pixels + metadata
struct OaJpegDecodeResult {
	OaVec<OaU8> Pixels;  // RGB pixel data (row-major)
	OaI32 Width;
	OaI32 Height;
	OaI32 Channels;       // Usually 3 (RGB)
};

// JPEG decoder (stateless, functional API)
class OaJpegDecoder {
public:
	// Decode JPEG from memory buffer
	// Returns decoded RGB pixels and dimensions
	static OaJpegDecodeResult Decode(const OaSpan<const OaU8>& InJpegData);

	// Decode JPEG from file path
	static OaJpegDecodeResult DecodeFile(OaStringView InPath);

	// Decode JPEG and upload directly to GPU device matrix
	// Output: OaMatrix [1, 3, height, width] in Float32
	// If InTargetWidth/Height > 0, decodes at native size first then resizes on GPU
	static OaMatrix DecodeToGpu(
		class OaEngine& InRt,
		const OaSpan<const OaU8>& InJpegData,
		OaU32 InTargetWidth  = 0,
		OaU32 InTargetHeight = 0,
		bool InNormalizeImageNet = false
	);

	// Decode JPEG file and upload to GPU
	static OaMatrix DecodeFileToGpu(
		class OaEngine& InRt,
		OaStringView InPath,
		OaU32 InTargetWidth  = 0,
		OaU32 InTargetHeight = 0,
		bool InNormalizeImageNet = false
	);

	// Batch decode N JPEG files in parallel using OaThreadPool
	// Each decoded to its own OaMatrix
	static OaStatus DecodeBatchToGpu(
		class OaEngine& InRt,
		const OaVec<OaString>& InPaths,
		OaVec<OaMatrix>& OutTensors,
		OaU32 InTargetWidth  = 0,
		OaU32 InTargetHeight = 0
	);
};
