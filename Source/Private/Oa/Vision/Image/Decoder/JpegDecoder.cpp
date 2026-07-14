// OA Vision — JPEG Decoder Implementation
// CPU-side JPEG decode via stb_image, upload to GPU device matrix

#include <Oa/Vision/JpegDecoder.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>

#include "../../../../../ThirdParty/stb/stb_image.h"

namespace {

OaMatrix UploadRgbNchw01(const OaJpegDecodeResult& InDecoded)
{
	if (InDecoded.Width <= 0 || InDecoded.Height <= 0 || InDecoded.Channels != 3) {
		return {};
	}

	const OaU64 elementCount =
		static_cast<OaU64>(InDecoded.Width) * InDecoded.Height * 3U;
	OaVec<OaF32> planar(elementCount);
	for (OaI32 c = 0; c < 3; ++c) {
		for (OaI32 h = 0; h < InDecoded.Height; ++h) {
			for (OaI32 w = 0; w < InDecoded.Width; ++w) {
				const OaU64 src = (static_cast<OaU64>(h) * InDecoded.Width + w) * 3U + c;
				const OaU64 dst = static_cast<OaU64>(c) * InDecoded.Height * InDecoded.Width +
					static_cast<OaU64>(h) * InDecoded.Width + w;
				planar[dst] = static_cast<OaF32>(InDecoded.Pixels[src]) / 255.0F;
			}
		}
	}

	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(planar.Data()),
			planar.Size() * sizeof(OaF32)),
		OaMatrixShape{1, 3, InDecoded.Height, InDecoded.Width},
		OaScalarType::Float32);
}

} // namespace

// Decode JPEG from memory buffer
OaJpegDecodeResult OaJpegDecoder::Decode(const OaSpan<const OaU8>& InJpegData)
{
	OaJpegDecodeResult result;
	
	int width, height, channels;
	stbi_uc* pixels = stbi_load_from_memory(
		InJpegData.Data(), 
		static_cast<int>(InJpegData.Size()), 
		&width, &height, &channels, 
		3); // Force RGB
	
	if (not pixels) {
		result.Width = 0;
		result.Height = 0;
		result.Channels = 0;
		return result;
	}

	// Copy pixels to result
	OaU64 pixelCount = static_cast<OaU64>(width) * height * 3;
	result.Pixels.Resize(pixelCount);
	memcpy(result.Pixels.Data(), pixels, pixelCount);

	result.Width = width;
	result.Height = height;
	result.Channels = 3;

	stbi_image_free(pixels);
	return result;
}

// Decode JPEG from file path
OaJpegDecodeResult OaJpegDecoder::DecodeFile(OaStringView InPath)
{
	OaJpegDecodeResult result;

	int width, height, channels;
	stbi_uc* pixels = stbi_load(
		InPath.Data(),
		&width, &height, &channels,
		3); // Force RGB

	if (not pixels) {
		// Decode failed
		result.Width = 0;
		result.Height = 0;
		result.Channels = 0;
		return result;
	}
	
	// Copy pixels to result
	OaU64 pixelCount = static_cast<OaU64>(width) * height * 3;
	result.Pixels.Resize(pixelCount);
	memcpy(result.Pixels.Data(), pixels, pixelCount);
	
	result.Width = width;
	result.Height = height;
	result.Channels = 3;
	
	stbi_image_free(pixels);
	return result;
}

// Decode JPEG and upload directly to GPU device matrix
OaMatrix OaJpegDecoder::DecodeToGpu(
	OaEngine& InRt,
	const OaSpan<const OaU8>& InJpegData,
	OaU32 InTargetWidth,
	OaU32 InTargetHeight,
	bool InNormalizeImageNet)
{
	// 1. CPU decode
	auto result = Decode(InJpegData);
	
	if (result.Width == 0) {
		// Decode failed, return empty tensor
		return OaMatrix();
	}
	
	// 2. Convert interleaved RGB8 to a normalized [1,3,H,W] tensor and upload.
	auto tensor = UploadRgbNchw01(result);

	// 3. Resize if needed
	if (InTargetWidth > 0 and InTargetHeight > 0) {
		tensor = OaFnImage::Resize(InRt, tensor, InTargetWidth, InTargetHeight);
	}
	
	// 4. Normalize if needed
	if (InNormalizeImageNet) {
		OaNormalizationParams params = {
			.Mean = {0.485f, 0.456f, 0.406f},
			.Std = {0.229f, 0.224f, 0.225f}
		};
		tensor = OaFnImage::Normalize(InRt, tensor, params);
	}
	if ((InTargetWidth > 0 && InTargetHeight > 0) || InNormalizeImageNet) {
		auto& ctx = OaContext::GetDefault();
		if (!ctx.Execute().IsOk() || !ctx.Sync().IsOk()) {
			return {};
		}
	}
	
	return tensor;
}

// Decode JPEG file and upload to GPU
OaMatrix OaJpegDecoder::DecodeFileToGpu(
	OaEngine& InRt,
	OaStringView InPath,
	OaU32 InTargetWidth,
	OaU32 InTargetHeight,
	bool InNormalizeImageNet)
{
	// 1. CPU decode
	auto result = DecodeFile(InPath);
	
	if (result.Width == 0) {
		// Decode failed, return empty tensor
		return OaMatrix();
	}
	
	// 2. Use the same [0,1] NCHW contract as DecodeToGpu(memory).
	auto tensor = UploadRgbNchw01(result);
	if (tensor.IsEmpty()) {
		return {};
	}

	// 3. Resize if needed
	if (InTargetWidth > 0 and InTargetHeight > 0) {
		tensor = OaFnImage::Resize(InRt, tensor, InTargetWidth, InTargetHeight);
	}
	
	// 4. Normalize if needed
	if (InNormalizeImageNet) {
		OaNormalizationParams params = {
			.Mean = {0.485f, 0.456f, 0.406f},
			.Std = {0.229f, 0.224f, 0.225f}
		};
		tensor = OaFnImage::Normalize(InRt, tensor, params);
	}
	if ((InTargetWidth > 0 && InTargetHeight > 0) || InNormalizeImageNet) {
		auto& ctx = OaContext::GetDefault();
		if (!ctx.Execute().IsOk() || !ctx.Sync().IsOk()) {
			return {};
		}
	}
	
	return tensor;
}

// Batch decode N JPEG files in parallel using OaThreadPool
OaStatus OaJpegDecoder::DecodeBatchToGpu(
	OaEngine& InRt,
	const OaVec<OaString>& InPaths,
	OaVec<OaMatrix>& OutTensors,
	OaU32 InTargetWidth,
	OaU32 InTargetHeight)
{
	OutTensors.Clear();
	OutTensors.Reserve(InPaths.Size());
	
	// TODO: Parallel decode using OaThreadPool
	// For now, serial decode
	for (const auto& path : InPaths) {
		auto tensor = DecodeFileToGpu(InRt, path, InTargetWidth, InTargetHeight, false);
		
		if (tensor.IsEmpty()) {
			return OaStatus::Error("Failed to decode: " + path);
		}
		
		OutTensors.PushBack(tensor);
	}
	
	return OaStatus::Ok();
}
