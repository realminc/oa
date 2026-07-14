// Test Vision geometric transformation kernels
// Verifies GPU kernel output against CPU reference implementations

#include "../../../OaTest.h"
#include <Oa/Vision.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <algorithm>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// Nearest-neighbor resize (simple reference)
void CpuResizeNearest(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 srcH, OaU32 srcW, OaU32 dstH, OaU32 dstW, OaU32 channels) {
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < dstH; ++y) {
			for (OaU32 x = 0; x < dstW; ++x) {
				// Map destination pixel to source pixel
				OaU32 srcY = (y * srcH) / dstH;
				OaU32 srcX = (x * srcW) / dstW;
				OaU32 srcIdx = c * srcH * srcW + srcY * srcW + srcX;
				OaU32 dstIdx = c * dstH * dstW + y * dstW + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

void CpuResizeBilinear(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 srcH, OaU32 srcW, OaU32 dstH, OaU32 dstW, OaU32 channels) {
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < dstH; ++y) {
			const float sy = std::clamp(
				(static_cast<float>(y) + 0.5F) * srcH / dstH - 0.5F,
				0.0F, static_cast<float>(srcH - 1));
			const OaU32 y0 = static_cast<OaU32>(std::floor(sy));
			const OaU32 y1 = std::min(y0 + 1, srcH - 1);
			const float fy = sy - y0;
			for (OaU32 x = 0; x < dstW; ++x) {
				const float sx = std::clamp(
					(static_cast<float>(x) + 0.5F) * srcW / dstW - 0.5F,
					0.0F, static_cast<float>(srcW - 1));
				const OaU32 x0 = static_cast<OaU32>(std::floor(sx));
				const OaU32 x1 = std::min(x0 + 1, srcW - 1);
				const float fx = sx - x0;
				const OaU32 base = c * srcH * srcW;
				const float v0 = std::lerp(src[base + y0 * srcW + x0],
					src[base + y0 * srcW + x1], fx);
				const float v1 = std::lerp(src[base + y1 * srcW + x0],
					src[base + y1 * srcW + x1], fx);
				dst[c * dstH * dstW + y * dstW + x] = std::lerp(v0, v1, fy);
			}
		}
	}
}

// Horizontal flip
void CpuFlipHorizontal(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 h, OaU32 w, OaU32 channels) {
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < h; ++y) {
			for (OaU32 x = 0; x < w; ++x) {
				OaU32 srcIdx = c * h * w + y * w + x;
				OaU32 dstIdx = c * h * w + y * w + (w - 1 - x);
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// Vertical flip
void CpuFlipVertical(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 h, OaU32 w, OaU32 channels) {
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < h; ++y) {
			for (OaU32 x = 0; x < w; ++x) {
				OaU32 srcIdx = c * h * w + y * w + x;
				OaU32 dstIdx = c * h * w + (h - 1 - y) * w + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// 90-degree rotation (clockwise)
void CpuRotate90(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 h, OaU32 w, OaU32 channels) {
	// After 90° rotation: new_h = w, new_w = h
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < h; ++y) {
			for (OaU32 x = 0; x < w; ++x) {
				OaU32 srcIdx = c * h * w + y * w + x;
				// (x, y) -> (y, w-1-x) after 90° CW rotation
				OaU32 newY = x;
				OaU32 newX = h - 1 - y;
				OaU32 dstIdx = c * w * h + newY * h + newX;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// Crop operation
void CpuCrop(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 srcH, OaU32 srcW, OaU32 cropX, OaU32 cropY,
	OaU32 cropW, OaU32 cropH, OaU32 channels) {
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = 0; y < cropH; ++y) {
			for (OaU32 x = 0; x < cropW; ++x) {
				OaU32 srcIdx = c * srcH * srcW + (cropY + y) * srcW + (cropX + x);
				OaU32 dstIdx = c * cropH * cropW + y * cropW + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class GeometricKernels : public OaVkEngineTestFixture {
protected:
	void Materialize() {
		auto& ctx = OaContext::GetDefault();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}

	// Helper: compute max absolute error
	double ComputeMaxAbsError(const std::vector<float>& ref, const OaMatrix& gpu) {
		Materialize();
		double maxError = 0.0;
		OaI64 n = ref.size();
		for (OaI64 i = 0; i < n; ++i) {
			double absErr = std::abs(gpu.At(i) - ref[i]);
			maxError = std::max(maxError, absErr);
		}
		return maxError;
	}
};

// ═══════════════════════════════════════════════════════════════════════════════
// Resize Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Resize_Downscale) {
	const OaU32 srcH = 64;
	const OaU32 srcW = 64;
	const OaU32 dstH = 32;
	const OaU32 dstW = 32;
	const OaU32 channels = 3;
	
	// Create input
	auto input = OaFnMatrix::Rand({1, channels, srcH, srcW});
	
	// GPU resize
	auto resized = OaFnImage::Resize(input, dstW, dstH);
	Materialize();
	
	// Verify shape
	EXPECT_EQ(resized.GetShape(), OaMatrixShape({1, channels, dstH, dstW}));
	
	// Note: We can't easily verify bilinear interpolation values without
	// implementing the exact same algorithm. Just verify it runs and produces
	// reasonable output (no NaN/Inf, values in expected range)
	for (OaI64 i = 0; i < resized.NumElements(); ++i) {
		float val = resized.At(i);
		EXPECT_TRUE(std::isfinite(val)) << "Non-finite value at index " << i;
		EXPECT_GE(val, 0.0F) << "Value below 0 at index " << i;
		EXPECT_LE(val, 1.0F) << "Value above 1 at index " << i;
	}
}

TEST_F(GeometricKernels, Resize_Upscale) {
	const OaU32 srcH = 32;
	const OaU32 srcW = 32;
	const OaU32 dstH = 128;
	const OaU32 dstW = 128;
	
	auto input = OaFnMatrix::Rand({1, 3, srcH, srcW});
	auto resized = OaFnImage::Resize(input, dstW, dstH);
	
	EXPECT_EQ(resized.GetShape(), OaMatrixShape({1, 3, dstH, dstW}));
}

TEST_F(GeometricKernels, Resize_BilinearMatchesCpuReference) {
	constexpr OaU32 srcH = 3, srcW = 4, dstH = 7, dstW = 5, channels = 2;
	auto input = OaFnMatrix::Empty({1, channels, srcH, srcW});
	std::vector<float> host(channels * srcH * srcW);
	for (OaU32 i = 0; i < host.size(); ++i) {
		host[i] = static_cast<float>((i * 7U) % 19U) / 18.0F;
		input.Set(i, host[i]);
	}
	std::vector<float> reference(channels * dstH * dstW);
	CpuResizeBilinear(host, reference, srcH, srcW, dstH, dstW, channels);

	auto output = OaFnImage::Resize(Rt(), input, dstW, dstH, OaInterpolationMode::Bilinear);
	EXPECT_EQ(output.GetShape(), OaMatrixShape({1, channels, dstH, dstW}));
	EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(GeometricKernels, Resize_NearestMatchesCpuReference) {
	constexpr OaU32 srcH = 4, srcW = 5, dstH = 3, dstW = 8, channels = 1;
	auto input = OaFnMatrix::Empty({1, channels, srcH, srcW});
	std::vector<float> host(channels * srcH * srcW);
	for (OaU32 i = 0; i < host.size(); ++i) {
		host[i] = static_cast<float>(i);
		input.Set(i, host[i]);
	}
	std::vector<float> reference(channels * dstH * dstW);
	CpuResizeNearest(host, reference, srcH, srcW, dstH, dstW, channels);

	auto output = OaFnImage::Resize(Rt(), input, dstW, dstH, OaInterpolationMode::Nearest);
	EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-6);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flip Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Flip_Horizontal) {
	const OaU32 h = 16;
	const OaU32 w = 16;
	const OaU32 channels = 3;
	
	auto input = OaFnMatrix::Rand({1, channels, h, w});
	Materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (OaI64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> flippedRef(channels * h * w);
	CpuFlipHorizontal(inputHost, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = OaFnImage::Flip(input, true, false);
	
	// Verify
	EXPECT_EQ(flippedGpu.GetShape(), input.GetShape());
	double maxError = ComputeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Flip_Vertical) {
	const OaU32 h = 16;
	const OaU32 w = 16;
	const OaU32 channels = 3;
	
	auto input = OaFnMatrix::Rand({1, channels, h, w});
	Materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (OaI64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> flippedRef(channels * h * w);
	CpuFlipVertical(inputHost, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = OaFnImage::Flip(input, false, true);
	
	// Verify
	double maxError = ComputeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Flip_Both) {
	const OaU32 h = 16;
	const OaU32 w = 16;
	const OaU32 channels = 3;
	
	auto input = OaFnMatrix::Rand({1, channels, h, w});
	Materialize();
	
	// CPU reference (flip both = flip horizontal then vertical)
	std::vector<float> inputHost(channels * h * w);
	for (OaI64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> temp(channels * h * w);
	std::vector<float> flippedRef(channels * h * w);
	CpuFlipHorizontal(inputHost, temp, h, w, channels);
	CpuFlipVertical(temp, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = OaFnImage::Flip(input, true, true);
	
	// Verify
	double maxError = ComputeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rotate Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Rotate_90Degrees) {
	const OaU32 h = 16;
	const OaU32 w = 16;
	const OaU32 channels = 3;
	
	auto input = OaFnMatrix::Rand({1, channels, h, w});
	Materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (OaI64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> rotatedRef(channels * w * h);
	CpuRotate90(inputHost, rotatedRef, h, w, channels);
	
	// GPU kernel
	auto rotatedGpu = OaFnImage::Rotate(input, 90);
	
	// Verify shape (width and height swap)
	EXPECT_EQ(rotatedGpu.GetShape(), OaMatrixShape({1, channels, w, h}));
	
	// Verify values
	double maxError = ComputeMaxAbsError(rotatedRef, rotatedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Crop Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Crop_CenterRegion) {
	const OaU32 srcH = 64;
	const OaU32 srcW = 64;
	const OaU32 cropX = 16;
	const OaU32 cropY = 16;
	const OaU32 cropW = 32;
	const OaU32 cropH = 32;
	const OaU32 channels = 3;
	
	auto input = OaFnMatrix::Rand({1, channels, srcH, srcW});
	Materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * srcH * srcW);
	for (OaI64 i = 0; i < channels * srcH * srcW; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> croppedRef(channels * cropH * cropW);
	CpuCrop(inputHost, croppedRef, srcH, srcW, cropX, cropY, cropW, cropH, channels);
	
	// GPU kernel
	auto croppedGpu = OaFnImage::Crop(input, cropX, cropY, cropW, cropH);
	
	// Verify shape
	EXPECT_EQ(croppedGpu.GetShape(), OaMatrixShape({1, channels, cropH, cropW}));
	
	// Verify values
	double maxError = ComputeMaxAbsError(croppedRef, croppedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Crop_TopLeftCorner) {
	const OaU32 srcH = 128;
	const OaU32 srcW = 128;
	const OaU32 cropW = 64;
	const OaU32 cropH = 64;
	
	auto input = OaFnMatrix::Rand({1, 3, srcH, srcW});
	
	// Crop from (0, 0)
	auto cropped = OaFnImage::Crop(input, 0, 0, cropW, cropH);
	
	EXPECT_EQ(cropped.GetShape(), OaMatrixShape({1, 3, cropH, cropW}));
}
