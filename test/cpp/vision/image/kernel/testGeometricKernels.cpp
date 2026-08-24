// Test Vision geometric transformation kernels
// Verifies GPU kernel output against CPU reference implementations

#include "../../../oaTest.h"
#include <oa/vision.h>
#include <algorithm>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// Nearest-neighbor resize (simple reference)
void cpuResizeNearest(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 srcH, oa::U32 srcW, oa::U32 dstH, oa::U32 dstW, oa::U32 channels) {
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < dstH; ++y) {
			for (oa::U32 x = 0; x < dstW; ++x) {
				// map destination pixel to source pixel
				oa::U32 srcY = (y * srcH) / dstH;
				oa::U32 srcX = (x * srcW) / dstW;
				oa::U32 srcIdx = c * srcH * srcW + srcY * srcW + srcX;
				oa::U32 dstIdx = c * dstH * dstW + y * dstW + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

void cpuResizeBilinear(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 srcH, oa::U32 srcW, oa::U32 dstH, oa::U32 dstW, oa::U32 channels) {
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < dstH; ++y) {
			const float sy = std::clamp(
				(static_cast<float>(y) + 0.5F) * srcH / dstH - 0.5F,
				0.0F, static_cast<float>(srcH - 1));
			const oa::U32 y0 = static_cast<oa::U32>(std::floor(sy));
			const oa::U32 y1 = std::min(y0 + 1, srcH - 1);
			const float fy = sy - y0;
			for (oa::U32 x = 0; x < dstW; ++x) {
				const float sx = std::clamp(
					(static_cast<float>(x) + 0.5F) * srcW / dstW - 0.5F,
					0.0F, static_cast<float>(srcW - 1));
				const oa::U32 x0 = static_cast<oa::U32>(std::floor(sx));
				const oa::U32 x1 = std::min(x0 + 1, srcW - 1);
				const float fx = sx - x0;
				const oa::U32 base = c * srcH * srcW;
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
void cpuFlipHorizontal(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 h, oa::U32 w, oa::U32 channels) {
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < h; ++y) {
			for (oa::U32 x = 0; x < w; ++x) {
				oa::U32 srcIdx = c * h * w + y * w + x;
				oa::U32 dstIdx = c * h * w + y * w + (w - 1 - x);
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// Vertical flip
void cpuFlipVertical(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 h, oa::U32 w, oa::U32 channels) {
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < h; ++y) {
			for (oa::U32 x = 0; x < w; ++x) {
				oa::U32 srcIdx = c * h * w + y * w + x;
				oa::U32 dstIdx = c * h * w + (h - 1 - y) * w + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// 90-degree rotation (clockwise)
void cpuRotate90(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 h, oa::U32 w, oa::U32 channels) {
	// After 90° rotation: new_h = w, new_w = h
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < h; ++y) {
			for (oa::U32 x = 0; x < w; ++x) {
				oa::U32 srcIdx = c * h * w + y * w + x;
				// (x, y) -> (y, w-1-x) after 90° CW rotation
				oa::U32 newY = x;
				oa::U32 newX = h - 1 - y;
				oa::U32 dstIdx = c * w * h + newY * h + newX;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// Crop operation
void cpuCrop(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 srcH, oa::U32 srcW, oa::U32 cropX, oa::U32 cropY,
	oa::U32 cropW, oa::U32 cropH, oa::U32 channels) {
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = 0; y < cropH; ++y) {
			for (oa::U32 x = 0; x < cropW; ++x) {
				oa::U32 srcIdx = c * srcH * srcW + (cropY + y) * srcW + (cropX + x);
				oa::U32 dstIdx = c * cropH * cropW + y * cropW + x;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class GeometricKernels : public VkEngineTestFixture {
protected:
	void materialize() {
		auto& ctx = oa::ExecutionSession::getActive();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	}

	// Helper: compute max absolute error
	double computeMaxAbsError(const std::vector<float>& ref, const oa::Matrix& gpu) {
		materialize();
		double maxError = 0.0;
		oa::I64 n = ref.size();
		for (oa::I64 i = 0; i < n; ++i) {
			double absErr = std::abs(gpu.at(i) - ref[i]);
			maxError = std::max(maxError, absErr);
		}
		return maxError;
	}
};

// ═══════════════════════════════════════════════════════════════════════════════
// resize Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Resize_Downscale) {
	const oa::U32 srcH = 64;
	const oa::U32 srcW = 64;
	const oa::U32 dstH = 32;
	const oa::U32 dstW = 32;
	const oa::U32 channels = 3;
	
	// Create input
	auto input = oa::FnMatrix::rand({1, channels, srcH, srcW});
	
	// GPU resize
	auto resized = oa::FnImage::resize(input, dstW, dstH);
	materialize();
	
	// verify shape
	EXPECT_EQ(resized.getShape(), oa::MatrixShape({1, channels, dstH, dstW}));
	
	// Note: We can't easily verify bilinear interpolation values without
	// implementing the exact same algorithm. Just verify it runs and produces
	// reasonable output (no NaN/Inf, values in expected range)
	for (oa::I64 i = 0; i < resized.numElements(); ++i) {
		float val = resized.at(i);
		EXPECT_TRUE(std::isfinite(val)) << "Non-finite value at index " << i;
		EXPECT_GE(val, 0.0F) << "Value below 0 at index " << i;
		EXPECT_LE(val, 1.0F) << "Value above 1 at index " << i;
	}
}

TEST_F(GeometricKernels, Resize_Upscale) {
	const oa::U32 srcH = 32;
	const oa::U32 srcW = 32;
	const oa::U32 dstH = 128;
	const oa::U32 dstW = 128;
	
	auto input = oa::FnMatrix::rand({1, 3, srcH, srcW});
	auto resized = oa::FnImage::resize(input, dstW, dstH);
	
	EXPECT_EQ(resized.getShape(), oa::MatrixShape({1, 3, dstH, dstW}));
}

TEST_F(GeometricKernels, Resize_BilinearMatchesCpuReference) {
	constexpr oa::U32 srcH = 3, srcW = 4, dstH = 7, dstW = 5, channels = 2;
	auto input = oa::FnMatrix::empty({1, channels, srcH, srcW});
	std::vector<float> host(channels * srcH * srcW);
	for (oa::U32 i = 0; i < host.size(); ++i) {
		host[i] = static_cast<float>((i * 7U) % 19U) / 18.0F;
		input.set(i, host[i]);
	}
	std::vector<float> reference(channels * dstH * dstW);
	cpuResizeBilinear(host, reference, srcH, srcW, dstH, dstW, channels);

	auto output = oa::FnImage::resize(rt(), input, dstW, dstH, oa::InterpolationMode::Bilinear);
	EXPECT_EQ(output.getShape(), oa::MatrixShape({1, channels, dstH, dstW}));
	EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(GeometricKernels, Resize_NearestMatchesCpuReference) {
	constexpr oa::U32 srcH = 4, srcW = 5, dstH = 3, dstW = 8, channels = 1;
	auto input = oa::FnMatrix::empty({1, channels, srcH, srcW});
	std::vector<float> host(channels * srcH * srcW);
	for (oa::U32 i = 0; i < host.size(); ++i) {
		host[i] = static_cast<float>(i);
		input.set(i, host[i]);
	}
	std::vector<float> reference(channels * dstH * dstW);
	cpuResizeNearest(host, reference, srcH, srcW, dstH, dstW, channels);

	auto output = oa::FnImage::resize(rt(), input, dstW, dstH, oa::InterpolationMode::Nearest);
	EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-6);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flip Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Flip_Horizontal) {
	const oa::U32 h = 16;
	const oa::U32 w = 16;
	const oa::U32 channels = 3;
	
	auto input = oa::FnMatrix::rand({1, channels, h, w});
	materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (oa::I64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> flippedRef(channels * h * w);
	cpuFlipHorizontal(inputHost, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = oa::FnImage::flip(input, true, false);
	
	// verify
	EXPECT_EQ(flippedGpu.getShape(), input.getShape());
	double maxError = computeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Flip_Vertical) {
	const oa::U32 h = 16;
	const oa::U32 w = 16;
	const oa::U32 channels = 3;
	
	auto input = oa::FnMatrix::rand({1, channels, h, w});
	materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (oa::I64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> flippedRef(channels * h * w);
	cpuFlipVertical(inputHost, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = oa::FnImage::flip(input, false, true);
	
	// verify
	double maxError = computeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Flip_Both) {
	const oa::U32 h = 16;
	const oa::U32 w = 16;
	const oa::U32 channels = 3;
	
	auto input = oa::FnMatrix::rand({1, channels, h, w});
	materialize();
	
	// CPU reference (flip both = flip horizontal then vertical)
	std::vector<float> inputHost(channels * h * w);
	for (oa::I64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> temp(channels * h * w);
	std::vector<float> flippedRef(channels * h * w);
	cpuFlipHorizontal(inputHost, temp, h, w, channels);
	cpuFlipVertical(temp, flippedRef, h, w, channels);
	
	// GPU kernel
	auto flippedGpu = oa::FnImage::flip(input, true, true);
	
	// verify
	double maxError = computeMaxAbsError(flippedRef, flippedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// rotate Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Rotate_90Degrees) {
	const oa::U32 h = 16;
	const oa::U32 w = 16;
	const oa::U32 channels = 3;
	
	auto input = oa::FnMatrix::rand({1, channels, h, w});
	materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (oa::I64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> rotatedRef(channels * w * h);
	cpuRotate90(inputHost, rotatedRef, h, w, channels);
	
	// GPU kernel
	auto rotatedGpu = oa::FnImage::rotate(input, 90);
	
	// verify shape (width and height swap)
	EXPECT_EQ(rotatedGpu.getShape(), oa::MatrixShape({1, channels, w, h}));
	
	// verify values
	double maxError = computeMaxAbsError(rotatedRef, rotatedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Crop Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GeometricKernels, Crop_CenterRegion) {
	const oa::U32 srcH = 64;
	const oa::U32 srcW = 64;
	const oa::U32 cropX = 16;
	const oa::U32 cropY = 16;
	const oa::U32 cropW = 32;
	const oa::U32 cropH = 32;
	const oa::U32 channels = 3;
	
	auto input = oa::FnMatrix::rand({1, channels, srcH, srcW});
	materialize();
	
	// CPU reference
	std::vector<float> inputHost(channels * srcH * srcW);
	for (oa::I64 i = 0; i < channels * srcH * srcW; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> croppedRef(channels * cropH * cropW);
	cpuCrop(inputHost, croppedRef, srcH, srcW, cropX, cropY, cropW, cropH, channels);
	
	// GPU kernel
	auto croppedGpu = oa::FnImage::crop(input, cropX, cropY, cropW, cropH);
	
	// verify shape
	EXPECT_EQ(croppedGpu.getShape(), oa::MatrixShape({1, channels, cropH, cropW}));
	
	// verify values
	double maxError = computeMaxAbsError(croppedRef, croppedGpu);
	EXPECT_LT(maxError, 1e-6) << "MaxAbsError: " << maxError;
}

TEST_F(GeometricKernels, Crop_TopLeftCorner) {
	const oa::U32 srcH = 128;
	const oa::U32 srcW = 128;
	const oa::U32 cropW = 64;
	const oa::U32 cropH = 64;
	
	auto input = oa::FnMatrix::rand({1, 3, srcH, srcW});
	
	// Crop from (0, 0)
	auto cropped = oa::FnImage::crop(input, 0, 0, cropW, cropH);
	
	EXPECT_EQ(cropped.getShape(), oa::MatrixShape({1, 3, cropH, cropW}));
}
