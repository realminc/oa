// Test Vision conversion kernels (CvtColor, Normalize, etc.)
// Verifies GPU kernel output against CPU reference implementations

#include "../../OaTest.h"
#include <Oa/Vision.h>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// RGB to Grayscale: Y = 0.299*R + 0.587*G + 0.114*B
void CpuRgbToGray(const std::vector<float>& rgb, std::vector<float>& gray, OaU32 h, OaU32 w) {
	for (OaU32 y = 0; y < h; ++y) {
		for (OaU32 x = 0; x < w; ++x) {
			OaU32 idx = y * w + x;
			float r = rgb[idx];
			float g = rgb[h * w + idx];
			float b = rgb[2 * h * w + idx];
			gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
		}
	}
}

// ImageNet normalization: (x - mean) / std
void CpuNormalize(const std::vector<float>& input, std::vector<float>& output,
	OaU32 c, OaU32 h, OaU32 w, const OaNormalizationParams& params) {
	for (OaU32 ch = 0; ch < c; ++ch) {
		float mean = params.Mean[ch];
		float std = params.Std[ch];
		for (OaU32 y = 0; y < h; ++y) {
			for (OaU32 x = 0; x < w; ++x) {
				OaU32 idx = ch * h * w + y * w + x;
				output[idx] = (input[idx] - mean) / std;
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class ConversionKernels : public ::testing::Test {
protected:
	void SetUp() override {
		// Runtime is set up by OaVkTestEnvironment (global fixture)
		// Just verify it's available
		ASSERT_NE(OaEngine::GetGlobal(), nullptr);
	}
	
	// Helper: compute max relative error between CPU and GPU results
	double ComputeMaxRelativeError(const std::vector<float>& ref, const OaMatrix& gpu) {
		double maxError = 0.0;
		OaI64 n = ref.size();
		for (OaI64 i = 0; i < n; ++i) {
			float refVal = ref[i];
			float gpuVal = gpu.At(i);
			double absErr = std::abs(gpuVal - refVal);
			double relErr = (std::abs(refVal) > 1e-6) ? absErr / std::abs(refVal) : absErr;
			maxError = std::max(maxError, relErr);
		}
		return maxError;
	}
};

// ═══════════════════════════════════════════════════════════════════════════════
// RGB to Grayscale Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ConversionKernels, RgbToGray_SmallImage) {
	const OaU32 h = 4, w = 4;
	
	// Create RGB input (3 channels)
	auto rgb = OaFnMatrix::Rand({1, 3, h, w});
	
	// CPU reference
	std::vector<float> rgbHost(3 * h * w);
	for (OaI64 i = 0; i < 3 * h * w; ++i) {
		rgbHost[i] = rgb.At(i);
	}
	std::vector<float> grayRef(h * w);
	CpuRgbToGray(rgbHost, grayRef, h, w);
	
	// GPU kernel (via OaFnMatrix API)
	auto grayGpu = OaFnMatrix::CvtColor(rgb, OaColorConversion::RgbToGray);
	
	// Verify shape
	EXPECT_EQ(grayGpu.GetShape(), OaMatrixShape({1, 1, h, w}));
	
	// Verify values
	double maxError = ComputeMaxRelativeError(grayRef, grayGpu);
	EXPECT_LT(maxError, 1e-5) << "MaxRelError: " << maxError;
}

TEST_F(ConversionKernels, RgbToGray_LargeImage) {
	const OaU32 h = 256, w = 256;
	
	auto rgb = OaFnMatrix::Rand({1, 3, h, w});
	
	// CPU reference
	std::vector<float> rgbHost(3 * h * w);
	for (OaI64 i = 0; i < 3 * h * w; ++i) {
		rgbHost[i] = rgb.At(i);
	}
	std::vector<float> grayRef(h * w);
	CpuRgbToGray(rgbHost, grayRef, h, w);
	
	// GPU kernel
	auto grayGpu = OaFnMatrix::CvtColor(rgb, OaColorConversion::RgbToGray);
	
	// Verify
	double maxError = ComputeMaxRelativeError(grayRef, grayGpu);
	EXPECT_LT(maxError, 1e-5) << "MaxRelError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Normalization Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ConversionKernels, Normalize_ImageNet) {
	const OaU32 h = 8, w = 8;
	
	// Create RGB input
	auto input = OaFnMatrix::Rand({1, 3, h, w});
	
	// ImageNet normalization params
	OaNormalizationParams params;
	params.Mean[0] = 0.485f;
	params.Mean[1] = 0.456f;
	params.Mean[2] = 0.406f;
	params.Std[0] = 0.229f;
	params.Std[1] = 0.224f;
	params.Std[2] = 0.225f;
	
	// CPU reference
	std::vector<float> inputHost(3 * h * w);
	for (OaI64 i = 0; i < 3 * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> normRef(3 * h * w);
	CpuNormalize(inputHost, normRef, 3, h, w, params);
	
	// GPU kernel
	auto normGpu = OaFnMatrix::Normalize(input, params);
	
	// Verify shape
	EXPECT_EQ(normGpu.GetShape(), input.GetShape());
	
	// Verify values
	double maxError = ComputeMaxRelativeError(normRef, normGpu);
	EXPECT_LT(maxError, 1e-5) << "MaxRelError: " << maxError;
}

TEST_F(ConversionKernels, Normalize_CustomParams) {
	const OaU32 h = 16, w = 16;
	
	auto input = OaFnMatrix::Rand({1, 3, h, w});
	
	// Custom normalization
	OaNormalizationParams params;
	params.Mean[0] = 0.5f;
	params.Mean[1] = 0.5f;
	params.Mean[2] = 0.5f;
	params.Std[0] = 0.5f;
	params.Std[1] = 0.5f;
	params.Std[2] = 0.5f;
	
	// CPU reference
	std::vector<float> inputHost(3 * h * w);
	for (OaI64 i = 0; i < 3 * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> normRef(3 * h * w);
	CpuNormalize(inputHost, normRef, 3, h, w, params);
	
	// GPU kernel
	auto normGpu = OaFnMatrix::Normalize(input, params);
	
	// Verify
	double maxError = ComputeMaxRelativeError(normRef, normGpu);
	EXPECT_LT(maxError, 1e-5) << "MaxRelError: " << maxError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch Processing Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ConversionKernels, RgbToGray_BatchProcessing) {
	const OaU32 b = 4, h = 32, w = 32;
	
	// Create batch of RGB images
	auto rgb = OaFnMatrix::Rand({b, 3, h, w});
	
	// GPU kernel
	auto gray = OaFnMatrix::CvtColor(rgb, OaColorConversion::RgbToGray);
	
	// Verify shape
	EXPECT_EQ(gray.GetShape(), OaMatrixShape({b, 1, h, w}));
	
	// Verify each image in batch
	for (OaU32 i = 0; i < b; ++i) {
		std::vector<float> rgbHost(3 * h * w);
		for (OaU32 c = 0; c < 3; ++c) {
			for (OaU32 y = 0; y < h; ++y) {
				for (OaU32 x = 0; x < w; ++x) {
					OaI64 idx = i * 3 * h * w + c * h * w + y * w + x;
					rgbHost[c * h * w + y * w + x] = rgb.At(idx);
				}
			}
		}
		
		std::vector<float> grayRef(h * w);
		CpuRgbToGray(rgbHost, grayRef, h, w);
		
		// Compare this batch element
		double maxError = 0.0;
		for (OaU32 y = 0; y < h; ++y) {
			for (OaU32 x = 0; x < w; ++x) {
				OaI64 idx = i * h * w + y * w + x;
				float refVal = grayRef[y * w + x];
				float gpuVal = gray.At(idx);
				double absErr = std::abs(gpuVal - refVal);
				double relErr = (std::abs(refVal) > 1e-6) ? absErr / std::abs(refVal) : absErr;
				maxError = std::max(maxError, relErr);
			}
		}
		
		EXPECT_LT(maxError, 1e-5) << "Batch " << i << " MaxRelError: " << maxError;
	}
}
