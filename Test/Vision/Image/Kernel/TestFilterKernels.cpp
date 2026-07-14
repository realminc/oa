// Test Vision filter kernels (GaussianBlur, etc.)
// Verifies GPU kernel output against CPU reference implementations

#include "../../../OaTest.h"
#include <Oa/Vision.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <cmath>
#include <limits>

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// Generate 1D Gaussian kernel
std::vector<float> GenerateGaussianKernel(OaU32 radius, float sigma) {
	OaU32 size = 2 * radius + 1;
	std::vector<float> kernel(size);
	float sum = 0.0F;
	
	for (OaU32 i = 0; i < size; ++i) {
		float x = static_cast<float>(static_cast<OaI32>(i) - static_cast<OaI32>(radius));
		kernel[i] = std::exp(-(x * x) / (2.0F * sigma * sigma));
		sum += kernel[i];
	}
	
	// Normalize
	for (OaU32 i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}
	
	return kernel;
}

// Apply 1D Gaussian blur (horizontal or vertical)
void ApplyGaussian1D(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 h, OaU32 w, const std::vector<float>& kernel, bool horizontal) {
	OaU32 radius = static_cast<OaU32>(kernel.size()) / 2;
	
	for (OaU32 y = 0; y < h; ++y) {
		for (OaU32 x = 0; x < w; ++x) {
			float sum = 0.0F;
			
			for (OaU32 k = 0; k < kernel.size(); ++k) {
				OaI32 offset = static_cast<OaI32>(k) - static_cast<OaI32>(radius);
				OaI32 sampleX = horizontal ? (static_cast<OaI32>(x) + offset) : static_cast<OaI32>(x);
				OaI32 sampleY = horizontal ? static_cast<OaI32>(y) : (static_cast<OaI32>(y) + offset);
				
				// Clamp to image bounds
				sampleX = std::max(0, std::min(sampleX, static_cast<OaI32>(w) - 1));
				sampleY = std::max(0, std::min(sampleY, static_cast<OaI32>(h) - 1));
				
				OaU32 idx = static_cast<OaU32>(sampleY) * w + static_cast<OaU32>(sampleX);
				sum += src[idx] * kernel[k];
			}
			
			dst[y * w + x] = sum;
		}
	}
}

// CPU Gaussian blur (separable: horizontal then vertical)
void CpuGaussianBlur(const std::vector<float>& src, std::vector<float>& dst,
	OaU32 h, OaU32 w, OaU32 channels, float sigma) {
	// Use radius = ceil(3*sigma) for good coverage
	OaU32 radius = static_cast<OaU32>(std::ceil(3.0F * sigma));
	auto kernel = GenerateGaussianKernel(radius, sigma);
	
	std::vector<float> temp(h * w);
	
	for (OaU32 c = 0; c < channels; ++c) {
		// Extract channel
		std::vector<float> channel(h * w);
		for (OaU32 i = 0; i < h * w; ++i) {
			channel[i] = src[c * h * w + i];
		}
		
		// Horizontal pass
		ApplyGaussian1D(channel, temp, h, w, kernel, true);
		
		// Vertical pass
		std::vector<float> result(h * w);
		ApplyGaussian1D(temp, result, h, w, kernel, false);
		
		// Store result
		for (OaU32 i = 0; i < h * w; ++i) {
			dst[c * h * w + i] = result[i];
		}
	}
}

OaI32 MapBorderCoordinate(OaI32 coordinate, OaI32 size, OaBorderMode border) {
	if (coordinate >= 0 && coordinate < size) return coordinate;
	if (border == OaBorderMode::Constant) return -1;
	if (size == 1) return 0;
	if (border == OaBorderMode::Replicate) return std::clamp(coordinate, 0, size - 1);
	auto positiveModulo = [](OaI32 value, OaI32 modulus) {
		const OaI32 result = value % modulus;
		return result < 0 ? result + modulus : result;
	};
	if (border == OaBorderMode::Wrap) return positiveModulo(coordinate, size);
	if (border == OaBorderMode::Reflect) {
		const OaI32 period = size * 2;
		OaI32 mapped = positiveModulo(coordinate, period);
		return mapped >= size ? period - mapped - 1 : mapped;
	}
	const OaI32 period = (size - 1) * 2;
	OaI32 mapped = positiveModulo(coordinate, period);
	return mapped >= size ? period - mapped : mapped;
}

std::vector<float> CpuConvolve2d(const std::vector<float>& input,
	OaU32 batch, OaU32 channels, OaU32 height, OaU32 width,
	const std::vector<float>& kernel, OaU32 kernelHeight, OaU32 kernelWidth,
	OaBorderMode border, float borderValue = 0.0F) {
	std::vector<float> output(input.size());
	const OaI32 radiusY = static_cast<OaI32>(kernelHeight / 2);
	const OaI32 radiusX = static_cast<OaI32>(kernelWidth / 2);
	for (OaU32 b = 0; b < batch; ++b) {
		for (OaU32 c = 0; c < channels; ++c) {
			for (OaU32 y = 0; y < height; ++y) {
				for (OaU32 x = 0; x < width; ++x) {
					float sum = 0.0F;
					for (OaU32 ky = 0; ky < kernelHeight; ++ky) {
						for (OaU32 kx = 0; kx < kernelWidth; ++kx) {
							const OaI32 sy = MapBorderCoordinate(
								static_cast<OaI32>(y + ky) - radiusY,
								static_cast<OaI32>(height), border);
							const OaI32 sx = MapBorderCoordinate(
								static_cast<OaI32>(x + kx) - radiusX,
								static_cast<OaI32>(width), border);
							float sample = borderValue;
							if (sy >= 0 && sx >= 0) {
								const OaU32 index = b * channels * height * width +
									c * height * width + static_cast<OaU32>(sy) * width +
									static_cast<OaU32>(sx);
								sample = input[index];
							}
							sum += sample * kernel[ky * kernelWidth + kx];
						}
					}
					output[b * channels * height * width + c * height * width + y * width + x] = sum;
				}
			}
		}
	}
	return output;
}

std::vector<float> CpuMorphology(const std::vector<float>& input,
	OaU32 batch, OaU32 channels, OaU32 height, OaU32 width,
	OaU32 kernelHeight, OaU32 kernelWidth, OaBorderMode border,
	float borderValue, bool dilate) {
	std::vector<float> output(input.size());
	const OaI32 radiusY = static_cast<OaI32>(kernelHeight / 2);
	const OaI32 radiusX = static_cast<OaI32>(kernelWidth / 2);
	for (OaU32 b = 0; b < batch; ++b) {
		for (OaU32 c = 0; c < channels; ++c) {
			for (OaU32 y = 0; y < height; ++y) {
				for (OaU32 x = 0; x < width; ++x) {
					float result = dilate
						? std::numeric_limits<float>::lowest()
						: std::numeric_limits<float>::max();
					for (OaU32 ky = 0; ky < kernelHeight; ++ky) {
						for (OaU32 kx = 0; kx < kernelWidth; ++kx) {
							const OaI32 sy = MapBorderCoordinate(
								static_cast<OaI32>(y + ky) - radiusY,
								static_cast<OaI32>(height), border);
							const OaI32 sx = MapBorderCoordinate(
								static_cast<OaI32>(x + kx) - radiusX,
								static_cast<OaI32>(width), border);
							float sample = borderValue;
							if (sy >= 0 && sx >= 0) {
								const OaU32 index = b * channels * height * width +
									c * height * width + static_cast<OaU32>(sy) * width +
									static_cast<OaU32>(sx);
								sample = input[index];
							}
							result = dilate ? std::max(result, sample) : std::min(result, sample);
						}
					}
					output[b * channels * height * width +
						c * height * width + y * width + x] = result;
				}
			}
		}
	}
	return output;
}

OaMatrix MatrixFromValues(OaMatrixShape shape, const std::vector<float>& values) {
	auto matrix = OaFnMatrix::Empty(shape);
	for (OaI64 i = 0; i < static_cast<OaI64>(values.size()); ++i) matrix.Set(i, values[i]);
	return matrix;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class FilterKernels : public ::testing::Test {
protected:
	void SetUp() override {
		ASSERT_NE(OaRuntimeGlobal::GetRuntime(), nullptr);
	}

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
	
	// Helper: compute mean absolute error
	double ComputeMeanAbsError(const std::vector<float>& ref, const OaMatrix& gpu) {
		Materialize();
		double sumError = 0.0;
		OaI64 n = ref.size();
		for (OaI64 i = 0; i < n; ++i) {
			sumError += std::abs(gpu.At(i) - ref[i]);
		}
		return sumError / static_cast<double>(n);
	}
};

// ═══════════════════════════════════════════════════════════════════════════════
// Gaussian Blur Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(FilterKernels, GaussianBlur_SmallSigma) {
	const OaU32 h = 32;
	const OaU32 w = 32;
	const OaU32 channels = 3;
	const float sigma = 1.0F;
	
	// Create input with some structure (not just random noise)
	auto input = OaFnMatrix::Zeros({1, channels, h, w});
	// Add a bright spot in the center
	for (OaU32 c = 0; c < channels; ++c) {
		for (OaU32 y = h/2 - 4; y < h/2 + 4; ++y) {
			for (OaU32 x = w/2 - 4; x < w/2 + 4; ++x) {
				OaI64 idx = c * h * w + y * w + x;
				input.Set(idx, 1.0F);
			}
		}
	}
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (OaI64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.At(i);
	}
	std::vector<float> blurredRef(channels * h * w);
	CpuGaussianBlur(inputHost, blurredRef, h, w, channels, sigma);
	
	// GPU kernel
	auto blurredGpu = OaFnImage::GaussianBlur(input, sigma);
	
	// Verify shape
	EXPECT_EQ(blurredGpu.GetShape(), input.GetShape());
	
	// Verify values (allow some tolerance due to different implementations)
	double maxError = ComputeMaxAbsError(blurredRef, blurredGpu);
	double meanError = ComputeMeanAbsError(blurredRef, blurredGpu);
	
	EXPECT_LT(maxError, 0.05) << "MaxAbsError: " << maxError;
	EXPECT_LT(meanError, 0.01) << "MeanAbsError: " << meanError;
}

TEST_F(FilterKernels, GaussianBlur_ClampBorderMatchesCpuReference) {
	constexpr OaU32 h = 5;
	constexpr OaU32 w = 7;
	constexpr float sigma = 1.0F;
	auto input = OaFnMatrix::Zeros({1, 1, h, w});
	// Exercise both negative-coordinate borders. The old unsigned conversion
	// incorrectly wrapped those taps to the last row/column.
	input.Set(w - 1, 1.0F);
	input.Set((h - 1) * w, 0.5F);

	std::vector<float> host(h * w);
	for (OaU32 i = 0; i < host.size(); ++i) host[i] = input.At(i);
	std::vector<float> reference(h * w);
	CpuGaussianBlur(host, reference, h, w, 1, sigma);

	auto output = OaFnImage::GaussianBlur(input, sigma);
	EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, GaussianBlur_InvalidParametersAreNoOp) {
	auto input = OaFnMatrix::Ones({1, 1, 4, 4});
	EXPECT_EQ(OaFnImage::GaussianBlur(input, 0.0F).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::GaussianBlur(input, 1.0F, 4).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
}

TEST_F(FilterKernels, GaussianBlur_LargeSigma) {
	const OaU32 h = 64;
	const OaU32 w = 64;
	const OaU32 channels = 1;
	const float sigma = 3.0F;
	
	// Create checkerboard pattern
	auto input = OaFnMatrix::Zeros({1, channels, h, w});
	for (OaU32 y = 0; y < h; ++y) {
		for (OaU32 x = 0; x < w; ++x) {
			if ((x / 8 + y / 8) % 2 == 0) {
				OaI64 idx = y * w + x;
				input.Set(idx, 1.0F);
			}
		}
	}
	
	// GPU kernel
	auto blurred = OaFnImage::GaussianBlur(input, sigma);
	Materialize();
	
	// Verify shape
	EXPECT_EQ(blurred.GetShape(), input.GetShape());
	
	// Verify output is smoothed (no sharp edges)
	// Check that values are between 0 and 1 and not equal to input
	bool hasSmoothing = false;
	for (OaI64 i = 0; i < blurred.NumElements(); ++i) {
		float val = blurred.At(i);
		EXPECT_TRUE(std::isfinite(val));
		EXPECT_GE(val, 0.0F);
		EXPECT_LE(val, 1.0F);
		
		// Check if any value differs from input (indicating blur happened)
		if (std::abs(val - input.At(i)) > 0.01F) {
			hasSmoothing = true;
		}
	}
	EXPECT_TRUE(hasSmoothing) << "Blur should modify the image";
}

TEST_F(FilterKernels, GaussianBlur_MultiChannel) {
	const OaU32 h = 16;
	const OaU32 w = 16;
	const OaU32 channels = 3;
	const float sigma = 1.5F;
	
	// Create input with different patterns per channel
	auto input = OaFnMatrix::Zeros({1, channels, h, w});
	
	// Channel 0: horizontal stripes
	for (OaU32 y = 0; y < h; ++y) {
		if (y % 4 < 2) {
			for (OaU32 x = 0; x < w; ++x) {
				input.Set(y * w + x, 1.0F);
			}
		}
	}
	
	// Channel 1: vertical stripes
	for (OaU32 y = 0; y < h; ++y) {
		for (OaU32 x = 0; x < w; ++x) {
			if (x % 4 < 2) {
				input.Set(h * w + y * w + x, 1.0F);
			}
		}
	}
	
	// Channel 2: diagonal pattern
	for (OaU32 y = 0; y < h; ++y) {
		for (OaU32 x = 0; x < w; ++x) {
			if ((x + y) % 4 < 2) {
				input.Set(2 * h * w + y * w + x, 1.0F);
			}
		}
	}
	
	// GPU kernel
	auto blurred = OaFnImage::GaussianBlur(input, sigma);
	Materialize();
	
	// Verify shape
	EXPECT_EQ(blurred.GetShape(), input.GetShape());
	
	// Verify each channel is smoothed independently
	for (OaU32 c = 0; c < channels; ++c) {
		bool hasSmoothing = false;
		for (OaU32 i = 0; i < h * w; ++i) {
			OaI64 idx = c * h * w + i;
			float inVal = input.At(idx);
			float outVal = blurred.At(idx);
			
			EXPECT_TRUE(std::isfinite(outVal));
			EXPECT_GE(outVal, 0.0F);
			EXPECT_LE(outVal, 1.0F);
			
			if (std::abs(outVal - inVal) > 0.01F) {
				hasSmoothing = true;
			}
		}
		EXPECT_TRUE(hasSmoothing) << "Channel " << c << " should be smoothed";
	}
}

TEST_F(FilterKernels, GaussianBlur_PreservesRange) {
	const OaU32 h = 32;
	const OaU32 w = 32;
	const float sigma = 2.0F;
	
	// Create input with values in [0.2, 0.8]
	auto input = OaFnMatrix::Rand({1, 1, h, w});
	Materialize();
	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		float val = input.At(i);
		input.Set(i, 0.2F + 0.6F * val);
	}
	
	// GPU kernel
	auto blurred = OaFnImage::GaussianBlur(input, sigma);
	Materialize();
	
	// Verify output stays within reasonable range
	// (Gaussian blur is a weighted average, so output should be within input range)
	for (OaI64 i = 0; i < blurred.NumElements(); ++i) {
		float val = blurred.At(i);
		EXPECT_GE(val, 0.15F) << "Value too low at index " << i;
		EXPECT_LE(val, 0.85F) << "Value too high at index " << i;
	}
}

TEST_F(FilterKernels, Convolve2d_AllBorderModesMatchCpuOracle) {
	const std::vector<float> inputValues{1, 2, 3, 4, 5, 6};
	const std::vector<float> kernelValues{1, 2, 0, -1, 3, 1, 2, 0, -2};
	auto input = MatrixFromValues({1, 1, 2, 3}, inputValues);
	auto kernel = MatrixFromValues({3, 3}, kernelValues);
	for (const OaBorderMode border : {
		OaBorderMode::Constant, OaBorderMode::Replicate, OaBorderMode::Reflect,
		OaBorderMode::Reflect101, OaBorderMode::Wrap}) {
		const float borderValue = border == OaBorderMode::Constant ? 7.0F : 0.0F;
		auto reference = CpuConvolve2d(
			inputValues, 1, 1, 2, 3, kernelValues, 3, 3, border, borderValue);
		auto output = OaFnImage::Convolve2d(input, kernel, border, borderValue);
		EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5)
			<< "border=" << static_cast<OaU32>(border);
	}
}

TEST_F(FilterKernels, Convolve2d_OnePixelBordersAreDefined) {
	const std::vector<float> inputValues{2.0F};
	const std::vector<float> kernelValues(9, 1.0F);
	auto input = MatrixFromValues({1, 1, 1, 1}, inputValues);
	auto kernel = MatrixFromValues({3, 3}, kernelValues);
	for (const OaBorderMode border : {
		OaBorderMode::Replicate, OaBorderMode::Reflect,
		OaBorderMode::Reflect101, OaBorderMode::Wrap}) {
		auto output = OaFnImage::Convolve2d(input, kernel, border);
		EXPECT_LT(ComputeMaxAbsError({18.0F}, output), 1.0e-5);
	}
}

TEST_F(FilterKernels, SeparableConvolve2d_MatchesFullOuterProduct) {
	const std::vector<float> inputValues{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15};
	const std::vector<float> kernelX{1, 2, 1};
	const std::vector<float> kernelY{-1, 0, 1};
	std::vector<float> fullKernel(9);
	for (OaU32 y = 0; y < 3; ++y)
		for (OaU32 x = 0; x < 3; ++x) fullKernel[y * 3 + x] = kernelY[y] * kernelX[x];
	auto input = MatrixFromValues({1, 1, 3, 5}, inputValues);
	auto kx = MatrixFromValues({3}, kernelX);
	auto ky = MatrixFromValues({3}, kernelY);
	auto reference = CpuConvolve2d(
		inputValues, 1, 1, 3, 5, fullKernel, 3, 3, OaBorderMode::Reflect101);
	auto output = OaFnImage::SeparableConvolve2d(
		input, kx, ky, OaBorderMode::Reflect101);
	EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, SemanticFiltersMatchIndependentKernels) {
	const std::vector<float> inputValues{
		0, 1, 4, 2,
		3, 7, 5, 1,
		8, 6, 9, 0};
	auto input = MatrixFromValues({1, 1, 3, 4}, inputValues);
	struct Case {
		std::vector<float> Kernel;
		OaMatrix (*Apply)(const OaMatrix&);
	};
	const auto sobelX = [](const OaMatrix& image) { return OaFnImage::Sobel(image, 1, 0); };
	const auto sobelY = [](const OaMatrix& image) { return OaFnImage::Sobel(image, 0, 1); };
	const auto scharrX = [](const OaMatrix& image) { return OaFnImage::Scharr(image, 1, 0); };
	const auto laplacian = [](const OaMatrix& image) { return OaFnImage::Laplacian(image); };
	const Case cases[] = {
		{{-1, 0, 1, -2, 0, 2, -1, 0, 1}, sobelX},
		{{-1, -2, -1, 0, 0, 0, 1, 2, 1}, sobelY},
		{{-3, 0, 3, -10, 0, 10, -3, 0, 3}, scharrX},
		{{0, 1, 0, 1, -4, 1, 0, 1, 0}, laplacian},
	};
	for (const auto& testCase : cases) {
		auto reference = CpuConvolve2d(inputValues, 1, 1, 3, 4,
			testCase.Kernel, 3, 3, OaBorderMode::Reflect101);
		auto output = testCase.Apply(input);
		EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5);
	}
}

TEST_F(FilterKernels, AverageBlur_MatchesCpuAndDeferredProducer) {
	auto base = OaFnMatrix::Full({2, 2, 4, 5}, 0.25F);
	auto produced = OaFnMatrix::Add(base, base);
	auto output = OaFnImage::AverageBlur(
		produced, 3, 3, OaBorderMode::Reflect101);
	std::vector<float> reference(static_cast<size_t>(output.NumElements()), 0.5F);
	EXPECT_LT(ComputeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, ErodeAndDilate_AllBordersMatchCpuOracle) {
	const std::vector<float> values{1, 7, 3, -2, 5, 4};
	auto input = MatrixFromValues({1, 1, 2, 3}, values);
	for (const OaBorderMode border : {
		OaBorderMode::Constant, OaBorderMode::Replicate, OaBorderMode::Reflect,
		OaBorderMode::Reflect101, OaBorderMode::Wrap}) {
		const float borderValue = border == OaBorderMode::Constant ? 2.5F : 0.0F;
		auto erodeRef = CpuMorphology(
			values, 1, 1, 2, 3, 3, 3, border, borderValue, false);
		auto dilateRef = CpuMorphology(
			values, 1, 1, 2, 3, 3, 3, border, borderValue, true);
		auto eroded = OaFnImage::Erode(input, 3, 3, border, borderValue);
		auto dilated = OaFnImage::Dilate(input, 3, 3, border, borderValue);
		EXPECT_LT(ComputeMaxAbsError(erodeRef, eroded), 1.0e-5);
		EXPECT_LT(ComputeMaxAbsError(dilateRef, dilated), 1.0e-5);
	}
}

TEST_F(FilterKernels, MorphologyCompositionsMatchIndependentCpuOracle) {
	const std::vector<float> values{
		0, 0, 0, 0, 0,
		0, 3, 1, 4, 0,
		0, 2, 9, 2, 0,
		0, 5, 1, 3, 0,
		0, 0, 0, 0, 0};
	auto input = MatrixFromValues({1, 1, 5, 5}, values);
	const auto eroded = CpuMorphology(
		values, 1, 1, 5, 5, 3, 3, OaBorderMode::Reflect101, 0.0F, false);
	const auto dilated = CpuMorphology(
		values, 1, 1, 5, 5, 3, 3, OaBorderMode::Reflect101, 0.0F, true);
	const auto opened = CpuMorphology(
		eroded, 1, 1, 5, 5, 3, 3, OaBorderMode::Reflect101, 0.0F, true);
	const auto closed = CpuMorphology(
		dilated, 1, 1, 5, 5, 3, 3, OaBorderMode::Reflect101, 0.0F, false);
	std::vector<float> gradient(values.size());
	for (OaUsize i = 0; i < gradient.size(); ++i) gradient[i] = dilated[i] - eroded[i];
	EXPECT_LT(ComputeMaxAbsError(opened, OaFnImage::MorphologyOpen(input)), 1.0e-5);
	EXPECT_LT(ComputeMaxAbsError(closed, OaFnImage::MorphologyClose(input)), 1.0e-5);
	EXPECT_LT(ComputeMaxAbsError(gradient, OaFnImage::MorphologyGradient(input)), 1.0e-5);
}

TEST_F(FilterKernels, InvalidFilterParametersAreNoOp) {
	auto input = OaFnMatrix::Ones({1, 1, 4, 4});
	auto evenKernel = OaFnMatrix::Ones({2, 2});
	auto wrongDtype = OaFnMatrix::Ones({3, 3}, OaScalarType::BFloat16);
	EXPECT_EQ(OaFnImage::Convolve2d(input, evenKernel).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::Convolve2d(input, wrongDtype).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::AverageBlur(input, 4, 3).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::Sobel(input, 1, 1).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::Erode(input, 2, 3).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::Dilate(input, 3, 33).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
}
