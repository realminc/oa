// Test Vision filter kernels (GaussianBlur, etc.)
// Verifies GPU kernel output against CPU reference implementations

#include "../../../oaTest.h"
#include <oa/core/matrixAccess.h>
#include <oa/vision.h>
#include <cmath>
#include <limits>

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════════

// generate 1D Gaussian kernel
std::vector<float> generateGaussianKernel(oa::U32 radius, float sigma) {
	oa::U32 size = 2 * radius + 1;
	std::vector<float> kernel(size);
	float sum = 0.0F;
	
	for (oa::U32 i = 0; i < size; ++i) {
		float x = static_cast<float>(static_cast<oa::I32>(i) - static_cast<oa::I32>(radius));
		kernel[i] = std::exp(-(x * x) / (2.0F * sigma * sigma));
		sum += kernel[i];
	}
	
	// normalize
	for (oa::U32 i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}
	
	return kernel;
}

// apply 1D Gaussian blur (horizontal or vertical)
void applyGaussian1D(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 h, oa::U32 w, const std::vector<float>& kernel, bool horizontal) {
	oa::U32 radius = static_cast<oa::U32>(kernel.size()) / 2;
	
	for (oa::U32 y = 0; y < h; ++y) {
		for (oa::U32 x = 0; x < w; ++x) {
			float sum = 0.0F;
			
			for (oa::U32 k = 0; k < kernel.size(); ++k) {
				oa::I32 offset = static_cast<oa::I32>(k) - static_cast<oa::I32>(radius);
				oa::I32 sampleX = horizontal ? (static_cast<oa::I32>(x) + offset) : static_cast<oa::I32>(x);
				oa::I32 sampleY = horizontal ? static_cast<oa::I32>(y) : (static_cast<oa::I32>(y) + offset);
				
				// Clamp to image bounds
				sampleX = std::max(0, std::min(sampleX, static_cast<oa::I32>(w) - 1));
				sampleY = std::max(0, std::min(sampleY, static_cast<oa::I32>(h) - 1));
				
				oa::U32 idx = static_cast<oa::U32>(sampleY) * w + static_cast<oa::U32>(sampleX);
				sum += src[idx] * kernel[k];
			}
			
			dst[y * w + x] = sum;
		}
	}
}

// CPU Gaussian blur (separable: horizontal then vertical)
void cpuGaussianBlur(const std::vector<float>& src, std::vector<float>& dst,
	oa::U32 h, oa::U32 w, oa::U32 channels, float sigma) {
	// Use radius = ceil(3*sigma) for good coverage
	oa::U32 radius = static_cast<oa::U32>(std::ceil(3.0F * sigma));
	auto kernel = generateGaussianKernel(radius, sigma);
	
	std::vector<float> temp(h * w);
	
	for (oa::U32 c = 0; c < channels; ++c) {
		// Extract channel
		std::vector<float> channel(h * w);
		for (oa::U32 i = 0; i < h * w; ++i) {
			channel[i] = src[c * h * w + i];
		}
		
		// Horizontal pass
		applyGaussian1D(channel, temp, h, w, kernel, true);
		
		// Vertical pass
		std::vector<float> result(h * w);
		applyGaussian1D(temp, result, h, w, kernel, false);
		
		// store result
		for (oa::U32 i = 0; i < h * w; ++i) {
			dst[c * h * w + i] = result[i];
		}
	}
}

oa::I32 mapBorderCoordinate(oa::I32 coordinate, oa::I32 size, oa::BorderMode border) {
	if (coordinate >= 0 && coordinate < size) return coordinate;
	if (border == oa::BorderMode::Constant) return -1;
	if (size == 1) return 0;
	if (border == oa::BorderMode::Replicate) return std::clamp(coordinate, 0, size - 1);
	auto positiveModulo = [](oa::I32 value, oa::I32 modulus) {
		const oa::I32 result = value % modulus;
		return result < 0 ? result + modulus : result;
	};
	if (border == oa::BorderMode::Wrap) return positiveModulo(coordinate, size);
	if (border == oa::BorderMode::Reflect) {
		const oa::I32 period = size * 2;
		oa::I32 mapped = positiveModulo(coordinate, period);
		return mapped >= size ? period - mapped - 1 : mapped;
	}
	const oa::I32 period = (size - 1) * 2;
	oa::I32 mapped = positiveModulo(coordinate, period);
	return mapped >= size ? period - mapped : mapped;
}

std::vector<float> cpuConvolve2d(const std::vector<float>& input,
	oa::U32 batch, oa::U32 channels, oa::U32 height, oa::U32 width,
	const std::vector<float>& kernel, oa::U32 kernelHeight, oa::U32 kernelWidth,
	oa::BorderMode border, float borderValue = 0.0F) {
	std::vector<float> output(input.size());
	const oa::I32 radiusY = static_cast<oa::I32>(kernelHeight / 2);
	const oa::I32 radiusX = static_cast<oa::I32>(kernelWidth / 2);
	for (oa::U32 b = 0; b < batch; ++b) {
		for (oa::U32 c = 0; c < channels; ++c) {
			for (oa::U32 y = 0; y < height; ++y) {
				for (oa::U32 x = 0; x < width; ++x) {
					float sum = 0.0F;
					for (oa::U32 ky = 0; ky < kernelHeight; ++ky) {
						for (oa::U32 kx = 0; kx < kernelWidth; ++kx) {
							const oa::I32 sy = mapBorderCoordinate(
								static_cast<oa::I32>(y + ky) - radiusY,
								static_cast<oa::I32>(height), border);
							const oa::I32 sx = mapBorderCoordinate(
								static_cast<oa::I32>(x + kx) - radiusX,
								static_cast<oa::I32>(width), border);
							float sample = borderValue;
							if (sy >= 0 && sx >= 0) {
								const oa::U32 index = b * channels * height * width +
									c * height * width + static_cast<oa::U32>(sy) * width +
									static_cast<oa::U32>(sx);
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

std::vector<float> cpuMorphology(const std::vector<float>& input,
	oa::U32 batch, oa::U32 channels, oa::U32 height, oa::U32 width,
	oa::U32 kernelHeight, oa::U32 kernelWidth, oa::BorderMode border,
	float borderValue, bool dilate) {
	std::vector<float> output(input.size());
	const oa::I32 radiusY = static_cast<oa::I32>(kernelHeight / 2);
	const oa::I32 radiusX = static_cast<oa::I32>(kernelWidth / 2);
	for (oa::U32 b = 0; b < batch; ++b) {
		for (oa::U32 c = 0; c < channels; ++c) {
			for (oa::U32 y = 0; y < height; ++y) {
				for (oa::U32 x = 0; x < width; ++x) {
					float result = dilate
						? std::numeric_limits<float>::lowest()
						: std::numeric_limits<float>::max();
					for (oa::U32 ky = 0; ky < kernelHeight; ++ky) {
						for (oa::U32 kx = 0; kx < kernelWidth; ++kx) {
							const oa::I32 sy = mapBorderCoordinate(
								static_cast<oa::I32>(y + ky) - radiusY,
								static_cast<oa::I32>(height), border);
							const oa::I32 sx = mapBorderCoordinate(
								static_cast<oa::I32>(x + kx) - radiusX,
								static_cast<oa::I32>(width), border);
							float sample = borderValue;
							if (sy >= 0 && sx >= 0) {
								const oa::U32 index = b * channels * height * width +
									c * height * width + static_cast<oa::U32>(sy) * width +
									static_cast<oa::U32>(sx);
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

oa::Matrix matrixFromValues(oa::MatrixShape shape, const std::vector<float>& values) {
	auto matrix = oa::FnMatrix::empty(shape);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(values.size()); ++i) matrix.set(i, values[i]);
	return matrix;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class FilterKernels : public ::testing::Test {
protected:
	void SetUp() override {
		ASSERT_NE(testEnginePtr(), nullptr);
	}

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
	
	// Helper: compute mean absolute error
	double computeMeanAbsError(const std::vector<float>& ref, const oa::Matrix& gpu) {
		materialize();
		double sumError = 0.0;
		oa::I64 n = ref.size();
		for (oa::I64 i = 0; i < n; ++i) {
			sumError += std::abs(gpu.at(i) - ref[i]);
		}
		return sumError / static_cast<double>(n);
	}
};

// ═══════════════════════════════════════════════════════════════════════════════
// Gaussian Blur Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(FilterKernels, GaussianBlur_SmallSigma) {
	const oa::U32 h = 32;
	const oa::U32 w = 32;
	const oa::U32 channels = 3;
	const float sigma = 1.0F;
	
	// Create input with some structure (not just random noise)
	auto input = oa::FnMatrix::zeros({1, channels, h, w});
	// Add a bright spot in the center
	for (oa::U32 c = 0; c < channels; ++c) {
		for (oa::U32 y = h/2 - 4; y < h/2 + 4; ++y) {
			for (oa::U32 x = w/2 - 4; x < w/2 + 4; ++x) {
				oa::I64 idx = c * h * w + y * w + x;
				input.set(idx, 1.0F);
			}
		}
	}
	
	// CPU reference
	std::vector<float> inputHost(channels * h * w);
	for (oa::I64 i = 0; i < channels * h * w; ++i) {
		inputHost[i] = input.at(i);
	}
	std::vector<float> blurredRef(channels * h * w);
	cpuGaussianBlur(inputHost, blurredRef, h, w, channels, sigma);
	
	// GPU kernel
	auto blurredGpu = oa::FnImage::gaussianBlur(input, sigma);
	
	// verify shape
	EXPECT_EQ(blurredGpu.getShape(), input.getShape());
	
	// verify values (allow some tolerance due to different implementations)
	double maxError = computeMaxAbsError(blurredRef, blurredGpu);
	double meanError = computeMeanAbsError(blurredRef, blurredGpu);
	
	EXPECT_LT(maxError, 0.05) << "MaxAbsError: " << maxError;
	EXPECT_LT(meanError, 0.01) << "MeanAbsError: " << meanError;
}

TEST_F(FilterKernels, GaussianBlur_ClampBorderMatchesCpuReference) {
	constexpr oa::U32 h = 5;
	constexpr oa::U32 w = 7;
	constexpr float sigma = 1.0F;
	auto input = oa::FnMatrix::zeros({1, 1, h, w});
	// Exercise both negative-coordinate borders. The old unsigned conversion
	// incorrectly wrapped those taps to the last row/column.
	input.set(w - 1, 1.0F);
	input.set((h - 1) * w, 0.5F);

	std::vector<float> host(h * w);
	for (oa::U32 i = 0; i < host.size(); ++i) host[i] = input.at(i);
	std::vector<float> reference(h * w);
	cpuGaussianBlur(host, reference, h, w, 1, sigma);

	auto output = oa::FnImage::gaussianBlur(input, sigma);
	EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, GaussianBlur_InvalidParametersAreNoOp) {
	auto input = oa::FnMatrix::ones({1, 1, 4, 4});
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::gaussianBlur(input, 0.0F)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::gaussianBlur(input, 1.0F, 4)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
}

TEST_F(FilterKernels, GaussianBlur_LargeSigma) {
	const oa::U32 h = 64;
	const oa::U32 w = 64;
	const oa::U32 channels = 1;
	const float sigma = 3.0F;
	
	// Create checkerboard pattern
	auto input = oa::FnMatrix::zeros({1, channels, h, w});
	for (oa::U32 y = 0; y < h; ++y) {
		for (oa::U32 x = 0; x < w; ++x) {
			if ((x / 8 + y / 8) % 2 == 0) {
				oa::I64 idx = y * w + x;
				input.set(idx, 1.0F);
			}
		}
	}
	
	// GPU kernel
	auto blurred = oa::FnImage::gaussianBlur(input, sigma);
	materialize();
	
	// verify shape
	EXPECT_EQ(blurred.getShape(), input.getShape());
	
	// verify output is smoothed (no sharp edges)
	// Check that values are between 0 and 1 and not equal to input
	bool hasSmoothing = false;
	for (oa::I64 i = 0; i < blurred.numElements(); ++i) {
		float val = blurred.at(i);
		EXPECT_TRUE(std::isfinite(val));
		EXPECT_GE(val, 0.0F);
		EXPECT_LE(val, 1.0F);
		
		// Check if any value differs from input (indicating blur happened)
		if (std::abs(val - input.at(i)) > 0.01F) {
			hasSmoothing = true;
		}
	}
	EXPECT_TRUE(hasSmoothing) << "Blur should modify the image";
}

TEST_F(FilterKernels, GaussianBlur_MultiChannel) {
	const oa::U32 h = 16;
	const oa::U32 w = 16;
	const oa::U32 channels = 3;
	const float sigma = 1.5F;
	
	// Create input with different patterns per channel
	auto input = oa::FnMatrix::zeros({1, channels, h, w});
	
	// Channel 0: horizontal stripes
	for (oa::U32 y = 0; y < h; ++y) {
		if (y % 4 < 2) {
			for (oa::U32 x = 0; x < w; ++x) {
				input.set(y * w + x, 1.0F);
			}
		}
	}
	
	// Channel 1: vertical stripes
	for (oa::U32 y = 0; y < h; ++y) {
		for (oa::U32 x = 0; x < w; ++x) {
			if (x % 4 < 2) {
				input.set(h * w + y * w + x, 1.0F);
			}
		}
	}
	
	// Channel 2: diagonal pattern
	for (oa::U32 y = 0; y < h; ++y) {
		for (oa::U32 x = 0; x < w; ++x) {
			if ((x + y) % 4 < 2) {
				input.set(2 * h * w + y * w + x, 1.0F);
			}
		}
	}
	
	// GPU kernel
	auto blurred = oa::FnImage::gaussianBlur(input, sigma);
	materialize();
	
	// verify shape
	EXPECT_EQ(blurred.getShape(), input.getShape());
	
	// verify each channel is smoothed independently
	for (oa::U32 c = 0; c < channels; ++c) {
		bool hasSmoothing = false;
		for (oa::U32 i = 0; i < h * w; ++i) {
			oa::I64 idx = c * h * w + i;
			float inVal = input.at(idx);
			float outVal = blurred.at(idx);
			
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
	const oa::U32 h = 32;
	const oa::U32 w = 32;
	const float sigma = 2.0F;
	
	// Create input with values in [0.2, 0.8]
	auto input = oa::FnMatrix::rand({1, 1, h, w});
	materialize();
	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		float val = input.at(i);
		input.set(i, 0.2F + 0.6F * val);
	}
	
	// GPU kernel
	auto blurred = oa::FnImage::gaussianBlur(input, sigma);
	materialize();
	
	// verify output stays within reasonable range
	// (Gaussian blur is a weighted average, so output should be within input range)
	for (oa::I64 i = 0; i < blurred.numElements(); ++i) {
		float val = blurred.at(i);
		EXPECT_GE(val, 0.15F) << "Value too low at index " << i;
		EXPECT_LE(val, 0.85F) << "Value too high at index " << i;
	}
}

TEST_F(FilterKernels, Convolve2d_AllBorderModesMatchCpuOracle) {
	const std::vector<float> inputValues{1, 2, 3, 4, 5, 6};
	const std::vector<float> kernelValues{1, 2, 0, -1, 3, 1, 2, 0, -2};
	auto input = matrixFromValues({1, 1, 2, 3}, inputValues);
	auto kernel = matrixFromValues({3, 3}, kernelValues);
	for (const oa::BorderMode border : {
		oa::BorderMode::Constant, oa::BorderMode::Replicate, oa::BorderMode::Reflect,
		oa::BorderMode::Reflect101, oa::BorderMode::Wrap}) {
		const float borderValue = border == oa::BorderMode::Constant ? 7.0F : 0.0F;
		auto reference = cpuConvolve2d(
			inputValues, 1, 1, 2, 3, kernelValues, 3, 3, border, borderValue);
		auto output = oa::FnImage::convolve2d(input, kernel, border, borderValue);
		EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5)
			<< "border=" << static_cast<oa::U32>(border);
	}
}

TEST_F(FilterKernels, Convolve2d_OnePixelBordersAreDefined) {
	const std::vector<float> inputValues{2.0F};
	const std::vector<float> kernelValues(9, 1.0F);
	auto input = matrixFromValues({1, 1, 1, 1}, inputValues);
	auto kernel = matrixFromValues({3, 3}, kernelValues);
	for (const oa::BorderMode border : {
		oa::BorderMode::Replicate, oa::BorderMode::Reflect,
		oa::BorderMode::Reflect101, oa::BorderMode::Wrap}) {
		auto output = oa::FnImage::convolve2d(input, kernel, border);
		EXPECT_LT(computeMaxAbsError({18.0F}, output), 1.0e-5);
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
	for (oa::U32 y = 0; y < 3; ++y)
		for (oa::U32 x = 0; x < 3; ++x) fullKernel[y * 3 + x] = kernelY[y] * kernelX[x];
	auto input = matrixFromValues({1, 1, 3, 5}, inputValues);
	auto kx = matrixFromValues({3}, kernelX);
	auto ky = matrixFromValues({3}, kernelY);
	auto reference = cpuConvolve2d(
		inputValues, 1, 1, 3, 5, fullKernel, 3, 3, oa::BorderMode::Reflect101);
	auto output = oa::FnImage::separableConvolve2d(
		input, kx, ky, oa::BorderMode::Reflect101);
	EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, SemanticFiltersMatchIndependentKernels) {
	const std::vector<float> inputValues{
		0, 1, 4, 2,
		3, 7, 5, 1,
		8, 6, 9, 0};
	auto input = matrixFromValues({1, 1, 3, 4}, inputValues);
	struct Case {
		std::vector<float> kernel;
		oa::Matrix (*apply)(const oa::Matrix&);
	};
	const auto sobelX = [](const oa::Matrix& image) { return oa::FnImage::sobel(image, 1, 0); };
	const auto sobelY = [](const oa::Matrix& image) { return oa::FnImage::sobel(image, 0, 1); };
	const auto scharrX = [](const oa::Matrix& image) { return oa::FnImage::scharr(image, 1, 0); };
	const auto laplacian = [](const oa::Matrix& image) { return oa::FnImage::laplacian(image); };
	const Case cases[] = {
		{{-1, 0, 1, -2, 0, 2, -1, 0, 1}, sobelX},
		{{-1, -2, -1, 0, 0, 0, 1, 2, 1}, sobelY},
		{{-3, 0, 3, -10, 0, 10, -3, 0, 3}, scharrX},
		{{0, 1, 0, 1, -4, 1, 0, 1, 0}, laplacian},
	};
	for (const auto& testCase : cases) {
		auto reference = cpuConvolve2d(inputValues, 1, 1, 3, 4,
			testCase.kernel, 3, 3, oa::BorderMode::Reflect101);
		auto output = testCase.apply(input);
		EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5);
	}
}

TEST_F(FilterKernels, AverageBlur_MatchesCpuAndDeferredProducer) {
	auto base = oa::FnMatrix::full({2, 2, 4, 5}, 0.25F);
	auto produced = oa::FnMatrix::add(base, base);
	auto output = oa::FnImage::averageBlur(
		produced, 3, 3, oa::BorderMode::Reflect101);
	std::vector<float> reference(static_cast<size_t>(output.numElements()), 0.5F);
	EXPECT_LT(computeMaxAbsError(reference, output), 1.0e-5);
}

TEST_F(FilterKernels, ErodeAndDilate_AllBordersMatchCpuOracle) {
	const std::vector<float> values{1, 7, 3, -2, 5, 4};
	auto input = matrixFromValues({1, 1, 2, 3}, values);
	for (const oa::BorderMode border : {
		oa::BorderMode::Constant, oa::BorderMode::Replicate, oa::BorderMode::Reflect,
		oa::BorderMode::Reflect101, oa::BorderMode::Wrap}) {
		const float borderValue = border == oa::BorderMode::Constant ? 2.5F : 0.0F;
		auto erodeRef = cpuMorphology(
			values, 1, 1, 2, 3, 3, 3, border, borderValue, false);
		auto dilateRef = cpuMorphology(
			values, 1, 1, 2, 3, 3, 3, border, borderValue, true);
		auto eroded = oa::FnImage::erode(input, 3, 3, border, borderValue);
		auto dilated = oa::FnImage::dilate(input, 3, 3, border, borderValue);
		EXPECT_LT(computeMaxAbsError(erodeRef, eroded), 1.0e-5);
		EXPECT_LT(computeMaxAbsError(dilateRef, dilated), 1.0e-5);
	}
}

TEST_F(FilterKernels, MorphologyCompositionsMatchIndependentCpuOracle) {
	const std::vector<float> values{
		0, 0, 0, 0, 0,
		0, 3, 1, 4, 0,
		0, 2, 9, 2, 0,
		0, 5, 1, 3, 0,
		0, 0, 0, 0, 0};
	auto input = matrixFromValues({1, 1, 5, 5}, values);
	const auto eroded = cpuMorphology(
		values, 1, 1, 5, 5, 3, 3, oa::BorderMode::Reflect101, 0.0F, false);
	const auto dilated = cpuMorphology(
		values, 1, 1, 5, 5, 3, 3, oa::BorderMode::Reflect101, 0.0F, true);
	const auto opened = cpuMorphology(
		eroded, 1, 1, 5, 5, 3, 3, oa::BorderMode::Reflect101, 0.0F, true);
	const auto closed = cpuMorphology(
		dilated, 1, 1, 5, 5, 3, 3, oa::BorderMode::Reflect101, 0.0F, false);
	std::vector<float> gradient(values.size());
	for (oa::Usize i = 0; i < gradient.size(); ++i) gradient[i] = dilated[i] - eroded[i];
	EXPECT_LT(computeMaxAbsError(opened, oa::FnImage::morphologyOpen(input)), 1.0e-5);
	EXPECT_LT(computeMaxAbsError(closed, oa::FnImage::morphologyClose(input)), 1.0e-5);
	EXPECT_LT(computeMaxAbsError(gradient, oa::FnImage::morphologyGradient(input)), 1.0e-5);
}

TEST_F(FilterKernels, InvalidFilterParametersAreNoOp) {
	auto input = oa::FnMatrix::ones({1, 1, 4, 4});
	auto evenKernel = oa::FnMatrix::ones({2, 2});
	auto wrongDtype = oa::FnMatrix::ones({3, 3}, oa::ScalarType::BFloat16);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::convolve2d(input, evenKernel)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::convolve2d(input, wrongDtype)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::averageBlur(input, 4, 3)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::sobel(input, 1, 1)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::erode(input, 2, 3)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::dilate(input, 3, 33)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
}
