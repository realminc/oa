// Test/Ml/FnMatrix/Conv/TestFnMatrixConvBwd.cpp
// Tests for Conv2d backward passes (Conv2dBwdData, Conv2dBwdWeight)

#include <gtest/gtest.h>
#include <Oa/Core.h>
#include <Oa/Ml.h>
#include <vector>
#include <cmath>

// Helper to create matrix from host data
static OaMatrix CreateMatrixFromHost(const std::vector<float>& data, const OaMatrixShape& shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Helper to copy matrix to host
static std::vector<float> CopyMatrixToHost(const OaMatrix& mat) {
	std::vector<float> result(mat.NumElements());
	[[maybe_unused]] auto copy_result = OaFnMatrix::CopyToHost(mat, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to check if all values are finite
static void OaExpectFinite(const std::vector<float>& data, const char* name) {
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_TRUE(std::isfinite(data[i])) << name << " contains non-finite value at index " << i;
	}
}

class ConvBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize runtime if needed
	}
};

// ============================================================================
// Conv2dBwdData Tests (gradient w.r.t. input)
// ============================================================================

TEST_VK(ConvBwd, Conv2dBwdDataBasic) {
	// Test basic Conv2dBwdData: gradient w.r.t. input
	// Forward: input [1,1,4,4] * weight [1,1,3,3] -> output [1,1,2,2]
	// Backward: grad_output [1,1,2,2] -> grad_input [1,1,4,4]
	
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);  // 3x3 kernel of ones
	
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto grad_input = OaFnMatrix::Conv2dBwdData(
		grad_output, weight, 
		1,  // stride
		0,  // padding
		OaMatrixShape{1, 1, 4, 4},  // input_shape
		1   // groups
	);
	
	auto result = CopyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 16);  // 4x4
	EXPECT_EQ(grad_input.GetShape()[0], 1);
	EXPECT_EQ(grad_input.GetShape()[1], 1);
	EXPECT_EQ(grad_input.GetShape()[2], 4);
	EXPECT_EQ(grad_input.GetShape()[3], 4);
	
	OaExpectFinite(result, "Conv2dBwdData output");
}

TEST_VK(ConvBwd, Conv2dBwdDataWithPadding) {
	// Test Conv2dBwdData with padding
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> weight_data(9, 0.5f);
	
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto grad_input = OaFnMatrix::Conv2dBwdData(
		grad_output, weight,
		1,  // stride
		1,  // padding
		OaMatrixShape{1, 1, 2, 2},  // input_shape (same as output due to padding)
		1   // groups
	);
	
	auto result = CopyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 4);
	OaExpectFinite(result, "Conv2dBwdData with padding");
}

TEST_VK(ConvBwd, Conv2dBwdDataWithStride) {
	// Test Conv2dBwdData with stride > 1
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);
	
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto grad_input = OaFnMatrix::Conv2dBwdData(
		grad_output, weight,
		2,  // stride
		0,  // padding
		OaMatrixShape{1, 1, 6, 6},  // input_shape
		1   // groups
	);
	
	auto result = CopyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 36);  // 6x6
	OaExpectFinite(result, "Conv2dBwdData with stride");
}

TEST_VK(ConvBwd, Conv2dBwdDataMultiChannel) {
	// Test Conv2dBwdData with multiple channels
	const OaI32 in_channels = 3;
	const OaI32 out_channels = 2;
	
	std::vector<float> grad_output_data(out_channels * 2 * 2, 1.0f);
	std::vector<float> weight_data(out_channels * in_channels * 3 * 3, 0.5f);
	
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, out_channels, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{out_channels, in_channels, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto grad_input = OaFnMatrix::Conv2dBwdData(
		grad_output, weight,
		1,  // stride
		0,  // padding
		OaMatrixShape{1, in_channels, 4, 4},
		1   // groups
	);
	
	auto result = CopyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), in_channels * 4 * 4);
	EXPECT_EQ(grad_input.GetShape()[1], in_channels);
	OaExpectFinite(result, "Conv2dBwdData multi-channel");
}

// ============================================================================
// Conv2dBwdWeight Tests (gradient w.r.t. weight and bias)
// ============================================================================

TEST_VK(ConvBwd, Conv2dBwdWeightBasic) {
	// Test basic Conv2dBwdWeight: gradients w.r.t. weight and bias
	std::vector<float> input_data(16);
	for (OaI32 i = 0; i < 16; ++i) input_data[i] = static_cast<float>(i + 1);
	
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);  // For shape reference
	
	auto input = CreateMatrixFromHost(input_data, OaMatrixShape{1, 1, 4, 4});
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::Conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		0,  // padding
		1   // groups
	);
	
	auto grad_weight = CopyMatrixToHost(bwd_result.GradWeight);
	auto grad_bias = CopyMatrixToHost(bwd_result.GradBias);
	
	ASSERT_EQ(grad_weight.size(), 9);  // 3x3 kernel
	ASSERT_EQ(grad_bias.size(), 1);    // 1 output channel
	
	EXPECT_EQ(bwd_result.GradWeight.GetShape()[0], 1);
	EXPECT_EQ(bwd_result.GradWeight.GetShape()[1], 1);
	EXPECT_EQ(bwd_result.GradWeight.GetShape()[2], 3);
	EXPECT_EQ(bwd_result.GradWeight.GetShape()[3], 3);
	
	OaExpectFinite(grad_weight, "Conv2dBwdWeight grad_weight");
	OaExpectFinite(grad_bias, "Conv2dBwdWeight grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightMultiChannel) {
	// Test Conv2dBwdWeight with multiple channels
	const OaI32 in_channels = 2;
	const OaI32 out_channels = 3;
	
	std::vector<float> input_data(in_channels * 4 * 4, 1.0f);
	std::vector<float> grad_output_data(out_channels * 2 * 2, 1.0f);
	std::vector<float> weight_data(out_channels * in_channels * 3 * 3, 1.0f);
	
	auto input = CreateMatrixFromHost(input_data, OaMatrixShape{1, in_channels, 4, 4});
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, out_channels, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{out_channels, in_channels, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::Conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		0,  // padding
		1   // groups
	);
	
	auto grad_weight = CopyMatrixToHost(bwd_result.GradWeight);
	auto grad_bias = CopyMatrixToHost(bwd_result.GradBias);
	
	ASSERT_EQ(grad_weight.size(), out_channels * in_channels * 3 * 3);
	ASSERT_EQ(grad_bias.size(), out_channels);
	
	OaExpectFinite(grad_weight, "Conv2dBwdWeight multi-channel grad_weight");
	OaExpectFinite(grad_bias, "Conv2dBwdWeight multi-channel grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightWithPadding) {
	// Test Conv2dBwdWeight with padding
	std::vector<float> input_data(4, 1.0f);
	std::vector<float> grad_output_data(4, 1.0f);
	std::vector<float> weight_data(9, 1.0f);
	
	auto input = CreateMatrixFromHost(input_data, OaMatrixShape{1, 1, 2, 2});
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::Conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		1,  // padding
		1   // groups
	);
	
	auto grad_weight = CopyMatrixToHost(bwd_result.GradWeight);
	auto grad_bias = CopyMatrixToHost(bwd_result.GradBias);
	
	OaExpectFinite(grad_weight, "Conv2dBwdWeight with padding grad_weight");
	OaExpectFinite(grad_bias, "Conv2dBwdWeight with padding grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightBiasSum) {
	// Verify that bias gradient is sum of grad_output over spatial dimensions
	std::vector<float> input_data(16, 1.0f);
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);
	
	auto input = CreateMatrixFromHost(input_data, OaMatrixShape{1, 1, 4, 4});
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, 3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::Conv2dBwdWeight(
		input, grad_output, weight,
		1, 0, 1
	);
	
	auto grad_bias = CopyMatrixToHost(bwd_result.GradBias);
	
	// Bias gradient should be sum of grad_output: 1+2+3+4 = 10
	ASSERT_EQ(grad_bias.size(), 1);
	EXPECT_NEAR(grad_bias[0], 10.0f, 1e-5f);
}

TEST_VK(ConvBwd, Conv2dBwdWeightLargeKernel) {
	// Test with larger kernel size
	const OaI32 kernel_size = 5;
	std::vector<float> input_data(8 * 8, 1.0f);
	std::vector<float> grad_output_data(4 * 4, 1.0f);
	std::vector<float> weight_data(kernel_size * kernel_size, 1.0f);
	
	auto input = CreateMatrixFromHost(input_data, OaMatrixShape{1, 1, 8, 8});
	auto grad_output = CreateMatrixFromHost(grad_output_data, OaMatrixShape{1, 1, 4, 4});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{1, 1, kernel_size, kernel_size});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::Conv2dBwdWeight(
		input, grad_output, weight,
		1, 0, 1
	);
	
	auto grad_weight = CopyMatrixToHost(bwd_result.GradWeight);
	
	ASSERT_EQ(grad_weight.size(), kernel_size * kernel_size);
	OaExpectFinite(grad_weight, "Conv2dBwdWeight large kernel");
}

