// Test/Ml/FnMatrix/Conv/TestFnMatrixConvBwd.cpp
// Tests for Conv2d backward passes (Conv2dBwdData, Conv2dBwdWeight)

#include <gtest/gtest.h>
#include <oa/core.h>
#include <oa/ml.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>

// Helper to create matrix from host data
static oa::Matrix createMatrixFromHost(const std::vector<float>& data, const oa::MatrixShape& shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Helper to copy matrix to host
static std::vector<float> copyMatrixToHost(const oa::Matrix& mat) {
	std::vector<float> result(mat.numElements());
	[[maybe_unused]] auto copy_result = oa::FnMatrix::copyToHost(mat, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to check if all values are finite
static void expectFinite(const std::vector<float>& data, const char* name) {
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_TRUE(std::isfinite(data[i])) << name << " contains non-finite value at index " << i;
	}
}

class ConvBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// initialize runtime if needed
	}
};

// ============================================================================
// Conv2dBwdData tests (gradient w.r.t. input)
// ============================================================================

TEST_VK(ConvBwd, Conv2dBwdDataBasic) {
	// Test basic Conv2dBwdData: gradient w.r.t. input
	// forward: input [1,1,4,4] * weight [1,1,3,3] -> output [1,1,2,2]
	// backward: grad_output [1,1,2,2] -> grad_input [1,1,4,4]
	
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);  // 3x3 kernel of ones
	
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto grad_input = oa::FnMatrix::conv2dBwdData(
		grad_output, weight, 
		1,  // stride
		0,  // padding
		oa::MatrixShape{1, 1, 4, 4},  // input_shape
		1   // groups
	);
	
	auto result = copyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 16);  // 4x4
	EXPECT_EQ(grad_input.getShape()[0], 1);
	EXPECT_EQ(grad_input.getShape()[1], 1);
	EXPECT_EQ(grad_input.getShape()[2], 4);
	EXPECT_EQ(grad_input.getShape()[3], 4);
	
	expectFinite(result, "Conv2dBwdData output");
}

TEST_VK(ConvBwd, Conv2dBwdDataWithPadding) {
	// Test Conv2dBwdData with padding
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> weight_data(9, 0.5f);
	
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto grad_input = oa::FnMatrix::conv2dBwdData(
		grad_output, weight,
		1,  // stride
		1,  // padding
		oa::MatrixShape{1, 1, 2, 2},  // input_shape (same as output due to padding)
		1   // groups
	);
	
	auto result = copyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 4);
	expectFinite(result, "Conv2dBwdData with padding");
}

TEST_VK(ConvBwd, Conv2dBwdDataWithStride) {
	// Test Conv2dBwdData with stride > 1
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);
	
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto grad_input = oa::FnMatrix::conv2dBwdData(
		grad_output, weight,
		2,  // stride
		0,  // padding
		oa::MatrixShape{1, 1, 6, 6},  // input_shape
		1   // groups
	);
	
	auto result = copyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), 36);  // 6x6
	expectFinite(result, "Conv2dBwdData with stride");
}

TEST_VK(ConvBwd, Conv2dBwdDataMultiChannel) {
	// Test Conv2dBwdData with multiple channels
	const oa::I32 in_channels = 3;
	const oa::I32 out_channels = 2;
	
	std::vector<float> grad_output_data(out_channels * 2 * 2, 1.0f);
	std::vector<float> weight_data(out_channels * in_channels * 3 * 3, 0.5f);
	
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, out_channels, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{out_channels, in_channels, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto grad_input = oa::FnMatrix::conv2dBwdData(
		grad_output, weight,
		1,  // stride
		0,  // padding
		oa::MatrixShape{1, in_channels, 4, 4},
		1   // groups
	);
	
	auto result = copyMatrixToHost(grad_input);
	
	ASSERT_EQ(result.size(), in_channels * 4 * 4);
	EXPECT_EQ(grad_input.getShape()[1], in_channels);
	expectFinite(result, "Conv2dBwdData multi-channel");
}

// ============================================================================
// Conv2dBwdWeight tests (gradient w.r.t. weight and bias)
// ============================================================================

TEST_VK(ConvBwd, Conv2dBwdWeightBasic) {
	// Test basic Conv2dBwdWeight: gradients w.r.t. weight and bias
	std::vector<float> input_data(16);
	for (oa::I32 i = 0; i < 16; ++i) input_data[i] = static_cast<float>(i + 1);
	
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);  // For shape reference
	
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 4, 4});
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		0,  // padding
		1   // groups
	);
	
	auto grad_weight = copyMatrixToHost(bwd_result.gradWeight);
	auto grad_bias = copyMatrixToHost(bwd_result.gradBias);
	
	ASSERT_EQ(grad_weight.size(), 9);  // 3x3 kernel
	ASSERT_EQ(grad_bias.size(), 1);    // 1 output channel
	
	EXPECT_EQ(bwd_result.gradWeight.getShape()[0], 1);
	EXPECT_EQ(bwd_result.gradWeight.getShape()[1], 1);
	EXPECT_EQ(bwd_result.gradWeight.getShape()[2], 3);
	EXPECT_EQ(bwd_result.gradWeight.getShape()[3], 3);
	
	expectFinite(grad_weight, "Conv2dBwdWeight grad_weight");
	expectFinite(grad_bias, "Conv2dBwdWeight grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightMultiChannel) {
	// Test Conv2dBwdWeight with multiple channels
	const oa::I32 in_channels = 2;
	const oa::I32 out_channels = 3;
	
	std::vector<float> input_data(in_channels * 4 * 4, 1.0f);
	std::vector<float> grad_output_data(out_channels * 2 * 2, 1.0f);
	std::vector<float> weight_data(out_channels * in_channels * 3 * 3, 1.0f);
	
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, in_channels, 4, 4});
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, out_channels, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{out_channels, in_channels, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		0,  // padding
		1   // groups
	);
	
	auto grad_weight = copyMatrixToHost(bwd_result.gradWeight);
	auto grad_bias = copyMatrixToHost(bwd_result.gradBias);
	
	ASSERT_EQ(grad_weight.size(), out_channels * in_channels * 3 * 3);
	ASSERT_EQ(grad_bias.size(), out_channels);
	
	expectFinite(grad_weight, "Conv2dBwdWeight multi-channel grad_weight");
	expectFinite(grad_bias, "Conv2dBwdWeight multi-channel grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightWithPadding) {
	// Test Conv2dBwdWeight with padding
	std::vector<float> input_data(4, 1.0f);
	std::vector<float> grad_output_data(4, 1.0f);
	std::vector<float> weight_data(9, 1.0f);
	
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::conv2dBwdWeight(
		input, grad_output, weight,
		1,  // stride
		1,  // padding
		1   // groups
	);
	
	auto grad_weight = copyMatrixToHost(bwd_result.gradWeight);
	auto grad_bias = copyMatrixToHost(bwd_result.gradBias);
	
	expectFinite(grad_weight, "Conv2dBwdWeight with padding grad_weight");
	expectFinite(grad_bias, "Conv2dBwdWeight with padding grad_bias");
}

TEST_VK(ConvBwd, Conv2dBwdWeightBiasSum) {
	// verify that bias gradient is sum of grad_output over spatial dimensions
	std::vector<float> input_data(16, 1.0f);
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data(9, 1.0f);
	
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 4, 4});
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::conv2dBwdWeight(
		input, grad_output, weight,
		1, 0, 1
	);
	
	auto grad_bias = copyMatrixToHost(bwd_result.gradBias);
	
	// Bias gradient should be sum of grad_output: 1+2+3+4 = 10
	ASSERT_EQ(grad_bias.size(), 1);
	EXPECT_NEAR(grad_bias[0], 10.0f, 1e-5f);
}

TEST_VK(ConvBwd, Conv2dBwdWeightLargeKernel) {
	// Test with larger kernel size
	const oa::I32 kernel_size = 5;
	std::vector<float> input_data(8 * 8, 1.0f);
	std::vector<float> grad_output_data(4 * 4, 1.0f);
	std::vector<float> weight_data(kernel_size * kernel_size, 1.0f);
	
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 8, 8});
	auto grad_output = createMatrixFromHost(grad_output_data, oa::MatrixShape{1, 1, 4, 4});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, kernel_size, kernel_size});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::conv2dBwdWeight(
		input, grad_output, weight,
		1, 0, 1
	);
	
	auto grad_weight = copyMatrixToHost(bwd_result.gradWeight);
	
	ASSERT_EQ(grad_weight.size(), kernel_size * kernel_size);
	expectFinite(grad_weight, "Conv2dBwdWeight large kernel");
}
