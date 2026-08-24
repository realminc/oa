// Test/Ml/FnMatrix/Conv/TestFnMatrixConvTranspose2d.cpp
// Tests for ConvTranspose2d forward and backward passes.

#include <gtest/gtest.h>
#include <oa/core.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>

static oa::Matrix createMatrixFromHost(const std::vector<float>& data, const oa::MatrixShape& shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

static std::vector<float> copyMatrixToHost(const oa::Matrix& mat) {
	std::vector<float> result(static_cast<size_t>(mat.numElements()));
	[[maybe_unused]] auto copy_result = oa::FnMatrix::copyToHost(mat, result.data(), result.size() * sizeof(float));
	return result;
}

static void expectFinite(const std::vector<float>& data, const char* name) {
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_TRUE(std::isfinite(data[i])) << name << " contains non-finite value at index " << i;
	}
}

class ConvTranspose2d : public ::testing::Test {};

TEST_VK(ConvTranspose2d, ForwardShape) {
	// input [1, 1, 2, 2], weight [1, 1, 3, 3], stride 1, padding 0
	// output: (2-1)*1 - 0 + 3 = 4 per side -> [1, 1, 4, 4]
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> weight_data(9, 1.0F);
	std::vector<float> bias_data(1, 0.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{1});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 1, 0);
	auto result = copyMatrixToHost(out);

	EXPECT_EQ(out.getShape()[0], 1);
	EXPECT_EQ(out.getShape()[1], 1);
	EXPECT_EQ(out.getShape()[2], 4);
	EXPECT_EQ(out.getShape()[3], 4);
	EXPECT_EQ(result.size(), 16);
	expectFinite(result, "ConvTranspose2d forward");
}

TEST_VK(ConvTranspose2d, ForwardWithStride) {
	// input [1, 1, 2, 2], weight [1, 1, 3, 3], stride 2, padding 0
	// output: (2-1)*2 + 3 = 5 per side -> [1, 1, 5, 5]
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> weight_data(9, 0.5F);
	std::vector<float> bias_data(1, 0.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{1});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 2, 0);
	auto result = copyMatrixToHost(out);

	EXPECT_EQ(out.getShape()[2], 5);
	EXPECT_EQ(out.getShape()[3], 5);
	EXPECT_EQ(result.size(), 25);
	expectFinite(result, "ConvTranspose2d forward with stride");
}

TEST_VK(ConvTranspose2d, ForwardWithPadding) {
	// input [1, 1, 4, 4], weight [1, 1, 3, 3], stride 1, padding 1
	// output: (4-1)*1 - 2*1 + 3 = 4 per side -> [1, 1, 4, 4]
	std::vector<float> input_data(16, 1.0F);
	std::vector<float> weight_data(9, 1.0F);
	std::vector<float> bias_data(1, 2.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 4, 4});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{1});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 1, 1);
	auto result = copyMatrixToHost(out);

	EXPECT_EQ(out.getShape()[2], 4);
	EXPECT_EQ(out.getShape()[3], 4);
	EXPECT_EQ(result.size(), 16);
	expectFinite(result, "ConvTranspose2d forward with padding");
}

TEST_VK(ConvTranspose2d, ForwardMultiChannel) {
	// input [1, 2, 3, 3], weight [2, 3, 3, 3], bias [3]
	// output: (3-1)*1 + 3 = 5 -> [1, 3, 5, 5]
	const oa::I32 inC = 2;
	const oa::I32 outC = 3;
	std::vector<float> input_data(static_cast<size_t>(inC) * 3 * 3, 1.0F);
	std::vector<float> weight_data(static_cast<size_t>(inC) * outC * 3 * 3, 0.25F);
	std::vector<float> bias_data(outC, 0.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, inC, 3, 3});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{inC, outC, 3, 3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{outC});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 1, 0);
	auto result = copyMatrixToHost(out);

	EXPECT_EQ(out.getShape()[1], outC);
	EXPECT_EQ(out.getShape()[2], 5);
	EXPECT_EQ(out.getShape()[3], 5);
	EXPECT_EQ(result.size(), outC * 5 * 5);
	expectFinite(result, "ConvTranspose2d forward multi-channel");
}

TEST_VK(ConvTranspose2d, ForwardNumericalReference) {
	// 1x1x2x2 input, 1x1x2x2 weight of ones, stride 1, padding 0.
	// output dim = (2-1)*1 + 2 = 3.
	// CPU reference: y[oh,ow] = sum_{ih+kh=oh, iw+kw=ow} x[ih,iw] * w[kh,kw]
	// 1 1 * 1 1  -> 1 2 1
	// 1 1         -> 2 4 2
	//               -> 1 2 1
	std::vector<float> input_data = {1.0F, 1.0F, 1.0F, 1.0F};
	std::vector<float> weight_data(4, 1.0F);
	std::vector<float> bias_data(1, 0.0F);
	std::vector<float> expected = {
		1.0F, 2.0F, 1.0F,
		2.0F, 4.0F, 2.0F,
		1.0F, 2.0F, 1.0F
	};

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 2, 2});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{1});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 1, 0);
	auto result = copyMatrixToHost(out);

	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5F) << " at index " << i;
	}
}

TEST_VK(ConvTranspose2d, BwdDataShape) {
	// d_out [1, 1, 4, 4], weight [1, 1, 3, 3], stride 1, padding 0
	// d_input: Conv2d(d_out, weight) -> [1, 1, 2, 2]
	std::vector<float> d_out_data(16, 1.0F);
	std::vector<float> weight_data(9, 1.0F);

	auto d_out = createMatrixFromHost(d_out_data, oa::MatrixShape{1, 1, 4, 4});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto d_input = oa::FnMatrix::convTranspose2dBwdData(d_out, weight, 1, 0, oa::MatrixShape{1, 1, 2, 2});
	auto result = copyMatrixToHost(d_input);

	EXPECT_EQ(d_input.getShape()[0], 1);
	EXPECT_EQ(d_input.getShape()[1], 1);
	EXPECT_EQ(d_input.getShape()[2], 2);
	EXPECT_EQ(d_input.getShape()[3], 2);
	EXPECT_EQ(result.size(), 4);
	expectFinite(result, "ConvTranspose2dBwdData");
}

TEST_VK(ConvTranspose2d, BwdDataNumericalReference) {
	// d_out [1,1,3,3] with values 1..9, weight [1,1,2,2] of ones, S=1, P=0.
	// d_input = Conv2d(d_out, W) -> [1,1,2,2].
	// d_input[0,0] = 1+2+4+5 = 12
	// d_input[0,1] = 2+3+5+6 = 16
	// d_input[1,0] = 4+5+7+8 = 24
	// d_input[1,1] = 5+6+8+9 = 28
	std::vector<float> d_out_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
	std::vector<float> weight_data(4, 1.0F);
	std::vector<float> expected = {12.0F, 16.0F, 24.0F, 28.0F};

	auto d_out = createMatrixFromHost(d_out_data, oa::MatrixShape{1, 1, 3, 3});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 2, 2});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto d_input = oa::FnMatrix::convTranspose2dBwdData(d_out, weight, 1, 0, oa::MatrixShape{1, 1, 2, 2});
	auto result = copyMatrixToHost(d_input);

	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5F) << " d_input at index " << i;
	}
}

TEST_VK(ConvTranspose2d, BwdWeightBiasSum) {
	// input [1,1,2,2] of ones, d_out [1,1,3,3] with values 1..9, weight [1,1,2,2]
	// output dims: H_out = (2-1)*1 + 2 = 3, W_out = (2-1)*1 + 2 = 3.
	// d_bias = sum(d_out) = 45.
	// d_weight for 2x2 kernel, S=1, P=0:
	//   [0,0]: d_out[0:2,0:2] = 1+2+4+5 = 12
	//   [0,1]: d_out[0:2,1:3] = 2+3+5+6 = 16
	//   [1,0]: d_out[1:3,0:2] = 4+5+7+8 = 24
	//   [1,1]: d_out[1:3,1:3] = 5+6+8+9 = 28
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> d_out_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
	std::vector<float> weight_data(4, 1.0F);
	std::vector<float> expected_dw = {12.0F, 16.0F, 24.0F, 28.0F};

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto d_out = createMatrixFromHost(d_out_data, oa::MatrixShape{1, 1, 3, 3});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 2, 2});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd = oa::FnMatrix::convTranspose2dBwdWeight(input, d_out, weight, 1, 0);
	auto grad_bias = copyMatrixToHost(bwd.gradBias);
	auto grad_weight = copyMatrixToHost(bwd.gradWeight);

	ASSERT_EQ(grad_bias.size(), 1);
	EXPECT_NEAR(grad_bias[0], 45.0F, 1e-5F);

	ASSERT_EQ(grad_weight.size(), 4);
	for (size_t i = 0; i < grad_weight.size(); ++i) {
		EXPECT_NEAR(grad_weight[i], expected_dw[i], 1e-5F) << " d_weight at index " << i;
	}
}

TEST_VK(ConvTranspose2d, BwdWeightShape) {
	// input [1, 1, 2, 2], d_out [1, 1, 4, 4], weight [1, 1, 3, 3]
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> d_out_data(16, 1.0F);
	std::vector<float> weight_data(9, 1.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto d_out = createMatrixFromHost(d_out_data, oa::MatrixShape{1, 1, 4, 4});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd = oa::FnMatrix::convTranspose2dBwdWeight(input, d_out, weight, 1, 0);
	auto grad_weight = copyMatrixToHost(bwd.gradWeight);
	auto grad_bias = copyMatrixToHost(bwd.gradBias);

	EXPECT_EQ(bwd.gradWeight.getShape()[0], 1);
	EXPECT_EQ(bwd.gradWeight.getShape()[1], 1);
	EXPECT_EQ(bwd.gradWeight.getShape()[2], 3);
	EXPECT_EQ(bwd.gradWeight.getShape()[3], 3);
	EXPECT_EQ(grad_weight.size(), 9);
	EXPECT_EQ(grad_bias.size(), 1);
	expectFinite(grad_weight, "ConvTranspose2dBwdWeight grad_weight");
	expectFinite(grad_bias, "ConvTranspose2dBwdWeight grad_bias");
}

TEST_VK(ConvTranspose2d, LayerAutogradGradCorrect) {
	// compare module-level autograd gradients against explicit backward functions.
	// input [1,1,2,2], weight [1,1,3,3], d_out [1,1,4,4] of ones.
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> d_out_data(16, 1.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto d_out = createMatrixFromHost(d_out_data, oa::MatrixShape{1, 1, 4, 4});

	auto layer = oa::ConvTranspose2d(1, 1, 3, 1, 0);
	auto& weightParam = layer.parameters()[0];
	auto& biasParam = layer.parameters()[1];
	weightParam.data.setRequiresGrad(true);
	biasParam.data.setRequiresGrad(true);
	input.setRequiresGrad(true);

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::GradientTape tape;
	auto out = layer.forward(input);
	auto loss = oa::FnMatrix::sum(out);
	tape.backward(loss);

	auto dx_ag = copyMatrixToHost(input.gradMatrix());
	auto dw_ag = copyMatrixToHost(weightParam.data.gradMatrix());
	auto db_ag = copyMatrixToHost(biasParam.data.gradMatrix());

	auto weight = weightParam.data;
	auto dx_ex = copyMatrixToHost(oa::FnMatrix::convTranspose2dBwdData(d_out, weight, 1, 0, oa::MatrixShape{1, 1, 2, 2}));
	auto bwd_ex = oa::FnMatrix::convTranspose2dBwdWeight(input, d_out, weight, 1, 0);
	auto dw_ex = copyMatrixToHost(bwd_ex.gradWeight);
	auto db_ex = copyMatrixToHost(bwd_ex.gradBias);

	ASSERT_EQ(dx_ag.size(), dx_ex.size());
	ASSERT_EQ(dw_ag.size(), dw_ex.size());
	ASSERT_EQ(db_ag.size(), db_ex.size());
	for (size_t i = 0; i < dx_ag.size(); ++i) {
		EXPECT_NEAR(dx_ag[i], dx_ex[i], 1e-5F) << " d_input autograd vs explicit at index " << i;
	}
	for (size_t i = 0; i < dw_ag.size(); ++i) {
		EXPECT_NEAR(dw_ag[i], dw_ex[i], 1e-5F) << " d_weight autograd vs explicit at index " << i;
	}
	for (size_t i = 0; i < db_ag.size(); ++i) {
		EXPECT_NEAR(db_ag[i], db_ex[i], 1e-5F) << " d_bias autograd vs explicit at index " << i;
	}
}

TEST_VK(ConvTranspose2d, AutogradFinite) {
	// verify that gradients flow through ConvTranspose2d.
	std::vector<float> input_data(4, 1.0F);
	std::vector<float> weight_data(9, 0.5F);
	std::vector<float> bias_data(1, 0.0F);
	std::vector<float> target_data(16, 0.0F);

	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{1, 1, 3, 3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{1});
	auto target = createMatrixFromHost(target_data, oa::MatrixShape{1, 1, 4, 4});

	input.setRequiresGrad(true);
	weight.setRequiresGrad(true);
	bias.setRequiresGrad(true);

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::GradientTape tape;
	auto out = oa::FnMatrix::convTranspose2d(input, weight, bias, 1, 0);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);

	auto dx = copyMatrixToHost(input.gradMatrix());
	auto dw = copyMatrixToHost(weight.gradMatrix());
	auto db = copyMatrixToHost(bias.gradMatrix());

	EXPECT_EQ(dx.size(), 4);
	EXPECT_EQ(dw.size(), 9);
	EXPECT_EQ(db.size(), 1);
	expectFinite(dx, "ConvTranspose2d autograd d_input");
	expectFinite(dw, "ConvTranspose2d autograd d_weight");
	expectFinite(db, "ConvTranspose2d autograd d_bias");
}

TEST_VK(ConvTranspose2d, LayerForward) {
	// Use the oa::ConvTranspose2d module directly.
	auto layer = oa::ConvTranspose2d(1, 1, 3, 1, 0);
	std::vector<float> input_data(4, 1.0F);
	auto input = createMatrixFromHost(input_data, oa::MatrixShape{1, 1, 2, 2});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto out = layer.forward(input);
	auto result = copyMatrixToHost(out);

	EXPECT_EQ(out.getShape()[0], 1);
	EXPECT_EQ(out.getShape()[1], 1);
	EXPECT_EQ(out.getShape()[2], 4);
	EXPECT_EQ(out.getShape()[3], 4);
	EXPECT_EQ(result.size(), 16);
	expectFinite(result, "oa::ConvTranspose2d layer forward");
}
