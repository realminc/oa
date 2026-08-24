// Test/Ml/FnMatrix/norm/TestFnMatrixNormAdvanced.cpp
// Advanced normalization tests: RmsNormGated, RmsNormGatedBwd

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

// Numerical gradient helper for RmsNormGated
static oa::Matrix numericalGradientRmsNormGated(
	const std::vector<float>& x_data, const std::vector<float>& weight_data,
	const std::vector<float>& bias_data, const std::vector<float>& z_data,
	const oa::MatrixShape& shape, oa::F32 eps, bool norm_before_gate, oa::I32 param_idx)
{
	const oa::F32 h = 1e-4f;
	const oa::I32 n = static_cast<oa::I32>(x_data.size());
	std::vector<float> grad(param_idx == 0 ? x_data.size() : 
	                        param_idx == 1 ? weight_data.size() :
	                        param_idx == 2 ? bias_data.size() : z_data.size());
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	for (oa::I32 i = 0; i < static_cast<oa::I32>(grad.size()); ++i) {
		// forward pass with +h
		std::vector<float> x_plus = x_data, w_plus = weight_data, b_plus = bias_data, z_plus = z_data;
		if (param_idx == 0) x_plus[i] += h;
		else if (param_idx == 1) w_plus[i] += h;
		else if (param_idx == 2) b_plus[i] += h;
		else z_plus[i] += h;
		
		auto x_mat_plus = createMatrixFromHost(x_plus, shape);
		auto w_mat_plus = createMatrixFromHost(w_plus, oa::MatrixShape{shape[1]});
		auto b_mat_plus = createMatrixFromHost(b_plus, oa::MatrixShape{shape[1]});
		auto z_mat_plus = createMatrixFromHost(z_plus, shape);
		auto out_plus = oa::FnMatrix::rmsNormGated(x_mat_plus, w_mat_plus, b_mat_plus, z_mat_plus, eps, norm_before_gate);
		auto result_plus = copyMatrixToHost(out_plus);
		
		// forward pass with -h
		std::vector<float> x_minus = x_data, w_minus = weight_data, b_minus = bias_data, z_minus = z_data;
		if (param_idx == 0) x_minus[i] -= h;
		else if (param_idx == 1) w_minus[i] -= h;
		else if (param_idx == 2) b_minus[i] -= h;
		else z_minus[i] -= h;
		
		auto x_mat_minus = createMatrixFromHost(x_minus, shape);
		auto w_mat_minus = createMatrixFromHost(w_minus, oa::MatrixShape{shape[1]});
		auto b_mat_minus = createMatrixFromHost(b_minus, oa::MatrixShape{shape[1]});
		auto z_mat_minus = createMatrixFromHost(z_minus, shape);
		auto out_minus = oa::FnMatrix::rmsNormGated(x_mat_minus, w_mat_minus, b_mat_minus, z_mat_minus, eps, norm_before_gate);
		auto result_minus = copyMatrixToHost(out_minus);
		
		// Compute gradient: (f(x+h) - f(x-h)) / (2h)
		oa::F32 sum_grad = 0.0f;
		for (size_t j = 0; j < result_plus.size(); ++j) {
			sum_grad += (result_plus[j] - result_minus[j]) / (2.0f * h);
		}
		grad[i] = sum_grad;
	}
	
	return createMatrixFromHost(grad, param_idx == 0 ? shape : 
	                                   param_idx == 1 ? oa::MatrixShape{shape[1]} :
	                                   param_idx == 2 ? oa::MatrixShape{shape[1]} : shape);
}

class NormAdvanced : public ::testing::Test {
protected:
	void SetUp() override {
		// initialize runtime if needed
	}
};

// ============================================================================
// RmsNormGated Tests
// ============================================================================

TEST_VK(NormAdvanced, RmsNormGatedBasic) {
	// Test basic RmsNormGated forward pass
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data = {1.0f, 1.0f};
	std::vector<float> bias_data = {0.0f, 0.0f};
	std::vector<float> z_data = {0.5f, 0.5f, 0.5f, 0.5f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{2});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{2});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::rmsNormGated(x, weight, bias, z, 1e-5f, true);
	
	auto result = copyMatrixToHost(output);
	
	ASSERT_EQ(result.size(), 4);
	expectFinite(result, "RmsNormGated output");
	
	// output shape should match input
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 2);
}

TEST_VK(NormAdvanced, RmsNormGatedNormBeforeGate) {
	// Test norm_before_gate = true vs false
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> weight_data = {1.0f, 1.0f, 1.0f};
	std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};
	std::vector<float> z_data = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{2, 3});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{3});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{2, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output_before = oa::FnMatrix::rmsNormGated(x, weight, bias, z, 1e-5f, true);
	auto output_after = oa::FnMatrix::rmsNormGated(x, weight, bias, z, 1e-5f, false);
	
	auto result_before = copyMatrixToHost(output_before);
	auto result_after = copyMatrixToHost(output_after);
	
	// results should be different
	bool different = false;
	for (size_t i = 0; i < result_before.size(); ++i) {
		if (std::abs(result_before[i] - result_after[i]) > 1e-5f) {
			different = true;
			break;
		}
	}
	EXPECT_TRUE(different) << "norm_before_gate should produce different results";
}

TEST_VK(NormAdvanced, RmsNormGatedLargeBatch) {
	// Test with larger batch size
	const oa::I32 batch = 8;
	const oa::I32 dim = 16;
	std::vector<float> x_data(batch * dim);
	std::vector<float> weight_data(dim, 1.0f);
	std::vector<float> bias_data(dim, 0.0f);
	std::vector<float> z_data(batch * dim);
	
	for (oa::I32 i = 0; i < batch * dim; ++i) {
		x_data[i] = static_cast<float>(i % 10) * 0.1f;
		z_data[i] = 0.5f + static_cast<float>(i % 5) * 0.1f;
	}
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{batch, dim});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{dim});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{dim});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{batch, dim});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::rmsNormGated(x, weight, bias, z, 1e-5f, true);
	
	auto result = copyMatrixToHost(output);
	
	ASSERT_EQ(result.size(), batch * dim);
	expectFinite(result, "RmsNormGated large batch output");
}

// ============================================================================
// RmsNormGatedBwd Tests
// ============================================================================

TEST_VK(NormAdvanced, RmsNormGatedBwdNumericalGradientX) {
	// verify gradient w.r.t. x using numerical differentiation
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data = {1.0f, 1.0f};
	std::vector<float> bias_data = {0.0f, 0.0f};
	std::vector<float> z_data = {0.5f, 0.5f, 0.5f, 0.5f};
	std::vector<float> grad_out_data = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{2});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{2});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{2, 2});
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::rmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	auto analytical_grad = copyMatrixToHost(bwd_result.dX);
	auto numerical_grad_mat = numericalGradientRmsNormGated(x_data, weight_data, bias_data, z_data, oa::MatrixShape{2, 2}, 1e-5f, true, 0);
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_VK(NormAdvanced, RmsNormGatedBwdGradientShapes) {
	// verify all gradient shapes are correct
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> weight_data = {1.0f, 1.0f, 1.0f};
	std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};
	std::vector<float> z_data = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
	std::vector<float> grad_out_data(6, 1.0f);
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{2, 3});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{3});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{3});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{2, 3});
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{2, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::rmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	// Check shapes
	EXPECT_EQ(bwd_result.dX.getShape()[0], 2);
	EXPECT_EQ(bwd_result.dX.getShape()[1], 3);
	EXPECT_EQ(bwd_result.dWeight.getShape()[0], 3);
	EXPECT_EQ(bwd_result.dBias.getShape()[0], 3);
	EXPECT_EQ(bwd_result.dZ.getShape()[0], 2);
	EXPECT_EQ(bwd_result.dZ.getShape()[1], 3);
	
	// Check all gradients are finite
	auto dx = copyMatrixToHost(bwd_result.dX);
	auto dw = copyMatrixToHost(bwd_result.dWeight);
	auto db = copyMatrixToHost(bwd_result.dBias);
	auto dz = copyMatrixToHost(bwd_result.dZ);
	
	expectFinite(dx, "DX");
	expectFinite(dw, "dWeight");
	expectFinite(db, "dBias");
	expectFinite(dz, "DZ");
}

TEST_VK(NormAdvanced, RmsNormGatedBwdZeroGradient) {
	// Test with zero gradient input
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data = {1.0f, 1.0f};
	std::vector<float> bias_data = {0.0f, 0.0f};
	std::vector<float> z_data = {0.5f, 0.5f, 0.5f, 0.5f};
	std::vector<float> grad_out_data = {0.0f, 0.0f, 0.0f, 0.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{2, 2});
	auto weight = createMatrixFromHost(weight_data, oa::MatrixShape{2});
	auto bias = createMatrixFromHost(bias_data, oa::MatrixShape{2});
	auto z = createMatrixFromHost(z_data, oa::MatrixShape{2, 2});
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto bwd_result = oa::FnMatrix::rmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	auto dx = copyMatrixToHost(bwd_result.dX);
	auto dw = copyMatrixToHost(bwd_result.dWeight);
	auto db = copyMatrixToHost(bwd_result.dBias);
	auto dz = copyMatrixToHost(bwd_result.dZ);
	
	// All gradients should be zero or very small
	for (float val : dx) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : dw) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : db) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : dz) EXPECT_NEAR(val, 0.0f, 1e-5f);
}
