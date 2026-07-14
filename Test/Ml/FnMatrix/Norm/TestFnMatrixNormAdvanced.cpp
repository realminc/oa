// Test/Ml/FnMatrix/Norm/TestFnMatrixNormAdvanced.cpp
// Advanced normalization tests: RmsNormGated, RmsNormGatedBwd

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

// Numerical gradient helper for RmsNormGated
static OaMatrix NumericalGradientRmsNormGated(
	const std::vector<float>& x_data, const std::vector<float>& weight_data,
	const std::vector<float>& bias_data, const std::vector<float>& z_data,
	const OaMatrixShape& shape, OaF32 eps, bool norm_before_gate, OaI32 param_idx)
{
	const OaF32 h = 1e-4f;
	const OaI32 n = static_cast<OaI32>(x_data.size());
	std::vector<float> grad(param_idx == 0 ? x_data.size() : 
	                        param_idx == 1 ? weight_data.size() :
	                        param_idx == 2 ? bias_data.size() : z_data.size());
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	for (OaI32 i = 0; i < static_cast<OaI32>(grad.size()); ++i) {
		// Forward pass with +h
		std::vector<float> x_plus = x_data, w_plus = weight_data, b_plus = bias_data, z_plus = z_data;
		if (param_idx == 0) x_plus[i] += h;
		else if (param_idx == 1) w_plus[i] += h;
		else if (param_idx == 2) b_plus[i] += h;
		else z_plus[i] += h;
		
		auto x_mat_plus = CreateMatrixFromHost(x_plus, shape);
		auto w_mat_plus = CreateMatrixFromHost(w_plus, OaMatrixShape{shape[1]});
		auto b_mat_plus = CreateMatrixFromHost(b_plus, OaMatrixShape{shape[1]});
		auto z_mat_plus = CreateMatrixFromHost(z_plus, shape);
		auto out_plus = OaFnMatrix::RmsNormGated(x_mat_plus, w_mat_plus, b_mat_plus, z_mat_plus, eps, norm_before_gate);
		auto result_plus = CopyMatrixToHost(out_plus);
		
		// Forward pass with -h
		std::vector<float> x_minus = x_data, w_minus = weight_data, b_minus = bias_data, z_minus = z_data;
		if (param_idx == 0) x_minus[i] -= h;
		else if (param_idx == 1) w_minus[i] -= h;
		else if (param_idx == 2) b_minus[i] -= h;
		else z_minus[i] -= h;
		
		auto x_mat_minus = CreateMatrixFromHost(x_minus, shape);
		auto w_mat_minus = CreateMatrixFromHost(w_minus, OaMatrixShape{shape[1]});
		auto b_mat_minus = CreateMatrixFromHost(b_minus, OaMatrixShape{shape[1]});
		auto z_mat_minus = CreateMatrixFromHost(z_minus, shape);
		auto out_minus = OaFnMatrix::RmsNormGated(x_mat_minus, w_mat_minus, b_mat_minus, z_mat_minus, eps, norm_before_gate);
		auto result_minus = CopyMatrixToHost(out_minus);
		
		// Compute gradient: (f(x+h) - f(x-h)) / (2h)
		OaF32 sum_grad = 0.0f;
		for (size_t j = 0; j < result_plus.size(); ++j) {
			sum_grad += (result_plus[j] - result_minus[j]) / (2.0f * h);
		}
		grad[i] = sum_grad;
	}
	
	return CreateMatrixFromHost(grad, param_idx == 0 ? shape : 
	                                   param_idx == 1 ? OaMatrixShape{shape[1]} :
	                                   param_idx == 2 ? OaMatrixShape{shape[1]} : shape);
}

class NormAdvanced : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize runtime if needed
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
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{2});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{2});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{2, 2});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = OaFnMatrix::RmsNormGated(x, weight, bias, z, 1e-5f, true);
	
	auto result = CopyMatrixToHost(output);
	
	ASSERT_EQ(result.size(), 4);
	OaExpectFinite(result, "RmsNormGated output");
	
	// Output shape should match input
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 2);
}

TEST_VK(NormAdvanced, RmsNormGatedNormBeforeGate) {
	// Test norm_before_gate = true vs false
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> weight_data = {1.0f, 1.0f, 1.0f};
	std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};
	std::vector<float> z_data = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{2, 3});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{3});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{3});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{2, 3});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output_before = OaFnMatrix::RmsNormGated(x, weight, bias, z, 1e-5f, true);
	auto output_after = OaFnMatrix::RmsNormGated(x, weight, bias, z, 1e-5f, false);
	
	auto result_before = CopyMatrixToHost(output_before);
	auto result_after = CopyMatrixToHost(output_after);
	
	// Results should be different
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
	const OaI32 batch = 8;
	const OaI32 dim = 16;
	std::vector<float> x_data(batch * dim);
	std::vector<float> weight_data(dim, 1.0f);
	std::vector<float> bias_data(dim, 0.0f);
	std::vector<float> z_data(batch * dim);
	
	for (OaI32 i = 0; i < batch * dim; ++i) {
		x_data[i] = static_cast<float>(i % 10) * 0.1f;
		z_data[i] = 0.5f + static_cast<float>(i % 5) * 0.1f;
	}
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{batch, dim});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{dim});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{dim});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{batch, dim});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = OaFnMatrix::RmsNormGated(x, weight, bias, z, 1e-5f, true);
	
	auto result = CopyMatrixToHost(output);
	
	ASSERT_EQ(result.size(), batch * dim);
	OaExpectFinite(result, "RmsNormGated large batch output");
}

// ============================================================================
// RmsNormGatedBwd Tests
// ============================================================================

TEST_VK(NormAdvanced, RmsNormGatedBwdNumericalGradientX) {
	// Verify gradient w.r.t. x using numerical differentiation
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data = {1.0f, 1.0f};
	std::vector<float> bias_data = {0.0f, 0.0f};
	std::vector<float> z_data = {0.5f, 0.5f, 0.5f, 0.5f};
	std::vector<float> grad_out_data = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{2});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{2});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{2, 2});
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{2, 2});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::RmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	auto analytical_grad = CopyMatrixToHost(bwd_result.DX);
	auto numerical_grad_mat = NumericalGradientRmsNormGated(x_data, weight_data, bias_data, z_data, OaMatrixShape{2, 2}, 1e-5f, true, 0);
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_VK(NormAdvanced, RmsNormGatedBwdGradientShapes) {
	// Verify all gradient shapes are correct
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> weight_data = {1.0f, 1.0f, 1.0f};
	std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};
	std::vector<float> z_data = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
	std::vector<float> grad_out_data(6, 1.0f);
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{2, 3});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{3});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{3});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{2, 3});
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{2, 3});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::RmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	// Check shapes
	EXPECT_EQ(bwd_result.DX.GetShape()[0], 2);
	EXPECT_EQ(bwd_result.DX.GetShape()[1], 3);
	EXPECT_EQ(bwd_result.DWeight.GetShape()[0], 3);
	EXPECT_EQ(bwd_result.DBias.GetShape()[0], 3);
	EXPECT_EQ(bwd_result.DZ.GetShape()[0], 2);
	EXPECT_EQ(bwd_result.DZ.GetShape()[1], 3);
	
	// Check all gradients are finite
	auto dx = CopyMatrixToHost(bwd_result.DX);
	auto dw = CopyMatrixToHost(bwd_result.DWeight);
	auto db = CopyMatrixToHost(bwd_result.DBias);
	auto dz = CopyMatrixToHost(bwd_result.DZ);
	
	OaExpectFinite(dx, "DX");
	OaExpectFinite(dw, "DWeight");
	OaExpectFinite(db, "DBias");
	OaExpectFinite(dz, "DZ");
}

TEST_VK(NormAdvanced, RmsNormGatedBwdZeroGradient) {
	// Test with zero gradient input
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> weight_data = {1.0f, 1.0f};
	std::vector<float> bias_data = {0.0f, 0.0f};
	std::vector<float> z_data = {0.5f, 0.5f, 0.5f, 0.5f};
	std::vector<float> grad_out_data = {0.0f, 0.0f, 0.0f, 0.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{2, 2});
	auto weight = CreateMatrixFromHost(weight_data, OaMatrixShape{2});
	auto bias = CreateMatrixFromHost(bias_data, OaMatrixShape{2});
	auto z = CreateMatrixFromHost(z_data, OaMatrixShape{2, 2});
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{2, 2});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto bwd_result = OaFnMatrix::RmsNormGatedBwd(x, weight, bias, z, grad_out, 1e-5f);
	
	auto dx = CopyMatrixToHost(bwd_result.DX);
	auto dw = CopyMatrixToHost(bwd_result.DWeight);
	auto db = CopyMatrixToHost(bwd_result.DBias);
	auto dz = CopyMatrixToHost(bwd_result.DZ);
	
	// All gradients should be zero or very small
	for (float val : dx) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : dw) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : db) EXPECT_NEAR(val, 0.0f, 1e-5f);
	for (float val : dz) EXPECT_NEAR(val, 0.0f, 1e-5f);
}

