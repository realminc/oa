// Numerical gradient tests for activation backward passes
// MishBwd, SoftplusBwd, GeluBwd variants

#include <gtest/gtest.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <vector>
#include <cmath>

static OaEngine* GRt = nullptr;

class TestFnMatrixActivationBwd : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixActivationBwd";
		auto r = OaEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> CopyToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to create matrix from host data
static OaMatrix CreateFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Numerical gradient checker
// Computes (f(x+h) - f(x-h)) / (2*h) for each element
template<typename ForwardFn>
static std::vector<float> NumericalGradient(
	const std::vector<float>& input,
	OaMatrixShape shape,
	ForwardFn forward_fn,
	// eps=1e-4 is too small for fp32 central differences: the f(x+h)-f(x-h)
	// subtraction loses ~machine_eps/eps ≈ 1e-3 to cancellation, which exceeds the
	// 1e-3 gradcheck tolerance. 1e-3 balances cancellation vs O(h²) truncation.
	float epsilon = 1e-3f
) {
	std::vector<float> grad(input.size());
	
	for (size_t i = 0; i < input.size(); ++i) {
		// f(x + h)
		std::vector<float> input_plus = input;
		input_plus[i] += epsilon;
		auto x_plus = CreateFromHost(input_plus, shape);
		auto y_plus = forward_fn(x_plus);
		auto y_plus_data = CopyToHost(y_plus);
		float sum_plus = 0.0f;
		for (float val : y_plus_data) sum_plus += val;
		
		// f(x - h)
		std::vector<float> input_minus = input;
		input_minus[i] -= epsilon;
		auto x_minus = CreateFromHost(input_minus, shape);
		auto y_minus = forward_fn(x_minus);
		auto y_minus_data = CopyToHost(y_minus);
		float sum_minus = 0.0f;
		for (float val : y_minus_data) sum_minus += val;
		
		// Numerical gradient
		grad[i] = (sum_plus - sum_minus) / (2.0f * epsilon);
	}
	
	return grad;
}

// ============================================================================
// MishBwd Tests
// ============================================================================

TEST_VK(TestFnMatrixActivationBwd, MishBwd_NumericalGradient) {
	// Test Mish backward pass against numerical gradient
	// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	// Forward pass
	auto output = OaFnMatrix::Mish(input);
	
	// Analytical gradient (from MishBwd)
	std::vector<float> grad_output_data(5, 1.0f);  // All ones
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{5});
	auto grad_input = OaFnMatrix::MishBwd(input, grad_output);
	auto analytical_grad = CopyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = NumericalGradient(input_data, OaMatrixShape{5}, 
		[](const OaMatrix& x) { return OaFnMatrix::Mish(x); });
	
	// Compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, MishBwd_ZeroGradient) {
	// Test that zero grad_output produces zero grad_input
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-1.0f, 0.0f, 1.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{3});
	
	std::vector<float> grad_output_data(3, 0.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{3});
	
	auto grad_input = OaFnMatrix::MishBwd(input, grad_output);
	auto result = CopyToHost(grad_input);
	
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixActivationBwd, MishBwd_LargeValues) {
	// Test Mish backward with large positive/negative values
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-10.0f, -5.0f, 5.0f, 10.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{4});
	
	std::vector<float> grad_output_data(4, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{4});
	
	auto grad_input = OaFnMatrix::MishBwd(input, grad_output);
	auto result = CopyToHost(grad_input);
	
	// For large positive x, Mish(x) ≈ x, so gradient ≈ 1
	EXPECT_NEAR(result[2], 1.0f, 0.1f) << "Large positive gradient";
	EXPECT_NEAR(result[3], 1.0f, 0.1f) << "Large positive gradient";
	
	// For large negative x, Mish(x) ≈ 0, so gradient ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 0.1f) << "Large negative gradient";
}

// ============================================================================
// SoftplusBwd Tests
// ============================================================================

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_NumericalGradient) {
	// Test Softplus backward pass against numerical gradient
	// Softplus(x) = ln(1 + e^x)
	// d/dx Softplus(x) = sigmoid(x) = 1 / (1 + e^(-x))
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	// Forward pass
	auto output = OaFnMatrix::Softplus(input);
	
	// Analytical gradient. SoftplusBwd takes the forward OUTPUT y=softplus(x)
	// (it computes sigmoid(x)=1-e^-y), unlike Gelu/Silu/Mish which take the input.
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{5});
	auto grad_input = OaFnMatrix::SoftplusBwd(output, grad_output);
	auto analytical_grad = CopyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = NumericalGradient(input_data, OaMatrixShape{5},
		[](const OaMatrix& x) { return OaFnMatrix::Softplus(x); });
	
	// Compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f)
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_SigmoidProperty) {
	// Test that Softplus gradient equals Sigmoid
	// d/dx Softplus(x) = Sigmoid(x)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	// Softplus gradient. SoftplusBwd takes the forward OUTPUT y=softplus(x).
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{5});
	auto softplus_out = OaFnMatrix::Softplus(input);
	auto softplus_grad = OaFnMatrix::SoftplusBwd(softplus_out, grad_output);
	auto softplus_grad_data = CopyToHost(softplus_grad);
	
	// Sigmoid
	auto sigmoid_output = OaFnMatrix::Sigmoid(input);
	auto sigmoid_data = CopyToHost(sigmoid_output);
	
	// Compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(softplus_grad_data[i], sigmoid_data[i], 1e-5f)
			<< "Softplus gradient should equal Sigmoid at index " << i;
	}
}

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_ExtremeValues) {
	// Test Softplus backward with extreme values
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-20.0f, -10.0f, 10.0f, 20.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{4});
	
	std::vector<float> grad_output_data(4, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{4});

	// SoftplusBwd takes the forward OUTPUT y=softplus(x).
	auto softplus_out = OaFnMatrix::Softplus(input);
	auto grad_input = OaFnMatrix::SoftplusBwd(softplus_out, grad_output);
	auto result = CopyToHost(grad_input);
	
	// For large positive x, sigmoid(x) ≈ 1
	EXPECT_NEAR(result[2], 1.0f, 1e-4f) << "Large positive";
	EXPECT_NEAR(result[3], 1.0f, 1e-4f) << "Large positive";
	
	// For large negative x, sigmoid(x) ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 1e-4f) << "Large negative";
	EXPECT_NEAR(result[1], 0.0f, 1e-4f) << "Large negative";
}

// ============================================================================
// GeluBwd Tests (Approximation Variants)
// ============================================================================

TEST_VK(TestFnMatrixActivationBwd, GeluForwardMatchesFusedGeglu) {
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());

	const std::vector<float> x = {-3.0f, -1.5f, -0.75f, -0.1f,
		0.0f, 0.1f, 0.75f, 1.5f, 3.0f};
	const OaU32 n = static_cast<OaU32>(x.size());
	auto standalone = OaFnMatrix::Gelu(CreateFromHost(x, OaMatrixShape{n}));

	// Geglu([up; gate]) = up * GELU(gate). Unit up isolates the shared GELU
	// curve and catches constant drift between standalone and fused paths.
	std::vector<float> fusedInput(2U * n, 1.0f);
	for (OaU32 i = 0; i < n; ++i) fusedInput[n + i] = x[i];
	auto fused = OaFnMatrix::Geglu(
		CreateFromHost(fusedInput, OaMatrixShape{2U * n}), n);

	const auto standaloneHost = CopyToHost(standalone);
	const auto fusedHost = CopyToHost(fused);
	ASSERT_EQ(standaloneHost.size(), n);
	ASSERT_EQ(fusedHost.size(), n);
	for (OaU32 i = 0; i < n; ++i) {
		EXPECT_NEAR(standaloneHost[i], fusedHost[i], 1e-7f)
			<< "GELU curve mismatch at x=" << x[i];
	}
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_NumericalGradient) {
	// Test GELU backward pass against numerical gradient
	// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	// Forward pass
	auto output = OaFnMatrix::Gelu(input);
	
	// Analytical gradient
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{5});
	auto grad_input = OaFnMatrix::GeluBwd(input, grad_output);
	auto analytical_grad = CopyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = NumericalGradient(input_data, OaMatrixShape{5},
		[](const OaMatrix& x) { return OaFnMatrix::Gelu(x); });
	
	// Compare (GELU uses approximation, so tolerance is higher)
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-2f)
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_ZeroPoint) {
	// Test GELU gradient at x=0
	// GELU'(0) = 0.5 (by symmetry)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {0.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{1});
	
	std::vector<float> grad_output_data = {1.0f};
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{1});
	
	auto grad_input = OaFnMatrix::GeluBwd(input, grad_output);
	auto result = CopyToHost(grad_input);
	
	EXPECT_NEAR(result[0], 0.5f, 0.05f) << "GELU gradient at x=0 should be ~0.5";
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_Symmetry) {
	// Test that GELU gradient has expected symmetry properties
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-1.0f, 1.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{2});
	
	std::vector<float> grad_output_data(2, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{2});
	
	auto grad_input = OaFnMatrix::GeluBwd(input, grad_output);
	auto result = CopyToHost(grad_input);
	
	// GELU'(-x) + GELU'(x) should be close to 1 (by symmetry of GELU)
	float sum = result[0] + result[1];
	EXPECT_NEAR(sum, 1.0f, 0.1f) << "GELU gradient symmetry";
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_LargeValues) {
	// Test GELU backward with large values
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data = {-5.0f, 5.0f};
	auto input = CreateFromHost(input_data, OaMatrixShape{2});
	
	std::vector<float> grad_output_data(2, 1.0f);
	auto grad_output = CreateFromHost(grad_output_data, OaMatrixShape{2});
	
	auto grad_input = OaFnMatrix::GeluBwd(input, grad_output);
	auto result = CopyToHost(grad_input);
	
	// For large positive x, GELU(x) ≈ x, so gradient ≈ 1
	EXPECT_NEAR(result[1], 1.0f, 0.1f) << "Large positive gradient";
	
	// For large negative x, GELU(x) ≈ 0, so gradient ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 0.1f) << "Large negative gradient";
}
