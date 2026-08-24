// Numerical gradient tests for activation backward passes
// MishBwd, SoftplusBwd, GeluBwd variants

#include <gtest/gtest.h>
#include <oa/ml/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>

static oa::Engine* GRt = nullptr;

class TestFnMatrixActivationBwd : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixActivationBwd";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> copyToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to create matrix from host data
static oa::Matrix createFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Numerical gradient checker
// computes (f(x+h) - f(x-h)) / (2*h) for each element
template<typename ForwardFn>
static std::vector<float> numericalGradient(
	const std::vector<float>& input,
	oa::MatrixShape shape,
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
		auto x_plus = createFromHost(input_plus, shape);
		auto y_plus = forward_fn(x_plus);
		auto y_plus_data = copyToHost(y_plus);
		float sum_plus = 0.0f;
		for (float val : y_plus_data) sum_plus += val;
		
		// f(x - h)
		std::vector<float> input_minus = input;
		input_minus[i] -= epsilon;
		auto x_minus = createFromHost(input_minus, shape);
		auto y_minus = forward_fn(x_minus);
		auto y_minus_data = copyToHost(y_minus);
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
	// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	// forward pass
	auto output = oa::FnMatrix::mish(input);
	
	// Analytical gradient (from MishBwd)
	std::vector<float> grad_output_data(5, 1.0f);  // All ones
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{5});
	auto grad_input = oa::FnMatrix::mishBwd(input, grad_output);
	auto analytical_grad = copyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = numericalGradient(input_data, oa::MatrixShape{5}, 
		[](const oa::Matrix& x) { return oa::FnMatrix::mish(x); });
	
	// compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, MishBwd_ZeroGradient) {
	// Test that zero grad_output produces zero grad_input
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-1.0f, 0.0f, 1.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{3});
	
	std::vector<float> grad_output_data(3, 0.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{3});
	
	auto grad_input = oa::FnMatrix::mishBwd(input, grad_output);
	auto result = copyToHost(grad_input);
	
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "index " << i;
	}
}

TEST_VK(TestFnMatrixActivationBwd, MishBwd_LargeValues) {
	// Test Mish backward with large positive/negative values
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-10.0f, -5.0f, 5.0f, 10.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{4});
	
	std::vector<float> grad_output_data(4, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{4});
	
	auto grad_input = oa::FnMatrix::mishBwd(input, grad_output);
	auto result = copyToHost(grad_input);
	
	// For large positive x, mish(x) ≈ x, so gradient ≈ 1
	EXPECT_NEAR(result[2], 1.0f, 0.1f) << "Large positive gradient";
	EXPECT_NEAR(result[3], 1.0f, 0.1f) << "Large positive gradient";
	
	// For large negative x, mish(x) ≈ 0, so gradient ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 0.1f) << "Large negative gradient";
}

// ============================================================================
// SoftplusBwd Tests
// ============================================================================

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_NumericalGradient) {
	// Test Softplus backward pass against numerical gradient
	// softplus(x) = ln(1 + e^x)
	// d/dx softplus(x) = sigmoid(x) = 1 / (1 + e^(-x))
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	// forward pass
	auto output = oa::FnMatrix::softplus(input);
	
	// Analytical gradient. SoftplusBwd takes the forward OUTPUT y=softplus(x)
	// (it computes sigmoid(x)=1-e^-y), unlike Gelu/Silu/Mish which take the input.
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{5});
	auto grad_input = oa::FnMatrix::softplusBwd(output, grad_output);
	auto analytical_grad = copyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = numericalGradient(input_data, oa::MatrixShape{5},
		[](const oa::Matrix& x) { return oa::FnMatrix::softplus(x); });
	
	// compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f)
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_SigmoidProperty) {
	// Test that Softplus gradient equals Sigmoid
	// d/dx softplus(x) = sigmoid(x)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	// Softplus gradient. SoftplusBwd takes the forward OUTPUT y=softplus(x).
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{5});
	auto softplus_out = oa::FnMatrix::softplus(input);
	auto softplus_grad = oa::FnMatrix::softplusBwd(softplus_out, grad_output);
	auto softplus_grad_data = copyToHost(softplus_grad);
	
	// Sigmoid
	auto sigmoid_output = oa::FnMatrix::sigmoid(input);
	auto sigmoid_data = copyToHost(sigmoid_output);
	
	// compare
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(softplus_grad_data[i], sigmoid_data[i], 1e-5f)
			<< "Softplus gradient should equal Sigmoid at index " << i;
	}
}

TEST_VK(TestFnMatrixActivationBwd, SoftplusBwd_ExtremeValues) {
	// Test Softplus backward with extreme values
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-20.0f, -10.0f, 10.0f, 20.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{4});
	
	std::vector<float> grad_output_data(4, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{4});

	// SoftplusBwd takes the forward OUTPUT y=softplus(x).
	auto softplus_out = oa::FnMatrix::softplus(input);
	auto grad_input = oa::FnMatrix::softplusBwd(softplus_out, grad_output);
	auto result = copyToHost(grad_input);
	
	// For large positive x, sigmoid(x) ≈ 1
	EXPECT_NEAR(result[2], 1.0f, 1e-4f) << "Large positive";
	EXPECT_NEAR(result[3], 1.0f, 1e-4f) << "Large positive";
	
	// For large negative x, sigmoid(x) ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 1e-4f) << "Large negative";
	EXPECT_NEAR(result[1], 0.0f, 1e-4f) << "Large negative";
}

// ============================================================================
// GeluBwd tests (Approximation Variants)
// ============================================================================

TEST_VK(TestFnMatrixActivationBwd, GeluForwardMatchesFusedGeglu) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());

	const std::vector<float> x = {-3.0f, -1.5f, -0.75f, -0.1f,
		0.0f, 0.1f, 0.75f, 1.5f, 3.0f};
	const oa::U32 n = static_cast<oa::U32>(x.size());
	auto standalone = oa::FnMatrix::gelu(createFromHost(x, oa::MatrixShape{n}));

	// geglu([up; gate]) = up * GELU(gate). Unit up isolates the shared GELU
	// curve and catches constant drift between standalone and fused paths.
	std::vector<float> fusedInput(2U * n, 1.0f);
	for (oa::U32 i = 0; i < n; ++i) fusedInput[n + i] = x[i];
	auto fused = oa::FnMatrix::geglu(
		createFromHost(fusedInput, oa::MatrixShape{2U * n}), n);

	const auto standaloneHost = copyToHost(standalone);
	const auto fusedHost = copyToHost(fused);
	ASSERT_EQ(standaloneHost.size(), n);
	ASSERT_EQ(fusedHost.size(), n);
	for (oa::U32 i = 0; i < n; ++i) {
		EXPECT_NEAR(standaloneHost[i], fusedHost[i], 1e-7f)
			<< "GELU curve mismatch at x=" << x[i];
	}
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_NumericalGradient) {
	// Test GELU backward pass against numerical gradient
	// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	// forward pass
	auto output = oa::FnMatrix::gelu(input);
	
	// Analytical gradient
	std::vector<float> grad_output_data(5, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{5});
	auto grad_input = oa::FnMatrix::geluBwd(input, grad_output);
	auto analytical_grad = copyToHost(grad_input);
	
	// Numerical gradient
	auto numerical_grad = numericalGradient(input_data, oa::MatrixShape{5},
		[](const oa::Matrix& x) { return oa::FnMatrix::gelu(x); });
	
	// compare (GELU uses approximation, so tolerance is higher)
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-2f)
			<< "Mismatch at index " << i << " (input=" << input_data[i] << ")";
	}
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_ZeroPoint) {
	// Test GELU gradient at x=0
	// GELU'(0) = 0.5 (by symmetry)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {0.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{1});
	
	std::vector<float> grad_output_data = {1.0f};
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{1});
	
	auto grad_input = oa::FnMatrix::geluBwd(input, grad_output);
	auto result = copyToHost(grad_input);
	
	EXPECT_NEAR(result[0], 0.5f, 0.05f) << "GELU gradient at x=0 should be ~0.5";
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_Symmetry) {
	// Test that GELU gradient has expected symmetry properties
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-1.0f, 1.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{2});
	
	std::vector<float> grad_output_data(2, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{2});
	
	auto grad_input = oa::FnMatrix::geluBwd(input, grad_output);
	auto result = copyToHost(grad_input);
	
	// GELU'(-x) + GELU'(x) should be close to 1 (by symmetry of GELU)
	float sum = result[0] + result[1];
	EXPECT_NEAR(sum, 1.0f, 0.1f) << "GELU gradient symmetry";
}

TEST_VK(TestFnMatrixActivationBwd, GeluBwd_LargeValues) {
	// Test GELU backward with large values
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	std::vector<float> input_data = {-5.0f, 5.0f};
	auto input = createFromHost(input_data, oa::MatrixShape{2});
	
	std::vector<float> grad_output_data(2, 1.0f);
	auto grad_output = createFromHost(grad_output_data, oa::MatrixShape{2});
	
	auto grad_input = oa::FnMatrix::geluBwd(input, grad_output);
	auto result = copyToHost(grad_input);
	
	// For large positive x, GELU(x) ≈ x, so gradient ≈ 1
	EXPECT_NEAR(result[1], 1.0f, 0.1f) << "Large positive gradient";
	
	// For large negative x, GELU(x) ≈ 0, so gradient ≈ 0
	EXPECT_NEAR(result[0], 0.0f, 0.1f) << "Large negative gradient";
}
