// Test/Core/FnMatrix/Elemwise/TestFnMatrixElemwiseBwd.cpp
// Numerical gradient tests for elemwise backward passes

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

// Numerical gradient helper
template<typename Func>
static OaMatrix NumericalGradient(const std::vector<float>& input_data, const OaMatrixShape& shape, Func forward_fn) {
	const OaF32 h = 1e-4f;
	std::vector<float> grad(input_data.size());
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	for (size_t i = 0; i < input_data.size(); ++i) {
		// f(x + h)
		std::vector<float> x_plus = input_data;
		x_plus[i] += h;
		auto mat_plus = CreateMatrixFromHost(x_plus, shape);
		auto out_plus = forward_fn(mat_plus);
		auto result_plus = CopyMatrixToHost(out_plus);
		
		// f(x - h)
		std::vector<float> x_minus = input_data;
		x_minus[i] -= h;
		auto mat_minus = CreateMatrixFromHost(x_minus, shape);
		auto out_minus = forward_fn(mat_minus);
		auto result_minus = CopyMatrixToHost(out_minus);
		
		// (f(x+h) - f(x-h)) / (2h)
		OaF32 sum_grad = 0.0f;
		for (size_t j = 0; j < result_plus.size(); ++j) {
			sum_grad += (result_plus[j] - result_minus[j]) / (2.0f * h);
		}
		grad[i] = sum_grad;
	}
	
	return CreateMatrixFromHost(grad, shape);
}

class ElemwiseBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize runtime if needed
	}
};

// ============================================================================
// Div Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, DivBwdNumericalGradient) {
	// Test Div backward: d/dx (x / y)
	std::vector<float> x_data = {2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> y_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	auto y = CreateMatrixFromHost(y_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	// Forward with autograd
	x.RequiresGrad_(true);
	auto result = OaFnMatrix::Div(x, y);
	
	// Backward
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[&y_data](const OaMatrix& x_mat) {
			auto y_mat = CreateMatrixFromHost(y_data, OaMatrixShape{4});
			return OaFnMatrix::Div(x_mat, y_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, DivBwdZeroDenominator) {
	// Test Div backward with small denominator
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> y_data = {0.1f, 0.2f, 0.3f, 0.4f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	auto y = CreateMatrixFromHost(y_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Div(x, y);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
	// Gradient should be 1/y for each element
	for (size_t i = 0; i < grad.size(); ++i) {
		EXPECT_NEAR(grad[i], 1.0f / y_data[i], 1e-4f);
	}
}

// ============================================================================
// Pow Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, PowBwdNumericalGradient) {
	// Test Pow backward: d/dx (x^n)
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	const OaF32 exponent = 2.0f;
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Pow(x, exponent);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[exponent](const OaMatrix& x_mat) {
			return OaFnMatrix::Pow(x_mat, exponent);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, PowBwdFractionalExponent) {
	// Test Pow backward with fractional exponent
	std::vector<float> x_data = {1.0f, 4.0f, 9.0f, 16.0f};
	const OaF32 exponent = 0.5f;  // Square root
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Pow(x, exponent);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
	// Gradient should be 0.5 * x^(-0.5) = 0.5 / sqrt(x)
	for (size_t i = 0; i < grad.size(); ++i) {
		OaF32 expected = 0.5f / std::sqrt(x_data[i]);
		EXPECT_NEAR(grad[i], expected, 1e-4f);
	}
}

// ============================================================================
// Log Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, LogBwdNumericalGradient) {
	// Test Log backward: d/dx log(x) = 1/x
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Log(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Log(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, LogBwdSmallValues) {
	// Test Log backward with small positive values
	std::vector<float> x_data = {0.1f, 0.5f, 1.0f, 2.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Log(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
	// Gradient should be 1/x
	for (size_t i = 0; i < grad.size(); ++i) {
		EXPECT_NEAR(grad[i], 1.0f / x_data[i], 1e-4f);
	}
}

// ============================================================================
// Sqrt Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, SqrtBwdNumericalGradient) {
	// Test Sqrt backward: d/dx sqrt(x) = 1/(2*sqrt(x))
	std::vector<float> x_data = {1.0f, 4.0f, 9.0f, 16.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Sqrt(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Sqrt(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Exp Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, ExpBwdNumericalGradient) {
	// Test Exp backward: d/dx exp(x) = exp(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Exp(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Exp(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Sin/Cos Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, SinBwdNumericalGradient) {
	// Test Sin backward: d/dx sin(x) = cos(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Sin(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Sin(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, CosBwdNumericalGradient) {
	// Test Cos backward: d/dx cos(x) = -sin(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Cos(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Cos(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Reciprocal Backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, ReciprocalBwdNumericalGradient) {
	// Test Reciprocal backward: d/dx (1/x) = -1/x^2
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	auto result = OaFnMatrix::Reciprocal(x);
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto analytical_grad = CopyMatrixToHost(x.Grad());
	
	// Numerical gradient
	auto numerical_grad_mat = NumericalGradient(x_data, OaMatrixShape{4}, 
		[](const OaMatrix& x_mat) {
			return OaFnMatrix::Reciprocal(x_mat);
		});
	auto numerical_grad = CopyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

