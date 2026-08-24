// Test/Core/FnMatrix/Elemwise/TestFnMatrixElemwiseBwd.cpp
// Numerical gradient tests for elemwise backward passes

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

// Numerical gradient helper
template<typename Func>
static oa::Matrix numericalGradient(const std::vector<float>& input_data, const oa::MatrixShape& shape, Func forward_fn) {
	const oa::F32 h = 1e-4f;
	std::vector<float> grad(input_data.size());
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	for (size_t i = 0; i < input_data.size(); ++i) {
		// f(x + h)
		std::vector<float> x_plus = input_data;
		x_plus[i] += h;
		auto mat_plus = createMatrixFromHost(x_plus, shape);
		auto out_plus = forward_fn(mat_plus);
		auto result_plus = copyMatrixToHost(out_plus);
		
		// f(x - h)
		std::vector<float> x_minus = input_data;
		x_minus[i] -= h;
		auto mat_minus = createMatrixFromHost(x_minus, shape);
		auto out_minus = forward_fn(mat_minus);
		auto result_minus = copyMatrixToHost(out_minus);
		
		// (f(x+h) - f(x-h)) / (2h)
		oa::F32 sum_grad = 0.0f;
		for (size_t j = 0; j < result_plus.size(); ++j) {
			sum_grad += (result_plus[j] - result_minus[j]) / (2.0f * h);
		}
		grad[i] = sum_grad;
	}
	
	return createMatrixFromHost(grad, shape);
}

class ElemwiseBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// initialize runtime if needed
	}
};

// ============================================================================
// Div backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, DivBwdNumericalGradient) {
	// Test Div backward: d/dx (x / y)
	std::vector<float> x_data = {2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> y_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	auto y = createMatrixFromHost(y_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	// forward with autograd
	x.requiresGrad_(true);
	auto result = oa::FnMatrix::div(x, y);
	
	// backward
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[&y_data](const oa::Matrix& x_mat) {
			auto y_mat = createMatrixFromHost(y_data, oa::MatrixShape{4});
			return oa::FnMatrix::div(x_mat, y_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
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
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	auto y = createMatrixFromHost(y_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::div(x, y);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be 1/y for each element
	for (size_t i = 0; i < grad.size(); ++i) {
		EXPECT_NEAR(grad[i], 1.0f / y_data[i], 1e-4f);
	}
}

// ============================================================================
// Pow backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, PowBwdNumericalGradient) {
	// Test Pow backward: d/dx (x^n)
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	const oa::F32 exponent = 2.0f;
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::pow(x, exponent);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[exponent](const oa::Matrix& x_mat) {
			return oa::FnMatrix::pow(x_mat, exponent);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, PowBwdFractionalExponent) {
	// Test Pow backward with fractional exponent
	std::vector<float> x_data = {1.0f, 4.0f, 9.0f, 16.0f};
	const oa::F32 exponent = 0.5f;  // Square root
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::pow(x, exponent);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be 0.5 * x^(-0.5) = 0.5 / sqrt(x)
	for (size_t i = 0; i < grad.size(); ++i) {
		oa::F32 expected = 0.5f / std::sqrt(x_data[i]);
		EXPECT_NEAR(grad[i], expected, 1e-4f);
	}
}

// ============================================================================
// log backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, LogBwdNumericalGradient) {
	// Test log backward: d/dx log(x) = 1/x
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::log(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::log(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, LogBwdSmallValues) {
	// Test log backward with small positive values
	std::vector<float> x_data = {0.1f, 0.5f, 1.0f, 2.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::log(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be 1/x
	for (size_t i = 0; i < grad.size(); ++i) {
		EXPECT_NEAR(grad[i], 1.0f / x_data[i], 1e-4f);
	}
}

// ============================================================================
// Sqrt backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, SqrtBwdNumericalGradient) {
	// Test Sqrt backward: d/dx sqrt(x) = 1/(2*sqrt(x))
	std::vector<float> x_data = {1.0f, 4.0f, 9.0f, 16.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::sqrt(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::sqrt(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Exp backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, ExpBwdNumericalGradient) {
	// Test Exp backward: d/dx exp(x) = exp(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::exp(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::exp(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Sin/Cos backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, SinBwdNumericalGradient) {
	// Test Sin backward: d/dx sin(x) = cos(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::sin(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::sin(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

TEST_F(ElemwiseBwd, CosBwdNumericalGradient) {
	// Test Cos backward: d/dx cos(x) = -sin(x)
	std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::cos(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::cos(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}

// ============================================================================
// Reciprocal backward Tests
// ============================================================================

TEST_F(ElemwiseBwd, ReciprocalBwdNumericalGradient) {
	// Test Reciprocal backward: d/dx (1/x) = -1/x^2
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	auto result = oa::FnMatrix::reciprocal(x);
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto analytical_grad = copyMatrixToHost(x.grad());
	
	// Numerical gradient
	auto numerical_grad_mat = numericalGradient(x_data, oa::MatrixShape{4}, 
		[](const oa::Matrix& x_mat) {
			return oa::FnMatrix::reciprocal(x_mat);
		});
	auto numerical_grad = copyMatrixToHost(numerical_grad_mat);
	
	ASSERT_EQ(analytical_grad.size(), numerical_grad.size());
	for (size_t i = 0; i < analytical_grad.size(); ++i) {
		EXPECT_NEAR(analytical_grad[i], numerical_grad[i], 1e-3f) 
			<< "Gradient mismatch at index " << i;
	}
}
