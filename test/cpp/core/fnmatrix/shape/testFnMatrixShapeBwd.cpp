// Test/Core/FnMatrix/Shape/TestFnMatrixShapeBwd.cpp
// backward pass tests for shape operations (Concat, Slice)

#include <gtest/gtest.h>
#include <oa/core.h>
#include <oa/ml.h>
#include <oa/runtime/executionSession.h>
#include <vector>

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

class ShapeBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// initialize runtime if needed
	}
};

// ============================================================================
// Concat backward Tests
// ============================================================================

TEST_F(ShapeBwd, ConcatBwdDim0) {
	// Test Concat backward along dimension 0
	std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
	
	auto a = createMatrixFromHost(a_data, oa::MatrixShape{2, 2});
	auto b = createMatrixFromHost(b_data, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	a.requiresGrad_(true);
	b.requiresGrad_(true);
	
	// Concat along dim 0: [2,2] + [2,2] -> [4,2]
	auto result = oa::FnMatrix::concat({a, b}, 0);
	
	// backward with ones
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad_a = copyMatrixToHost(a.grad());
	auto grad_b = copyMatrixToHost(b.grad());
	
	// Each input should receive its portion of the gradient
	ASSERT_EQ(grad_a.size(), 4);
	ASSERT_EQ(grad_b.size(), 4);
	
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

TEST_F(ShapeBwd, ConcatBwdDim1) {
	// Test Concat backward along dimension 1
	std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
	
	auto a = createMatrixFromHost(a_data, oa::MatrixShape{2, 2});
	auto b = createMatrixFromHost(b_data, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	a.requiresGrad_(true);
	b.requiresGrad_(true);
	
	// Concat along dim 1: [2,2] + [2,2] -> [2,4]
	auto result = oa::FnMatrix::concat({a, b}, 1);
	
	// backward with ones
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad_a = copyMatrixToHost(a.grad());
	auto grad_b = copyMatrixToHost(b.grad());
	
	ASSERT_EQ(grad_a.size(), 4);
	ASSERT_EQ(grad_b.size(), 4);
	
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

TEST_F(ShapeBwd, ConcatBwdMultipleInputs) {
	// Test Concat backward with 3 inputs
	std::vector<float> a_data = {1.0f, 2.0f};
	std::vector<float> b_data = {3.0f, 4.0f};
	std::vector<float> c_data = {5.0f, 6.0f};
	
	auto a = createMatrixFromHost(a_data, oa::MatrixShape{2});
	auto b = createMatrixFromHost(b_data, oa::MatrixShape{2});
	auto c = createMatrixFromHost(c_data, oa::MatrixShape{2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	a.requiresGrad_(true);
	b.requiresGrad_(true);
	c.requiresGrad_(true);
	
	// Concat: [2] + [2] + [2] -> [6]
	auto result = oa::FnMatrix::concat({a, b, c}, 0);
	
	// backward with custom gradient
	std::vector<float> grad_out_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{6});
	result.backward(grad_out);
	
	auto grad_a = copyMatrixToHost(a.grad());
	auto grad_b = copyMatrixToHost(b.grad());
	auto grad_c = copyMatrixToHost(c.grad());
	
	// Each input should receive its corresponding slice of gradient
	ASSERT_EQ(grad_a.size(), 2);
	EXPECT_NEAR(grad_a[0], 1.0f, 1e-5f);
	EXPECT_NEAR(grad_a[1], 2.0f, 1e-5f);
	
	ASSERT_EQ(grad_b.size(), 2);
	EXPECT_NEAR(grad_b[0], 3.0f, 1e-5f);
	EXPECT_NEAR(grad_b[1], 4.0f, 1e-5f);
	
	ASSERT_EQ(grad_c.size(), 2);
	EXPECT_NEAR(grad_c[0], 5.0f, 1e-5f);
	EXPECT_NEAR(grad_c[1], 6.0f, 1e-5f);
}

TEST_F(ShapeBwd, ConcatBwdDifferentSizes) {
	// Test Concat backward with different sized inputs
	std::vector<float> a_data = {1.0f, 2.0f};
	std::vector<float> b_data = {3.0f, 4.0f, 5.0f, 6.0f};
	
	auto a = createMatrixFromHost(a_data, oa::MatrixShape{2});
	auto b = createMatrixFromHost(b_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	a.requiresGrad_(true);
	b.requiresGrad_(true);
	
	// Concat: [2] + [4] -> [6]
	auto result = oa::FnMatrix::concat({a, b}, 0);
	
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad_a = copyMatrixToHost(a.grad());
	auto grad_b = copyMatrixToHost(b.grad());
	
	ASSERT_EQ(grad_a.size(), 2);
	ASSERT_EQ(grad_b.size(), 4);
	
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

// ============================================================================
// Slice backward Tests
// ============================================================================

TEST_F(ShapeBwd, SliceBwdBasic) {
	// Test Slice backward: gradient should be padded with zeros
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{6});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	// Slice [1:4] from [6] -> [3]
	auto result = oa::FnMatrix::slice(x, 0, 1, 4);
	
	// backward with ones
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be [0, 1, 1, 1, 0, 0]
	ASSERT_EQ(grad.size(), 6);
	EXPECT_NEAR(grad[0], 0.0f, 1e-5f);  // Before slice
	EXPECT_NEAR(grad[1], 1.0f, 1e-5f);  // Slice start
	EXPECT_NEAR(grad[2], 1.0f, 1e-5f);  // Slice middle
	EXPECT_NEAR(grad[3], 1.0f, 1e-5f);  // Slice end
	EXPECT_NEAR(grad[4], 0.0f, 1e-5f);  // After slice
	EXPECT_NEAR(grad[5], 0.0f, 1e-5f);  // After slice
}

TEST_F(ShapeBwd, SliceBwd2D) {
	// Test Slice backward on 2D tensor
	std::vector<float> x_data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{3, 3});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	// Slice rows [0:2] from [3,3] -> [2,3]
	auto result = oa::FnMatrix::slice(x, 0, 0, 2);
	
	// backward with custom gradient
	std::vector<float> grad_out_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{2, 3});
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be [[1,2,3], [4,5,6], [0,0,0]]
	ASSERT_EQ(grad.size(), 9);
	EXPECT_NEAR(grad[0], 1.0f, 1e-5f);
	EXPECT_NEAR(grad[1], 2.0f, 1e-5f);
	EXPECT_NEAR(grad[2], 3.0f, 1e-5f);
	EXPECT_NEAR(grad[3], 4.0f, 1e-5f);
	EXPECT_NEAR(grad[4], 5.0f, 1e-5f);
	EXPECT_NEAR(grad[5], 6.0f, 1e-5f);
	EXPECT_NEAR(grad[6], 0.0f, 1e-5f);
	EXPECT_NEAR(grad[7], 0.0f, 1e-5f);
	EXPECT_NEAR(grad[8], 0.0f, 1e-5f);
}

TEST_F(ShapeBwd, SliceBwdMiddle) {
	// Test Slice backward for middle slice
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{5});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	// Slice [1:4] from [5] -> [3]
	auto result = oa::FnMatrix::slice(x, 0, 1, 4);
	
	std::vector<float> grad_out_data = {10.0f, 20.0f, 30.0f};
	auto grad_out = createMatrixFromHost(grad_out_data, oa::MatrixShape{3});
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be [0, 10, 20, 30, 0]
	ASSERT_EQ(grad.size(), 5);
	EXPECT_NEAR(grad[0], 0.0f, 1e-5f);
	EXPECT_NEAR(grad[1], 10.0f, 1e-5f);
	EXPECT_NEAR(grad[2], 20.0f, 1e-5f);
	EXPECT_NEAR(grad[3], 30.0f, 1e-5f);
	EXPECT_NEAR(grad[4], 0.0f, 1e-5f);
}

TEST_F(ShapeBwd, SliceBwdFullRange) {
	// Test Slice backward for full range (should be identity)
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto x = createMatrixFromHost(x_data, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	x.requiresGrad_(true);
	
	// Slice [0:4] from [4] -> [4] (full range)
	auto result = oa::FnMatrix::slice(x, 0, 0, 4);
	
	auto grad_out = oa::FnMatrix::ones(result.getShape(), result.getDtype());
	result.backward(grad_out);
	
	auto grad = copyMatrixToHost(x.grad());
	
	// Gradient should be all ones (identity)
	ASSERT_EQ(grad.size(), 4);
	for (float val : grad) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

// ============================================================================
// Concat + Slice Round-Trip Tests
// ============================================================================

TEST_F(ShapeBwd, ConcatSliceRoundTrip) {
	// Test that Concat followed by Slice preserves gradients
	std::vector<float> a_data = {1.0f, 2.0f};
	std::vector<float> b_data = {3.0f, 4.0f};
	
	auto a = createMatrixFromHost(a_data, oa::MatrixShape{2});
	auto b = createMatrixFromHost(b_data, oa::MatrixShape{2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	a.requiresGrad_(true);
	b.requiresGrad_(true);
	
	// Concat then slice back to original
	auto concat = oa::FnMatrix::concat({a, b}, 0);  // [4]
	auto slice_a = oa::FnMatrix::slice(concat, 0, 0, 2);  // [2]
	auto slice_b = oa::FnMatrix::slice(concat, 0, 2, 4);  // [2]
	
	// backward on both slices
	auto grad_out = oa::FnMatrix::ones(oa::MatrixShape{2}, oa::ScalarType::Float32);
	slice_a.backward(grad_out);
	slice_b.backward(grad_out);
	
	auto grad_a = copyMatrixToHost(a.grad());
	auto grad_b = copyMatrixToHost(b.grad());
	
	// Both should receive gradient of 1
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}
