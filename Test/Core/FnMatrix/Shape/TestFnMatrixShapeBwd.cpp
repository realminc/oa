// Test/Core/FnMatrix/Shape/TestFnMatrixShapeBwd.cpp
// Backward pass tests for shape operations (Concat, Slice)

#include <gtest/gtest.h>
#include <Oa/Core.h>
#include <Oa/Ml.h>
#include <vector>

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

class ShapeBwd : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize runtime if needed
	}
};

// ============================================================================
// Concat Backward Tests
// ============================================================================

TEST_F(ShapeBwd, ConcatBwdDim0) {
	// Test Concat backward along dimension 0
	std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
	
	auto a = CreateMatrixFromHost(a_data, OaMatrixShape{2, 2});
	auto b = CreateMatrixFromHost(b_data, OaMatrixShape{2, 2});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	a.RequiresGrad_(true);
	b.RequiresGrad_(true);
	
	// Concat along dim 0: [2,2] + [2,2] -> [4,2]
	auto result = OaFnMatrix::Concat({a, b}, 0);
	
	// Backward with ones
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad_a = CopyMatrixToHost(a.Grad());
	auto grad_b = CopyMatrixToHost(b.Grad());
	
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
	
	auto a = CreateMatrixFromHost(a_data, OaMatrixShape{2, 2});
	auto b = CreateMatrixFromHost(b_data, OaMatrixShape{2, 2});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	a.RequiresGrad_(true);
	b.RequiresGrad_(true);
	
	// Concat along dim 1: [2,2] + [2,2] -> [2,4]
	auto result = OaFnMatrix::Concat({a, b}, 1);
	
	// Backward with ones
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad_a = CopyMatrixToHost(a.Grad());
	auto grad_b = CopyMatrixToHost(b.Grad());
	
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
	
	auto a = CreateMatrixFromHost(a_data, OaMatrixShape{2});
	auto b = CreateMatrixFromHost(b_data, OaMatrixShape{2});
	auto c = CreateMatrixFromHost(c_data, OaMatrixShape{2});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	a.RequiresGrad_(true);
	b.RequiresGrad_(true);
	c.RequiresGrad_(true);
	
	// Concat: [2] + [2] + [2] -> [6]
	auto result = OaFnMatrix::Concat({a, b, c}, 0);
	
	// Backward with custom gradient
	std::vector<float> grad_out_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{6});
	result.Backward(grad_out);
	
	auto grad_a = CopyMatrixToHost(a.Grad());
	auto grad_b = CopyMatrixToHost(b.Grad());
	auto grad_c = CopyMatrixToHost(c.Grad());
	
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
	
	auto a = CreateMatrixFromHost(a_data, OaMatrixShape{2});
	auto b = CreateMatrixFromHost(b_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	a.RequiresGrad_(true);
	b.RequiresGrad_(true);
	
	// Concat: [2] + [4] -> [6]
	auto result = OaFnMatrix::Concat({a, b}, 0);
	
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad_a = CopyMatrixToHost(a.Grad());
	auto grad_b = CopyMatrixToHost(b.Grad());
	
	ASSERT_EQ(grad_a.size(), 2);
	ASSERT_EQ(grad_b.size(), 4);
	
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

// ============================================================================
// Slice Backward Tests
// ============================================================================

TEST_F(ShapeBwd, SliceBwdBasic) {
	// Test Slice backward: gradient should be padded with zeros
	std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{6});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	// Slice [1:4] from [6] -> [3]
	auto result = OaFnMatrix::Slice(x, 0, 1, 4);
	
	// Backward with ones
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
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
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{3, 3});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	// Slice rows [0:2] from [3,3] -> [2,3]
	auto result = OaFnMatrix::Slice(x, 0, 0, 2);
	
	// Backward with custom gradient
	std::vector<float> grad_out_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{2, 3});
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
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
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{5});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	// Slice [1:4] from [5] -> [3]
	auto result = OaFnMatrix::Slice(x, 0, 1, 4);
	
	std::vector<float> grad_out_data = {10.0f, 20.0f, 30.0f};
	auto grad_out = CreateMatrixFromHost(grad_out_data, OaMatrixShape{3});
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
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
	
	auto x = CreateMatrixFromHost(x_data, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	x.RequiresGrad_(true);
	
	// Slice [0:4] from [4] -> [4] (full range)
	auto result = OaFnMatrix::Slice(x, 0, 0, 4);
	
	auto grad_out = OaFnMatrix::Ones(result.GetShape(), result.GetDtype());
	result.Backward(grad_out);
	
	auto grad = CopyMatrixToHost(x.Grad());
	
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
	
	auto a = CreateMatrixFromHost(a_data, OaMatrixShape{2});
	auto b = CreateMatrixFromHost(b_data, OaMatrixShape{2});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	a.RequiresGrad_(true);
	b.RequiresGrad_(true);
	
	// Concat then slice back to original
	auto concat = OaFnMatrix::Concat({a, b}, 0);  // [4]
	auto slice_a = OaFnMatrix::Slice(concat, 0, 0, 2);  // [2]
	auto slice_b = OaFnMatrix::Slice(concat, 0, 2, 4);  // [2]
	
	// Backward on both slices
	auto grad_out = OaFnMatrix::Ones(OaMatrixShape{2}, OaScalarType::Float32);
	slice_a.Backward(grad_out);
	slice_b.Backward(grad_out);
	
	auto grad_a = CopyMatrixToHost(a.Grad());
	auto grad_b = CopyMatrixToHost(b.Grad());
	
	// Both should receive gradient of 1
	for (float val : grad_a) EXPECT_NEAR(val, 1.0f, 1e-5f);
	for (float val : grad_b) EXPECT_NEAR(val, 1.0f, 1e-5f);
}

