// Manual tests for Core/FnMatrix index operations
// These operations use manual session lowering because their shape rules are complex.

#include <gtest/gtest.h>
#include "../../../oaTest.h"
#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <algorithm>
#include <numeric>
#include <vector>

// Helper to create matrix from host data
static oa::Matrix createMatrixFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

static oa::Engine* GRt = nullptr;

class TestFnMatrixIndex : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixIndex";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// ============================================================================
// Argmax Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Argmax_1D) {
	// Test argmax on 1D tensor
	constexpr oa::U32 N = 10;
	std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f, 8.0f, 0.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::I64 idx = oa::FnMatrix::argmax(a);
	
	// CPU reference: find index of maximum value
	auto max_it = std::max_element(data.begin(), data.end());
	oa::I64 expected_idx = std::distance(data.begin(), max_it);
	
	EXPECT_EQ(idx, expected_idx) << "expected index " << expected_idx << " (value=" << *max_it << ")";
}

TEST_VK(TestFnMatrixIndex, Argmax_AllNegative) {
	// Test argmax with all negative values
	constexpr oa::U32 N = 5;
	std::vector<float> data = {-5.0f, -2.0f, -8.0f, -1.0f, -3.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::I64 idx = oa::FnMatrix::argmax(a);
	
	// CPU reference: -1.0f is maximum at index 3
	EXPECT_EQ(idx, 3);
}

TEST_VK(TestFnMatrixIndex, Argmax_Duplicates) {
	// Test argmax with duplicate maximum values (should return first occurrence)
	constexpr oa::U32 N = 6;
	std::vector<float> data = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f, 4.0f};
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	oa::I64 idx = oa::FnMatrix::argmax(a);
	
	// CPU reference: first occurrence of 5.0f is at index 1
	EXPECT_EQ(idx, 1);
}

TEST_VK(TestFnMatrixIndex, MaskedCategoricalAccuracyCount) {
	const std::vector<float> logits = {
		5.0F, 1.0F, 0.0F,
		0.0F, 5.0F, 1.0F,
		0.0F, 1.0F, 5.0F,
		5.0F, 1.0F, 0.0F};
	const std::vector<oa::I32> labels = {0, 1, 2, 0};
	const std::vector<float> mask = {1.0F, 0.0F, 1.0F, 0.0F};
	auto logitsM = createMatrixFromHost(logits, oa::MatrixShape{4, 3});
	auto labelsM = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(labels.data()),
			labels.size() * sizeof(oa::I32)), oa::MatrixShape{4}, oa::ScalarType::Int32);
	auto maskM = createMatrixFromHost(mask, oa::MatrixShape{4});

	auto count = oa::FnMatrix::maskedCategoricalAccuracyCount(logitsM, labelsM, maskM);
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ASSERT_TRUE(count.hasStorage());
	EXPECT_EQ(count.dataAs<const oa::U32>()[0], 2u);
}

// ============================================================================
// reshape Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Reshape_1Dto2D) {
	// Test reshape from 1D to 2D
	constexpr oa::U32 N = 12;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);  // 0, 1, 2, ..., 11
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto reshaped = oa::FnMatrix::reshape(a, {3, 4});
	
	EXPECT_EQ(reshaped.getShape().rank, 2);
	EXPECT_EQ(reshaped.getShape()[0], 3);
	EXPECT_EQ(reshaped.getShape()[1], 4);
	
	// verify data is unchanged
	std::vector<float> got(N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(reshaped, got.data(), N * sizeof(float)).isOk());
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Reshape_2Dto1D) {
	// Test reshape from 2D to 1D
	constexpr oa::U32 M = 3, N = 4;
	std::vector<float> data(M * N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{M, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto reshaped = oa::FnMatrix::reshape(a, {M * N});
	
	EXPECT_EQ(reshaped.getShape().rank, 1);
	EXPECT_EQ(reshaped.getShape()[0], M * N);
	
	// verify data is unchanged
	std::vector<float> got(M * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(reshaped, got.data(), M * N * sizeof(float)).isOk());
	for (oa::U32 i = 0; i < M * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Reshape_3D) {
	// Test reshape to 3D
	constexpr oa::U32 N = 24;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto reshaped = oa::FnMatrix::reshape(a, {2, 3, 4});
	
	EXPECT_EQ(reshaped.getShape().rank, 3);
	EXPECT_EQ(reshaped.getShape()[0], 2);
	EXPECT_EQ(reshaped.getShape()[1], 3);
	EXPECT_EQ(reshaped.getShape()[2], 4);
	
	// verify data is unchanged
	std::vector<float> got(N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(reshaped, got.data(), N * sizeof(float)).isOk());
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "index " << i;
	}
}

// ============================================================================
// Slice Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Slice_1D_Basic) {
	// Test basic 1D slice
	constexpr oa::U32 N = 10;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);  // 0, 1, 2, ..., 9
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto sliced = oa::FnMatrix::slice(a, 0, 2, 7);  // [2:7] = [2, 3, 4, 5, 6]
	
	EXPECT_EQ(sliced.getShape()[0], 5);
	
	std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> got(5);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(sliced, got.data(), 5 * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 5; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Slice_2D_Rows) {
	// Test slicing rows from 2D tensor
	constexpr oa::U32 M = 5, N = 3;
	std::vector<float> data(M * N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = createMatrixFromHost(data, oa::MatrixShape{M, N});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto sliced = oa::FnMatrix::slice(a, 0, 1, 4);  // rows [1:4] = rows 1, 2, 3
	
	EXPECT_EQ(sliced.getShape()[0], 3);
	EXPECT_EQ(sliced.getShape()[1], N);
	
	// expected: rows 1, 2, 3 from original
	std::vector<float> expected = {
		3.0f, 4.0f, 5.0f,   // Row 1
		6.0f, 7.0f, 8.0f,   // Row 2
		9.0f, 10.0f, 11.0f  // Row 3
	};
	std::vector<float> got(3 * N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(sliced, got.data(), 3 * N * sizeof(float)).isOk());
	
	for (oa::U32 i = 0; i < 3 * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Slice_BackwardPadsOutsideInterval) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope ctx_scope(ctx);
	auto input = createMatrixFromHost(
		{1.0F, 2.0F, 3.0F, 4.0F, 5.0F}, oa::MatrixShape{5});
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	auto sliced = oa::FnMatrix::slice(input, 0, 1, 4);
	auto loss = oa::FnMatrix::sum(sliced);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> gradient(5);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		input.gradMatrix(), gradient.data(),
		gradient.size() * sizeof(float)).isOk());
	const std::vector<float> expected = {0.0F, 1.0F, 1.0F, 1.0F, 0.0F};
	EXPECT_EQ(gradient, expected);
}

// Note: Slice API doesn't support step parameter
// Only supports [start:end] slicing, not [start:end:step]

// Note: CopyAtOffset is not exposed in the public oa::FnMatrix API
// It exists as a kernel but is not available for direct testing
