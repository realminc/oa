// Manual tests for Core/FnMatrix Index operations
// These operations are too complex for autogen (manual_context body, complex shapes)

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <algorithm>
#include <numeric>
#include <vector>

// Helper to create matrix from host data
static OaMatrix CreateMatrixFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixIndex : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixIndex";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// ============================================================================
// Argmax Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Argmax_1D) {
	// Test argmax on 1D tensor
	constexpr OaU32 N = 10;
	std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f, 8.0f, 0.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaI64 idx = OaFnMatrix::Argmax(a);
	
	// CPU reference: find index of maximum value
	auto max_it = std::max_element(data.begin(), data.end());
	OaI64 expected_idx = std::distance(data.begin(), max_it);
	
	EXPECT_EQ(idx, expected_idx) << "Expected index " << expected_idx << " (value=" << *max_it << ")";
}

TEST_VK(TestFnMatrixIndex, Argmax_AllNegative) {
	// Test argmax with all negative values
	constexpr OaU32 N = 5;
	std::vector<float> data = {-5.0f, -2.0f, -8.0f, -1.0f, -3.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaI64 idx = OaFnMatrix::Argmax(a);
	
	// CPU reference: -1.0f is maximum at index 3
	EXPECT_EQ(idx, 3);
}

TEST_VK(TestFnMatrixIndex, Argmax_Duplicates) {
	// Test argmax with duplicate maximum values (should return first occurrence)
	constexpr OaU32 N = 6;
	std::vector<float> data = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f, 4.0f};
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	OaI64 idx = OaFnMatrix::Argmax(a);
	
	// CPU reference: first occurrence of 5.0f is at index 1
	EXPECT_EQ(idx, 1);
}

TEST_VK(TestFnMatrixIndex, MaskedCategoricalAccuracyCount) {
	const std::vector<float> logits = {
		5.0F, 1.0F, 0.0F,
		0.0F, 5.0F, 1.0F,
		0.0F, 1.0F, 5.0F,
		5.0F, 1.0F, 0.0F};
	const std::vector<OaI32> labels = {0, 1, 2, 0};
	const std::vector<float> mask = {1.0F, 0.0F, 1.0F, 0.0F};
	auto logitsM = CreateMatrixFromHost(logits, OaMatrixShape{4, 3});
	auto labelsM = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(labels.data()),
			labels.size() * sizeof(OaI32)), OaMatrixShape{4}, OaScalarType::Int32);
	auto maskM = CreateMatrixFromHost(mask, OaMatrixShape{4});

	auto count = OaFnMatrix::MaskedCategoricalAccuracyCount(logitsM, labelsM, maskM);
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ASSERT_TRUE(count.HasStorage());
	EXPECT_EQ(count.DataAs<const OaU32>()[0], 2u);
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Reshape_1Dto2D) {
	// Test reshape from 1D to 2D
	constexpr OaU32 N = 12;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);  // 0, 1, 2, ..., 11
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto reshaped = OaFnMatrix::Reshape(a, {3, 4});
	
	EXPECT_EQ(reshaped.GetShape().Rank, 2);
	EXPECT_EQ(reshaped.GetShape()[0], 3);
	EXPECT_EQ(reshaped.GetShape()[1], 4);
	
	// Verify data is unchanged
	std::vector<float> got(N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(reshaped, got.data(), N * sizeof(float)).IsOk());
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Reshape_2Dto1D) {
	// Test reshape from 2D to 1D
	constexpr OaU32 M = 3, N = 4;
	std::vector<float> data(M * N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{M, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto reshaped = OaFnMatrix::Reshape(a, {M * N});
	
	EXPECT_EQ(reshaped.GetShape().Rank, 1);
	EXPECT_EQ(reshaped.GetShape()[0], M * N);
	
	// Verify data is unchanged
	std::vector<float> got(M * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(reshaped, got.data(), M * N * sizeof(float)).IsOk());
	for (OaU32 i = 0; i < M * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Reshape_3D) {
	// Test reshape to 3D
	constexpr OaU32 N = 24;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto reshaped = OaFnMatrix::Reshape(a, {2, 3, 4});
	
	EXPECT_EQ(reshaped.GetShape().Rank, 3);
	EXPECT_EQ(reshaped.GetShape()[0], 2);
	EXPECT_EQ(reshaped.GetShape()[1], 3);
	EXPECT_EQ(reshaped.GetShape()[2], 4);
	
	// Verify data is unchanged
	std::vector<float> got(N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(reshaped, got.data(), N * sizeof(float)).IsOk());
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(got[i], data[i]) << "Index " << i;
	}
}

// ============================================================================
// Slice Tests
// ============================================================================

TEST_VK(TestFnMatrixIndex, Slice_1D_Basic) {
	// Test basic 1D slice
	constexpr OaU32 N = 10;
	std::vector<float> data(N);
	std::iota(data.begin(), data.end(), 0.0f);  // 0, 1, 2, ..., 9
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto sliced = OaFnMatrix::Slice(a, 0, 2, 7);  // [2:7] = [2, 3, 4, 5, 6]
	
	EXPECT_EQ(sliced.GetShape()[0], 5);
	
	std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	std::vector<float> got(5);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(sliced, got.data(), 5 * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 5; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

TEST_VK(TestFnMatrixIndex, Slice_2D_Rows) {
	// Test slicing rows from 2D tensor
	constexpr OaU32 M = 5, N = 3;
	std::vector<float> data(M * N);
	std::iota(data.begin(), data.end(), 0.0f);
	
	auto a = CreateMatrixFromHost(data, OaMatrixShape{M, N});
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto sliced = OaFnMatrix::Slice(a, 0, 1, 4);  // Rows [1:4] = rows 1, 2, 3
	
	EXPECT_EQ(sliced.GetShape()[0], 3);
	EXPECT_EQ(sliced.GetShape()[1], N);
	
	// Expected: rows 1, 2, 3 from original
	std::vector<float> expected = {
		3.0f, 4.0f, 5.0f,   // Row 1
		6.0f, 7.0f, 8.0f,   // Row 2
		9.0f, 10.0f, 11.0f  // Row 3
	};
	std::vector<float> got(3 * N);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(sliced, got.data(), 3 * N * sizeof(float)).IsOk());
	
	for (OaU32 i = 0; i < 3 * N; ++i) {
		EXPECT_FLOAT_EQ(got[i], expected[i]) << "Index " << i;
	}
}

// Note: Slice API doesn't support step parameter
// Only supports [start:end] slicing, not [start:end:step]

// Note: CopyAtOffset is not exposed in the public OaFnMatrix API
// It exists as a kernel but is not available for direct testing
