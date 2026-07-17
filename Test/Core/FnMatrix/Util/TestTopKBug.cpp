// Minimal test case to reproduce TopK crash
// Bug discovered during test coverage expansion

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <vector>

static OaComputeEngine* GRt = nullptr;

class TestTopKBug : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestTopKBug";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to create matrix from host data
static OaMatrix CreateFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Helper to copy matrix to host
static std::vector<float> CopyToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

TEST_VK(TestTopKBug, TopK_MinimalCrash) {
	// Minimal test case that crashes
	// Input: [3, 1, 4, 1, 5] - find top 2
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	// This line crashes with SIGSEGV
	auto result = OaFnMatrix::TopK(input, 2, 0);
	
	// If we get here, TopK worked
	EXPECT_EQ(result.Values.GetShape()[0], 2);
	
	auto values = CopyToHost(result.Values);
	
	// Top 2 should be 5.0 and 4.0
	EXPECT_FLOAT_EQ(values[0], 5.0f);
	EXPECT_FLOAT_EQ(values[1], 4.0f);
}

TEST_VK(TestTopKBug, TopK_DifferentDim) {
	// Try with explicit dim=-1 (last dimension)
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	
	auto result = OaFnMatrix::TopK(input, 2, -1);
	
	EXPECT_EQ(result.Values.GetShape()[0], 2);
}

TEST_VK(TestTopKBug, TopK_2DInput) {
	// Try with 2D input
	std::vector<float> input_data = {
		3.0f, 1.0f, 4.0f,
		5.0f, 2.0f, 6.0f
	};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	
	// Find top-2 along last dimension
	auto result = OaFnMatrix::TopK(input, 2, -1);
	
	EXPECT_EQ(result.Values.GetShape().Rank, 2);
	EXPECT_EQ(result.Values.GetShape()[0], 2);
	EXPECT_EQ(result.Values.GetShape()[1], 2);
}

TEST_VK(TestTopKBug, TopK_K1) {
	// Try with k=1 (simplest case)
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{3});
	
	auto result = OaFnMatrix::TopK(input, 1, 0);
	
	EXPECT_EQ(result.Values.GetShape()[0], 1);
	
	auto values = CopyToHost(result.Values);
	EXPECT_FLOAT_EQ(values[0], 4.0f);  // Max value
}

