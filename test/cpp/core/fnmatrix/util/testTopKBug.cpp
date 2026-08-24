// Minimal test case to reproduce topK crash
// Bug discovered during test coverage expansion

#include <gtest/gtest.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>

static oa::Engine* GRt = nullptr;

class TestTopKBug : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestTopKBug";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to create matrix from host data
static oa::Matrix createFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// Helper to copy matrix to host
static std::vector<float> copyToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

TEST_VK(TestTopKBug, TopK_MinimalCrash) {
	// Minimal test case that crashes
	// input: [3, 1, 4, 1, 5] - find top 2
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	// This line crashes with SIGSEGV
	auto result = oa::FnMatrix::topK(input, 2, 0);
	
	// If we get here, topK worked
	EXPECT_EQ(result.values.getShape()[0], 2);
	
	auto values = copyToHost(result.values);
	
	// top 2 should be 5.0 and 4.0
	EXPECT_FLOAT_EQ(values[0], 5.0f);
	EXPECT_FLOAT_EQ(values[1], 4.0f);
}

TEST_VK(TestTopKBug, TopK_DifferentDim) {
	// Try with explicit dim=-1 (last dimension)
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	
	auto result = oa::FnMatrix::topK(input, 2, -1);
	
	EXPECT_EQ(result.values.getShape()[0], 2);
}

TEST_VK(TestTopKBug, TopK_2DInput) {
	// Try with 2D input
	std::vector<float> input_data = {
		3.0f, 1.0f, 4.0f,
		5.0f, 2.0f, 6.0f
	};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	
	// find top-2 along last dimension
	auto result = oa::FnMatrix::topK(input, 2, -1);
	
	EXPECT_EQ(result.values.getShape().rank, 2);
	EXPECT_EQ(result.values.getShape()[0], 2);
	EXPECT_EQ(result.values.getShape()[1], 2);
}

TEST_VK(TestTopKBug, TopK_K1) {
	// Try with k=1 (simplest case)
	std::vector<float> input_data = {3.0f, 1.0f, 4.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{3});
	
	auto result = oa::FnMatrix::topK(input, 1, 0);
	
	EXPECT_EQ(result.values.getShape()[0], 1);
	
	auto values = copyToHost(result.values);
	EXPECT_FLOAT_EQ(values[0], 4.0f);  // Max value
}
