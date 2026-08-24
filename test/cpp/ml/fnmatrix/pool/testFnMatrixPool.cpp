// Tests for Ml/FnMatrix pooling operations
// MaxPool2d, AvgPool2d

#include <gtest/gtest.h>
#include <oa/ml/fnMatrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <vector>
#include <algorithm>

static oa::Engine* GRt = nullptr;

class TestFnMatrixPool : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixPool";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to create matrix from host data
static oa::Matrix createMatrixFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

// Helper to copy matrix to host
static std::vector<float> copyMatrixToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// ============================================================================
// MaxPool2d Tests
// ============================================================================

TEST_VK(TestFnMatrixPool, MaxPool2d_Simple2x2) {
	// Test 2x2 max pooling on 4x4 input
	// input: [1, 1, 4, 4]
	std::vector<float> input = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 1, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::maxPool2d(input_mat, 2, 2, 0).out;
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [1, 1, 2, 2]
	// Each 2x2 window takes the max
	std::vector<float> expected = {
		6.0f, 8.0f,
		14.0f, 16.0f
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, MaxPool2d_WithStride) {
	// Test max pooling with stride != kernel_size
	// input: [1, 1, 5, 5], kernel=2, stride=1
	std::vector<float> input = {
		1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
		6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
		11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
		16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
		21.0f, 22.0f, 23.0f, 24.0f, 25.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 1, 5, 5});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::maxPool2d(input_mat, 2, 1, 0).out;
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [1, 1, 4, 4] (stride=1 gives overlapping windows)
	// first window: max(1,2,6,7) = 7
	std::vector<float> expected = {
		7.0f, 8.0f, 9.0f, 10.0f,
		12.0f, 13.0f, 14.0f, 15.0f,
		17.0f, 18.0f, 19.0f, 20.0f,
		22.0f, 23.0f, 24.0f, 25.0f
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, MaxPool2d_MultiChannel) {
	// Test max pooling with multiple channels
	// input: [1, 2, 4, 4] (2 channels)
	std::vector<float> input = {
		// Channel 0
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f,
		// Channel 1
		16.0f, 15.0f, 14.0f, 13.0f,
		12.0f, 11.0f, 10.0f, 9.0f,
		8.0f, 7.0f, 6.0f, 5.0f,
		4.0f, 3.0f, 2.0f, 1.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 2, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto pool_result = oa::FnMatrix::maxPool2d(input_mat, 2, 2, 0);
	
	auto result = copyMatrixToHost(pool_result.out);
	
	// expected output: [1, 2, 2, 2]
	std::vector<float> expected = {
		// Channel 0
		6.0f, 8.0f,
		14.0f, 16.0f,
		// Channel 1 (max of each 2x2 window)
		16.0f, 14.0f,  // max(16,15,12,11), max(14,13,10,9)
		8.0f, 6.0f     // max(8,7,4,3), max(6,5,2,1)
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, MaxPool2d_Batch) {
	// Test max pooling with batch size > 1
	// input: [2, 1, 2, 2] (batch=2)
	std::vector<float> input = {
		// Batch 0
		1.0f, 2.0f,
		3.0f, 4.0f,
		// Batch 1
		5.0f, 6.0f,
		7.0f, 8.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{2, 1, 2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::maxPool2d(input_mat, 2, 2, 0).out;
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [2, 1, 1, 1]
	std::vector<float> expected = {
		4.0f,  // max of batch 0
		8.0f   // max of batch 1
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

// ============================================================================
// AvgPool2d Tests
// ============================================================================

TEST_VK(TestFnMatrixPool, AvgPool2d_Simple2x2) {
	// Test 2x2 average pooling on 4x4 input
	// input: [1, 1, 4, 4]
	std::vector<float> input = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 1, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::avgPool2d(input_mat, 2, 2, 0);
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [1, 1, 2, 2]
	// Each 2x2 window takes the average
	std::vector<float> expected = {
		3.5f,  // avg(1,2,5,6)
		5.5f,  // avg(3,4,7,8)
		11.5f, // avg(9,10,13,14)
		13.5f  // avg(11,12,15,16)
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, AvgPool2d_WithStride) {
	// Test average pooling with stride != kernel_size
	// input: [1, 1, 4, 4], kernel=2, stride=1
	std::vector<float> input = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 1, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::avgPool2d(input_mat, 2, 1, 0);
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [1, 1, 3, 3] (stride=1 gives overlapping windows)
	std::vector<float> expected = {
		3.5f, 4.5f, 5.5f,    // avg(1,2,5,6), avg(2,3,6,7), avg(3,4,7,8)
		7.5f, 8.5f, 9.5f,    // avg(5,6,9,10), avg(6,7,10,11), avg(7,8,11,12)
		11.5f, 12.5f, 13.5f  // avg(9,10,13,14), avg(10,11,14,15), avg(11,12,15,16)
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, AvgPool2d_MultiChannel) {
	// Test average pooling with multiple channels
	// input: [1, 2, 4, 4] (2 channels)
	std::vector<float> input = {
		// Channel 0
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f,
		// Channel 1
		2.0f, 4.0f, 6.0f, 8.0f,
		10.0f, 12.0f, 14.0f, 16.0f,
		18.0f, 20.0f, 22.0f, 24.0f,
		26.0f, 28.0f, 30.0f, 32.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 2, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::avgPool2d(input_mat, 2, 2, 0);
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [1, 2, 2, 2]
	std::vector<float> expected = {
		// Channel 0
		3.5f, 5.5f,
		11.5f, 13.5f,
		// Channel 1
		7.0f, 11.0f,
		23.0f, 27.0f
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixPool, AvgPool2d_Batch) {
	// Test average pooling with batch size > 1
	// input: [2, 1, 2, 2] (batch=2)
	std::vector<float> input = {
		// Batch 0
		1.0f, 2.0f,
		3.0f, 4.0f,
		// Batch 1
		5.0f, 6.0f,
		7.0f, 8.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{2, 1, 2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = oa::FnMatrix::avgPool2d(input_mat, 2, 2, 0);
	
	auto result = copyMatrixToHost(output);
	
	// expected output: [2, 1, 1, 1]
	std::vector<float> expected = {
		2.5f,  // avg of batch 0: (1+2+3+4)/4
		6.5f   // avg of batch 1: (5+6+7+8)/4
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
	}
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_VK(TestFnMatrixPool, Compare_MaxVsAvg) {
	// compare max and average pooling on same input
	std::vector<float> input = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	
	auto input_mat = createMatrixFromHost(input, oa::MatrixShape{1, 1, 4, 4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto max_pool_result = oa::FnMatrix::maxPool2d(input_mat, 2, 2, 0);
	auto avg_output = oa::FnMatrix::avgPool2d(input_mat, 2, 2, 0);
	
	auto max_result = copyMatrixToHost(max_pool_result.out);
	auto avg_result = copyMatrixToHost(avg_output);
	
	// Max should always be >= average
	ASSERT_EQ(max_result.size(), avg_result.size());
	for (size_t i = 0; i < max_result.size(); ++i) {
		EXPECT_GE(max_result[i], avg_result[i]) << "Max should be >= avg at index " << i;
	}
}
