// Tests for Core/FnMatrix RNG operations
// PhiloxNormal, PhiloxUniform - statistical validation

#include <gtest/gtest.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>
#include <algorithm>

static oa::Engine* GRt = nullptr;

class TestFnMatrixRngManual : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixRngManual";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> copyMatrixToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to compute mean
static float computeMean(const std::vector<float>& data) {
	float sum = 0.0f;
	for (float v : data) sum += v;
	return sum / static_cast<float>(data.size());
}

// Helper to compute standard deviation
static float computeStddev(const std::vector<float>& data, float mean) {
	float sum_sq = 0.0f;
	for (float v : data) {
		float diff = v - mean;
		sum_sq += diff * diff;
	}
	return std::sqrt(sum_sq / static_cast<float>(data.size()));
}

// ============================================================================
// PhiloxNormal Tests
// ============================================================================

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_StandardNormal) {
	// Test standard normal distribution (mean=0, stddev=1)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	// generate 10000 samples for statistical validation
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{10000});
	auto samples = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, 42);
	auto data = copyMatrixToHost(samples);
	
	// Compute statistics
	float mean = computeMean(data);
	float stddev = computeStddev(data, mean);
	
	// For 10000 samples, mean should be close to 0, stddev close to 1
	// Using generous tolerance for statistical variation
	EXPECT_NEAR(mean, 0.0f, 0.05f) << "Mean should be close to 0";
	EXPECT_NEAR(stddev, 1.0f, 0.05f) << "stddev should be close to 1";
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_CustomMeanStddev) {
	// Test custom mean and stddev
	float target_mean = 5.0f;
	float target_stddev = 2.0f;
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{10000});
	auto samples = oa::FnMatrix::philoxNormal(shape_matrix, target_mean, target_stddev, 123);
	auto data = copyMatrixToHost(samples);
	
	float mean = computeMean(data);
	float stddev = computeStddev(data, mean);
	
	EXPECT_NEAR(mean, target_mean, 0.1f);
	EXPECT_NEAR(stddev, target_stddev, 0.1f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_Reproducibility) {
	// Test that same seed produces same results
	oa::U64 seed = 999;
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{1000});
	auto samples1 = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, seed);
	auto samples2 = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, seed);
	
	auto data1 = copyMatrixToHost(samples1);
	auto data2 = copyMatrixToHost(samples2);
	
	// Same seed should produce identical results
	ASSERT_EQ(data1.size(), data2.size());
	for (size_t i = 0; i < data1.size(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_DifferentSeeds) {
	// Test that different seeds produce different results
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{1000});
	auto samples1 = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, 111);
	auto samples2 = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, 222);
	
	auto data1 = copyMatrixToHost(samples1);
	auto data2 = copyMatrixToHost(samples2);
	
	// Different seeds should produce different results
	int differences = 0;
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i]) differences++;
	}
	
	// expect most values to be different (>99%)
	EXPECT_GT(differences, 990) << "Different seeds should produce different values";
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_2D) {
	// Test 2D shape
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{100, 100});
	auto samples = oa::FnMatrix::philoxNormal(shape_matrix, 0.0f, 1.0f, 42);
	
	EXPECT_EQ(samples.getShape().rank, 2);
	EXPECT_EQ(samples.getShape()[0], 100);
	EXPECT_EQ(samples.getShape()[1], 100);
	
	auto data = copyMatrixToHost(samples);
	EXPECT_EQ(data.size(), 10000u);
	
	float mean = computeMean(data);
	EXPECT_NEAR(mean, 0.0f, 0.05f);
}

// ============================================================================
// PhiloxUniform Tests
// ============================================================================

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_ZeroToOne) {
	// Test uniform distribution [0, 1)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{10000});
	auto samples = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, 42);
	auto data = copyMatrixToHost(samples);
	
	// Check all values are in range [0, 1)
	for (float v : data) {
		EXPECT_GE(v, 0.0f) << "Value should be >= 0";
		EXPECT_LT(v, 1.0f) << "Value should be < 1";
	}
	
	// Mean should be close to 0.5 for uniform [0, 1)
	float mean = computeMean(data);
	EXPECT_NEAR(mean, 0.5f, 0.02f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_CustomRange) {
	// Test custom range [-5, 5)
	float low = -5.0f;
	float high = 5.0f;
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{10000});
	auto samples = oa::FnMatrix::philoxUniform(shape_matrix, low, high, 123);
	auto data = copyMatrixToHost(samples);
	
	// Check all values are in range
	for (float v : data) {
		EXPECT_GE(v, low) << "Value should be >= low";
		EXPECT_LT(v, high) << "Value should be < high";
	}
	
	// Mean should be close to midpoint
	float mean = computeMean(data);
	float expected_mean = (low + high) / 2.0f;
	EXPECT_NEAR(mean, expected_mean, 0.1f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_Reproducibility) {
	// Test that same seed produces same results
	oa::U64 seed = 777;
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{1000});
	auto samples1 = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, seed);
	auto samples2 = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, seed);
	
	auto data1 = copyMatrixToHost(samples1);
	auto data2 = copyMatrixToHost(samples2);
	
	// Same seed should produce identical results
	ASSERT_EQ(data1.size(), data2.size());
	for (size_t i = 0; i < data1.size(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_DifferentSeeds) {
	// Test that different seeds produce different results
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{1000});
	auto samples1 = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, 333);
	auto samples2 = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, 444);
	
	auto data1 = copyMatrixToHost(samples1);
	auto data2 = copyMatrixToHost(samples2);
	
	// Different seeds should produce different results
	int differences = 0;
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i]) differences++;
	}
	
	EXPECT_GT(differences, 990) << "Different seeds should produce different values";
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_UsesAll64SeedBits) {
	oa::ExecutionSession::RecordingScope ctxScope(oa::ExecutionSession::getActive());
	auto shapeMatrix = oa::FnMatrix::empty(oa::MatrixShape{1000});
	constexpr oa::U64 seedLowOnly = 0x000000000000002aULL;
	constexpr oa::U64 seedWithHighBits = 0xdeadbeef0000002aULL;
	auto samples1 = oa::FnMatrix::philoxUniform(
		shapeMatrix, 0.0F, 1.0F, seedLowOnly);
	auto samples2 = oa::FnMatrix::philoxUniform(
		shapeMatrix, 0.0F, 1.0F, seedWithHighBits);

	const auto data1 = copyMatrixToHost(samples1);
	const auto data2 = copyMatrixToHost(samples2);
	oa::I32 differences = 0;
	for (oa::Usize index = 0; index < data1.size(); ++index) {
		if (data1[index] != data2[index]) ++differences;
	}
	EXPECT_GT(differences, 990)
		<< "The high 32 seed bits must participate in the Philox key";
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_Distribution) {
	// Test that uniform distribution is actually uniform
	// Divide [0, 1) into 10 bins and check counts
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{10000});
	auto samples = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, 555);
	auto data = copyMatrixToHost(samples);
	
	// Count values in each bin
	std::vector<int> bins(10, 0);
	for (float v : data) {
		int bin = static_cast<int>(v * 10.0f);
		if (bin >= 0 && bin < 10) bins[bin]++;
	}
	
	// Each bin should have roughly 1000 samples (10000 / 10)
	// Allow 20% deviation for statistical variation
	for (int i = 0; i < 10; ++i) {
		EXPECT_GT(bins[i], 800) << "bin " << i << " has too few samples";
		EXPECT_LT(bins[i], 1200) << "bin " << i << " has too many samples";
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_2D) {
	// Test 2D shape
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto shape_matrix = oa::FnMatrix::empty(oa::MatrixShape{50, 200});
	auto samples = oa::FnMatrix::philoxUniform(shape_matrix, 0.0f, 1.0f, 42);
	
	EXPECT_EQ(samples.getShape().rank, 2);
	EXPECT_EQ(samples.getShape()[0], 50);
	EXPECT_EQ(samples.getShape()[1], 200);
	
	auto data = copyMatrixToHost(samples);
	EXPECT_EQ(data.size(), 10000u);
	
	// Check range
	for (float v : data) {
		EXPECT_GE(v, 0.0f);
		EXPECT_LT(v, 1.0f);
	}
}
