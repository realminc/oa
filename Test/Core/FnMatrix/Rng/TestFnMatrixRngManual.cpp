// Tests for Core/FnMatrix RNG operations
// PhiloxNormal, PhiloxUniform - statistical validation

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <vector>
#include <cmath>
#include <algorithm>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixRngManual : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixRngManual";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
		OaRuntimeGlobal::SetRuntime(GRt);
	}
};

// Helper to copy matrix to host
static std::vector<float> CopyMatrixToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to compute mean
static float ComputeMean(const std::vector<float>& data) {
	float sum = 0.0f;
	for (float v : data) sum += v;
	return sum / static_cast<float>(data.size());
}

// Helper to compute standard deviation
static float ComputeStddev(const std::vector<float>& data, float mean) {
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
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	// Generate 10000 samples for statistical validation
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{10000});
	auto samples = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, 42);
	auto data = CopyMatrixToHost(samples);
	
	// Compute statistics
	float mean = ComputeMean(data);
	float stddev = ComputeStddev(data, mean);
	
	// For 10000 samples, mean should be close to 0, stddev close to 1
	// Using generous tolerance for statistical variation
	EXPECT_NEAR(mean, 0.0f, 0.05f) << "Mean should be close to 0";
	EXPECT_NEAR(stddev, 1.0f, 0.05f) << "Stddev should be close to 1";
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_CustomMeanStddev) {
	// Test custom mean and stddev
	float target_mean = 5.0f;
	float target_stddev = 2.0f;
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{10000});
	auto samples = OaFnMatrix::PhiloxNormal(shape_matrix, target_mean, target_stddev, 123);
	auto data = CopyMatrixToHost(samples);
	
	float mean = ComputeMean(data);
	float stddev = ComputeStddev(data, mean);
	
	EXPECT_NEAR(mean, target_mean, 0.1f);
	EXPECT_NEAR(stddev, target_stddev, 0.1f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_Reproducibility) {
	// Test that same seed produces same results
	OaU64 seed = 999;
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{1000});
	auto samples1 = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, seed);
	auto samples2 = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, seed);
	
	auto data1 = CopyMatrixToHost(samples1);
	auto data2 = CopyMatrixToHost(samples2);
	
	// Same seed should produce identical results
	ASSERT_EQ(data1.size(), data2.size());
	for (size_t i = 0; i < data1.size(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_DifferentSeeds) {
	// Test that different seeds produce different results
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{1000});
	auto samples1 = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, 111);
	auto samples2 = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, 222);
	
	auto data1 = CopyMatrixToHost(samples1);
	auto data2 = CopyMatrixToHost(samples2);
	
	// Different seeds should produce different results
	int differences = 0;
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i]) differences++;
	}
	
	// Expect most values to be different (>99%)
	EXPECT_GT(differences, 990) << "Different seeds should produce different values";
}

TEST_VK(TestFnMatrixRngManual, PhiloxNormal_2D) {
	// Test 2D shape
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{100, 100});
	auto samples = OaFnMatrix::PhiloxNormal(shape_matrix, 0.0f, 1.0f, 42);
	
	EXPECT_EQ(samples.GetShape().Rank, 2);
	EXPECT_EQ(samples.GetShape()[0], 100);
	EXPECT_EQ(samples.GetShape()[1], 100);
	
	auto data = CopyMatrixToHost(samples);
	EXPECT_EQ(data.size(), 10000u);
	
	float mean = ComputeMean(data);
	EXPECT_NEAR(mean, 0.0f, 0.05f);
}

// ============================================================================
// PhiloxUniform Tests
// ============================================================================

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_ZeroToOne) {
	// Test uniform distribution [0, 1)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{10000});
	auto samples = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, 42);
	auto data = CopyMatrixToHost(samples);
	
	// Check all values are in range [0, 1)
	for (float v : data) {
		EXPECT_GE(v, 0.0f) << "Value should be >= 0";
		EXPECT_LT(v, 1.0f) << "Value should be < 1";
	}
	
	// Mean should be close to 0.5 for uniform [0, 1)
	float mean = ComputeMean(data);
	EXPECT_NEAR(mean, 0.5f, 0.02f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_CustomRange) {
	// Test custom range [-5, 5)
	float low = -5.0f;
	float high = 5.0f;
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{10000});
	auto samples = OaFnMatrix::PhiloxUniform(shape_matrix, low, high, 123);
	auto data = CopyMatrixToHost(samples);
	
	// Check all values are in range
	for (float v : data) {
		EXPECT_GE(v, low) << "Value should be >= low";
		EXPECT_LT(v, high) << "Value should be < high";
	}
	
	// Mean should be close to midpoint
	float mean = ComputeMean(data);
	float expected_mean = (low + high) / 2.0f;
	EXPECT_NEAR(mean, expected_mean, 0.1f);
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_Reproducibility) {
	// Test that same seed produces same results
	OaU64 seed = 777;
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{1000});
	auto samples1 = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, seed);
	auto samples2 = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, seed);
	
	auto data1 = CopyMatrixToHost(samples1);
	auto data2 = CopyMatrixToHost(samples2);
	
	// Same seed should produce identical results
	ASSERT_EQ(data1.size(), data2.size());
	for (size_t i = 0; i < data1.size(); ++i) {
		EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_DifferentSeeds) {
	// Test that different seeds produce different results
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{1000});
	auto samples1 = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, 333);
	auto samples2 = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, 444);
	
	auto data1 = CopyMatrixToHost(samples1);
	auto data2 = CopyMatrixToHost(samples2);
	
	// Different seeds should produce different results
	int differences = 0;
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i]) differences++;
	}
	
	EXPECT_GT(differences, 990) << "Different seeds should produce different values";
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_Distribution) {
	// Test that uniform distribution is actually uniform
	// Divide [0, 1) into 10 bins and check counts
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{10000});
	auto samples = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, 555);
	auto data = CopyMatrixToHost(samples);
	
	// Count values in each bin
	std::vector<int> bins(10, 0);
	for (float v : data) {
		int bin = static_cast<int>(v * 10.0f);
		if (bin >= 0 && bin < 10) bins[bin]++;
	}
	
	// Each bin should have roughly 1000 samples (10000 / 10)
	// Allow 20% deviation for statistical variation
	for (int i = 0; i < 10; ++i) {
		EXPECT_GT(bins[i], 800) << "Bin " << i << " has too few samples";
		EXPECT_LT(bins[i], 1200) << "Bin " << i << " has too many samples";
	}
}

TEST_VK(TestFnMatrixRngManual, PhiloxUniform_2D) {
	// Test 2D shape
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto shape_matrix = OaFnMatrix::Empty(OaMatrixShape{50, 200});
	auto samples = OaFnMatrix::PhiloxUniform(shape_matrix, 0.0f, 1.0f, 42);
	
	EXPECT_EQ(samples.GetShape().Rank, 2);
	EXPECT_EQ(samples.GetShape()[0], 50);
	EXPECT_EQ(samples.GetShape()[1], 200);
	
	auto data = CopyMatrixToHost(samples);
	EXPECT_EQ(data.size(), 10000u);
	
	// Check range
	for (float v : data) {
		EXPECT_GE(v, 0.0f);
		EXPECT_LT(v, 1.0f);
	}
}

