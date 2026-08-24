// Manual tests for Core/FnMatrix allocation/factory operations
// Full, Eye, Arange, Linspace, CausalMask

#include <gtest/gtest.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <cmath>

static oa::Engine* GRt = nullptr;

class TestFnMatrixAllocManual : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixAllocManual";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> copyToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// ============================================================================
// Full Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Full_1D) {
	// Test creating 1D tensor filled with constant value
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::full(oa::MatrixShape{5}, 3.14);
	
	EXPECT_EQ(tensor.getShape().rank, 1);
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 3.14f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_2D) {
	// Test creating 2D tensor filled with constant value
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::full(oa::MatrixShape{3, 4}, -2.5);
	
	EXPECT_EQ(tensor.getShape().rank, 2);
	EXPECT_EQ(tensor.getShape()[0], 3);
	EXPECT_EQ(tensor.getShape()[1], 4);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -2.5f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_Zero) {
	// Test Full with zero value (should match Zeros)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::full(oa::MatrixShape{10}, 0.0);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_NegativeValue) {
	// Test Full with negative value
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::full(oa::MatrixShape{2, 3, 4}, -1.0);
	
	EXPECT_EQ(tensor.getShape().numElements(), 24);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -1.0f) << "index " << i;
	}
}

// ============================================================================
// Eye Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Eye_Square) {
	// Test creating square identity matrix
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::eye(4);
	
	EXPECT_EQ(tensor.getShape().rank, 2);
	EXPECT_EQ(tensor.getShape()[0], 4);
	EXPECT_EQ(tensor.getShape()[1], 4);
	
	auto result = copyToHost(tensor);
	
	// Check diagonal is 1, off-diagonal is 0
	for (oa::U32 i = 0; i < 4; ++i) {
		for (oa::U32 j = 0; j < 4; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 4 + j], expected) 
				<< "position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Rectangular_TallMatrix) {
	// Test creating tall rectangular identity matrix (more rows than cols)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::eye(5, 3);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	EXPECT_EQ(tensor.getShape()[1], 3);
	
	auto result = copyToHost(tensor);
	
	// Check: diagonal is 1 where i==j, rest is 0
	for (oa::U32 i = 0; i < 5; ++i) {
		for (oa::U32 j = 0; j < 3; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 3 + j], expected) 
				<< "position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Rectangular_WideMatrix) {
	// Test creating wide rectangular identity matrix (more cols than rows)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::eye(3, 5);
	
	EXPECT_EQ(tensor.getShape()[0], 3);
	EXPECT_EQ(tensor.getShape()[1], 5);
	
	auto result = copyToHost(tensor);
	
	// Check: diagonal is 1 where i==j, rest is 0
	for (oa::U32 i = 0; i < 3; ++i) {
		for (oa::U32 j = 0; j < 5; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 5 + j], expected) 
				<< "position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Size1) {
	// Test edge case: 1x1 identity matrix
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::eye(1);
	
	EXPECT_EQ(tensor.getShape()[0], 1);
	EXPECT_EQ(tensor.getShape()[1], 1);
	
	auto result = copyToHost(tensor);
	EXPECT_FLOAT_EQ(result[0], 1.0f);
}

// ============================================================================
// Arange Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Arange_Basic) {
	// Test basic arange: [0, 1, 2, 3, 4]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::arange(0.0, 5.0, 1.0);
	
	EXPECT_EQ(tensor.getShape().rank, 1);
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], static_cast<float>(i)) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_NonZeroStart) {
	// Test arange with non-zero start: [5, 6, 7, 8, 9]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::arange(5.0, 10.0, 1.0);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 5.0f + static_cast<float>(i)) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_StepSize2) {
	// Test arange with step=2: [0, 2, 4, 6, 8]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::arange(0.0, 10.0, 2.0);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	std::vector<float> expected = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_FractionalStep) {
	// Test arange with fractional step: [0.0, 0.5, 1.0, 1.5, 2.0]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::arange(0.0, 2.5, 0.5);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	std::vector<float> expected = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_NegativeRange) {
	// Test arange with negative values: [-5, -4, -3, -2, -1]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::arange(-5.0, 0.0, 1.0);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -5.0f + static_cast<float>(i)) << "index " << i;
	}
}

// ============================================================================
// Linspace Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Linspace_Basic) {
	// Test basic linspace: [0, 0.25, 0.5, 0.75, 1.0]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::linspace(0.0, 1.0, 5);
	
	EXPECT_EQ(tensor.getShape().rank, 1);
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	std::vector<float> expected = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_TwoPoints) {
	// Test linspace with 2 points (start and end)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::linspace(0.0, 10.0, 2);
	
	EXPECT_EQ(tensor.getShape()[0], 2);
	
	auto result = copyToHost(tensor);
	EXPECT_FLOAT_EQ(result[0], 0.0f);
	EXPECT_FLOAT_EQ(result[1], 10.0f);
}

TEST_F(TestFnMatrixAllocManual, Linspace_NegativeRange) {
	// Test linspace with negative range: [-1, -0.5, 0, 0.5, 1]
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::linspace(-1.0, 1.0, 5);
	
	EXPECT_EQ(tensor.getShape()[0], 5);
	
	auto result = copyToHost(tensor);
	std::vector<float> expected = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_LargeRange) {
	// Test linspace with large range
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::linspace(0.0, 100.0, 11);
	
	EXPECT_EQ(tensor.getShape()[0], 11);
	
	auto result = copyToHost(tensor);
	// Should be [0, 10, 20, ..., 100]
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], static_cast<float>(i * 10), 1e-4f) << "index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_ReverseRange) {
	// Test linspace with start > end (descending)
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto tensor = oa::FnMatrix::linspace(10.0, 0.0, 6);
	
	EXPECT_EQ(tensor.getShape()[0], 6);
	
	auto result = copyToHost(tensor);
	std::vector<float> expected = {10.0f, 8.0f, 6.0f, 4.0f, 2.0f, 0.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "index " << i;
	}
}

// ============================================================================
// CausalMask Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, CausalMask_Size4) {
	// Test causal mask for sequence length 4
	// expected: lower triangular matrix with 0s above diagonal, -inf below
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto mask = oa::FnMatrix::causalMask(4);
	
	EXPECT_EQ(mask.getShape().rank, 2);
	EXPECT_EQ(mask.getShape()[0], 4);
	EXPECT_EQ(mask.getShape()[1], 4);
	
	auto result = copyToHost(mask);
	
	// Causal mask: 0 on and below diagonal, -inf above
	// Row 0: [0, -inf, -inf, -inf]
	// Row 1: [0, 0, -inf, -inf]
	// Row 2: [0, 0, 0, -inf]
	// Row 3: [0, 0, 0, 0]
	for (oa::U32 i = 0; i < 4; ++i) {
		for (oa::U32 j = 0; j < 4; ++j) {
			float val = result[i * 4 + j];
			if (j <= i) {
				EXPECT_FLOAT_EQ(val, 0.0f) << "position [" << i << "," << j << "] should be 0";
			} else {
				EXPECT_TRUE(std::isinf(val) && val < 0) 
					<< "position [" << i << "," << j << "] should be -inf, got " << val;
			}
		}
	}
}

TEST_F(TestFnMatrixAllocManual, CausalMask_Size1) {
	// Test edge case: causal mask for sequence length 1
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto mask = oa::FnMatrix::causalMask(1);
	
	EXPECT_EQ(mask.getShape()[0], 1);
	EXPECT_EQ(mask.getShape()[1], 1);
	
	auto result = copyToHost(mask);
	EXPECT_FLOAT_EQ(result[0], 0.0f);
}

TEST_F(TestFnMatrixAllocManual, CausalMask_Size8) {
	// Test larger causal mask
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto mask = oa::FnMatrix::causalMask(8);
	
	EXPECT_EQ(mask.getShape()[0], 8);
	EXPECT_EQ(mask.getShape()[1], 8);
	
	auto result = copyToHost(mask);
	
	// verify structure: lower triangular with 0s, upper triangular with -inf
	for (oa::U32 i = 0; i < 8; ++i) {
		for (oa::U32 j = 0; j < 8; ++j) {
			float val = result[i * 8 + j];
			if (j <= i) {
				EXPECT_FLOAT_EQ(val, 0.0f) << "position [" << i << "," << j << "]";
			} else {
				EXPECT_TRUE(std::isinf(val) && val < 0) << "position [" << i << "," << j << "]";
			}
		}
	}
}
