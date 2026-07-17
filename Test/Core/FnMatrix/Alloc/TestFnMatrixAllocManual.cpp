// Manual tests for Core/FnMatrix allocation/factory operations
// Full, Eye, Arange, Linspace, CausalMask

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <vector>
#include <cmath>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixAllocManual : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixAllocManual";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> CopyToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// ============================================================================
// Full Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Full_1D) {
	// Test creating 1D tensor filled with constant value
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Full(OaMatrixShape{5}, 3.14);
	
	EXPECT_EQ(tensor.GetShape().Rank, 1);
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 3.14f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_2D) {
	// Test creating 2D tensor filled with constant value
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Full(OaMatrixShape{3, 4}, -2.5);
	
	EXPECT_EQ(tensor.GetShape().Rank, 2);
	EXPECT_EQ(tensor.GetShape()[0], 3);
	EXPECT_EQ(tensor.GetShape()[1], 4);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -2.5f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_Zero) {
	// Test Full with zero value (should match Zeros)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Full(OaMatrixShape{10}, 0.0);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Full_NegativeValue) {
	// Test Full with negative value
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Full(OaMatrixShape{2, 3, 4}, -1.0);
	
	EXPECT_EQ(tensor.GetShape().NumElements(), 24);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -1.0f) << "Index " << i;
	}
}

// ============================================================================
// Eye Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Eye_Square) {
	// Test creating square identity matrix
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Eye(4);
	
	EXPECT_EQ(tensor.GetShape().Rank, 2);
	EXPECT_EQ(tensor.GetShape()[0], 4);
	EXPECT_EQ(tensor.GetShape()[1], 4);
	
	auto result = CopyToHost(tensor);
	
	// Check diagonal is 1, off-diagonal is 0
	for (OaU32 i = 0; i < 4; ++i) {
		for (OaU32 j = 0; j < 4; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 4 + j], expected) 
				<< "Position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Rectangular_TallMatrix) {
	// Test creating tall rectangular identity matrix (more rows than cols)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Eye(5, 3);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	EXPECT_EQ(tensor.GetShape()[1], 3);
	
	auto result = CopyToHost(tensor);
	
	// Check: diagonal is 1 where i==j, rest is 0
	for (OaU32 i = 0; i < 5; ++i) {
		for (OaU32 j = 0; j < 3; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 3 + j], expected) 
				<< "Position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Rectangular_WideMatrix) {
	// Test creating wide rectangular identity matrix (more cols than rows)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Eye(3, 5);
	
	EXPECT_EQ(tensor.GetShape()[0], 3);
	EXPECT_EQ(tensor.GetShape()[1], 5);
	
	auto result = CopyToHost(tensor);
	
	// Check: diagonal is 1 where i==j, rest is 0
	for (OaU32 i = 0; i < 3; ++i) {
		for (OaU32 j = 0; j < 5; ++j) {
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_FLOAT_EQ(result[i * 5 + j], expected) 
				<< "Position [" << i << "," << j << "]";
		}
	}
}

TEST_F(TestFnMatrixAllocManual, Eye_Size1) {
	// Test edge case: 1x1 identity matrix
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Eye(1);
	
	EXPECT_EQ(tensor.GetShape()[0], 1);
	EXPECT_EQ(tensor.GetShape()[1], 1);
	
	auto result = CopyToHost(tensor);
	EXPECT_FLOAT_EQ(result[0], 1.0f);
}

// ============================================================================
// Arange Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Arange_Basic) {
	// Test basic arange: [0, 1, 2, 3, 4]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Arange(0.0, 5.0, 1.0);
	
	EXPECT_EQ(tensor.GetShape().Rank, 1);
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], static_cast<float>(i)) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_NonZeroStart) {
	// Test arange with non-zero start: [5, 6, 7, 8, 9]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Arange(5.0, 10.0, 1.0);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 5.0f + static_cast<float>(i)) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_StepSize2) {
	// Test arange with step=2: [0, 2, 4, 6, 8]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Arange(0.0, 10.0, 2.0);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	std::vector<float> expected = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_FractionalStep) {
	// Test arange with fractional step: [0.0, 0.5, 1.0, 1.5, 2.0]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Arange(0.0, 2.5, 0.5);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	std::vector<float> expected = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Arange_NegativeRange) {
	// Test arange with negative values: [-5, -4, -3, -2, -1]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Arange(-5.0, 0.0, 1.0);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], -5.0f + static_cast<float>(i)) << "Index " << i;
	}
}

// ============================================================================
// Linspace Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, Linspace_Basic) {
	// Test basic linspace: [0, 0.25, 0.5, 0.75, 1.0]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Linspace(0.0, 1.0, 5);
	
	EXPECT_EQ(tensor.GetShape().Rank, 1);
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	std::vector<float> expected = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_TwoPoints) {
	// Test linspace with 2 points (start and end)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Linspace(0.0, 10.0, 2);
	
	EXPECT_EQ(tensor.GetShape()[0], 2);
	
	auto result = CopyToHost(tensor);
	EXPECT_FLOAT_EQ(result[0], 0.0f);
	EXPECT_FLOAT_EQ(result[1], 10.0f);
}

TEST_F(TestFnMatrixAllocManual, Linspace_NegativeRange) {
	// Test linspace with negative range: [-1, -0.5, 0, 0.5, 1]
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Linspace(-1.0, 1.0, 5);
	
	EXPECT_EQ(tensor.GetShape()[0], 5);
	
	auto result = CopyToHost(tensor);
	std::vector<float> expected = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_LargeRange) {
	// Test linspace with large range
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Linspace(0.0, 100.0, 11);
	
	EXPECT_EQ(tensor.GetShape()[0], 11);
	
	auto result = CopyToHost(tensor);
	// Should be [0, 10, 20, ..., 100]
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], static_cast<float>(i * 10), 1e-4f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixAllocManual, Linspace_ReverseRange) {
	// Test linspace with start > end (descending)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto tensor = OaFnMatrix::Linspace(10.0, 0.0, 6);
	
	EXPECT_EQ(tensor.GetShape()[0], 6);
	
	auto result = CopyToHost(tensor);
	std::vector<float> expected = {10.0f, 8.0f, 6.0f, 4.0f, 2.0f, 0.0f};
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_NEAR(result[i], expected[i], 1e-6f) << "Index " << i;
	}
}

// ============================================================================
// CausalMask Tests
// ============================================================================

TEST_F(TestFnMatrixAllocManual, CausalMask_Size4) {
	// Test causal mask for sequence length 4
	// Expected: lower triangular matrix with 0s above diagonal, -inf below
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto mask = OaFnMatrix::CausalMask(4);
	
	EXPECT_EQ(mask.GetShape().Rank, 2);
	EXPECT_EQ(mask.GetShape()[0], 4);
	EXPECT_EQ(mask.GetShape()[1], 4);
	
	auto result = CopyToHost(mask);
	
	// Causal mask: 0 on and below diagonal, -inf above
	// Row 0: [0, -inf, -inf, -inf]
	// Row 1: [0, 0, -inf, -inf]
	// Row 2: [0, 0, 0, -inf]
	// Row 3: [0, 0, 0, 0]
	for (OaU32 i = 0; i < 4; ++i) {
		for (OaU32 j = 0; j < 4; ++j) {
			float val = result[i * 4 + j];
			if (j <= i) {
				EXPECT_FLOAT_EQ(val, 0.0f) << "Position [" << i << "," << j << "] should be 0";
			} else {
				EXPECT_TRUE(std::isinf(val) && val < 0) 
					<< "Position [" << i << "," << j << "] should be -inf, got " << val;
			}
		}
	}
}

TEST_F(TestFnMatrixAllocManual, CausalMask_Size1) {
	// Test edge case: causal mask for sequence length 1
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto mask = OaFnMatrix::CausalMask(1);
	
	EXPECT_EQ(mask.GetShape()[0], 1);
	EXPECT_EQ(mask.GetShape()[1], 1);
	
	auto result = CopyToHost(mask);
	EXPECT_FLOAT_EQ(result[0], 0.0f);
}

TEST_F(TestFnMatrixAllocManual, CausalMask_Size8) {
	// Test larger causal mask
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto mask = OaFnMatrix::CausalMask(8);
	
	EXPECT_EQ(mask.GetShape()[0], 8);
	EXPECT_EQ(mask.GetShape()[1], 8);
	
	auto result = CopyToHost(mask);
	
	// Verify structure: lower triangular with 0s, upper triangular with -inf
	for (OaU32 i = 0; i < 8; ++i) {
		for (OaU32 j = 0; j < 8; ++j) {
			float val = result[i * 8 + j];
			if (j <= i) {
				EXPECT_FLOAT_EQ(val, 0.0f) << "Position [" << i << "," << j << "]";
			} else {
				EXPECT_TRUE(std::isinf(val) && val < 0) << "Position [" << i << "," << j << "]";
			}
		}
	}
}

