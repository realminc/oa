// Tests for OaMatrix shape+fill constructor and arithmetic operator overloads.
//
// The shape+fill constructor delegates to OaFnMatrix::Full, so the braced-init
// form `OaMatrix a = {OaMatrixShape{3, 3}, 0.0F};` works directly. Each operator
// dispatches a real compute kernel — uses oa_add_ml_test for Vulkan setup.

#include "../OaTest.h"

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <gtest/gtest.h>

namespace {

OaComputeEngine* GRt = nullptr;

class TestMatrixOperators : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestMatrixOperators";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
	}

	// Flush + sync the default context so .At() reads committed values.
	static void Sync() {
		auto& ctx = OaContext::GetDefault();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
};

// ─── Constructor ──────────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ShapeFillConstructorFloat) {
	OaMatrix a = {OaMatrixShape{3, 3}, 5.0F};
	EXPECT_EQ(a.Rank(), 2);
	EXPECT_EQ(a.Size(0), 3);
	EXPECT_EQ(a.Size(1), 3);
	EXPECT_EQ(a.NumElements(), 9);
	EXPECT_TRUE(a.HasStorage());
	Sync();
	for (OaI64 i = 0; i < a.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(a.At(i), 5.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, ShapeFillConstructorZeros) {
	OaMatrix m = {OaMatrixShape{4}, 0.0F};
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(m.At(i), 0.0F) << "i=" << i;
}

// ─── Element-wise arithmetic ──────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ElementwiseAdd) {
	OaMatrix a = {OaMatrixShape{3, 3}, 1.0F};
	OaMatrix b = {OaMatrixShape{3, 3}, 2.0F};
	OaMatrix c = a + b;
	Sync();
	for (OaI64 i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(c.At(i), 3.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseSub) {
	OaMatrix a = {OaMatrixShape{2, 2}, 5.0F};
	OaMatrix b = {OaMatrixShape{2, 2}, 3.0F};
	OaMatrix c = a - b;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.At(i), 2.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseMul) {
	OaMatrix a = {OaMatrixShape{2, 2}, 3.0F};
	OaMatrix b = {OaMatrixShape{2, 2}, 4.0F};
	OaMatrix c = a * b;  // element-wise — NOT matmul
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.At(i), 12.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseDiv) {
	OaMatrix a = {OaMatrixShape{2, 2}, 8.0F};
	OaMatrix b = {OaMatrixShape{2, 2}, 2.0F};
	OaMatrix c = a / b;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.At(i), 4.0F) << "i=" << i;
}

// ─── Scalar arithmetic ────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ScalarMul) {
	OaMatrix a = {OaMatrixShape{8}, 1.5F};
	OaMatrix b = a * 4.0F;
	Sync();
	for (OaI64 i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(b.At(i), 6.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarAdd) {
	OaMatrix a = {OaMatrixShape{4}, 1.0F};
	OaMatrix b = a + 2.5F;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.At(i), 3.5F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarDiv) {
	OaMatrix a = {OaMatrixShape{4}, 10.0F};
	OaMatrix b = a / 4.0F;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.At(i), 2.5F) << "i=" << i;
}

// ─── Unary ────────────────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, Negate) {
	OaMatrix a = {OaMatrixShape{4}, 3.0F};
	OaMatrix b = -a;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.At(i), -3.0F) << "i=" << i;
}

// ─── Compound assignment ──────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, AddAssign) {
	OaMatrix a = {OaMatrixShape{4}, 1.0F};
	OaMatrix b = {OaMatrixShape{4}, 2.0F};
	a += b;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(a.At(i), 3.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarMulAssign) {
	OaMatrix a = {OaMatrixShape{4}, 2.0F};
	a *= 3.0F;
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(a.At(i), 6.0F) << "i=" << i;
}

// ─── Broadcast operations ──────────────────────────────────────────────

TEST_VK(TestMatrixOperators, BroadcastMul1DTo2D) {
	OaMatrix a = {OaMatrixShape{4}, 2.0F};
	OaMatrix b = {OaMatrixShape{3, 4}, 5.0F};
	OaMatrix c = a * b;  // [4] * [3,4] → [3,4] of 10.0
	Sync();
	EXPECT_EQ(c.Rank(), 2);
	EXPECT_EQ(c.Size(0), 3);
	EXPECT_EQ(c.Size(1), 4);
	for (OaI64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(c.At(i), 10.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastAddBias2D) {
	OaMatrix logits = {OaMatrixShape{3, 4}, 1.0F};
	OaMatrix bias   = {OaMatrixShape{4},     0.5F};
	OaMatrix r = logits + bias;  // [3,4] + [4] → [3,4] of 1.5
	Sync();
	EXPECT_EQ(r.Rank(), 2);
	EXPECT_EQ(r.Size(0), 3);
	EXPECT_EQ(r.Size(1), 4);
	for (OaI64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(r.At(i), 1.5F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastSubOnes) {
	OaMatrix a = {OaMatrixShape{2, 3}, 5.0F};
	OaMatrix b = {OaMatrixShape{1},    2.0F};
	OaMatrix c = a - b;  // [2,3] - [1] → [2,3] of 3.0
	Sync();
	EXPECT_EQ(c.Rank(), 2);
	for (OaI64 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(c.At(i), 3.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastDivRowVector) {
	OaMatrix a = {OaMatrixShape{2, 3}, 8.0F};
	OaMatrix b = {OaMatrixShape{3},    2.0F};
	OaMatrix c = a / b;  // [2,3] / [3] → [2,3] of 4.0
	Sync();
	EXPECT_EQ(c.Rank(), 2);
	for (OaI64 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(c.At(i), 4.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, ScalarAddBroadcast) {
	OaMatrix a = {OaMatrixShape{2, 2}, 3.0F};
	OaMatrix b = a + 1.5F;  // scalar → [1] broadcast to [2,2]
	Sync();
	for (OaI64 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(b.At(i), 4.5F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastInPlaceAdd) {
	OaMatrix a = {OaMatrixShape{3, 4}, 1.0F};
	OaMatrix b = {OaMatrixShape{4},     2.0F};
	a += b;  // [3,4] + [4] in-place
	Sync();
	for (OaI64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(a.At(i), 3.0F) << "i=" << i;
	}
}

// ─── Chained expression ──────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ChainedExpression) {
	OaMatrix a = {OaMatrixShape{4}, 2.0F};
	OaMatrix b = {OaMatrixShape{4}, 3.0F};
	OaMatrix r = (a + b) * 2.0F - a;  // (2+3)*2 - 2 = 8
	Sync();
	for (OaI64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r.At(i), 8.0F) << "i=" << i;
}

}  // namespace
