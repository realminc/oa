// Tests for oa::Matrix shape+fill constructor and arithmetic operator overloads.
//
// The shape+fill constructor delegates to oa::FnMatrix::full, so the braced-init
// form `oa::Matrix a = {oa::MatrixShape{3, 3}, 0.0F};` works directly. Each operator
// dispatches a real compute kernel — uses oa_add_ml_test for vulkan setup.

#include "../oaTest.h"

#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <gtest/gtest.h>

namespace {

oa::Engine* GRt = nullptr;

class TestMatrixOperators : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestMatrixOperators";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}

	// flush + sync the default context so .at() reads committed values.
	static void sync() {
		auto& ctx = oa::ExecutionSession::getActive();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	}
};

// ─── Constructor ──────────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ShapeFillConstructorFloat) {
	oa::Matrix a = {oa::MatrixShape{3, 3}, 5.0F};
	EXPECT_EQ(a.rank(), 2);
	EXPECT_EQ(a.size(0), 3);
	EXPECT_EQ(a.size(1), 3);
	EXPECT_EQ(a.numElements(), 9);
	EXPECT_TRUE(a.hasStorage());
	sync();
	for (oa::I64 i = 0; i < a.numElements(); ++i) {
		EXPECT_FLOAT_EQ(a.at(i), 5.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, ShapeFillConstructorZeros) {
	oa::Matrix m = {oa::MatrixShape{4}, 0.0F};
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(m.at(i), 0.0F) << "i=" << i;
}

// ─── Element-wise arithmetic ──────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ElementwiseAdd) {
	oa::Matrix a = {oa::MatrixShape{3, 3}, 1.0F};
	oa::Matrix b = {oa::MatrixShape{3, 3}, 2.0F};
	oa::Matrix c = a + b;
	sync();
	for (oa::I64 i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(c.at(i), 3.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseSub) {
	oa::Matrix a = {oa::MatrixShape{2, 2}, 5.0F};
	oa::Matrix b = {oa::MatrixShape{2, 2}, 3.0F};
	oa::Matrix c = a - b;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.at(i), 2.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseMul) {
	oa::Matrix a = {oa::MatrixShape{2, 2}, 3.0F};
	oa::Matrix b = {oa::MatrixShape{2, 2}, 4.0F};
	oa::Matrix c = a * b;  // element-wise — NOT matmul
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.at(i), 12.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ElementwiseDiv) {
	oa::Matrix a = {oa::MatrixShape{2, 2}, 8.0F};
	oa::Matrix b = {oa::MatrixShape{2, 2}, 2.0F};
	oa::Matrix c = a / b;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c.at(i), 4.0F) << "i=" << i;
}

// ─── scalar arithmetic ────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ScalarMul) {
	oa::Matrix a = {oa::MatrixShape{8}, 1.5F};
	oa::Matrix b = a * 4.0F;
	sync();
	for (oa::I64 i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(b.at(i), 6.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarAdd) {
	oa::Matrix a = {oa::MatrixShape{4}, 1.0F};
	oa::Matrix b = a + 2.5F;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.at(i), 3.5F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarDiv) {
	oa::Matrix a = {oa::MatrixShape{4}, 10.0F};
	oa::Matrix b = a / 4.0F;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.at(i), 2.5F) << "i=" << i;
}

// ─── Unary ────────────────────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, Negate) {
	oa::Matrix a = {oa::MatrixShape{4}, 3.0F};
	oa::Matrix b = -a;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(b.at(i), -3.0F) << "i=" << i;
}

// ─── Compound assignment ──────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, AddAssign) {
	oa::Matrix a = {oa::MatrixShape{4}, 1.0F};
	oa::Matrix b = {oa::MatrixShape{4}, 2.0F};
	a += b;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(a.at(i), 3.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, ScalarMulAssign) {
	oa::Matrix a = {oa::MatrixShape{4}, 2.0F};
	a *= 3.0F;
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(a.at(i), 6.0F) << "i=" << i;
}

TEST_VK(TestMatrixOperators, MatrixCompoundAssignmentsMutateSharedStorage) {
	oa::Matrix add = {oa::MatrixShape{4}, 5.0F};
	oa::Matrix addView = add.view(oa::MatrixShape{2, 2});
	const oa::U64 addVersion = addView.observeStorageMutationVersion();
	add += oa::Matrix{oa::MatrixShape{4}, 2.0F};
	EXPECT_GT(addView.currentStorageMutationVersion(), addVersion);

	oa::Matrix sub = {oa::MatrixShape{4}, 9.0F};
	oa::Matrix subAlias = sub;
	const oa::U64 subVersion = subAlias.observeStorageMutationVersion();
	sub -= oa::Matrix{oa::MatrixShape{4}, 2.0F};
	EXPECT_GT(subAlias.currentStorageMutationVersion(), subVersion);

	oa::Matrix mul = {oa::MatrixShape{4}, 3.0F};
	oa::Matrix mulAlias = mul;
	const oa::U64 mulVersion = mulAlias.observeStorageMutationVersion();
	mul *= oa::Matrix{oa::MatrixShape{4}, 4.0F};
	EXPECT_GT(mulAlias.currentStorageMutationVersion(), mulVersion);

	oa::Matrix div = {oa::MatrixShape{4}, 8.0F};
	oa::Matrix divAlias = div;
	const oa::U64 divVersion = divAlias.observeStorageMutationVersion();
	div /= oa::Matrix{oa::MatrixShape{4}, 2.0F};
	EXPECT_GT(divAlias.currentStorageMutationVersion(), divVersion);

	sync();
	for (oa::I64 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(add.at(i), 7.0F) << "add i=" << i;
		EXPECT_FLOAT_EQ(addView.at(i), 7.0F) << "add view i=" << i;
		EXPECT_FLOAT_EQ(sub.at(i), 7.0F) << "sub i=" << i;
		EXPECT_FLOAT_EQ(subAlias.at(i), 7.0F) << "sub alias i=" << i;
		EXPECT_FLOAT_EQ(mul.at(i), 12.0F) << "mul i=" << i;
		EXPECT_FLOAT_EQ(mulAlias.at(i), 12.0F) << "mul alias i=" << i;
		EXPECT_FLOAT_EQ(div.at(i), 4.0F) << "div i=" << i;
		EXPECT_FLOAT_EQ(divAlias.at(i), 4.0F) << "div alias i=" << i;
	}
}

// ─── Broadcast operations ──────────────────────────────────────────────

TEST_VK(TestMatrixOperators, BroadcastMul1DTo2D) {
	oa::Matrix a = {oa::MatrixShape{4}, 2.0F};
	oa::Matrix b = {oa::MatrixShape{3, 4}, 5.0F};
	oa::Matrix c = a * b;  // [4] * [3,4] → [3,4] of 10.0
	sync();
	EXPECT_EQ(c.rank(), 2);
	EXPECT_EQ(c.size(0), 3);
	EXPECT_EQ(c.size(1), 4);
	for (oa::I64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(c.at(i), 10.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastAddBias2D) {
	oa::Matrix logits = {oa::MatrixShape{3, 4}, 1.0F};
	oa::Matrix bias   = {oa::MatrixShape{4},     0.5F};
	oa::Matrix r = logits + bias;  // [3,4] + [4] → [3,4] of 1.5
	sync();
	EXPECT_EQ(r.rank(), 2);
	EXPECT_EQ(r.size(0), 3);
	EXPECT_EQ(r.size(1), 4);
	for (oa::I64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(r.at(i), 1.5F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastSubOnes) {
	oa::Matrix a = {oa::MatrixShape{2, 3}, 5.0F};
	oa::Matrix b = {oa::MatrixShape{1},    2.0F};
	oa::Matrix c = a - b;  // [2,3] - [1] → [2,3] of 3.0
	sync();
	EXPECT_EQ(c.rank(), 2);
	for (oa::I64 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(c.at(i), 3.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastDivRowVector) {
	oa::Matrix a = {oa::MatrixShape{2, 3}, 8.0F};
	oa::Matrix b = {oa::MatrixShape{3},    2.0F};
	oa::Matrix c = a / b;  // [2,3] / [3] → [2,3] of 4.0
	sync();
	EXPECT_EQ(c.rank(), 2);
	for (oa::I64 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(c.at(i), 4.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, ScalarAddBroadcast) {
	oa::Matrix a = {oa::MatrixShape{2, 2}, 3.0F};
	oa::Matrix b = a + 1.5F;  // scalar → [1] broadcast to [2,2]
	sync();
	for (oa::I64 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(b.at(i), 4.5F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastInPlaceAdd) {
	oa::Matrix a = {oa::MatrixShape{3, 4}, 1.0F};
	oa::Matrix b = {oa::MatrixShape{4},     2.0F};
	a += b;  // [3,4] + [4] in-place
	sync();
	for (oa::I64 i = 0; i < 12; ++i) {
		EXPECT_FLOAT_EQ(a.at(i), 3.0F) << "i=" << i;
	}
}

TEST_VK(TestMatrixOperators, BroadcastMatrixCompoundAssignments) {
	oa::Matrix sub = {oa::MatrixShape{2, 3}, 9.0F};
	oa::Matrix mul = {oa::MatrixShape{2, 3}, 3.0F};
	oa::Matrix div = {oa::MatrixShape{2, 3}, 8.0F};
	oa::Matrix rhs = {oa::MatrixShape{3}, 2.0F};
	sub -= rhs;
	mul *= rhs;
	div /= rhs;
	sync();
	for (oa::I64 i = 0; i < 6; ++i) {
		EXPECT_FLOAT_EQ(sub.at(i), 7.0F) << "sub i=" << i;
		EXPECT_FLOAT_EQ(mul.at(i), 6.0F) << "mul i=" << i;
		EXPECT_FLOAT_EQ(div.at(i), 4.0F) << "div i=" << i;
	}
}

// ─── Chained expression ──────────────────────────────────────────────────

TEST_VK(TestMatrixOperators, ChainedExpression) {
	oa::Matrix a = {oa::MatrixShape{4}, 2.0F};
	oa::Matrix b = {oa::MatrixShape{4}, 3.0F};
	oa::Matrix r = (a + b) * 2.0F - a;  // (2+3)*2 - 2 = 8
	sync();
	for (oa::I64 i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r.at(i), 8.0F) << "i=" << i;
}

}  // namespace
