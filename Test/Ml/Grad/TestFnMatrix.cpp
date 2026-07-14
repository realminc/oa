// OaFnMatrix smoke tests — Reshape, RepeatInterleave, CausalMask, TopK, Equal, CompactRows, ScatterRows

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrix : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrix";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
		OaRuntimeGlobal::SetRuntime(GRt);
	}
};

TEST_VK(TestFnMatrix, Reshape_InferDim) {
	auto a = OaFnMatrix::Zeros(OaMatrixShape{4, 6});
	auto b = OaFnMatrix::Reshape(a, {-1, 3});
	EXPECT_EQ(b.Rank(), 2);
	EXPECT_EQ(b.Size(0), 8);
	EXPECT_EQ(b.Size(1), 3);
}

TEST_VK(TestFnMatrix, RepeatInterleave_Dim1) {
	auto a = OaFnMatrix::Zeros(OaMatrixShape{2, 3, 4});
	auto b = OaFnMatrix::RepeatInterleave(a, 2, 1);
	EXPECT_EQ(b.Size(0), 2);
	EXPECT_EQ(b.Size(1), 6);
	EXPECT_EQ(b.Size(2), 4);
}

TEST_VK(TestFnMatrix, CausalMask_Shape) {
	auto scores = OaFnMatrix::Zeros(OaMatrixShape{4, 4});
	auto masked = OaFnMatrix::CausalMask(scores);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	EXPECT_EQ(masked.GetShape(), scores.GetShape());
	EXPECT_GT(masked.At(1 * 4 + 0), -1e8f);  // below diagonal: not masked
	EXPECT_LT(masked.At(0 * 4 + 1), -1e8f);  // above diagonal: masked
}

TEST_VK(TestFnMatrix, CausalMask_ConstructedOnGpu) {
	auto mask = OaFnMatrix::CausalMask(4);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	EXPECT_FLOAT_EQ(mask.At(3 * 4 + 0), 0.0f);
	EXPECT_LT(mask.At(0 * 4 + 3), -1e8f);
}

TEST_VK(TestFnMatrix, CausalMask_BackwardZerosFuturePositions) {
	auto scores = OaFnMatrix::Zeros(OaMatrixShape{3, 3});
	scores.SetRequiresGrad(true);
	OaGradientTape tape;
	auto loss = OaFnMatrix::Sum(OaFnMatrix::CausalMask(scores), -1);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();
	const auto& grad = scores.GradMatrix();
	for (OaI32 q = 0; q < 3; ++q)
		for (OaI32 k = 0; k < 3; ++k)
			EXPECT_FLOAT_EQ(grad.At(q * 3 + k), k <= q ? 1.0f : 0.0f);
}

TEST_VK(TestFnMatrix, TopK_Basic) {
	auto a = OaFnMatrix::Zeros(OaMatrixShape{2, 4});
	a.Set(0 * 4 + 3, 3.0f);
	a.Set(0 * 4 + 1, 1.0f);
	a.Set(1 * 4 + 0, 5.0f);
	a.Set(1 * 4 + 2, 2.0f);
	auto result = OaFnMatrix::TopK(a, 2);
	EXPECT_EQ(result.Values.Size(0), 2);
	EXPECT_EQ(result.Values.Size(1), 2);
	EXPECT_NEAR(result.Values.At(0 * 2 + 0), 3.0f, 1e-5f);
	EXPECT_NEAR(result.Values.At(1 * 2 + 0), 5.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, Equal_Float) {
	auto a = OaFnMatrix::Zeros(OaMatrixShape{2, 3});
	a.Set(0, 1.0f);
	a.Set(4, 1.0f);
	auto mask = OaFnMatrix::Equal(a, 1.0f);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	EXPECT_NEAR(mask.At(0), 1.0f, 1e-5f);
	EXPECT_NEAR(mask.At(1), 0.0f, 1e-5f);
	EXPECT_NEAR(mask.At(4), 1.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, Slice_Dim1) {
	auto a = OaFnMatrix::Zeros(OaMatrixShape{3, 4});
	a.Set(0 * 4 + 2, 7.0f);
	a.Set(1 * 4 + 2, 8.0f);
	auto s = OaFnMatrix::Slice(a, 1, 2, 3);  // [:, 2:3]
	EXPECT_EQ(s.Size(0), 3);
	EXPECT_EQ(s.Size(1), 1);
	// Slice records a deferred MatrixCopyRegion kernel; flush before host readback
	// (At() reads mapped memory directly and does NOT execute the context).
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	EXPECT_NEAR(s.At(0), 7.0f, 1e-5f);
	EXPECT_NEAR(s.At(1), 8.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, CompactScatterRows) {
	auto x = OaFnMatrix::Zeros(OaMatrixShape{4, 2});
	x.Set(0 * 2 + 0, 1.0f); x.Set(0 * 2 + 1, 2.0f);
	x.Set(2 * 2 + 0, 3.0f); x.Set(2 * 2 + 1, 4.0f);

	auto mask = OaFnMatrix::Zeros(OaMatrixShape{4, 1});
	mask.Set(0, 1.0f);
	mask.Set(2, 1.0f);

	auto compact = OaFnMatrix::CompactRows(x, mask);
	auto selected = compact.Values;
	EXPECT_EQ(selected.Size(0), 4);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	(void)OaContext::GetDefault().Sync();
	EXPECT_NEAR(selected.At(0 * 2 + 0), 1.0f, 1e-5f);
	EXPECT_NEAR(selected.At(1 * 2 + 0), 3.0f, 1e-5f);

	auto base = OaFnMatrix::Zeros(OaMatrixShape{4, 2});
	auto scattered = OaFnMatrix::ScatterRows(base, selected, compact.RowMap, compact.Count);
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	(void)OaContext::GetDefault().Sync();
	EXPECT_NEAR(scattered.At(0 * 2 + 0), 1.0f, 1e-5f);
	EXPECT_NEAR(scattered.At(2 * 2 + 0), 3.0f, 1e-5f);
	EXPECT_NEAR(scattered.At(1 * 2 + 0), 0.0f, 1e-5f);
}
