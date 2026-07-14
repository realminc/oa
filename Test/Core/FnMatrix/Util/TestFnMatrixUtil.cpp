// Tests for Core/FnMatrix utility helper operations
// RepeatInterleave, Equal, TopK, CompactRows, ScatterRows

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixUtil : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixUtil";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
		OaRuntimeGlobal::SetRuntime(GRt);
	}
};

// Helper to copy matrix to host
static std::vector<float> CopyToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to create matrix from host data
static OaMatrix CreateFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

TEST_VK(TestFnMatrixUtil, SampleLogits_GreedyRows) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto logits = CreateFromHost({1.0f, 4.0f, 2.0f, -1.0f, 3.0f, 3.0f}, OaMatrixShape{2, 3});
	auto ids = OaFnMatrix::SampleLogits(logits, 0.0f);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	ASSERT_EQ(ids.GetDtype(), OaScalarType::Int32);
	ASSERT_EQ(ids.NumElements(), 2);
	EXPECT_EQ(ids.DataAs<const OaI32>()[0], 1);
	EXPECT_EQ(ids.DataAs<const OaI32>()[1], 1); // equal logits resolve to lower index
}

TEST_VK(TestFnMatrixUtil, SampleLogits_TopKOneAlwaysArgmax) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto logits = CreateFromHost({-2.0f, 0.5f, 7.0f, 1.0f}, OaMatrixShape{1, 4});
	auto ids = OaFnMatrix::SampleLogits(logits, 0.8f, 1, 0.9f, 123);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	EXPECT_EQ(ids.DataAs<const OaI32>()[0], 2);
}

// ============================================================================
// RepeatInterleave Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Dim0) {
	// Test repeating along dimension 0: [2,3] -> [4,3] (repeat=2)
	std::vector<float> input_data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	auto output = OaFnMatrix::RepeatInterleave(input, 2, 0);
	
	EXPECT_EQ(output.GetShape().Rank, 2);
	EXPECT_EQ(output.GetShape()[0], 4);
	EXPECT_EQ(output.GetShape()[1], 3);
	
	auto result = CopyToHost(output);
	std::vector<float> expected = {
		1.0f, 2.0f, 3.0f,  // row 0 repeated
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,  // row 1 repeated
		4.0f, 5.0f, 6.0f
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Dim1) {
	// Test repeating along dimension 1: [2,3] -> [2,6] (repeat=2)
	std::vector<float> input_data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	auto output = OaFnMatrix::RepeatInterleave(input, 2, 1);
	
	EXPECT_EQ(output.GetShape().Rank, 2);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 6);
	
	auto result = CopyToHost(output);
	std::vector<float> expected = {
		1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
		4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f
	};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Repeat3) {
	// Test with repeat=3
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f};

	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{3});
	auto output = OaFnMatrix::RepeatInterleave(input, 3, 0);

	EXPECT_EQ(output.GetShape()[0], 9);

	auto result = CopyToHost(output);
	std::vector<float> expected = {
		1.0f, 1.0f, 1.0f,
		2.0f, 2.0f, 2.0f,
		3.0f, 3.0f, 3.0f
	};

	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

// ============================================================================
// RepeatInterleave Gradcheck — finite-difference vs autograd
// ============================================================================

static bool GradClose(OaF32 a, OaF32 n, OaF32 atol = 2e-3F, OaF32 rtol = 2e-2F) {
	return std::abs(a - n) <= (atol + rtol * std::abs(n));
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Gradcheck_Dim0) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	// Input [2, 3] with requires_grad
	std::vector<float> input_data = {0.5f, -1.2f, 3.1f, 0.8f, -2.4f, 1.7f};
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	input.SetRequiresGrad(true);

	// Tape must be active before forward so grad nodes are attached
	OaGradientTape tape;

	// Forward: RepeatInterleave(x, 2, 0) -> [4, 3], then sum to scalar
	auto repeated = OaFnMatrix::RepeatInterleave(input, 2, 0);
	auto loss = OaFnMatrix::Sum(repeated, -1);  // sum all -> scalar
	(void)ctx.Execute();
	(void)ctx.Sync();

	// Analytical backward
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto grad = CopyToHost(input.GradMatrix());

	// Finite-difference: perturb each input element, recompute loss
	const float eps = 1e-2F;
	float* d = input.DataAs<float>();
	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			OaGradNo noGrad;
			d[i] = orig + eps; (void)ctx.Sync();
			auto rp = OaFnMatrix::RepeatInterleave(input, 2, 0);
			auto lp = OaFnMatrix::Sum(rp, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vp = CopyToHost(lp)[0];

			d[i] = orig - eps; (void)ctx.Sync();
			auto rm = OaFnMatrix::RepeatInterleave(input, 2, 0);
			auto lm = OaFnMatrix::Sum(rm, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vm = CopyToHost(lm)[0];
		}
		d[i] = orig; (void)ctx.Sync();

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(GradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Gradcheck_Dim1) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	// Input [2, 3] with requires_grad
	std::vector<float> input_data = {0.3f, -0.7f, 1.5f, 2.1f, -0.4f, 0.9f};
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	input.SetRequiresGrad(true);

	// Tape must be active before forward so grad nodes are attached
	OaGradientTape tape;

	// Forward: RepeatInterleave(x, 3, 1) -> [2, 9], then sum to scalar
	auto repeated = OaFnMatrix::RepeatInterleave(input, 3, 1);
	auto loss = OaFnMatrix::Sum(repeated, -1);
	(void)ctx.Execute();
	(void)ctx.Sync();

	// Analytical backward
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto grad = CopyToHost(input.GradMatrix());

	// Finite-difference
	const float eps = 1e-2F;
	float* d = input.DataAs<float>();
	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			OaGradNo noGrad;
			d[i] = orig + eps; (void)ctx.Sync();
			auto rp = OaFnMatrix::RepeatInterleave(input, 3, 1);
			auto lp = OaFnMatrix::Sum(rp, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vp = CopyToHost(lp)[0];

			d[i] = orig - eps; (void)ctx.Sync();
			auto rm = OaFnMatrix::RepeatInterleave(input, 3, 1);
			auto lm = OaFnMatrix::Sum(rm, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vm = CopyToHost(lm)[0];
		}
		d[i] = orig; (void)ctx.Sync();

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(GradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}

// ============================================================================
// Equal Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, Equal_AllMatch) {
	// Test where all elements equal the value
	std::vector<float> input_data = {5.0f, 5.0f, 5.0f, 5.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{4});
	auto output = OaFnMatrix::Equal(input, 5.0f);
	
	auto result = CopyToHost(output);
	
	// Equal returns 1.0f for true, 0.0f for false
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 1.0f) << "Expected all 1.0f at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Equal_NoneMatch) {
	// Test where no elements equal the value
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{4});
	auto output = OaFnMatrix::Equal(input, 5.0f);
	
	auto result = CopyToHost(output);
	
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "Expected all 0.0f at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Equal_Mixed) {
	// Test with mixed matches
	std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 3.0f, 2.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	auto output = OaFnMatrix::Equal(input, 2.0f);
	
	auto result = CopyToHost(output);
	std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, GreaterEqual_Mixed) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(std::vector<float>{-1.0f, 0.25f, 0.5f, 2.0f}, OaMatrixShape{4});
	auto output = OaFnMatrix::GreaterEqual(input, 0.5f);
	auto result = CopyToHost(output);
	ASSERT_EQ(result.size(), 4u);
	EXPECT_FLOAT_EQ(result[0], 0.0f);
	EXPECT_FLOAT_EQ(result[1], 0.0f);
	EXPECT_FLOAT_EQ(result[2], 1.0f);
	EXPECT_FLOAT_EQ(result[3], 1.0f);
}

// ============================================================================
// Copy Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, Copy_1D) {
	// Test copying a 1D tensor
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	auto copied = OaFnMatrix::Copy(input);
	
	EXPECT_EQ(copied.GetShape().Rank, 1);
	EXPECT_EQ(copied.GetShape()[0], 5);
	
	auto result = CopyToHost(copied);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Copy_2D) {
	// Test copying a 2D tensor
	std::vector<float> input_data = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{2, 3});
	auto copied = OaFnMatrix::Copy(input);
	
	EXPECT_EQ(copied.GetShape().Rank, 2);
	EXPECT_EQ(copied.GetShape()[0], 2);
	EXPECT_EQ(copied.GetShape()[1], 3);
	
	auto result = CopyToHost(copied);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Copy_Independence) {
	// Test that copy creates independent tensor (modifying one doesn't affect the other)
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{3});
	auto copied = OaFnMatrix::Copy(input);
	
	// Modify the copy
	OaFnMatrix::Fill(copied, 99.0f);
	
	// Original should be unchanged
	auto original_result = CopyToHost(input);
	for (size_t i = 0; i < original_result.size(); ++i) {
		EXPECT_FLOAT_EQ(original_result[i], input_data[i]) << "Original modified at index " << i;
	}
	
	// Copy should be modified
	auto copied_result = CopyToHost(copied);
	for (size_t i = 0; i < copied_result.size(); ++i) {
		EXPECT_FLOAT_EQ(copied_result[i], 99.0f) << "Copy not modified at index " << i;
	}
}

// ============================================================================
// Detach Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, Detach_1D) {
	// Test detaching a 1D tensor (breaks autograd connection)
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{4});
	auto detached = OaFnMatrix::Detach(input);
	
	EXPECT_EQ(detached.GetShape().Rank, 1);
	EXPECT_EQ(detached.GetShape()[0], 4);
	
	auto result = CopyToHost(detached);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Detach_2D) {
	// Test detaching a 2D tensor
	std::vector<float> input_data = {
		1.0f, 2.0f,
		3.0f, 4.0f,
		5.0f, 6.0f
	};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{3, 2});
	auto detached = OaFnMatrix::Detach(input);
	
	EXPECT_EQ(detached.GetShape().Rank, 2);
	EXPECT_EQ(detached.GetShape()[0], 3);
	EXPECT_EQ(detached.GetShape()[1], 2);
	
	auto result = CopyToHost(detached);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Detach_PreservesData) {
	// Test that detach preserves data exactly
	std::vector<float> input_data = {-5.5f, 0.0f, 3.14f, 100.0f, -0.001f};
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto input = CreateFromHost(input_data, OaMatrixShape{5});
	auto detached = OaFnMatrix::Detach(input);
	
	auto result = CopyToHost(detached);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Data not preserved at index " << i;
	}
}

// ============================================================================
// CompactRows Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, CompactRows_Forward) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	auto input = CreateFromHost(data, OaMatrixShape{4, 2});
	std::vector<float> mask = {0.0f, 1.0f, 0.0f, 1.0f};
	auto maskM = CreateFromHost(mask, OaMatrixShape{4});

	auto compact = OaFnMatrix::CompactRows(input, maskM);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto result = CopyToHost(compact.Values);
	EXPECT_EQ(compact.Count.DataAs<const OaU32>()[0], 2u);
	ASSERT_EQ(result.size(), 8u);
	EXPECT_FLOAT_EQ(result[0], 3.0f);
	EXPECT_FLOAT_EQ(result[1], 4.0f);
	EXPECT_FLOAT_EQ(result[2], 7.0f);
	EXPECT_FLOAT_EQ(result[3], 8.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_DeferredMaskAndMultipleScanChunks) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();
	constexpr OaI64 T = 600;
	constexpr OaI64 D = 2;

	std::vector<float> data(static_cast<size_t>(T * D));
	std::vector<float> selector(static_cast<size_t>(T));
	OaU32 expectedCount = 0;
	for (OaI64 row = 0; row < T; ++row) {
		data[static_cast<size_t>(row * D)] = static_cast<float>(row);
		data[static_cast<size_t>(row * D + 1)] = static_cast<float>(-row);
		selector[static_cast<size_t>(row)] = row % 3 == 1 ? 1.0f : 0.0f;
		if (row % 3 == 1) ++expectedCount;
	}

	auto input = CreateFromHost(data, OaMatrixShape{T, D});
	auto selectorM = CreateFromHost(selector, OaMatrixShape{T});
	// Equal and CompactRows remain in one deferred GPU graph. This specifically
	// guards against reintroducing a hidden host read of a pending mask.
	auto mask = OaFnMatrix::Equal(selectorM, 1.0f);
	auto compact = OaFnMatrix::CompactRows(input, mask);
	(void)ctx.Execute();
	(void)ctx.Sync();

	EXPECT_EQ(compact.Count.DataAs<const OaU32>()[0], expectedCount);
	const auto values = CopyToHost(compact.Values);
	const auto* rowMap = compact.RowMap.DataAs<const OaU32>();
	for (OaU32 slot = 0; slot < expectedCount; ++slot) {
		const OaU32 expectedRow = slot * 3 + 1;
		EXPECT_EQ(rowMap[slot], expectedRow);
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(slot * D)], static_cast<float>(expectedRow));
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(slot * D + 1)], -static_cast<float>(expectedRow));
	}
	for (OaI64 i = static_cast<OaI64>(expectedCount) * D; i < T * D; ++i)
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(i)], 0.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_AllAndNone) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = CreateFromHost(data, OaMatrixShape{4, 1});
	auto all = CreateFromHost(std::vector<float>(4, 1.0f), OaMatrixShape{4});
	auto none = CreateFromHost(std::vector<float>(4, 0.0f), OaMatrixShape{4});
	auto compactAll = OaFnMatrix::CompactRows(input, all);
	auto compactNone = OaFnMatrix::CompactRows(input, none);
	(void)ctx.Execute();
	(void)ctx.Sync();

	EXPECT_EQ(compactAll.Count.DataAs<const OaU32>()[0], 4u);
	EXPECT_EQ(compactNone.Count.DataAs<const OaU32>()[0], 0u);
	EXPECT_EQ(CopyToHost(compactAll.Values), data);
	for (float value : CopyToHost(compactNone.Values)) EXPECT_FLOAT_EQ(value, 0.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_Gradcheck) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	std::vector<float> data = {0.5f, -1.2f, 3.1f, 0.8f, -2.4f, 1.7f, 0.3f, -0.9f};
	auto input = CreateFromHost(data, OaMatrixShape{4, 2});
	input.SetRequiresGrad(true);
	std::vector<float> mask = {0.0f, 1.0f, 0.0f, 1.0f};
	auto maskM = CreateFromHost(mask, OaMatrixShape{4});

	OaGradientTape tape;
	auto selected = OaFnMatrix::CompactRows(input, maskM).Values;
	auto loss = OaFnMatrix::Sum(selected, -1);
	(void)ctx.Execute(); (void)ctx.Sync();

	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto grad = CopyToHost(input.GradMatrix());

	const float eps = 1e-2F;
	float* d = input.DataAs<float>();
	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			OaGradNo noGrad;
			d[i] = orig + eps; (void)ctx.Sync();
			auto sp = OaFnMatrix::CompactRows(input, maskM).Values;
			auto lp = OaFnMatrix::Sum(sp, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vp = CopyToHost(lp)[0];

			d[i] = orig - eps; (void)ctx.Sync();
			auto sm = OaFnMatrix::CompactRows(input, maskM).Values;
			auto lm = OaFnMatrix::Sum(sm, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vm = CopyToHost(lm)[0];
		}
		d[i] = orig; (void)ctx.Sync();

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(GradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}

// ============================================================================
// ScatterRows Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, ScatterRows_Forward) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	std::vector<float> self_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
	auto self = CreateFromHost(self_data, OaMatrixShape{3, 2});
	std::vector<float> mask = {1.0f, 0.0f, 1.0f};
	auto maskM = CreateFromHost(mask, OaMatrixShape{3});
	std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
	auto source = CreateFromHost(src_data, OaMatrixShape{3, 2});

	auto plan = OaFnMatrix::CompactRows(self, maskM);
	auto out = OaFnMatrix::ScatterRows(self, source, plan.RowMap, plan.Count);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto result = CopyToHost(out);
	ASSERT_EQ(result.size(), 6u);
	EXPECT_FLOAT_EQ(result[0], 11.0f);
	EXPECT_FLOAT_EQ(result[1], 22.0f);
	EXPECT_FLOAT_EQ(result[2], 30.0f);
	EXPECT_FLOAT_EQ(result[3], 40.0f);
	EXPECT_FLOAT_EQ(result[4], 53.0f);
	EXPECT_FLOAT_EQ(result[5], 64.0f);
}

TEST_VK(TestFnMatrixUtil, ScatterRows_Gradcheck_Source) {
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto& ctx = OaContext::GetDefault();

	std::vector<float> self_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
	auto self = CreateFromHost(self_data, OaMatrixShape{3, 2});
	std::vector<float> mask = {1.0f, 0.0f, 1.0f};
	auto maskM = CreateFromHost(mask, OaMatrixShape{3});
	std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
	auto source = CreateFromHost(src_data, OaMatrixShape{3, 2});
	source.SetRequiresGrad(true);

	OaGradientTape tape;
	auto plan = OaFnMatrix::CompactRows(self, maskM);
	auto out = OaFnMatrix::ScatterRows(self, source, plan.RowMap, plan.Count);
	auto loss = OaFnMatrix::Sum(out, -1);
	(void)ctx.Execute(); (void)ctx.Sync();

	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto grad = CopyToHost(source.GradMatrix());

	const float eps = 1e-2F;
	float* d = source.DataAs<float>();
	for (OaI64 i = 0; i < source.NumElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			OaGradNo noGrad;
			d[i] = orig + eps; (void)ctx.Sync();
			auto p = OaFnMatrix::CompactRows(self, maskM);
			auto op = OaFnMatrix::ScatterRows(self, source, p.RowMap, p.Count);
			auto lp = OaFnMatrix::Sum(op, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vp = CopyToHost(lp)[0];

			d[i] = orig - eps; (void)ctx.Sync();
			auto m = OaFnMatrix::CompactRows(self, maskM);
			auto om = OaFnMatrix::ScatterRows(self, source, m.RowMap, m.Count);
			auto lm = OaFnMatrix::Sum(om, -1); (void)ctx.Execute(); (void)ctx.Sync();
			vm = CopyToHost(lm)[0];
		}
		d[i] = orig; (void)ctx.Sync();

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(GradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}
