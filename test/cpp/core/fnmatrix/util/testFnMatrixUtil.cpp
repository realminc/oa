// Tests for Core/FnMatrix utility helper operations
// RepeatInterleave, Equal, topK, CompactRows, ScatterRows

#include <gtest/gtest.h>
#include "../../../oaTest.h"
#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

static oa::Engine* GRt = nullptr;

class TestFnMatrixUtil : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixUtil";
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

// Helper to create matrix from host data
static oa::Matrix createFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

TEST_VK(TestFnMatrixUtil, SampleLogits_GreedyRows) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto logits = createFromHost({1.0f, 4.0f, 2.0f, -1.0f, 3.0f, 3.0f}, oa::MatrixShape{2, 3});
	auto ids = oa::FnMatrix::sampleLogits(logits, 0.0f);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	ASSERT_EQ(ids.getDtype(), oa::ScalarType::Int32);
	ASSERT_EQ(ids.numElements(), 2);
	EXPECT_EQ(ids.dataAs<const oa::I32>()[0], 1);
	EXPECT_EQ(ids.dataAs<const oa::I32>()[1], 1); // equal logits resolve to lower index
}

TEST_VK(TestFnMatrixUtil, SampleLogits_TopKOneAlwaysArgmax) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto logits = createFromHost({-2.0f, 0.5f, 7.0f, 1.0f}, oa::MatrixShape{1, 4});
	auto ids = oa::FnMatrix::sampleLogits(logits, 0.8f, 1, 0.9f, 123);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	EXPECT_EQ(ids.dataAs<const oa::I32>()[0], 2);
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	auto output = oa::FnMatrix::repeatInterleave(input, 2, 0);
	
	EXPECT_EQ(output.getShape().rank, 2);
	EXPECT_EQ(output.getShape()[0], 4);
	EXPECT_EQ(output.getShape()[1], 3);
	
	auto result = copyToHost(output);
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	auto output = oa::FnMatrix::repeatInterleave(input, 2, 1);
	
	EXPECT_EQ(output.getShape().rank, 2);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 6);
	
	auto result = copyToHost(output);
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

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{3});
	auto output = oa::FnMatrix::repeatInterleave(input, 3, 0);

	EXPECT_EQ(output.getShape()[0], 9);

	auto result = copyToHost(output);
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

static bool gradClose(oa::F32 a, oa::F32 n, oa::F32 atol = 2e-3F, oa::F32 rtol = 2e-2F) {
	return std::abs(a - n) <= (atol + rtol * std::abs(n));
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Gradcheck_Dim0) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	// input [2, 3] with requires_grad
	std::vector<float> input_data = {0.5f, -1.2f, 3.1f, 0.8f, -2.4f, 1.7f};
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	input.setRequiresGrad(true);

	// Tape must be active before forward so grad nodes are attached
	oa::GradientTape tape;

	// forward: repeatInterleave(x, 2, 0) -> [4, 3], then sum to scalar
	auto repeated = oa::FnMatrix::repeatInterleave(input, 2, 0);
	auto loss = oa::FnMatrix::sum(repeated, -1);  // sum all -> scalar
	(void)testSubmitAndWait(ctx);

	// Analytical backward
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto grad = copyToHost(input.gradMatrix());

	// Finite-difference: perturb each input element, recompute loss
	const float eps = 1e-2F;
	float* d = input.dataAs<float>();
	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			oa::GradNo noGrad;
			d[i] = orig + eps; (void)testSubmitAndWait(ctx);
			auto rp = oa::FnMatrix::repeatInterleave(input, 2, 0);
			auto lp = oa::FnMatrix::sum(rp, -1); (void)testSubmitAndWait(ctx);
			vp = copyToHost(lp)[0];

			d[i] = orig - eps; (void)testSubmitAndWait(ctx);
			auto rm = oa::FnMatrix::repeatInterleave(input, 2, 0);
			auto lm = oa::FnMatrix::sum(rm, -1); (void)testSubmitAndWait(ctx);
			vm = copyToHost(lm)[0];
		}
		d[i] = orig; (void)testSubmitAndWait(ctx);

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(gradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}

TEST_VK(TestFnMatrixUtil, RepeatInterleave_Gradcheck_Dim1) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	// input [2, 3] with requires_grad
	std::vector<float> input_data = {0.3f, -0.7f, 1.5f, 2.1f, -0.4f, 0.9f};
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	input.setRequiresGrad(true);

	// Tape must be active before forward so grad nodes are attached
	oa::GradientTape tape;

	// forward: repeatInterleave(x, 3, 1) -> [2, 9], then sum to scalar
	auto repeated = oa::FnMatrix::repeatInterleave(input, 3, 1);
	auto loss = oa::FnMatrix::sum(repeated, -1);
	(void)testSubmitAndWait(ctx);

	// Analytical backward
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto grad = copyToHost(input.gradMatrix());

	// Finite-difference
	const float eps = 1e-2F;
	float* d = input.dataAs<float>();
	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			oa::GradNo noGrad;
			d[i] = orig + eps; (void)testSubmitAndWait(ctx);
			auto rp = oa::FnMatrix::repeatInterleave(input, 3, 1);
			auto lp = oa::FnMatrix::sum(rp, -1); (void)testSubmitAndWait(ctx);
			vp = copyToHost(lp)[0];

			d[i] = orig - eps; (void)testSubmitAndWait(ctx);
			auto rm = oa::FnMatrix::repeatInterleave(input, 3, 1);
			auto lm = oa::FnMatrix::sum(rm, -1); (void)testSubmitAndWait(ctx);
			vm = copyToHost(lm)[0];
		}
		d[i] = orig; (void)testSubmitAndWait(ctx);

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(gradClose(grad[static_cast<size_t>(i)], numerical))
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{4});
	auto output = oa::FnMatrix::equal(input, 5.0f);
	
	auto result = copyToHost(output);
	
	// Equal returns 1.0f for true, 0.0f for false
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 1.0f) << "expected all 1.0f at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Equal_NoneMatch) {
	// Test where no elements equal the value
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{4});
	auto output = oa::FnMatrix::equal(input, 5.0f);
	
	auto result = copyToHost(output);
	
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "expected all 0.0f at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Equal_Mixed) {
	// Test with mixed matches
	std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 3.0f, 2.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	auto output = oa::FnMatrix::equal(input, 2.0f);
	
	auto result = copyToHost(output);
	std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
	
	ASSERT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, GreaterEqual_Mixed) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(std::vector<float>{-1.0f, 0.25f, 0.5f, 2.0f}, oa::MatrixShape{4});
	auto output = oa::FnMatrix::greaterEqual(input, 0.5f);
	auto result = copyToHost(output);
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	auto copied = oa::FnMatrix::copy(input);
	
	EXPECT_EQ(copied.getShape().rank, 1);
	EXPECT_EQ(copied.getShape()[0], 5);
	
	auto result = copyToHost(copied);
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{2, 3});
	auto copied = oa::FnMatrix::copy(input);
	
	EXPECT_EQ(copied.getShape().rank, 2);
	EXPECT_EQ(copied.getShape()[0], 2);
	EXPECT_EQ(copied.getShape()[1], 3);
	
	auto result = copyToHost(copied);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Copy_Independence) {
	// Test that copy creates independent tensor (modifying one doesn't affect the other)
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{3});
	auto copied = oa::FnMatrix::copy(input);
	
	// Modify the copy
	oa::FnMatrix::fillInPlace(copied, 99.0f);
	
	// Original should be unchanged
	auto original_result = copyToHost(input);
	for (size_t i = 0; i < original_result.size(); ++i) {
		EXPECT_FLOAT_EQ(original_result[i], input_data[i]) << "Original modified at index " << i;
	}
	
	// Copy should be modified
	auto copied_result = copyToHost(copied);
	for (size_t i = 0; i < copied_result.size(); ++i) {
		EXPECT_FLOAT_EQ(copied_result[i], 99.0f) << "Copy not modified at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Copy_BackwardIsIdentity) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();
	auto input = createFromHost(
		std::vector<float>{1.0F, -2.0F, 3.0F, -4.0F},
		oa::MatrixShape{2, 2});
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	auto copied = oa::FnMatrix::copy(input);
	auto loss = oa::FnMatrix::sum(copied);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	EXPECT_TRUE(input.gradMatrix().hasStorage());
	const auto gradient = copyToHost(input.gradMatrix());
	ASSERT_EQ(gradient.size(), 4U);
	for (const auto value : gradient) EXPECT_FLOAT_EQ(value, 1.0F);
}

// ============================================================================
// detach Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, Detach_1D) {
	// Test detaching a 1D tensor (breaks autograd connection)
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{4});
	auto detached = oa::FnMatrix::detach(input);
	
	EXPECT_EQ(detached.getShape().rank, 1);
	EXPECT_EQ(detached.getShape()[0], 4);
	
	auto result = copyToHost(detached);
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
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{3, 2});
	auto detached = oa::FnMatrix::detach(input);
	
	EXPECT_EQ(detached.getShape().rank, 2);
	EXPECT_EQ(detached.getShape()[0], 3);
	EXPECT_EQ(detached.getShape()[1], 2);
	
	auto result = copyToHost(detached);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Mismatch at index " << i;
	}
}

TEST_VK(TestFnMatrixUtil, Detach_PreservesData) {
	// Test that detach preserves data exactly
	std::vector<float> input_data = {-5.5f, 0.0f, 3.14f, 100.0f, -0.001f};
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto input = createFromHost(input_data, oa::MatrixShape{5});
	auto detached = oa::FnMatrix::detach(input);
	
	auto result = copyToHost(detached);
	ASSERT_EQ(result.size(), input_data.size());
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], input_data[i]) << "Data not preserved at index " << i;
	}
}

// ============================================================================
// CompactRows Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, CompactRows_Forward) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	auto input = createFromHost(data, oa::MatrixShape{4, 2});
	std::vector<float> mask = {0.0f, 1.0f, 0.0f, 1.0f};
	auto maskM = createFromHost(mask, oa::MatrixShape{4});

	auto compact = oa::FnMatrix::compactRows(input, maskM);
	(void)testSubmitAndWait(ctx);

	auto result = copyToHost(compact.values);
	EXPECT_EQ(compact.count.dataAs<const oa::U32>()[0], 2u);
	const oa::U32* dispatch = compact.dispatchArgs.dataAs<const oa::U32>();
	EXPECT_EQ(dispatch[0], 1u);
	EXPECT_EQ(dispatch[1], 1u);
	EXPECT_EQ(dispatch[2], 1u);
	ASSERT_EQ(result.size(), 8u);
	EXPECT_FLOAT_EQ(result[0], 3.0f);
	EXPECT_FLOAT_EQ(result[1], 4.0f);
	EXPECT_FLOAT_EQ(result[2], 7.0f);
	EXPECT_FLOAT_EQ(result[3], 8.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_DeferredMaskAndMultipleScanChunks) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::I64 T = 600;
	constexpr oa::I64 D = 2;

	std::vector<float> data(static_cast<size_t>(T * D));
	std::vector<float> selector(static_cast<size_t>(T));
	oa::U32 expectedCount = 0;
	for (oa::I64 row = 0; row < T; ++row) {
		data[static_cast<size_t>(row * D)] = static_cast<float>(row);
		data[static_cast<size_t>(row * D + 1)] = static_cast<float>(-row);
		selector[static_cast<size_t>(row)] = row % 3 == 1 ? 1.0f : 0.0f;
		if (row % 3 == 1) ++expectedCount;
	}

	auto input = createFromHost(data, oa::MatrixShape{T, D});
	auto selectorM = createFromHost(selector, oa::MatrixShape{T});
	// Equal and CompactRows remain in one deferred GPU graph. This specifically
	// guards against reintroducing a hidden host read of a pending mask.
	auto mask = oa::FnMatrix::equal(selectorM, 1.0f);
	auto compact = oa::FnMatrix::compactRows(input, mask);
	(void)testSubmitAndWait(ctx);

	EXPECT_EQ(compact.count.dataAs<const oa::U32>()[0], expectedCount);
	const auto values = copyToHost(compact.values);
	const auto* rowMap = compact.rowMap.dataAs<const oa::U32>();
	for (oa::U32 slot = 0; slot < expectedCount; ++slot) {
		const oa::U32 expectedRow = slot * 3 + 1;
		EXPECT_EQ(rowMap[slot], expectedRow);
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(slot * D)], static_cast<float>(expectedRow));
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(slot * D + 1)], -static_cast<float>(expectedRow));
	}
	for (oa::I64 i = static_cast<oa::I64>(expectedCount) * D; i < T * D; ++i)
		EXPECT_FLOAT_EQ(values[static_cast<size_t>(i)], 0.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_AllAndNone) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = createFromHost(data, oa::MatrixShape{4, 1});
	auto all = createFromHost(std::vector<float>(4, 1.0f), oa::MatrixShape{4});
	auto none = createFromHost(std::vector<float>(4, 0.0f), oa::MatrixShape{4});
	auto compactAll = oa::FnMatrix::compactRows(input, all);
	auto compactNone = oa::FnMatrix::compactRows(input, none);
	(void)testSubmitAndWait(ctx);

	EXPECT_EQ(compactAll.count.dataAs<const oa::U32>()[0], 4u);
	EXPECT_EQ(compactNone.count.dataAs<const oa::U32>()[0], 0u);
	EXPECT_EQ(compactAll.dispatchArgs.dataAs<const oa::U32>()[0], 1u);
	EXPECT_EQ(compactNone.dispatchArgs.dataAs<const oa::U32>()[0], 0u);
	EXPECT_EQ(copyToHost(compactAll.values), data);
	for (float value : copyToHost(compactNone.values)) EXPECT_FLOAT_EQ(value, 0.0f);
}

TEST_VK(TestFnMatrixUtil, CompactRows_Gradcheck) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> data = {0.5f, -1.2f, 3.1f, 0.8f, -2.4f, 1.7f, 0.3f, -0.9f};
	auto input = createFromHost(data, oa::MatrixShape{4, 2});
	input.setRequiresGrad(true);
	std::vector<float> mask = {0.0f, 1.0f, 0.0f, 1.0f};
	auto maskM = createFromHost(mask, oa::MatrixShape{4});

	oa::GradientTape tape;
	auto selected = oa::FnMatrix::compactRows(input, maskM).values;
	auto loss = oa::FnMatrix::sum(selected, -1);
	(void)testSubmitAndWait(ctx);

	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto grad = copyToHost(input.gradMatrix());

	const float eps = 1e-2F;
	float* d = input.dataAs<float>();
	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			oa::GradNo noGrad;
			d[i] = orig + eps; (void)testSubmitAndWait(ctx);
			auto sp = oa::FnMatrix::compactRows(input, maskM).values;
			auto lp = oa::FnMatrix::sum(sp, -1); (void)testSubmitAndWait(ctx);
			vp = copyToHost(lp)[0];

			d[i] = orig - eps; (void)testSubmitAndWait(ctx);
			auto sm = oa::FnMatrix::compactRows(input, maskM).values;
			auto lm = oa::FnMatrix::sum(sm, -1); (void)testSubmitAndWait(ctx);
			vm = copyToHost(lm)[0];
		}
		d[i] = orig; (void)testSubmitAndWait(ctx);

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(gradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}

// ============================================================================
// ScatterRows Tests
// ============================================================================

TEST_VK(TestFnMatrixUtil, ScatterRows_Forward) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> self_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
	auto self = createFromHost(self_data, oa::MatrixShape{3, 2});
	std::vector<float> mask = {1.0f, 0.0f, 1.0f};
	auto maskM = createFromHost(mask, oa::MatrixShape{3});
	std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
	auto source = createFromHost(src_data, oa::MatrixShape{3, 2});

	auto plan = oa::FnMatrix::compactRows(self, maskM);
	auto out = oa::FnMatrix::scatterRows(self, source, plan);
	(void)testSubmitAndWait(ctx);

	auto result = copyToHost(out);
	ASSERT_EQ(result.size(), 6u);
	EXPECT_FLOAT_EQ(result[0], 11.0f);
	EXPECT_FLOAT_EQ(result[1], 22.0f);
	EXPECT_FLOAT_EQ(result[2], 30.0f);
	EXPECT_FLOAT_EQ(result[3], 40.0f);
	EXPECT_FLOAT_EQ(result[4], 53.0f);
	EXPECT_FLOAT_EQ(result[5], 64.0f);
}

TEST_VK(TestFnMatrixUtil, ScatterRows_Gradcheck_Source) {
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> self_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
	auto self = createFromHost(self_data, oa::MatrixShape{3, 2});
	std::vector<float> mask = {1.0f, 0.0f, 1.0f};
	auto maskM = createFromHost(mask, oa::MatrixShape{3});
	std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
	auto source = createFromHost(src_data, oa::MatrixShape{3, 2});
	source.setRequiresGrad(true);

	oa::GradientTape tape;
	auto plan = oa::FnMatrix::compactRows(self, maskM);
	auto out = oa::FnMatrix::scatterRows(self, source, plan);
	auto loss = oa::FnMatrix::sum(out, -1);
	(void)testSubmitAndWait(ctx);

	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto grad = copyToHost(source.gradMatrix());

	const float eps = 1e-2F;
	float* d = source.dataAs<float>();
	for (oa::I64 i = 0; i < source.numElements(); ++i) {
		float orig = d[i];
		float vp, vm;
		{
			oa::GradNo noGrad;
			d[i] = orig + eps; (void)testSubmitAndWait(ctx);
			auto p = oa::FnMatrix::compactRows(self, maskM);
			auto op = oa::FnMatrix::scatterRows(self, source, p);
			auto lp = oa::FnMatrix::sum(op, -1); (void)testSubmitAndWait(ctx);
			vp = copyToHost(lp)[0];

			d[i] = orig - eps; (void)testSubmitAndWait(ctx);
			auto m = oa::FnMatrix::compactRows(self, maskM);
			auto om = oa::FnMatrix::scatterRows(self, source, m);
			auto lm = oa::FnMatrix::sum(om, -1); (void)testSubmitAndWait(ctx);
			vm = copyToHost(lm)[0];
		}
		d[i] = orig; (void)testSubmitAndWait(ctx);

		float numerical = (vp - vm) / (2.0F * eps);
		EXPECT_TRUE(gradClose(grad[static_cast<size_t>(i)], numerical))
			<< "idx " << i << ": analytical=" << grad[static_cast<size_t>(i)]
			<< " numerical=" << numerical;
	}
}
