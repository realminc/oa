// oa::FnMatrix smoke tests — reshape, RepeatInterleave, CausalMask, topK, Equal, CompactRows, ScatterRows

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>

static oa::Engine* GRt = nullptr;

class TestFnMatrix : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrix";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

TEST_VK(TestFnMatrix, Reshape_InferDim) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{4, 6});
	auto b = oa::FnMatrix::reshape(a, {-1, 3});
	EXPECT_EQ(b.rank(), 2);
	EXPECT_EQ(b.size(0), 8);
	EXPECT_EQ(b.size(1), 3);
}

TEST_VK(TestFnMatrix, RepeatInterleave_Dim1) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{2, 3, 4});
	auto b = oa::FnMatrix::repeatInterleave(a, 2, 1);
	EXPECT_EQ(b.size(0), 2);
	EXPECT_EQ(b.size(1), 6);
	EXPECT_EQ(b.size(2), 4);
}

TEST_VK(TestFnMatrix, CausalMask_Shape) {
	auto scores = oa::FnMatrix::zeros(oa::MatrixShape{4, 4});
	auto masked = oa::FnMatrix::causalMask(scores);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	EXPECT_EQ(masked.getShape(), scores.getShape());
	EXPECT_GT(masked.at(1 * 4 + 0), -1e8f);  // below diagonal: not masked
	EXPECT_LT(masked.at(0 * 4 + 1), -1e8f);  // above diagonal: masked
}

TEST_VK(TestFnMatrix, CausalMask_ConstructedOnGpu) {
	auto mask = oa::FnMatrix::causalMask(4);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	EXPECT_FLOAT_EQ(mask.at(3 * 4 + 0), 0.0f);
	EXPECT_LT(mask.at(0 * 4 + 3), -1e8f);
}

TEST_VK(TestFnMatrix, CausalMask_BackwardZerosFuturePositions) {
	auto scores = oa::FnMatrix::zeros(oa::MatrixShape{3, 3});
	scores.setRequiresGrad(true);
	oa::GradientTape tape;
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::causalMask(scores), -1);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	const auto& grad = scores.gradMatrix();
	for (oa::I32 q = 0; q < 3; ++q)
		for (oa::I32 k = 0; k < 3; ++k)
			EXPECT_FLOAT_EQ(grad.at(q * 3 + k), k <= q ? 1.0f : 0.0f);
}

TEST_VK(TestFnMatrix, AutogradRejectsSavedValueMutationBeforeBackward) {
	auto input = oa::FnMatrix::full(oa::MatrixShape{2, 2}, 2.0);
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(input, input));
	oa::FnMatrix::addScalarInPlace(input, 1.0F);

	const auto status = tape.tryBackward(loss);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_NE(status.getMessage().find("modified in place"), oa::String::Npos);
	EXPECT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	for (oa::I64 index = 0; index < input.numElements(); ++index) {
		EXPECT_FLOAT_EQ(input.gradMatrix().at(index), 0.0F);
	}
}

TEST_VK(TestFnMatrix, AutogradRejectsMutationThroughStorageAlias) {
	auto input = oa::FnMatrix::full(oa::MatrixShape{2, 2}, 2.0);
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(input, input));
	auto alias = input.view(oa::MatrixShape{4});
	oa::FnMatrix::scaleInPlace(alias, 3.0F);

	const auto status = tape.tryBackward(loss);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
}

TEST_VK(TestFnMatrix, AutogradRejectsMatrixCompoundAssignmentMutation) {
	auto input = oa::FnMatrix::full(oa::MatrixShape{2, 2}, 2.0);
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(input, input));
	input *= oa::FnMatrix::full(oa::MatrixShape{2, 2}, 3.0);

	const auto status = tape.tryBackward(loss);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_NE(status.getMessage().find("modified in place"), oa::String::Npos);
	EXPECT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
}

TEST_VK(TestFnMatrix, AutogradRejectsMutationThroughEarlierBufferDescriptor) {
	auto input = oa::FnMatrix::full(oa::MatrixShape{2, 2}, 2.0);
	input.setRequiresGrad(true);
	const oavk::Buffer earlierDescriptor = oa::MatrixAccess::descriptor(input);

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(input, input));
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());

	const oa::F32 replacement = 7.0F;
	ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(
		context.engine(), earlierDescriptor, 0U,
		&replacement, sizeof(replacement)).isOk());
	const auto status = tape.tryBackward(loss);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
}

TEST_VK(TestFnMatrix, TopK_Basic) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{2, 4});
	a.set(0 * 4 + 3, 3.0f);
	a.set(0 * 4 + 1, 1.0f);
	a.set(1 * 4 + 0, 5.0f);
	a.set(1 * 4 + 2, 2.0f);
	auto result = oa::FnMatrix::topK(a, 2);
	auto& context = oa::ExecutionSession::getActive();
	auto submitted = context.submit();
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();
	ASSERT_TRUE(context.wait(submitted.getValue()).isOk());
	EXPECT_EQ(result.values.size(0), 2);
	EXPECT_EQ(result.values.size(1), 2);
	EXPECT_NEAR(result.values.at(0 * 2 + 0), 3.0f, 1e-5f);
	EXPECT_NEAR(result.values.at(1 * 2 + 0), 5.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, Equal_Float) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{2, 3});
	a.set(0, 1.0f);
	a.set(4, 1.0f);
	auto mask = oa::FnMatrix::equal(a, 1.0f);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	EXPECT_NEAR(mask.at(0), 1.0f, 1e-5f);
	EXPECT_NEAR(mask.at(1), 0.0f, 1e-5f);
	EXPECT_NEAR(mask.at(4), 1.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, Slice_Dim1) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{3, 4});
	a.set(0 * 4 + 2, 7.0f);
	a.set(1 * 4 + 2, 8.0f);
	auto s = oa::FnMatrix::slice(a, 1, 2, 3);  // [:, 2:3]
	EXPECT_EQ(s.size(0), 3);
	EXPECT_EQ(s.size(1), 1);
	// Slice records a deferred MatrixCopyRegion kernel; flush before host readback
	// (at() reads mapped memory directly and does NOT execute the context).
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	EXPECT_NEAR(s.at(0), 7.0f, 1e-5f);
	EXPECT_NEAR(s.at(1), 8.0f, 1e-5f);
}

TEST_VK(TestFnMatrix, Concat_BackwardReachesEveryInput) {
	auto a = oa::FnMatrix::zeros(oa::MatrixShape{1, 1});
	auto b = oa::FnMatrix::zeros(oa::MatrixShape{1, 2});
	auto c = oa::FnMatrix::zeros(oa::MatrixShape{1, 3});
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);
	c.setRequiresGrad(true);

	oa::GradientTape tape;
	oa::Matrix inputs[] = {a, b, c};
	const auto concatenated = oa::FnMatrix::concat(
		oa::Span<oa::Matrix>(inputs, 3U), 1);
	const auto loss = oa::FnMatrix::sum(concatenated, -1);
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(context).isOk());

	for (oa::I64 index = 0; index < a.numElements(); ++index) {
		EXPECT_FLOAT_EQ(a.gradMatrix().at(index), 1.0F);
	}
	for (oa::I64 index = 0; index < b.numElements(); ++index) {
		EXPECT_FLOAT_EQ(b.gradMatrix().at(index), 1.0F);
	}
	for (oa::I64 index = 0; index < c.numElements(); ++index) {
		EXPECT_FLOAT_EQ(c.gradMatrix().at(index), 1.0F);
	}
}

TEST_VK(TestFnMatrix, Split_BackwardAccumulatesEveryOutput) {
	auto source = oa::FnMatrix::zeros(oa::MatrixShape{1, 6});
	source.setRequiresGrad(true);

	oa::GradientTape tape;
	oa::I64 sizes[] = {1, 2, 3};
	const auto outputs = oa::FnMatrix::split(
		source, oa::Span<oa::I64>(sizes, 3U), 1);
	ASSERT_EQ(outputs.size(), 3U);
	const auto loss0 = oa::FnMatrix::sum(
		oa::FnMatrix::scale(outputs[0], 2.0F), -1);
	const auto loss1 = oa::FnMatrix::sum(
		oa::FnMatrix::scale(outputs[1], 3.0F), -1);
	const auto loss2 = oa::FnMatrix::sum(
		oa::FnMatrix::scale(outputs[2], 4.0F), -1);
	const auto loss = oa::FnMatrix::add(
		oa::FnMatrix::add(loss0, loss1), loss2);
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(context).isOk());

	const oa::F32 expected[] = {2.0F, 3.0F, 3.0F, 4.0F, 4.0F, 4.0F};
	for (oa::I64 index = 0; index < source.numElements(); ++index) {
		EXPECT_FLOAT_EQ(
			source.gradMatrix().at(index),
			expected[static_cast<oa::Usize>(index)]);
	}
}

TEST_VK(TestFnMatrix, CompactScatterRows) {
	auto x = oa::FnMatrix::zeros(oa::MatrixShape{4, 2});
	x.set(0 * 2 + 0, 1.0f); x.set(0 * 2 + 1, 2.0f);
	x.set(2 * 2 + 0, 3.0f); x.set(2 * 2 + 1, 4.0f);

	auto mask = oa::FnMatrix::zeros(oa::MatrixShape{4, 1});
	mask.set(0, 1.0f);
	mask.set(2, 1.0f);

	auto compact = oa::FnMatrix::compactRows(x, mask);
	auto selected = compact.values;
	EXPECT_EQ(selected.size(0), 4);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	EXPECT_NEAR(selected.at(0 * 2 + 0), 1.0f, 1e-5f);
	EXPECT_NEAR(selected.at(1 * 2 + 0), 3.0f, 1e-5f);

	auto base = oa::FnMatrix::zeros(oa::MatrixShape{4, 2});
	auto scattered = oa::FnMatrix::scatterRows(base, selected, compact.rowMap, compact.count);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	EXPECT_NEAR(scattered.at(0 * 2 + 0), 1.0f, 1e-5f);
	EXPECT_NEAR(scattered.at(2 * 2 + 0), 3.0f, 1e-5f);
	EXPECT_NEAR(scattered.at(1 * 2 + 0), 0.0f, 1e-5f);
}
