#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

#include <gtest/gtest.h>
#include "../../oaTest.h"

#include <cmath>

namespace {

class AttentionTest : public ::testing::Test {};

oa::Matrix fromF32(oa::Span<const oa::F32> inValues, const oa::MatrixShape& inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inValues.data()), inValues.sizeBytes()),
		inShape, oa::ScalarType::Float32);
}

void setIdentity(oa::Matrix& inMatrix) {
	oa::F32* values = inMatrix.dataAs<oa::F32>();
	for (oa::I64 row = 0; row < inMatrix.size(0); ++row) {
		for (oa::I64 col = 0; col < inMatrix.size(1); ++col) {
			values[row * inMatrix.size(1) + col] = row == col ? 1.0F : 0.0F;
		}
	}
}

} // namespace

TEST_VK(AttentionTest, SplitMergeHeadsRoundtrip) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 values[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	};
	auto input = fromF32(values, oa::MatrixShape{6, 4});
	auto split = oa::FnMatrix::splitHeads(input, 2, 3, 2);
	auto merged = oa::FnMatrix::mergeHeads(split, 2, 3, 2);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ASSERT_EQ(merged.getShape(), input.getShape());
	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		EXPECT_FLOAT_EQ(merged.dataAs<const oa::F32>()[i], values[i]);
	}
}

TEST_VK(AttentionTest, SingleHeadSplitMergeIsDifferentiableView) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	auto input = oa::FnMatrix::randXavier({6, 4});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto split = oa::FnMatrix::splitHeads(input, 2, 3, 1);
	auto merged = oa::FnMatrix::mergeHeads(split, 2, 3, 1);

	EXPECT_EQ(split.getShape(), (oa::MatrixShape{2, 3, 4}));
	EXPECT_EQ(merged.getShape(), input.getShape());
	tape.backward(oa::FnMatrix::sum(merged));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	for (oa::I64 i = 0; i < input.numElements(); ++i) {
		EXPECT_FLOAT_EQ(merged.dataAs<const oa::F32>()[i], input.dataAs<const oa::F32>()[i]);
	}
	auto grad = input.gradMatrix();
	ASSERT_FALSE(grad.isEmpty());
	for (oa::I64 i = 0; i < grad.numElements(); ++i) {
		EXPECT_NEAR(grad.dataAs<const oa::F32>()[i], 1.0F, 1e-6F);
	}
}

TEST_VK(AttentionTest, SplitMergeHeadsBackwardIsIdentity) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 values[] = {0, 1, 2, 3, 4, 5, 6, 7};
	auto input = fromF32(values, oa::MatrixShape{2, 4});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto split = oa::FnMatrix::splitHeads(input, 1, 2, 2);
	auto merged = oa::FnMatrix::mergeHeads(split, 1, 2, 2);
	tape.backward(oa::FnMatrix::sum(merged));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto grad = input.gradMatrix();
	ASSERT_FALSE(grad.isEmpty());
	for (oa::I64 i = 0; i < grad.numElements(); ++i) EXPECT_FLOAT_EQ(grad.dataAs<const oa::F32>()[i], 1.0F);
}

TEST_VK(AttentionTest, BmmBackwardCrossRow) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 weights[] = {1, 0, 0.5F, 0.5F};
	const oa::F32 values[] = {2, 4};
	auto a = fromF32(weights, oa::MatrixShape{1, 2, 2});
	auto v = fromF32(values, oa::MatrixShape{1, 2, 1});
	v.setRequiresGrad(true);
	oa::GradientTape tape;
	auto output = oa::FnMatrix::bmm(a, v).reshape(oa::MatrixShape{2, 1});
	tape.backward(oa::FnMatrix::sum(oa::FnMatrix::slice(output, 0, 1, 2)));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto grad = v.gradMatrix();
	ASSERT_FALSE(grad.isEmpty());
	EXPECT_NEAR(grad.dataAs<const oa::F32>()[0], 0.5F, 1e-6F);
	EXPECT_NEAR(grad.dataAs<const oa::F32>()[1], 0.5F, 1e-6F);
}

TEST_VK(AttentionTest, FlashCausalForwardMatchesStandard) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 batchHeads = 4, seqLen = 5, headDim = 8;
	auto q = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	auto k = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	auto v = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	const oa::F32 scale = 1.0F / std::sqrt(static_cast<oa::F32>(headDim));
	auto mask = oa::FnMatrix::causalMask(oa::FnMatrix::zeros(
		{batchHeads, seqLen, seqLen}, oa::ScalarType::Float32));
	auto score = oa::FnMatrix::bmm(q, oa::FnMatrix::transpose(k, 1, 2));
	auto probability = oa::FnMatrix::softmaxScaledMasked(
		score.reshape({batchHeads * seqLen, seqLen}),
		mask.reshape({batchHeads * seqLen, seqLen}), scale);
	auto standard = oa::FnMatrix::bmm(
		probability.reshape({batchHeads, seqLen, seqLen}), v);
	auto flash = oa::FnMatrix::flashAttentionCausal(q, k, v, scale);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ASSERT_EQ(flash.getShape(), standard.getShape());
	for (oa::I64 i = 0; i < flash.numElements(); ++i) {
		EXPECT_NEAR(flash.dataAs<const oa::F32>()[i], standard.dataAs<const oa::F32>()[i], 2e-5F)
			<< "index " << i;
	}
}

TEST_VK(AttentionTest, FlashCausalBackwardMatchesStandard) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 batchHeads = 2, seqLen = 4, headDim = 4;
	const oa::F32 scale = 1.0F / std::sqrt(static_cast<oa::F32>(headDim));
	auto q0 = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	auto k0 = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	auto v0 = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	auto gradOutput = oa::FnMatrix::randN({batchHeads, seqLen, headDim});
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();

	auto qStandard = q0.clone(); qStandard.setRequiresGrad(true);
	auto kStandard = k0.clone(); kStandard.setRequiresGrad(true);
	auto vStandard = v0.clone(); vStandard.setRequiresGrad(true);
	oa::GradientTape standardTape;
	auto mask = oa::FnMatrix::causalMask(oa::FnMatrix::zeros(
		{batchHeads, seqLen, seqLen}, oa::ScalarType::Float32));
	auto score = oa::FnMatrix::bmm(qStandard, oa::FnMatrix::transpose(kStandard, 1, 2));
	auto probability = oa::FnMatrix::softmaxScaledMasked(
		score.reshape({batchHeads * seqLen, seqLen}),
		mask.reshape({batchHeads * seqLen, seqLen}), scale);
	auto standard = oa::FnMatrix::bmm(
		probability.reshape({batchHeads, seqLen, seqLen}), vStandard);
	standardTape.backward(oa::FnMatrix::sum(oa::FnMatrix::mul(standard, gradOutput)));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto dqReference = qStandard.gradMatrix().clone();
	auto dkReference = kStandard.gradMatrix().clone();
	auto dvReference = vStandard.gradMatrix().clone();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();

	auto qFlash = q0.clone(); qFlash.setRequiresGrad(true);
	auto kFlash = k0.clone(); kFlash.setRequiresGrad(true);
	auto vFlash = v0.clone(); vFlash.setRequiresGrad(true);
	oa::GradientTape flashTape;
	auto flash = oa::FnMatrix::flashAttentionCausal(qFlash, kFlash, vFlash, scale);
	flashTape.backward(oa::FnMatrix::sum(oa::FnMatrix::mul(flash, gradOutput)));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const oa::Matrix references[] = {dqReference, dkReference, dvReference};
	const oa::Matrix actual[] = {qFlash.gradMatrix(), kFlash.gradMatrix(), vFlash.gradMatrix()};
	for (oa::I32 matrix = 0; matrix < 3; ++matrix) {
		ASSERT_FALSE(actual[matrix].isEmpty());
		for (oa::I64 i = 0; i < actual[matrix].numElements(); ++i) {
			EXPECT_NEAR(actual[matrix].dataAs<const oa::F32>()[i],
				references[matrix].dataAs<const oa::F32>()[i], 8e-5F)
				<< "gradient " << matrix << " index " << i;
		}
	}
}

TEST_VK(AttentionTest, FlashRejectsUnverifiedBfloat16Storage) {
	auto q = oa::FnMatrix::empty({1, 4, 4}, oa::ScalarType::BFloat16);
	EXPECT_THROW((void)oa::FnMatrix::flashAttentionCausal(q, q, q, 0.5F), std::invalid_argument);
}

TEST_VK(AttentionTest, MultiHeadBackendPolicyIsExplicit) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::MultiHeadAttention attention(8, 2, 0.0F, true, oa::AttentionBackend::Auto);
	attention.setSeqLen(4);
	auto output = attention.forward(oa::FnMatrix::randN({8, 8}));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{8, 8}));
	EXPECT_EQ(attention.lastBackend(), oa::AttentionBackend::Standard);
	attention.setBackend(oa::AttentionBackend::Flash);
	(void)attention.forward(oa::FnMatrix::randN({8, 8}));
	EXPECT_EQ(attention.lastBackend(), oa::AttentionBackend::Flash);
	attention.setBackend(oa::AttentionBackend::Standard);
	(void)attention.forward(oa::FnMatrix::randN({8, 8}));
	EXPECT_EQ(attention.lastBackend(), oa::AttentionBackend::Standard);
	attention.setBackend(oa::AttentionBackend::Auto);
	{
		oa::GradientTape trainingTape;
		(void)attention.forward(oa::FnMatrix::randN({8, 8}));
	}
	EXPECT_EQ(attention.lastBackend(), oa::AttentionBackend::Standard);
}

TEST_VK(AttentionTest, MultiHeadCausalForwardMatchesCpuReference) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::MultiHeadAttention attention(4, 2, 0.0F, false);
	attention.setSeqLen(2);
	// finish the module's deferred parameter initialization before installing the
	// exact identity projection used by this reference test.
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	auto parameters = attention.allNamedParameterPtrs();
	ASSERT_EQ(parameters.size(), 4u);
	for (auto& parameter : parameters) setIdentity(parameter.param->data);

	const oa::F32 values[] = {1, 0, 0, 0, 0, 1, 10, 0};
	auto input = fromF32(values, oa::MatrixShape{2, 4});
	auto output = attention.forward(input);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const oa::F32 selfWeight = std::exp(1.0F / std::sqrt(2.0F));
	const oa::F32 firstWeight = 1.0F / (1.0F + selfWeight);
	const oa::F32 secondWeight = selfWeight / (1.0F + selfWeight);
	const oa::F32 expected[] = {1, 0, 0, 0, firstWeight, secondWeight, 10, 0};
	for (oa::I64 i = 0; i < output.numElements(); ++i) {
		EXPECT_NEAR(output.dataAs<const oa::F32>()[i], expected[i], 2e-5F) << "index " << i;
	}

	const oa::F32 maskValues[] = {0, -1e4F, 0, -1e4F, 0, -1e4F, 0, -1e4F};
	auto mask = fromF32(maskValues, oa::MatrixShape{4, 2});
	auto masked = attention.forwardMasked(input, mask);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const oa::F32 maskedExpected[] = {1, 0, 0, 0, 1, 0, 0, 0};
	for (oa::I64 i = 0; i < masked.numElements(); ++i) {
		EXPECT_NEAR(masked.dataAs<const oa::F32>()[i], maskedExpected[i], 2e-5F) << "masked index " << i;
	}
}

TEST_VK(AttentionTest, MultiHeadBidirectionalVisibilityIsExplicit) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::MultiHeadAttention attention(4, 2, 0.0F, false);
	attention.setSeqLen(2);
	attention.setMode(oa::AttentionMode::Bidirectional);
	ASSERT_EQ(attention.mode(), oa::AttentionMode::Bidirectional);

	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	for (auto& parameter : attention.allNamedParameterPtrs()) setIdentity(parameter.param->data);

	const oa::F32 values[] = {1, 0, 0, 0, 0, 1, 10, 0};
	auto output = attention.forward(fromF32(values, oa::MatrixShape{2, 4}));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(attention.lastBackend(), oa::AttentionBackend::Standard);

	const oa::F32 expScore = std::exp(1.0F / std::sqrt(2.0F));
	const oa::F32 otherWeight = 1.0F / (1.0F + expScore);
	const oa::F32 selfWeight = expScore / (1.0F + expScore);
	const oa::F32 expectedFirst[] = {selfWeight, otherWeight, 5.0F, 0.0F};
	for (oa::I64 i = 0; i < 4; ++i) {
		EXPECT_NEAR(output.dataAs<const oa::F32>()[i], expectedFirst[i], 2e-5F)
			<< "first-token index " << i;
	}

	attention.setBackend(oa::AttentionBackend::Flash);
	EXPECT_THROW((void)attention.forward(fromF32(values, oa::MatrixShape{2, 4})),
		std::invalid_argument);
}

TEST_VK(AttentionTest, MultiHeadBackwardReachesInput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::MultiHeadAttention attention(4, 1, 0.0F, false);
	attention.setSeqLen(2);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	for (auto& parameter : attention.allNamedParameterPtrs()) setIdentity(parameter.param->data);
	const oa::F32 values[] = {1, 0, 0, 0, 0, 1, 0, 0};
	auto input = fromF32(values, oa::MatrixShape{2, 4});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto output = attention.forward(input);
	// Only the second query contributes to the loss. Its gradient must still
	// reach the first (causally visible) token through attention.
	const oa::F32 lossMaskValues[] = {0, 0, 0, 0, 1, 0, 0, 0};
	auto lossMask = fromF32(lossMaskValues, oa::MatrixShape{2, 4});
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(output, lossMask));
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto grad = input.gradMatrix();
	ASSERT_FALSE(grad.isEmpty());
	oa::F64 firstTokenL1 = 0.0;
	for (oa::I64 i = 0; i < 4; ++i) firstTokenL1 += std::abs(grad.dataAs<const oa::F32>()[i]);
	EXPECT_GT(firstTokenL1, 1e-8);
}

TEST_VK(AttentionTest, TransformerBlockUsesSharedMultiHeadAttention) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::TransformerBlock block(8, 16, 3, 2, 1e-5F);
	EXPECT_EQ(block.numHeads(), 2);
	EXPECT_EQ(block.seqLen(), 3);
	EXPECT_EQ(block.attentionMode(), oa::AttentionMode::Causal);
	block.setAttentionMode(oa::AttentionMode::Bidirectional);
	EXPECT_EQ(block.attentionMode(), oa::AttentionMode::Bidirectional);

	auto input = oa::FnMatrix::randN(oa::MatrixShape{6, 8});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto output = block.forward(input);
	tape.backward(oa::FnMatrix::mean(output));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(output.getShape(), input.getShape());
	ASSERT_FALSE(input.gradMatrix().isEmpty());

	bool sawAttentionWeight = false;
	for (const auto& named : block.allNamedParameterPtrs()) {
		if (named.path == "attention.q_proj.weight") {
			sawAttentionWeight = true;
			ASSERT_FALSE(named.param->grad().isEmpty());
			oa::F64 gradL1 = 0.0;
			const oa::F32* grad = named.param->grad().dataAs<const oa::F32>();
			for (oa::I64 i = 0; i < named.param->grad().numElements(); ++i) {
				gradL1 += std::abs(grad[i]);
			}
			EXPECT_GT(gradL1, 1e-8);
		}
	}
	EXPECT_TRUE(sawAttentionWeight);

	block.setSeqLen(2);
	EXPECT_EQ(block.seqLen(), 2);
	auto shorter = block.forward(oa::FnMatrix::randN(oa::MatrixShape{4, 8}));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const oa::MatrixShape expectedShape{4, 8};
	EXPECT_EQ(shorter.getShape(), expectedShape);
}
