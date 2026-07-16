#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

class AttentionTest : public ::testing::Test {};

OaMatrix FromF32(OaSpan<const OaF32> InValues, const OaMatrixShape& InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InValues.Data()), InValues.SizeBytes()),
		InShape, OaScalarType::Float32);
}

void SetIdentity(OaMatrix& InMatrix) {
	OaF32* values = InMatrix.DataAs<OaF32>();
	for (OaI64 row = 0; row < InMatrix.Size(0); ++row) {
		for (OaI64 col = 0; col < InMatrix.Size(1); ++col) {
			values[row * InMatrix.Size(1) + col] = row == col ? 1.0F : 0.0F;
		}
	}
}

} // namespace

TEST_VK(AttentionTest, SplitMergeHeadsRoundtrip) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const OaF32 values[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	};
	auto input = FromF32(values, OaMatrixShape{6, 4});
	auto split = OaFnMatrix::SplitHeads(input, 2, 3, 2);
	auto merged = OaFnMatrix::MergeHeads(split, 2, 3, 2);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ASSERT_EQ(merged.GetShape(), input.GetShape());
	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(merged.DataAs<const OaF32>()[i], values[i]);
	}
}

TEST_VK(AttentionTest, SingleHeadSplitMergeIsDifferentiableView) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	auto input = OaFnMatrix::RandXavier({6, 4});
	input.SetRequiresGrad(true);
	OaGradientTape tape;
	auto split = OaFnMatrix::SplitHeads(input, 2, 3, 1);
	auto merged = OaFnMatrix::MergeHeads(split, 2, 3, 1);

	EXPECT_EQ(split.GetShape(), (OaMatrixShape{2, 3, 4}));
	EXPECT_EQ(merged.GetShape(), input.GetShape());
	tape.Backward(OaFnMatrix::Sum(merged));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	for (OaI64 i = 0; i < input.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(merged.DataAs<const OaF32>()[i], input.DataAs<const OaF32>()[i]);
	}
	auto grad = input.GradMatrix();
	ASSERT_FALSE(grad.IsEmpty());
	for (OaI64 i = 0; i < grad.NumElements(); ++i) {
		EXPECT_NEAR(grad.DataAs<const OaF32>()[i], 1.0F, 1e-6F);
	}
}

TEST_VK(AttentionTest, SplitMergeHeadsBackwardIsIdentity) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const OaF32 values[] = {0, 1, 2, 3, 4, 5, 6, 7};
	auto input = FromF32(values, OaMatrixShape{2, 4});
	input.SetRequiresGrad(true);
	OaGradientTape tape;
	auto split = OaFnMatrix::SplitHeads(input, 1, 2, 2);
	auto merged = OaFnMatrix::MergeHeads(split, 1, 2, 2);
	tape.Backward(OaFnMatrix::Sum(merged));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto grad = input.GradMatrix();
	ASSERT_FALSE(grad.IsEmpty());
	for (OaI64 i = 0; i < grad.NumElements(); ++i) EXPECT_FLOAT_EQ(grad.DataAs<const OaF32>()[i], 1.0F);
}

TEST_VK(AttentionTest, BmmBackwardCrossRow) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const OaF32 weights[] = {1, 0, 0.5F, 0.5F};
	const OaF32 values[] = {2, 4};
	auto a = FromF32(weights, OaMatrixShape{1, 2, 2});
	auto v = FromF32(values, OaMatrixShape{1, 2, 1});
	v.SetRequiresGrad(true);
	OaGradientTape tape;
	auto output = OaFnMatrix::Bmm(a, v).Reshape(OaMatrixShape{2, 1});
	tape.Backward(OaFnMatrix::Sum(OaFnMatrix::Slice(output, 0, 1, 2)));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto grad = v.GradMatrix();
	ASSERT_FALSE(grad.IsEmpty());
	EXPECT_NEAR(grad.DataAs<const OaF32>()[0], 0.5F, 1e-6F);
	EXPECT_NEAR(grad.DataAs<const OaF32>()[1], 0.5F, 1e-6F);
}

TEST_VK(AttentionTest, FlashCausalForwardMatchesStandard) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 batchHeads = 4, seqLen = 5, headDim = 8;
	auto q = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	auto k = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	auto v = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	const OaF32 scale = 1.0F / std::sqrt(static_cast<OaF32>(headDim));
	auto mask = OaFnMatrix::CausalMask(OaFnMatrix::Zeros(
		{batchHeads, seqLen, seqLen}, OaScalarType::Float32));
	auto score = OaFnMatrix::Bmm(q, OaFnMatrix::Transpose(k, 1, 2));
	auto probability = OaFnMatrix::SoftmaxScaledMasked(
		score.Reshape({batchHeads * seqLen, seqLen}),
		mask.Reshape({batchHeads * seqLen, seqLen}), scale);
	auto standard = OaFnMatrix::Bmm(
		probability.Reshape({batchHeads, seqLen, seqLen}), v);
	auto flash = OaFnMatrix::FlashAttentionCausal(q, k, v, scale);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ASSERT_EQ(flash.GetShape(), standard.GetShape());
	for (OaI64 i = 0; i < flash.NumElements(); ++i) {
		EXPECT_NEAR(flash.DataAs<const OaF32>()[i], standard.DataAs<const OaF32>()[i], 2e-5F)
			<< "index " << i;
	}
}

TEST_VK(AttentionTest, FlashCausalBackwardMatchesStandard) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 batchHeads = 2, seqLen = 4, headDim = 4;
	const OaF32 scale = 1.0F / std::sqrt(static_cast<OaF32>(headDim));
	auto q0 = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	auto k0 = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	auto v0 = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	auto gradOutput = OaFnMatrix::RandN({batchHeads, seqLen, headDim});
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();

	auto qStandard = q0.Clone(); qStandard.SetRequiresGrad(true);
	auto kStandard = k0.Clone(); kStandard.SetRequiresGrad(true);
	auto vStandard = v0.Clone(); vStandard.SetRequiresGrad(true);
	OaGradientTape standardTape;
	auto mask = OaFnMatrix::CausalMask(OaFnMatrix::Zeros(
		{batchHeads, seqLen, seqLen}, OaScalarType::Float32));
	auto score = OaFnMatrix::Bmm(qStandard, OaFnMatrix::Transpose(kStandard, 1, 2));
	auto probability = OaFnMatrix::SoftmaxScaledMasked(
		score.Reshape({batchHeads * seqLen, seqLen}),
		mask.Reshape({batchHeads * seqLen, seqLen}), scale);
	auto standard = OaFnMatrix::Bmm(
		probability.Reshape({batchHeads, seqLen, seqLen}), vStandard);
	standardTape.Backward(OaFnMatrix::Sum(OaFnMatrix::Mul(standard, gradOutput)));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto dqReference = qStandard.GradMatrix().Clone();
	auto dkReference = kStandard.GradMatrix().Clone();
	auto dvReference = vStandard.GradMatrix().Clone();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();

	auto qFlash = q0.Clone(); qFlash.SetRequiresGrad(true);
	auto kFlash = k0.Clone(); kFlash.SetRequiresGrad(true);
	auto vFlash = v0.Clone(); vFlash.SetRequiresGrad(true);
	OaGradientTape flashTape;
	auto flash = OaFnMatrix::FlashAttentionCausal(qFlash, kFlash, vFlash, scale);
	flashTape.Backward(OaFnMatrix::Sum(OaFnMatrix::Mul(flash, gradOutput)));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const OaMatrix references[] = {dqReference, dkReference, dvReference};
	const OaMatrix actual[] = {qFlash.GradMatrix(), kFlash.GradMatrix(), vFlash.GradMatrix()};
	for (OaI32 matrix = 0; matrix < 3; ++matrix) {
		ASSERT_FALSE(actual[matrix].IsEmpty());
		for (OaI64 i = 0; i < actual[matrix].NumElements(); ++i) {
			EXPECT_NEAR(actual[matrix].DataAs<const OaF32>()[i],
				references[matrix].DataAs<const OaF32>()[i], 8e-5F)
				<< "gradient " << matrix << " index " << i;
		}
	}
}

TEST_VK(AttentionTest, FlashRejectsUnverifiedBfloat16Storage) {
	auto q = OaFnMatrix::Empty({1, 4, 4}, OaScalarType::BFloat16);
	EXPECT_THROW((void)OaFnMatrix::FlashAttentionCausal(q, q, q, 0.5F), std::invalid_argument);
}

TEST_VK(AttentionTest, MultiHeadBackendPolicyIsExplicit) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaMultiHeadAttention attention(8, 2, 0.0F, true, OaAttentionBackend::Auto);
	attention.SetSeqLen(4);
	auto output = attention.Forward(OaFnMatrix::RandN({8, 8}));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{8, 8}));
	EXPECT_EQ(attention.LastBackend(), OaAttentionBackend::Flash);
	attention.SetBackend(OaAttentionBackend::Standard);
	(void)attention.Forward(OaFnMatrix::RandN({8, 8}));
	EXPECT_EQ(attention.LastBackend(), OaAttentionBackend::Standard);
	attention.SetBackend(OaAttentionBackend::Auto);
	{
		OaGradientTape trainingTape;
		(void)attention.Forward(OaFnMatrix::RandN({8, 8}));
	}
	EXPECT_EQ(attention.LastBackend(), OaAttentionBackend::Standard);
}

TEST_VK(AttentionTest, MultiHeadCausalForwardMatchesCpuReference) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaMultiHeadAttention attention(4, 2, 0.0F, false);
	attention.SetSeqLen(2);
	// Finish the module's deferred parameter initialization before installing the
	// exact identity projection used by this reference test.
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	auto parameters = attention.AllNamedParameterPtrs();
	ASSERT_EQ(parameters.Size(), 4u);
	for (auto& parameter : parameters) SetIdentity(parameter.Param->Data);

	const OaF32 values[] = {1, 0, 0, 0, 0, 1, 10, 0};
	auto input = FromF32(values, OaMatrixShape{2, 4});
	auto output = attention.Forward(input);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const OaF32 selfWeight = std::exp(1.0F / std::sqrt(2.0F));
	const OaF32 firstWeight = 1.0F / (1.0F + selfWeight);
	const OaF32 secondWeight = selfWeight / (1.0F + selfWeight);
	const OaF32 expected[] = {1, 0, 0, 0, firstWeight, secondWeight, 10, 0};
	for (OaI64 i = 0; i < output.NumElements(); ++i) {
		EXPECT_NEAR(output.DataAs<const OaF32>()[i], expected[i], 2e-5F) << "index " << i;
	}

	const OaF32 maskValues[] = {0, -1e4F, 0, -1e4F, 0, -1e4F, 0, -1e4F};
	auto mask = FromF32(maskValues, OaMatrixShape{4, 2});
	auto masked = attention.ForwardMasked(input, mask);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const OaF32 maskedExpected[] = {1, 0, 0, 0, 1, 0, 0, 0};
	for (OaI64 i = 0; i < masked.NumElements(); ++i) {
		EXPECT_NEAR(masked.DataAs<const OaF32>()[i], maskedExpected[i], 2e-5F) << "masked index " << i;
	}
}

TEST_VK(AttentionTest, MultiHeadBackwardReachesInput) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaMultiHeadAttention attention(4, 1, 0.0F, false);
	attention.SetSeqLen(2);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	for (auto& parameter : attention.AllNamedParameterPtrs()) SetIdentity(parameter.Param->Data);
	const OaF32 values[] = {1, 0, 0, 0, 0, 1, 0, 0};
	auto input = FromF32(values, OaMatrixShape{2, 4});
	input.SetRequiresGrad(true);
	OaGradientTape tape;
	auto output = attention.Forward(input);
	// Only the second query contributes to the loss. Its gradient must still
	// reach the first (causally visible) token through attention.
	const OaF32 lossMaskValues[] = {0, 0, 0, 0, 1, 0, 0, 0};
	auto lossMask = FromF32(lossMaskValues, OaMatrixShape{2, 4});
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(output, lossMask));
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto grad = input.GradMatrix();
	ASSERT_FALSE(grad.IsEmpty());
	OaF64 firstTokenL1 = 0.0;
	for (OaI64 i = 0; i < 4; ++i) firstTokenL1 += std::abs(grad.DataAs<const OaF32>()[i]);
	EXPECT_GT(firstTokenL1, 1e-8);
}

TEST_VK(AttentionTest, TransformerBlockUsesSharedMultiHeadAttention) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaTransformerBlock block(8, 16, 3, 2, 1e-5F);
	EXPECT_EQ(block.NumHeads(), 2);
	EXPECT_EQ(block.SeqLen(), 3);

	auto input = OaFnMatrix::RandN(OaMatrixShape{6, 8});
	input.SetRequiresGrad(true);
	OaGradientTape tape;
	auto output = block.Forward(input);
	tape.Backward(OaFnMatrix::Mean(output));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_EQ(output.GetShape(), input.GetShape());
	ASSERT_FALSE(input.GradMatrix().IsEmpty());

	bool sawAttentionWeight = false;
	for (const auto& named : block.AllNamedParameterPtrs()) {
		if (named.Path == "attention.q_proj.weight") {
			sawAttentionWeight = true;
			ASSERT_FALSE(named.Param->Grad().IsEmpty());
			OaF64 gradL1 = 0.0;
			const OaF32* grad = named.Param->Grad().DataAs<const OaF32>();
			for (OaI64 i = 0; i < named.Param->Grad().NumElements(); ++i) {
				gradL1 += std::abs(grad[i]);
			}
			EXPECT_GT(gradL1, 1e-8);
		}
	}
	EXPECT_TRUE(sawAttentionWeight);

	block.SetSeqLen(2);
	EXPECT_EQ(block.SeqLen(), 2);
	auto shorter = block.Forward(OaFnMatrix::RandN(OaMatrixShape{4, 8}));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const OaMatrixShape expectedShape{4, 8};
	EXPECT_EQ(shorter.GetShape(), expectedShape);
}
