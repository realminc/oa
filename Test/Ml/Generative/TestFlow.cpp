#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Data/DsHumanMl3d.h>
#include <Oa/Runtime/Context.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

class FlowTest : public ::testing::Test {};

OaMatrix FromF32(OaSpan<const OaF32> InValues, const OaMatrixShape& InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InValues.Data()),
			InValues.SizeBytes()),
		InShape, OaScalarType::Float32);
}

} // namespace

TEST_VK(FlowTest, LinearMatchBroadcastsBatchTimeAndEulerReconstructs) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	const OaF32 cleanValues[] = {0, 2, 4, 6, 8, 10};
	const OaF32 noiseValues[] = {10, 12, 14, 16, 18, 20};
	const OaF32 timeValues[] = {0.25F, 0.75F};
	auto clean = FromF32(cleanValues, {2, 3});
	auto noise = FromF32(noiseValues, {2, 3});
	auto time = FromF32(timeValues, {2});
	auto match = OaFnFlow::LinearMatch(clean, noise, time);
	auto reconstructed = OaFnFlow::EulerStep(match.State, match.Velocity, -0.25F);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const OaF32 expectedState[] = {2.5F, 4.5F, 6.5F, 13.5F, 15.5F, 17.5F};
	for (OaI64 index = 0; index < match.State.NumElements(); ++index) {
		EXPECT_NEAR(match.State.DataAs<const OaF32>()[index], expectedState[index], 1e-6F);
		EXPECT_NEAR(match.Velocity.DataAs<const OaF32>()[index], 10.0F, 1e-6F);
	}
	for (OaI64 index = 0; index < 3; ++index) {
		EXPECT_NEAR(reconstructed.DataAs<const OaF32>()[index], cleanValues[index], 1e-6F);
	}
}

TEST_VK(FlowTest, TimeEmbeddingRunsOnGpuAndMatchesCpuOracle) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	OaFlowTimeEmbedding embedding(4, 100.0F, 10.0F);
	const OaF32 timeValues[] = {0.0F, 0.5F};
	auto output = embedding.Forward(FromF32(timeValues, {2}));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ASSERT_EQ(output.GetShape(), (OaMatrixShape{2, 4}));

	const OaF32 frequencies[] = {10.0F, 1.0F};
	for (OaI64 batch = 0; batch < 2; ++batch) {
		for (OaI64 index = 0; index < 2; ++index) {
			const OaF32 phase = timeValues[batch] * frequencies[index];
			EXPECT_NEAR(output.DataAs<const OaF32>()[batch * 4 + index],
				std::sin(phase), 2e-6F);
			EXPECT_NEAR(output.DataAs<const OaF32>()[batch * 4 + 2 + index],
				std::cos(phase), 2e-6F);
		}
	}
}

TEST_VK(FlowTest, LinearMatchAutogradReachesBothEndpoints) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	const OaF32 cleanValues[] = {1, 2};
	const OaF32 noiseValues[] = {5, 8};
	const OaF32 timeValues[] = {0.25F};
	auto clean = FromF32(cleanValues, {1, 2});
	auto noise = FromF32(noiseValues, {1, 2});
	clean.SetRequiresGrad(true);
	noise.SetRequiresGrad(true);
	OaGradientTape tape;
	auto match = OaFnFlow::LinearMatch(clean, noise, FromF32(timeValues, {1}));
	tape.Backward(OaFnMatrix::Sum(match.State));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	ASSERT_FALSE(clean.GradMatrix().IsEmpty());
	ASSERT_FALSE(noise.GradMatrix().IsEmpty());
	for (OaI64 index = 0; index < 2; ++index) {
		EXPECT_NEAR(clean.GradMatrix().DataAs<const OaF32>()[index], 0.75F, 1e-6F);
		EXPECT_NEAR(noise.GradMatrix().DataAs<const OaF32>()[index], 0.25F, 1e-6F);
	}
}

TEST_VK(FlowTest, MaskedMseExcludesPaddingAndPreservesAutograd) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	const OaF32 predictionValues[] = {
		1, 3, 5, 7,
		100, 100, 100, 100,
	};
	const OaF32 targetValues[] = {
		0, 1, 2, 3,
		0, 0, 0, 0,
	};
	const OaF32 maskValues[] = {1, 0};
	auto prediction = FromF32(predictionValues, {1, 2, 4});
	prediction.SetRequiresGrad(true);
	auto target = FromF32(targetValues, {1, 2, 4});
	auto mask = FromF32(maskValues, {1, 2, 1});
	OaGradientTape tape;
	auto loss = OaFnFlow::MaskedMse(prediction, target, mask);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	// (1^2 + 2^2 + 3^2 + 4^2) / 4 = 7.5. The padded row must
	// not affect either the value or its gradient.
	EXPECT_NEAR(loss.DataAs<const OaF32>()[0], 7.5F, 1e-5F);
	ASSERT_FALSE(prediction.GradMatrix().IsEmpty());
	const OaF32 expectedGrad[] = {0.5F, 1.0F, 1.5F, 2.0F, 0, 0, 0, 0};
	for (OaI64 index = 0; index < prediction.NumElements(); ++index) {
		EXPECT_NEAR(prediction.GradMatrix().DataAs<const OaF32>()[index],
			expectedGrad[index], 1e-5F);
	}
}

TEST_VK(FlowTest, MaskedMseAllPaddingReturnsZeroAndRejectsBadShape) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	auto prediction = OaFnMatrix::Ones({2, 3, 4}, OaScalarType::Float32);
	auto target = OaFnMatrix::Zeros({2, 3, 4}, OaScalarType::Float32);
	auto mask = OaFnMatrix::Zeros({2, 3, 1}, OaScalarType::Float32);
	auto loss = OaFnFlow::MaskedMse(prediction, target, mask);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_FLOAT_EQ(loss.DataAs<const OaF32>()[0], 0.0F);
	EXPECT_THROW((void)OaFnFlow::MaskedMse(
		prediction, target, OaFnMatrix::Ones({2, 2}, OaScalarType::Float32)),
		std::invalid_argument);
}

TEST_VK(FlowTest, MaskedMseLargeMotionShapeRemainsNonNegative) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	auto prediction = OaFnMatrix::Ones({4, 64, 263}, OaScalarType::Float32);
	auto target = OaFnMatrix::Zeros({4, 64, 263}, OaScalarType::Float32);
	auto mask = OaFnMatrix::Ones({4, 64, 1}, OaScalarType::Float32);
	auto loss = OaFnFlow::MaskedMse(prediction, target, mask);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_NEAR(loss.DataAs<const OaF32>()[0], 1.0F, 1e-5F);
}

TEST_VK(FlowTest, RejectsInvalidContracts) {
	EXPECT_THROW((void)OaFlowTimeEmbedding(3), std::invalid_argument);
}

TEST_VK(FlowTest, HumanMl3dGeometryMetricsHaveExactIdentityOracle) {
	constexpr OaI32 frames = 3;
	constexpr OaI32 features = 263;
	OaVec<OaF32> target(frames * features, 0.0F);
	// Plant all four feet in the target and prediction.
	for (OaI32 frame = 0; frame < frames; ++frame) {
		for (OaI32 contact = 0; contact < 4; ++contact) {
			target[frame * features + features - 4 + contact] = 1.0F;
		}
	}
	auto identical = OaHumanMl3dEvaluateMotion(
		OaSpan<const OaF32>(target.Data(), target.Size()),
		OaSpan<const OaF32>(target.Data(), target.Size()), frames, features);
	ASSERT_TRUE(identical.Ok);
	EXPECT_DOUBLE_EQ(identical.MpjpeCm, 0.0);
	EXPECT_DOUBLE_EQ(identical.VelocityErrorCmPerFrame, 0.0);
	EXPECT_DOUBLE_EQ(identical.FootSkateCmPerFrame, 0.0);
	EXPECT_DOUBLE_EQ(identical.ContactAccuracy, 1.0);

	auto changed = target;
	changed[features + 1] = 1.0F; // root X velocity changes subsequent frames.
	changed[features + features - 4] = 0.0F;
	auto perturbed = OaHumanMl3dEvaluateMotion(
		OaSpan<const OaF32>(changed.Data(), changed.Size()),
		OaSpan<const OaF32>(target.Data(), target.Size()), frames, features);
	ASSERT_TRUE(perturbed.Ok);
	EXPECT_GT(perturbed.MpjpeCm, 0.0);
	EXPECT_GT(perturbed.VelocityErrorCmPerFrame, 0.0);
	EXPECT_LT(perturbed.ContactAccuracy, 1.0);
}

TEST_VK(FlowTest, DenseAndMoeTransformersShareBidirectionalContract) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	const OaFlowTransformerConfig denseConfig{
		.DModel = 4,
		.HiddenDim = 8,
		.SequenceLength = 2,
		.NumLayers = 1,
		.NumHeads = 1,
	};
	const OaFlowTransformerConfig moeConfig{
		.DModel = 4,
		.HiddenDim = 4,
		.SequenceLength = 2,
		.NumLayers = 1,
		.NumHeads = 1,
		.NumExperts = 2,
		.ExpertsPerToken = 1,
	};
	OaFlowTransformer dense(denseConfig);
	OaFlowTransformer moe(moeConfig);
	EXPECT_FALSE(dense.IsMoe());
	EXPECT_TRUE(moe.IsMoe());
	EXPECT_EQ(dense.Block(0).AttentionMode(), OaAttentionMode::Bidirectional);
	EXPECT_EQ(moe.Block(0).AttentionMode(), OaAttentionMode::Bidirectional);

	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 2, 4});
	input.SetRequiresGrad(true);
	OaGradientTape tape;
	auto denseOutput = dense.Forward(input);
	auto moeOutput = moe.Forward(input);
	tape.Backward(OaFnMatrix::Mean(denseOutput + moeOutput));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_EQ(denseOutput.GetShape(), input.GetShape());
	EXPECT_EQ(moeOutput.GetShape(), input.GetShape());
	EXPECT_FALSE(input.GradMatrix().IsEmpty());

	dense.SetSequenceLength(1);
	EXPECT_EQ(dense.Config().SequenceLength, 1);
	EXPECT_EQ(dense.Forward(input.Reshape(OaMatrixShape{4, 1, 4})).GetShape(),
		(OaMatrixShape{4, 1, 4}));
}

TEST_VK(FlowTest, TransformerRejectsAmbiguousShapesAndMoeConfig) {
	EXPECT_THROW((void)OaFlowTransformer(OaFlowTransformerConfig{
		.DModel = 4,
		.HiddenDim = 8,
		.SequenceLength = 2,
		.NumExperts = 2,
		.ExpertsPerToken = 3,
	}), std::invalid_argument);
	OaFlowTransformer model(OaFlowTransformerConfig{
		.DModel = 4,
		.HiddenDim = 8,
		.SequenceLength = 2,
	});
	EXPECT_THROW((void)model.Forward(OaFnMatrix::Zeros({2, 3, 4})),
		std::invalid_argument);
	EXPECT_THROW((void)model.Block(1), std::out_of_range);
}

TEST_VK(FlowTest, PaddingMaskPreventsInvalidKeysFromChangingValidTokens) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	const OaF32 firstValues[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
	};
	const OaF32 changedPaddingValues[] = {
		1, 2, 3, 4,
		500, -600, 700, -800,
	};
	const OaF32 maskValues[] = {1, 0};
	auto first = FromF32(firstValues, {1, 2, 4});
	auto changed = FromF32(changedPaddingValues, {1, 2, 4});
	auto mask = FromF32(maskValues, {1, 2, 1});

	for (bool moe : {false, true}) {
		OaFlowTransformer model(OaFlowTransformerConfig{
			.DModel = 4,
			.HiddenDim = moe ? 4 : 8,
			.SequenceLength = 2,
			.NumLayers = 1,
			.NumHeads = 1,
			.NumExperts = moe ? 2 : 0,
			.ExpertsPerToken = moe ? 1 : 0,
		});
		auto a = model.ForwardMasked(first, mask);
		auto b = model.ForwardMasked(changed, mask);
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		for (OaI64 feature = 0; feature < 4; ++feature) {
			EXPECT_NEAR(a.DataAs<const OaF32>()[feature],
				b.DataAs<const OaF32>()[feature], 2e-5F)
				<< "moe=" << moe << " feature=" << feature;
		}
	}

	OaFlowTransformer model(OaFlowTransformerConfig{
		.DModel = 4, .HiddenDim = 8, .SequenceLength = 2});
	EXPECT_THROW((void)model.ForwardMasked(first,
		OaFnMatrix::Ones({1, 3}, OaScalarType::Float32)),
		std::invalid_argument);
}

TEST_VK(FlowTest, DenoiserSharesConditionedDenseAndMoeContract) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	auto config = OaFlowDenoiserConfig{
		.InputDim = 2,
		.ConditionDim = 3,
		.Backbone = {
			.DModel = 4,
			.HiddenDim = 8,
			.SequenceLength = 2,
			.NumLayers = 1,
			.NumHeads = 1,
		},
		.TimeScale = 1.0F,
	};
	OaFlowDenoiser dense(config);
	config.Backbone.HiddenDim = 4;
	config.Backbone.NumExperts = 2;
	config.Backbone.ExpertsPerToken = 1;
	OaFlowDenoiser moe(config);
	auto sample = OaFnMatrix::RandN({2, 2, 2});
	auto time = OaFnMatrix::Full({2, 1}, 0.5F);
	auto condition = OaFnMatrix::RandN({2, 3});
	sample.SetRequiresGrad(true);
	OaGradientTape tape;
	auto denseOutput = dense.ForwardConditioned(sample, time, condition);
	auto moeOutput = moe.ForwardConditioned(sample, time, condition);
	tape.Backward(OaFnMatrix::Mean(denseOutput + moeOutput));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_EQ(denseOutput.GetShape(), sample.GetShape());
	EXPECT_EQ(moeOutput.GetShape(), sample.GetShape());
	EXPECT_FALSE(dense.IsMoe());
	EXPECT_TRUE(moe.IsMoe());
	EXPECT_FALSE(sample.GradMatrix().IsEmpty());
}

TEST_VK(FlowTest, AdaLnZeroAndClassifierFreeGuidanceAreSharedContracts) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	auto config = OaFlowDenoiserConfig{
		.InputDim = 2,
		.ConditionDim = 3,
		.Backbone = {
			.DModel = 4,
			.HiddenDim = 8,
			.SequenceLength = 2,
			.NumLayers = 1,
			.NumHeads = 1,
		},
		.TimeScale = 1.0F,
		.ConditionDropoutP = 0.25F,
	};
	OaFlowDenoiser model(config);
	auto named = model.AllNamedParameterPtrs();
	bool foundAdaptiveWeight = false;
	bool foundAdaptiveBias = false;
	for (const auto& parameter : named) {
		if (parameter.Path.find("adaptive_modulation.weight") != OaString::npos) {
			foundAdaptiveWeight = true;
			EXPECT_TRUE(parameter.Param->Data.RequiresGrad());
		}
		if (parameter.Path.find("adaptive_modulation.bias") != OaString::npos) {
			foundAdaptiveBias = true;
			EXPECT_TRUE(parameter.Param->Data.RequiresGrad());
		}
	}
	EXPECT_TRUE(foundAdaptiveWeight);
	EXPECT_TRUE(foundAdaptiveBias);

	auto sample = OaFnMatrix::RandN({2, 2, 2});
	auto time = OaFnMatrix::Full({2, 1}, 0.5F);
	auto condition = OaFnMatrix::Ones({2, 3}, OaScalarType::Float32);
	OaModule::ScopedEval eval(model);
	auto unconditional = model.ForwardConditioned(
		sample, time, OaFnMatrix::Zeros({2, 3}, OaScalarType::Float32));
	auto conditional = model.ForwardConditioned(sample, time, condition);
	auto guidanceZero = model.ForwardGuided(sample, time, condition, 0.0F);
	auto guidanceOne = model.ForwardGuided(sample, time, condition, 1.0F);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (OaI64 index = 0; index < sample.NumElements(); ++index) {
		EXPECT_NEAR(guidanceZero.DataAs<const OaF32>()[index],
			unconditional.DataAs<const OaF32>()[index], 2e-5F);
		EXPECT_NEAR(guidanceOne.DataAs<const OaF32>()[index],
			conditional.DataAs<const OaF32>()[index], 2e-5F);
	}
	EXPECT_THROW((void)model.ForwardGuided(
		sample, time, condition, -1.0F), std::invalid_argument);
	EXPECT_THROW((void)OaFlowDenoiser(OaFlowDenoiserConfig{
		.InputDim = 2,
		.ConditionDim = 3,
		.Backbone = {.DModel = 4, .HiddenDim = 8, .SequenceLength = 2},
		.ConditionDropoutP = 1.0F,
	}), std::invalid_argument);
}
