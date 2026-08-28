#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <data/dsHumanMl3d.h>
#include <oa/runtime/executionSession.h>

#include <gtest/gtest.h>
#include "../../oaTest.h"

#include <cmath>
#include <stdexcept>

namespace {

class FlowTest : public ::testing::Test {};

oa::Matrix fromF32(oa::Span<const oa::F32> inValues, const oa::MatrixShape& inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.sizeBytes()),
		inShape, oa::ScalarType::Float32);
}

} // namespace

TEST_VK(FlowTest, LinearMatchBroadcastsBatchTimeAndEulerReconstructs) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 cleanValues[] = {0, 2, 4, 6, 8, 10};
	const oa::F32 noiseValues[] = {10, 12, 14, 16, 18, 20};
	const oa::F32 timeValues[] = {0.25F, 0.75F};
	auto clean = fromF32(cleanValues, {2, 3});
	auto noise = fromF32(noiseValues, {2, 3});
	auto time = fromF32(timeValues, {2});
	auto match = oa::FnFlow::linearMatch(clean, noise, time);
	auto reconstructed = oa::FnFlow::eulerStep(match.state, match.velocity, -0.25F);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const oa::F32 expectedState[] = {2.5F, 4.5F, 6.5F, 13.5F, 15.5F, 17.5F};
	for (oa::I64 index = 0; index < match.state.numElements(); ++index) {
		EXPECT_NEAR(match.state.dataAs<const oa::F32>()[index], expectedState[index], 1e-6F);
		EXPECT_NEAR(match.velocity.dataAs<const oa::F32>()[index], 10.0F, 1e-6F);
	}
	for (oa::I64 index = 0; index < 3; ++index) {
		EXPECT_NEAR(reconstructed.dataAs<const oa::F32>()[index], cleanValues[index], 1e-6F);
	}
}

TEST_VK(FlowTest, TimeEmbeddingRunsOnGpuAndMatchesCpuOracle) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::FlowTimeEmbedding embedding(4, 100.0F, 10.0F);
	const oa::F32 timeValues[] = {0.0F, 0.5F};
	auto output = embedding.forward(fromF32(timeValues, {2}));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ASSERT_EQ(output.getShape(), (oa::MatrixShape{2, 4}));

	const oa::F32 frequencies[] = {10.0F, 1.0F};
	for (oa::I64 batch = 0; batch < 2; ++batch) {
		for (oa::I64 index = 0; index < 2; ++index) {
			const oa::F32 phase = timeValues[batch] * frequencies[index];
			EXPECT_NEAR(output.dataAs<const oa::F32>()[batch * 4 + index],
				std::sin(phase), 2e-6F);
			EXPECT_NEAR(output.dataAs<const oa::F32>()[batch * 4 + 2 + index],
				std::cos(phase), 2e-6F);
		}
	}
}

TEST_VK(FlowTest, LinearMatchAutogradReachesBothEndpoints) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 cleanValues[] = {1, 2};
	const oa::F32 noiseValues[] = {5, 8};
	const oa::F32 timeValues[] = {0.25F};
	auto clean = fromF32(cleanValues, {1, 2});
	auto noise = fromF32(noiseValues, {1, 2});
	clean.setRequiresGrad(true);
	noise.setRequiresGrad(true);
	oa::GradientTape tape;
	auto match = oa::FnFlow::linearMatch(clean, noise, fromF32(timeValues, {1}));
	tape.backward(oa::FnMatrix::sum(match.state));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	ASSERT_FALSE(clean.gradMatrix().isEmpty());
	ASSERT_FALSE(noise.gradMatrix().isEmpty());
	for (oa::I64 index = 0; index < 2; ++index) {
		EXPECT_NEAR(clean.gradMatrix().dataAs<const oa::F32>()[index], 0.75F, 1e-6F);
		EXPECT_NEAR(noise.gradMatrix().dataAs<const oa::F32>()[index], 0.25F, 1e-6F);
	}
}

TEST_VK(FlowTest, MaskedMseExcludesPaddingAndPreservesAutograd) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 predictionValues[] = {
		1, 3, 5, 7,
		100, 100, 100, 100,
	};
	const oa::F32 targetValues[] = {
		0, 1, 2, 3,
		0, 0, 0, 0,
	};
	const oa::F32 maskValues[] = {1, 0};
	auto prediction = fromF32(predictionValues, {1, 2, 4});
	prediction.setRequiresGrad(true);
	auto target = fromF32(targetValues, {1, 2, 4});
	auto mask = fromF32(maskValues, {1, 2, 1});
	oa::GradientTape tape;
	auto loss = oa::FnFlow::maskedMse(prediction, target, mask);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	// (1^2 + 2^2 + 3^2 + 4^2) / 4 = 7.5. The padded row must
	// not affect either the value or its gradient.
	EXPECT_NEAR(loss.dataAs<const oa::F32>()[0], 7.5F, 1e-5F);
	ASSERT_FALSE(prediction.gradMatrix().isEmpty());
	const oa::F32 expectedGrad[] = {0.5F, 1.0F, 1.5F, 2.0F, 0, 0, 0, 0};
	for (oa::I64 index = 0; index < prediction.numElements(); ++index) {
		EXPECT_NEAR(prediction.gradMatrix().dataAs<const oa::F32>()[index],
			expectedGrad[index], 1e-5F);
	}
}

TEST_VK(FlowTest, MaskedMseAllPaddingReturnsZeroAndRejectsBadShape) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto prediction = oa::FnMatrix::ones({2, 3, 4}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::zeros({2, 3, 4}, oa::ScalarType::Float32);
	auto mask = oa::FnMatrix::zeros({2, 3, 1}, oa::ScalarType::Float32);
	auto loss = oa::FnFlow::maskedMse(prediction, target, mask);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_FLOAT_EQ(loss.dataAs<const oa::F32>()[0], 0.0F);
	EXPECT_DEATH((void)oa::FnFlow::maskedMse(
		prediction, target, oa::FnMatrix::ones({2, 2}, oa::ScalarType::Float32)),
		"OA contract failed");
}

TEST_VK(FlowTest, MaskedMseLargeMotionShapeRemainsNonNegative) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto prediction = oa::FnMatrix::ones({4, 64, 263}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::zeros({4, 64, 263}, oa::ScalarType::Float32);
	auto mask = oa::FnMatrix::ones({4, 64, 1}, oa::ScalarType::Float32);
	auto loss = oa::FnFlow::maskedMse(prediction, target, mask);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_NEAR(loss.dataAs<const oa::F32>()[0], 1.0F, 1e-5F);
}

TEST_VK(FlowTest, RejectsInvalidContracts) {
	EXPECT_DEATH((void)oa::FlowTimeEmbedding(3), "OA contract failed");
}

TEST_VK(FlowTest, HumanMl3dGeometryMetricsHaveExactIdentityOracle) {
	constexpr oa::I32 frames = 3;
	constexpr oa::I32 features = 263;
	oa::Vector<oa::F32> target(frames * features, 0.0F);
	// Plant all four feet in the target and prediction.
	for (oa::I32 frame = 0; frame < frames; ++frame) {
		for (oa::I32 contact = 0; contact < 4; ++contact) {
			target[frame * features + features - 4 + contact] = 1.0F;
		}
	}
	auto identical = oa::humanMl3dEvaluateMotion(
		oa::Span<const oa::F32>(target.data(), target.size()),
		oa::Span<const oa::F32>(target.data(), target.size()), frames, features);
	ASSERT_TRUE(identical.ok);
	EXPECT_DOUBLE_EQ(identical.mpjpeCm, 0.0);
	EXPECT_DOUBLE_EQ(identical.velocityErrorCmPerFrame, 0.0);
	EXPECT_DOUBLE_EQ(identical.footSkateCmPerFrame, 0.0);
	EXPECT_DOUBLE_EQ(identical.contactAccuracy, 1.0);

	auto changed = target;
	changed[features + 1] = 1.0F; // root X velocity changes subsequent frames.
	changed[features + features - 4] = 0.0F;
	auto perturbed = oa::humanMl3dEvaluateMotion(
		oa::Span<const oa::F32>(changed.data(), changed.size()),
		oa::Span<const oa::F32>(target.data(), target.size()), frames, features);
	ASSERT_TRUE(perturbed.ok);
	EXPECT_GT(perturbed.mpjpeCm, 0.0);
	EXPECT_GT(perturbed.velocityErrorCmPerFrame, 0.0);
	EXPECT_LT(perturbed.contactAccuracy, 1.0);
}

TEST_VK(FlowTest, DenseAndMoeTransformersShareBidirectionalContract) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::FlowTransformerConfig denseConfig{
		.dModel = 4,
		.hiddenDim = 8,
		.sequenceLength = 2,
		.numLayers = 1,
		.numHeads = 1,
	};
	const oa::FlowTransformerConfig moeConfig{
		.dModel = 4,
		.hiddenDim = 4,
		.sequenceLength = 2,
		.numLayers = 1,
		.numHeads = 1,
		.numExperts = 2,
		.expertsPerToken = 1,
	};
	oa::FlowTransformer dense(denseConfig);
	oa::FlowTransformer moe(moeConfig);
	EXPECT_FALSE(dense.isMoe());
	EXPECT_TRUE(moe.isMoe());
	EXPECT_EQ(dense.block(0).attentionMode(), oa::AttentionMode::Bidirectional);
	EXPECT_EQ(moe.block(0).attentionMode(), oa::AttentionMode::Bidirectional);

	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 2, 4});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto denseOutput = dense.forward(input);
	auto moeOutput = moe.forward(input);
	tape.backward(oa::FnMatrix::mean(denseOutput + moeOutput));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(denseOutput.getShape(), input.getShape());
	EXPECT_EQ(moeOutput.getShape(), input.getShape());
	EXPECT_FALSE(input.gradMatrix().isEmpty());

	dense.setSequenceLength(1);
	EXPECT_EQ(dense.config().sequenceLength, 1);
	EXPECT_EQ(dense.forward(input.reshape(oa::MatrixShape{4, 1, 4})).getShape(),
		(oa::MatrixShape{4, 1, 4}));
}

TEST_VK(FlowTest, TransformerRejectsAmbiguousShapesAndMoeConfig) {
	EXPECT_DEATH((void)oa::FlowTransformer(oa::FlowTransformerConfig{
		.dModel = 4,
		.hiddenDim = 8,
		.sequenceLength = 2,
		.numExperts = 2,
		.expertsPerToken = 3,
	}), "OA contract failed");
	oa::FlowTransformer model(oa::FlowTransformerConfig{
		.dModel = 4,
		.hiddenDim = 8,
		.sequenceLength = 2,
	});
	EXPECT_DEATH((void)model.forward(oa::FnMatrix::zeros({2, 3, 4})),
		"OA contract failed");
	EXPECT_DEATH((void)model.block(1), "OA contract failed");
}

TEST_VK(FlowTest, PaddingMaskPreventsInvalidKeysFromChangingValidTokens) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 firstValues[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
	};
	const oa::F32 changedPaddingValues[] = {
		1, 2, 3, 4,
		500, -600, 700, -800,
	};
	const oa::F32 maskValues[] = {1, 0};
	auto first = fromF32(firstValues, {1, 2, 4});
	auto changed = fromF32(changedPaddingValues, {1, 2, 4});
	auto mask = fromF32(maskValues, {1, 2, 1});

	for (bool moe : {false, true}) {
		oa::FlowTransformer model(oa::FlowTransformerConfig{
			.dModel = 4,
			.hiddenDim = moe ? 4 : 8,
			.sequenceLength = 2,
			.numLayers = 1,
			.numHeads = 1,
			.numExperts = moe ? 2 : 0,
			.expertsPerToken = moe ? 1 : 0,
		});
		auto a = model.forwardMasked(first, mask);
		auto b = model.forwardMasked(changed, mask);
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		for (oa::I64 feature = 0; feature < 4; ++feature) {
			EXPECT_NEAR(a.dataAs<const oa::F32>()[feature],
				b.dataAs<const oa::F32>()[feature], 2e-5F)
				<< "moe=" << moe << " feature=" << feature;
		}
	}

	oa::FlowTransformer model(oa::FlowTransformerConfig{
		.dModel = 4, .hiddenDim = 8, .sequenceLength = 2});
	EXPECT_DEATH((void)model.forwardMasked(first,
		oa::FnMatrix::ones({1, 3}, oa::ScalarType::Float32)),
		"OA contract failed");
}

TEST_VK(FlowTest, DenoiserSharesConditionedDenseAndMoeContract) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto config = oa::FlowDenoiserConfig{
		.inputDim = 2,
		.conditionDim = 3,
		.backbone = {
			.dModel = 4,
			.hiddenDim = 8,
			.sequenceLength = 2,
			.numLayers = 1,
			.numHeads = 1,
		},
		.timeScale = 1.0F,
	};
	oa::FlowDenoiser dense(config);
	config.backbone.hiddenDim = 4;
	config.backbone.numExperts = 2;
	config.backbone.expertsPerToken = 1;
	oa::FlowDenoiser moe(config);
	auto sample = oa::FnMatrix::randN({2, 2, 2});
	auto time = oa::FnMatrix::full({2, 1}, 0.5F);
	auto condition = oa::FnMatrix::randN({2, 3});
	sample.setRequiresGrad(true);
	oa::GradientTape tape;
	auto denseOutput = dense.forwardConditioned(sample, time, condition);
	auto moeOutput = moe.forwardConditioned(sample, time, condition);
	tape.backward(oa::FnMatrix::mean(denseOutput + moeOutput));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(denseOutput.getShape(), sample.getShape());
	EXPECT_EQ(moeOutput.getShape(), sample.getShape());
	EXPECT_FALSE(dense.isMoe());
	EXPECT_TRUE(moe.isMoe());
	EXPECT_FALSE(sample.gradMatrix().isEmpty());
}

TEST_VK(FlowTest, AdaLnZeroAndClassifierFreeGuidanceAreSharedContracts) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto config = oa::FlowDenoiserConfig{
		.inputDim = 2,
		.conditionDim = 3,
		.backbone = {
			.dModel = 4,
			.hiddenDim = 8,
			.sequenceLength = 2,
			.numLayers = 1,
			.numHeads = 1,
		},
		.timeScale = 1.0F,
		.conditionDropoutP = 0.25F,
	};
	oa::FlowDenoiser model(config);
	auto named = model.allNamedParameterPtrs();
	bool foundAdaptiveWeight = false;
	bool foundAdaptiveBias = false;
	for (const auto& parameter : named) {
		if (parameter.path.find("adaptive_modulation.weight") != oa::String::Npos) {
			foundAdaptiveWeight = true;
			EXPECT_TRUE(parameter.param->data.requiresGrad());
		}
		if (parameter.path.find("adaptive_modulation.bias") != oa::String::Npos) {
			foundAdaptiveBias = true;
			EXPECT_TRUE(parameter.param->data.requiresGrad());
		}
	}
	EXPECT_TRUE(foundAdaptiveWeight);
	EXPECT_TRUE(foundAdaptiveBias);

	auto sample = oa::FnMatrix::randN({2, 2, 2});
	auto time = oa::FnMatrix::full({2, 1}, 0.5F);
	auto condition = oa::FnMatrix::ones({2, 3}, oa::ScalarType::Float32);
	oa::Module::ScopedEval eval(model);
	auto unconditional = model.forwardConditioned(
		sample, time, oa::FnMatrix::zeros({2, 3}, oa::ScalarType::Float32));
	auto conditional = model.forwardConditioned(sample, time, condition);
	auto guidanceZero = model.forwardGuided(sample, time, condition, 0.0F);
	auto guidanceOne = model.forwardGuided(sample, time, condition, 1.0F);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	for (oa::I64 index = 0; index < sample.numElements(); ++index) {
		EXPECT_NEAR(guidanceZero.dataAs<const oa::F32>()[index],
			unconditional.dataAs<const oa::F32>()[index], 2e-5F);
		EXPECT_NEAR(guidanceOne.dataAs<const oa::F32>()[index],
			conditional.dataAs<const oa::F32>()[index], 2e-5F);
	}
	EXPECT_DEATH((void)model.forwardGuided(
		sample, time, condition, -1.0F), "OA contract failed");
	EXPECT_DEATH((void)oa::FlowDenoiser(oa::FlowDenoiserConfig{
		.inputDim = 2,
		.conditionDim = 3,
		.backbone = {.dModel = 4, .hiddenDim = 8, .sequenceLength = 2},
		.conditionDropoutP = 1.0F,
	}), "OA contract failed");
}
