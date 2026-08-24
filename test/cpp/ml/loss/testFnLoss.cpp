// Tests for Ml/FnLoss operations
// MSE, BCE, L1 loss functions

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/ml/fnLoss.h>
#include <oa/ml/autograd.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <cmath>
#include <vector>

// MlTestMain owns this executable's shared test engine. The fixture intentionally
// carries no redundant device state of its own.
class TestFnLoss : public ::testing::Test {};

// Helper to create matrix from host data
static oa::Matrix createMatrixFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

// Helper to get scalar value from matrix
static float getScalar(const oa::Matrix& m) {
	float value = 0.0f;
	[[maybe_unused]] auto result = oa::FnMatrix::copyToHost(m, &value, sizeof(float));
	return value;
}

static float roundBfloat16NearestEven(float value) {
	union { float f32; oa::U32 bits; } encoded{.f32 = value};
	const oa::U32 rounding = 0x7FFFU + ((encoded.bits >> 16U) & 1U);
	return oa::bf16ToF32(static_cast<oa::U16>((encoded.bits + rounding) >> 16U));
}

static oa::Matrix createTargetsFromHost(const std::vector<oa::U32>& data) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()),
			data.size() * sizeof(oa::U32)),
		oa::MatrixShape{static_cast<oa::I64>(data.size())}, oa::ScalarType::UInt32);
}

static oa::Matrix createSignedTargetsFromHost(const std::vector<oa::I32>& data) {
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(data.data(), data.size()),
		oa::MatrixShape{static_cast<oa::I64>(data.size())}, oa::ScalarType::Int32);
}

static oa::Matrix createByteTargetsFromHost(const std::vector<oa::U8>& data) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(data.data(), data.size()),
		oa::MatrixShape{static_cast<oa::I64>(data.size())}, oa::ScalarType::UInt8);
}

// ============================================================================
// CrossEntropy Tests
// ============================================================================

TEST_VK(TestFnLoss, CrossEntropy_PublicForwardMatchesStableCpuOracle) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::I32> targets = {0, 2};
	auto logitsMatrix = createMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = createSignedTargetsFromHost(targets);

	oa::ExecutionSession::RecordingScope ctxScope(oa::ExecutionSession::getActive());
	const auto loss = oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.isEmpty());
	EXPECT_EQ(loss.getShape(), (oa::MatrixShape{1}));
	EXPECT_EQ(loss.getDtype(), oa::ScalarType::Float32);

	float expected = 0.0F;
	for (oa::U32 row = 0; row < 2U; ++row) {
		const oa::U32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (oa::U32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		expected += std::log(denominator) + maximum
			- logits[base + static_cast<oa::U32>(targets[row])];
	}
	expected /= 2.0F;
	EXPECT_NEAR(getScalar(loss), expected, 2.0e-6F);
}

TEST_VK(TestFnLoss, CrossEntropy_AutogradPreservesNonUnitUpstream) {
	constexpr float kUpstream = 2.5F;
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::U32> targets = {0U, 2U};
	auto logitsMatrix = createMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = createTargetsFromHost(targets);
	logitsMatrix.setRequiresGrad(true);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::scale(
		oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		logitsMatrix.gradMatrix(), gradient.data(),
		gradient.size() * sizeof(float)).isOk());
	for (oa::U32 row = 0; row < 2U; ++row) {
		const oa::U32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (oa::U32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (oa::U32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected = kUpstream *
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 3.0e-6F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropyBwd_Bfloat16MatchesCpuOracle) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::U32> targets = {0U, 2U};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto logitsMatrix = oa::FnMatrix::cast(
		createMatrixFromHost(logits, {2, 3}), oa::ScalarType::BFloat16);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();

	const auto gradientF32 = oa::FnMatrix::cast(
		oa::FnLoss::crossEntropyBwd(
			logitsMatrix, createTargetsFromHost(targets)),
		oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradientF32, gradient.data(), gradient.size() * sizeof(float)).isOk());

	for (oa::U32 row = 0; row < 2U; ++row) {
		const oa::U32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (oa::U32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (oa::U32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected =
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 2.0e-3F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_Bfloat16AutogradPreservesNonUnitUpstream) {
	// 1.003 rounds to 1.0 in BF16. The reference therefore distinguishes
	// FP32 upstream scaling from prematurely casting the upstream scalar.
	constexpr float kUpstream = 1.003F;
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::U32> targets = {0U, 2U};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	auto logitsMatrix = oa::FnMatrix::cast(
		createMatrixFromHost(logits, {2, 3}), oa::ScalarType::BFloat16);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	auto targetsMatrix = createTargetsFromHost(targets);
	const auto referenceF32 = oa::FnMatrix::cast(
		oa::FnMatrix::cast(
			oa::FnMatrix::scale(
				oa::FnMatrix::cast(
					oa::FnLoss::crossEntropyBwd(logitsMatrix, targetsMatrix),
					oa::ScalarType::Float32),
				kUpstream),
			oa::ScalarType::BFloat16),
		oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	std::vector<float> reference(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		referenceF32, reference.data(), reference.size() * sizeof(float)).isOk());
	ctx.clear();
	logitsMatrix.setRequiresGrad(true);

	oa::GradientTape tape;
	const auto loss = oa::FnMatrix::scale(
		oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.backward(loss);
	const auto gradientF32 = oa::FnMatrix::cast(
		logitsMatrix.gradMatrix(), oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradientF32, gradient.data(), gradient.size() * sizeof(float)).isOk());
	for (oa::Usize i = 0; i < gradient.size(); ++i) {
		EXPECT_FLOAT_EQ(gradient[i], reference[i]) << "element=" << i;
	}
	for (oa::U32 row = 0; row < 2U; ++row) {
		const oa::U32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (oa::U32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (oa::U32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected = kUpstream *
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 4.0e-3F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_OddBfloat16WithByteTargetIsTailSafe) {
	// Three BF16 values occupy six logical bytes. storage.slang reads/stores the
	// final value through its enclosing 32-bit word, so this is the minimal odd
	// element-count case that proves the descriptor exposes the padded tail.
	constexpr float kUpstream = 1.003F;
	const std::vector<float> logits = {2.0F, -1.0F, 0.5F};
	const std::vector<oa::U8> targets = {2U};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	auto logitsMatrix = oa::FnMatrix::cast(
		createMatrixFromHost(logits, {1, 3}), oa::ScalarType::BFloat16);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	const auto targetsMatrix = createByteTargetsFromHost(targets);

	const auto loss = oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix);
	const auto directGradientF32 = oa::FnMatrix::cast(
		oa::FnLoss::crossEntropyBwd(logitsMatrix, targetsMatrix),
		oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const float maximum = 2.0F;
	const float denominator = std::exp(2.0F - maximum)
		+ std::exp(-1.0F - maximum) + std::exp(0.5F - maximum);
	const float expectedLoss = std::log(denominator) + maximum - 0.5F;
	EXPECT_NEAR(getScalar(loss), expectedLoss, 2.0e-6F);

	std::vector<float> directGradient(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		directGradientF32, directGradient.data(),
		directGradient.size() * sizeof(float)).isOk());
	for (oa::U32 col = 0; col < 3U; ++col) {
		const float probability = std::exp(logits[col] - maximum) / denominator;
		const float expected = oa::bf16ToF32(oa::f32ToBf16(
			probability - (col == targets[0] ? 1.0F : 0.0F)));
		EXPECT_FLOAT_EQ(directGradient[col], expected) << "col=" << col;
	}

	const auto& forwardPipeline =
		oa::EnginePipelineAccess::get(ctx.engine()).getPipeline("CrossEntropy", 1U);
	const auto& backwardPipeline =
		oa::EnginePipelineAccess::get(ctx.engine()).getPipeline("CrossEntropyBwd", 1U);
	ASSERT_NE(forwardPipeline.pipeline, nullptr);
	ASSERT_NE(backwardPipeline.pipeline, nullptr);
	EXPECT_EQ(forwardPipeline.nativeDtype, 1U);
	EXPECT_EQ(backwardPipeline.nativeDtype, 1U);

	ctx.clear();
	logitsMatrix.setRequiresGrad(true);
	oa::GradientTape tape;
	const auto scaledLoss = oa::FnMatrix::scale(
		oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.backward(scaledLoss);
	const auto autogradF32 = oa::FnMatrix::cast(
		logitsMatrix.gradMatrix(), oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> autograd(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		autogradF32, autograd.data(), autograd.size() * sizeof(float)).isOk());
	for (oa::Usize col = 0; col < autograd.size(); ++col) {
		const float expected = roundBfloat16NearestEven(
			directGradient[col] * kUpstream);
		EXPECT_FLOAT_EQ(autograd[col], expected) << "col=" << col;
	}
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_InvalidSignedTargetsProduceNanWithoutTargetIndexedRead) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::I32> invalidTargets = {-1, 3};
	auto logitsMatrix = createMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = createSignedTargetsFromHost(invalidTargets);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const auto loss = oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix);
	const auto gradient = oa::FnLoss::crossEntropyBwd(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.isEmpty());
	ASSERT_FALSE(gradient.isEmpty());
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	EXPECT_TRUE(std::isnan(getScalar(loss)));
	std::vector<float> gradientValues(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradient, gradientValues.data(),
		gradientValues.size() * sizeof(float)).isOk());
	for (float value : gradientValues) EXPECT_TRUE(std::isnan(value));
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_InvalidByteTargetsProduceNanWithoutTargetIndexedRead) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::U8> invalidTargets = {3U, 255U};
	auto logitsMatrix = createMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = createByteTargetsFromHost(invalidTargets);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const auto loss = oa::FnLoss::crossEntropy(logitsMatrix, targetsMatrix);
	const auto gradient = oa::FnLoss::crossEntropyBwd(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.isEmpty());
	ASSERT_FALSE(gradient.isEmpty());
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	EXPECT_TRUE(std::isnan(getScalar(loss)));
	std::vector<float> gradientValues(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradient, gradientValues.data(),
		gradientValues.size() * sizeof(float)).isOk());
	for (float value : gradientValues) EXPECT_TRUE(std::isnan(value));
	ctx.clear();
}

TEST_VK(TestFnLoss, MaskedCrossEntropy_InvalidActiveByteTargetIsNanAndMaskedRowIsZero) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<oa::U8> targets = {3U, 255U};
	const std::vector<float> mask = {1.0F, 0.0F};
	auto logitsMatrix = createMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = createByteTargetsFromHost(targets);
	auto maskMatrix = createMatrixFromHost(mask, {2});
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const auto loss = oa::FnLoss::maskedCrossEntropy(
		logitsMatrix, targetsMatrix, maskMatrix, 1);
	const auto gradient = oa::FnLoss::maskedCrossEntropyBwd(
		logitsMatrix, targetsMatrix, maskMatrix, 1);
	ASSERT_FALSE(loss.isEmpty());
	ASSERT_FALSE(gradient.isEmpty());
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	EXPECT_TRUE(std::isnan(getScalar(loss)));
	std::vector<float> gradientValues(logits.size());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradient, gradientValues.data(),
		gradientValues.size() * sizeof(float)).isOk());
	for (oa::U32 col = 0; col < 3U; ++col) {
		EXPECT_TRUE(std::isnan(gradientValues[col]));
		EXPECT_FLOAT_EQ(gradientValues[3U + col], 0.0F);
	}
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_RejectsMalformedDirectCalls) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto logits = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto wrongTargetCount = oa::FnMatrix::empty({3}, oa::ScalarType::UInt32);
	const auto wrongTargetRank = oa::FnMatrix::empty({2, 1}, oa::ScalarType::UInt32);
	EXPECT_TRUE(oa::FnLoss::crossEntropy(logits, wrongTargetCount).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropyBwd(logits, wrongTargetCount).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropy(logits, wrongTargetRank).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropyBwd(logits, wrongTargetRank).isEmpty());

	const oa::I32 permutation[] = {1, 0};
	const auto permutedLogits = logits.permute(
		oa::Span<const oa::I32>(permutation, 2));
	const auto threeTargets = oa::FnMatrix::empty({3}, oa::ScalarType::UInt32);
	EXPECT_TRUE(oa::FnLoss::crossEntropy(permutedLogits, threeTargets).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropyBwd(permutedLogits, threeTargets).isEmpty());

	// view currently permits a shape larger than its shared allocation. Direct
	// loss entry points must reject that descriptor before recording a dispatch.
	const auto oversizedLogits = logits.view({2, 4});
	const auto twoTargets = oa::FnMatrix::empty({2}, oa::ScalarType::UInt32);
	EXPECT_TRUE(oa::FnLoss::crossEntropy(oversizedLogits, twoTargets).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropyBwd(oversizedLogits, twoTargets).isEmpty());
	EXPECT_TRUE(ctx.graph()->nodes().empty());
	ctx.clear();
}

TEST_VK(TestFnLoss, CrossEntropy_RejectsDescriptorRangeBeyondLiveDeviceLimit) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto logits = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto targets = oa::FnMatrix::empty({2}, oa::ScalarType::UInt8);
	const auto mask = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	auto& maximumRange =
		oa::EngineDeviceAccess::get(ctx.engine()).info.hardware.maxStorageBufferRangeBytes;
	const oa::U64 savedMaximumRange = maximumRange;
	VkPhysicalDeviceProperties properties{};
	oa::EngineDeviceAccess::get(ctx.engine()).instanceDispatch.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(
			oa::EngineDeviceAccess::get(ctx.engine()).physicalDevice),
		&properties);
	EXPECT_GT(savedMaximumRange, 0U);
	EXPECT_EQ(savedMaximumRange,
		static_cast<oa::U64>(properties.limits.maxStorageBufferRange));

	maximumRange = 8U;
	EXPECT_TRUE(oa::FnLoss::crossEntropy(logits, targets).isEmpty());
	EXPECT_TRUE(oa::FnLoss::crossEntropyBwd(logits, targets).isEmpty());
	EXPECT_TRUE(oa::FnLoss::maskedCrossEntropy(
		logits, targets, mask, 2).isEmpty());
	EXPECT_TRUE(oa::FnLoss::maskedCrossEntropyBwd(
		logits, targets, mask, 2).isEmpty());
	EXPECT_TRUE(ctx.graph()->nodes().empty());

	maximumRange = savedMaximumRange;
	ctx.clear();
}

// ============================================================================
// MSE (Mean Squared Error) Tests
// ============================================================================

TEST_VK(TestFnLoss, Mse_Perfect) {
	// Test MSE when predictions match targets perfectly
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::mse(pred_mat, target_mat);
	
	// expected: mean((0, 0, 0, 0)^2) = 0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, Mse_Simple) {
	// Test MSE with simple values
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {2.0f, 3.0f, 4.0f, 5.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::mse(pred_mat, target_mat);
	
	// expected: mean((1, 1, 1, 1)^2) = mean(1, 1, 1, 1) = 1.0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

TEST_VK(TestFnLoss, Mse_Mixed) {
	// Test MSE with mixed positive/negative errors
	std::vector<float> pred = {1.0f, 4.0f, 3.0f, 6.0f};
	std::vector<float> target = {2.0f, 2.0f, 5.0f, 4.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::mse(pred_mat, target_mat);
	
	// Errors: [-1, 2, -2, 2]
	// Squared: [1, 4, 4, 4]
	// Mean: (1 + 4 + 4 + 4) / 4 = 3.25
	float got = getScalar(loss);
	EXPECT_NEAR(got, 3.25f, 1e-6f);
}

TEST_VK(TestFnLoss, Mse_2D) {
	// Test MSE with 2D tensors
	std::vector<float> pred = {
		1.0f, 2.0f,
		3.0f, 4.0f
	};
	std::vector<float> target = {
		2.0f, 3.0f,
		4.0f, 5.0f
	};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{2, 2});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{2, 2});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::mse(pred_mat, target_mat);
	
	// All errors are 1, so MSE = 1.0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

// ============================================================================
// L1 (Mean Absolute Error) Tests
// ============================================================================

TEST_VK(TestFnLoss, L1_Perfect) {
	// Test L1 when predictions match targets perfectly
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::l1(pred_mat, target_mat);
	
	// expected: mean(|0, 0, 0, 0|) = 0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_Simple) {
	// Test L1 with simple values
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {2.0f, 3.0f, 4.0f, 5.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::l1(pred_mat, target_mat);
	
	// expected: mean(|1, 1, 1, 1|) = 1.0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_Mixed) {
	// Test L1 with mixed positive/negative errors
	std::vector<float> pred = {1.0f, 4.0f, 3.0f, 6.0f};
	std::vector<float> target = {2.0f, 2.0f, 5.0f, 4.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::l1(pred_mat, target_mat);
	
	// Errors: [-1, 2, -2, 2]
	// Absolute: [1, 2, 2, 2]
	// Mean: (1 + 2 + 2 + 2) / 4 = 1.75
	float got = getScalar(loss);
	EXPECT_NEAR(got, 1.75f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_vs_Mse) {
	// Test that L1 is less sensitive to outliers than MSE
	std::vector<float> pred = {1.0f, 1.0f, 1.0f, 10.0f};
	std::vector<float> target = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto l1_loss = oa::FnLoss::l1(pred_mat, target_mat);
	auto mse_loss = oa::FnLoss::mse(pred_mat, target_mat);
	
	// L1: mean(|0, 0, 0, 9|) = 9/4 = 2.25
	// MSE: mean(0, 0, 0, 81) = 81/4 = 20.25
	float l1 = getScalar(l1_loss);
	float mse = getScalar(mse_loss);
	
	EXPECT_NEAR(l1, 2.25f, 1e-6f);
	EXPECT_NEAR(mse, 20.25f, 1e-6f);
	EXPECT_LT(l1, mse);  // L1 is less sensitive to outliers
}

// ============================================================================
// BCE (Binary Cross-entropy) Tests
// ============================================================================

TEST_VK(TestFnLoss, Bce_Perfect) {
	// Test BCE with near-perfect predictions (avoiding exact 0/1 for numerical stability)
	// BCE = -[y*log(p) + (1-y)*log(1-p)]
	// Using 0.999 and 0.001 instead of exact 1.0 and 0.0
	std::vector<float> pred = {0.001f, 0.999f, 0.001f, 0.999f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::bce(pred_mat, target_mat);
	
	float got = getScalar(loss);
	
	// CPU reference: BCE = -mean[y*log(p) + (1-y)*log(1-p)]
	// For near-perfect predictions, loss should be very small
	float expected = 0.0f;
	for (size_t i = 0; i < pred.size(); ++i) {
		float p = pred[i];
		float y = target[i];
		expected += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
	}
	expected /= static_cast<float>(pred.size());
	
	EXPECT_NEAR(got, expected, 1e-4f);
	EXPECT_LT(got, 0.01f); // Should be very small for near-perfect predictions
}

TEST_VK(TestFnLoss, Bce_Probabilities) {
	// Test BCE with proper probabilities (avoiding 0/1)
	std::vector<float> pred = {0.1f, 0.9f, 0.2f, 0.8f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::bce(pred_mat, target_mat);
	
	// Manual calculation:
	// BCE = -mean(target * log(pred) + (1-target) * log(1-pred))
	// For each element:
	// [0]: 0 * log(0.1) + 1 * log(0.9) = log(0.9) ≈ -0.105
	// [1]: 1 * log(0.9) + 0 * log(0.1) = log(0.9) ≈ -0.105
	// [2]: 0 * log(0.2) + 1 * log(0.8) = log(0.8) ≈ -0.223
	// [3]: 1 * log(0.8) + 0 * log(0.2) = log(0.8) ≈ -0.223
	// Mean: (-0.105 - 0.105 - 0.223 - 0.223) / 4 ≈ -0.164
	// BCE = -(-0.164) = 0.164
	
	float got = getScalar(loss);
	EXPECT_GT(got, 0.0f);  // loss should be positive
	EXPECT_LT(got, 1.0f);  // Should be reasonable
	EXPECT_NEAR(got, 0.164f, 0.05f);  // Approximate check
}

TEST_VK(TestFnLoss, Bce_Confident) {
	// Test BCE with confident correct predictions
	std::vector<float> pred = {0.01f, 0.99f, 0.01f, 0.99f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::bce(pred_mat, target_mat);
	
	float got = getScalar(loss);
	EXPECT_GT(got, 0.0f);
	EXPECT_LT(got, 0.1f);  // Should be very low for confident correct predictions
}

TEST_VK(TestFnLoss, Bce_Wrong) {
	// Test BCE with wrong predictions
	std::vector<float> pred = {0.9f, 0.1f, 0.9f, 0.1f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::bce(pred_mat, target_mat);
	
	float got = getScalar(loss);
	EXPECT_GT(got, 1.0f);  // Should be high for wrong predictions
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_VK(TestFnLoss, Compare_Losses) {
	// compare different loss functions on same data
	std::vector<float> pred = {0.5f, 1.5f, 2.5f, 3.5f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto mse = oa::FnLoss::mse(pred_mat, target_mat);
	auto l1 = oa::FnLoss::l1(pred_mat, target_mat);
	
	float mse_val = getScalar(mse);
	float l1_val = getScalar(l1);
	
	// All errors are 0.5, so:
	// L1 = mean(|0.5, 0.5, 0.5, 0.5|) = 0.5
	// MSE = mean(0.25, 0.25, 0.25, 0.25) = 0.25
	EXPECT_NEAR(l1_val, 0.5f, 1e-6f);
	EXPECT_NEAR(mse_val, 0.25f, 1e-6f);
}

// ============================================================================
// smoothL1 (Huber loss) Tests
// ============================================================================

TEST_VK(TestFnLoss, SmoothL1_Perfect) {
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::smoothL1(pred_mat, target_mat);

	float got = getScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, SmoothL1_SmallError) {
	// |diff| < 1 → quadratic region: 0.5 * diff^2
	std::vector<float> pred = {1.5f, 2.5f, 3.5f, 4.5f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::smoothL1(pred_mat, target_mat);

	// Each diff = 0.5, loss = 0.5 * 0.25 = 0.125, mean = 0.125
	float got = getScalar(loss);
	EXPECT_NEAR(got, 0.125f, 1e-5f);
}

TEST_VK(TestFnLoss, SmoothL1_LargeError) {
	// |diff| >= 1 → linear region: |diff| - 0.5
	std::vector<float> pred = {3.0f, 5.0f, 7.0f, 9.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::smoothL1(pred_mat, target_mat);

	// diffs = [2, 3, 4, 5], losses = [1.5, 2.5, 3.5, 4.5], mean = 3.0
	float got = getScalar(loss);
	EXPECT_NEAR(got, 3.0f, 1e-5f);
}

TEST_VK(TestFnLoss, SmoothL1_Mixed) {
	// mix of quadratic and linear regions
	std::vector<float> pred = {1.3f, 4.0f, 3.5f, 6.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto loss = oa::FnLoss::smoothL1(pred_mat, target_mat);

	// diffs = [0.3, 2.0, 0.5, 2.0]
	// [0]: 0.5*0.09 = 0.045
	// [1]: 2.0 - 0.5 = 1.5
	// [2]: 0.5*0.25 = 0.125
	// [3]: 2.0 - 0.5 = 1.5
	// mean = (0.045 + 1.5 + 0.125 + 1.5) / 4 = 3.17 / 4 = 0.7925
	float got = getScalar(loss);
	EXPECT_NEAR(got, 0.7925f, 1e-4f);
}

TEST_VK(TestFnLoss, SmoothL1_vs_L1) {
	// SmoothL1 should be <= L1 for same errors (quadratic region is cheaper)
	std::vector<float> pred = {1.5f, 1.5f, 1.5f, 1.5f};
	std::vector<float> target = {1.0f, 1.0f, 1.0f, 1.0f};

	auto pred_mat = createMatrixFromHost(pred, oa::MatrixShape{4});
	auto target_mat = createMatrixFromHost(target, oa::MatrixShape{4});

	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto smooth = oa::FnLoss::smoothL1(pred_mat, target_mat);
	auto l1 = oa::FnLoss::l1(pred_mat, target_mat);

	// L1 = 0.5, SmoothL1 = 0.125 (quadratic region)
	float sv = getScalar(smooth);
	float lv = getScalar(l1);
	EXPECT_NEAR(lv, 0.5f, 1e-6f);
	EXPECT_NEAR(sv, 0.125f, 1e-5f);
	EXPECT_LT(sv, lv);
}
