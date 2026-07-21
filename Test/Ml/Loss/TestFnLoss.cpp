// Tests for Ml/FnLoss operations
// MSE, BCE, L1 loss functions

#include <gtest/gtest.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <cmath>
#include <vector>

// MlTestMain owns the single Vulkan engine for this executable. Creating a
// second static engine here replaced the process-global runtime and crashed
// teardown; the fixture intentionally carries no device state of its own.
class TestFnLoss : public ::testing::Test {};

// Helper to create matrix from host data
static OaMatrix CreateMatrixFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape);
}

// Helper to get scalar value from matrix
static float GetScalar(const OaMatrix& m) {
	float value = 0.0f;
	[[maybe_unused]] auto result = OaFnMatrix::CopyToHost(m, &value, sizeof(float));
	return value;
}

static float RoundBfloat16NearestEven(float value) {
	union { float Float; OaU32 Bits; } encoded{.Float = value};
	const OaU32 rounding = 0x7FFFU + ((encoded.Bits >> 16U) & 1U);
	return OaBf16ToF32(static_cast<OaU16>((encoded.Bits + rounding) >> 16U));
}

static OaMatrix CreateTargetsFromHost(const std::vector<OaU32>& data) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
			data.size() * sizeof(OaU32)),
		OaMatrixShape{static_cast<OaI64>(data.size())}, OaScalarType::UInt32);
}

static OaMatrix CreateSignedTargetsFromHost(const std::vector<OaI32>& data) {
	return OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(data.data(), data.size()),
		OaMatrixShape{static_cast<OaI64>(data.size())}, OaScalarType::Int32);
}

static OaMatrix CreateByteTargetsFromHost(const std::vector<OaU8>& data) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(data.data(), data.size()),
		OaMatrixShape{static_cast<OaI64>(data.size())}, OaScalarType::UInt8);
}

// ============================================================================
// CrossEntropy Tests
// ============================================================================

TEST_VK(TestFnLoss, CrossEntropy_PublicForwardMatchesStableCpuOracle) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaI32> targets = {0, 2};
	auto logitsMatrix = CreateMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = CreateSignedTargetsFromHost(targets);

	OaContext::RecordingScope ctxScope(OaContext::GetDefault());
	const auto loss = OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.IsEmpty());
	EXPECT_EQ(loss.GetShape(), (OaMatrixShape{1}));
	EXPECT_EQ(loss.GetDtype(), OaScalarType::Float32);

	float expected = 0.0F;
	for (OaU32 row = 0; row < 2U; ++row) {
		const OaU32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (OaU32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		expected += std::log(denominator) + maximum
			- logits[base + static_cast<OaU32>(targets[row])];
	}
	expected /= 2.0F;
	EXPECT_NEAR(GetScalar(loss), expected, 2.0e-6F);
}

TEST_VK(TestFnLoss, CrossEntropy_AutogradPreservesNonUnitUpstream) {
	constexpr float kUpstream = 2.5F;
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaU32> targets = {0U, 2U};
	auto logitsMatrix = CreateMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = CreateTargetsFromHost(targets);
	logitsMatrix.SetRequiresGrad(true);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaGradientTape tape;
	const auto loss = OaFnMatrix::Scale(
		OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		logitsMatrix.GradMatrix(), gradient.data(),
		gradient.size() * sizeof(float)).IsOk());
	for (OaU32 row = 0; row < 2U; ++row) {
		const OaU32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (OaU32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (OaU32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected = kUpstream *
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 3.0e-6F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropyBwd_Bfloat16MatchesCpuOracle) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaU32> targets = {0U, 2U};
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto logitsMatrix = OaFnMatrix::Cast(
		CreateMatrixFromHost(logits, {2, 3}), OaScalarType::BFloat16);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();

	const auto gradientF32 = OaFnMatrix::Cast(
		OaFnLoss::CrossEntropyBwd(
			logitsMatrix, CreateTargetsFromHost(targets)),
		OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradientF32, gradient.data(), gradient.size() * sizeof(float)).IsOk());

	for (OaU32 row = 0; row < 2U; ++row) {
		const OaU32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (OaU32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (OaU32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected =
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 2.0e-3F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropy_Bfloat16AutogradPreservesNonUnitUpstream) {
	// 1.003 rounds to 1.0 in BF16. The reference therefore distinguishes
	// FP32 upstream scaling from prematurely casting the upstream scalar.
	constexpr float kUpstream = 1.003F;
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaU32> targets = {0U, 2U};
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	auto logitsMatrix = OaFnMatrix::Cast(
		CreateMatrixFromHost(logits, {2, 3}), OaScalarType::BFloat16);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	auto targetsMatrix = CreateTargetsFromHost(targets);
	const auto referenceF32 = OaFnMatrix::Cast(
		OaFnMatrix::Cast(
			OaFnMatrix::Scale(
				OaFnMatrix::Cast(
					OaFnLoss::CrossEntropyBwd(logitsMatrix, targetsMatrix),
					OaScalarType::Float32),
				kUpstream),
			OaScalarType::BFloat16),
		OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	std::vector<float> reference(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		referenceF32, reference.data(), reference.size() * sizeof(float)).IsOk());
	ctx.Clear();
	logitsMatrix.SetRequiresGrad(true);

	OaGradientTape tape;
	const auto loss = OaFnMatrix::Scale(
		OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.Backward(loss);
	const auto gradientF32 = OaFnMatrix::Cast(
		logitsMatrix.GradMatrix(), OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> gradient(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradientF32, gradient.data(), gradient.size() * sizeof(float)).IsOk());
	for (OaUsize i = 0; i < gradient.size(); ++i) {
		EXPECT_FLOAT_EQ(gradient[i], reference[i]) << "element=" << i;
	}
	for (OaU32 row = 0; row < 2U; ++row) {
		const OaU32 base = row * 3U;
		const float maximum = std::max(
			logits[base], std::max(logits[base + 1U], logits[base + 2U]));
		float denominator = 0.0F;
		for (OaU32 col = 0; col < 3U; ++col) {
			denominator += std::exp(logits[base + col] - maximum);
		}
		for (OaU32 col = 0; col < 3U; ++col) {
			const float probability =
				std::exp(logits[base + col] - maximum) / denominator;
			const float expected = kUpstream *
				(probability - (col == targets[row] ? 1.0F : 0.0F)) / 2.0F;
			EXPECT_NEAR(gradient[base + col], expected, 4.0e-3F)
				<< "row=" << row << " col=" << col;
		}
	}
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropy_OddBfloat16WithByteTargetIsTailSafe) {
	// Three BF16 values occupy six logical bytes. Storage.slang reads/stores the
	// final value through its enclosing 32-bit word, so this is the minimal odd
	// element-count case that proves the descriptor exposes the padded tail.
	constexpr float kUpstream = 1.003F;
	const std::vector<float> logits = {2.0F, -1.0F, 0.5F};
	const std::vector<OaU8> targets = {2U};
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	auto logitsMatrix = OaFnMatrix::Cast(
		CreateMatrixFromHost(logits, {1, 3}), OaScalarType::BFloat16);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	const auto targetsMatrix = CreateByteTargetsFromHost(targets);

	const auto loss = OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix);
	const auto directGradientF32 = OaFnMatrix::Cast(
		OaFnLoss::CrossEntropyBwd(logitsMatrix, targetsMatrix),
		OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const float maximum = 2.0F;
	const float denominator = std::exp(2.0F - maximum)
		+ std::exp(-1.0F - maximum) + std::exp(0.5F - maximum);
	const float expectedLoss = std::log(denominator) + maximum - 0.5F;
	EXPECT_NEAR(GetScalar(loss), expectedLoss, 2.0e-6F);

	std::vector<float> directGradient(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		directGradientF32, directGradient.data(),
		directGradient.size() * sizeof(float)).IsOk());
	for (OaU32 col = 0; col < 3U; ++col) {
		const float probability = std::exp(logits[col] - maximum) / denominator;
		const float expected = OaBf16ToF32(OaF32ToBf16(
			probability - (col == targets[0] ? 1.0F : 0.0F)));
		EXPECT_FLOAT_EQ(directGradient[col], expected) << "col=" << col;
	}

	ASSERT_NE(ctx.GetEngine(), nullptr);
	const auto& forwardPipeline =
		ctx.GetEngine()->Pipelines.GetPipeline("CrossEntropy", 1U);
	const auto& backwardPipeline =
		ctx.GetEngine()->Pipelines.GetPipeline("CrossEntropyBwd", 1U);
	ASSERT_NE(forwardPipeline.Pipeline, nullptr);
	ASSERT_NE(backwardPipeline.Pipeline, nullptr);
	EXPECT_EQ(forwardPipeline.NativeDtype, 1U);
	EXPECT_EQ(backwardPipeline.NativeDtype, 1U);

	ctx.Clear();
	logitsMatrix.SetRequiresGrad(true);
	OaGradientTape tape;
	const auto scaledLoss = OaFnMatrix::Scale(
		OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix), kUpstream);
	tape.Backward(scaledLoss);
	const auto autogradF32 = OaFnMatrix::Cast(
		logitsMatrix.GradMatrix(), OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::vector<float> autograd(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		autogradF32, autograd.data(), autograd.size() * sizeof(float)).IsOk());
	for (OaUsize col = 0; col < autograd.size(); ++col) {
		const float expected = RoundBfloat16NearestEven(
			directGradient[col] * kUpstream);
		EXPECT_FLOAT_EQ(autograd[col], expected) << "col=" << col;
	}
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropy_InvalidSignedTargetsProduceNanWithoutTargetIndexedRead) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaI32> invalidTargets = {-1, 3};
	auto logitsMatrix = CreateMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = CreateSignedTargetsFromHost(invalidTargets);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	const auto loss = OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix);
	const auto gradient = OaFnLoss::CrossEntropyBwd(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.IsEmpty());
	ASSERT_FALSE(gradient.IsEmpty());
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	EXPECT_TRUE(std::isnan(GetScalar(loss)));
	std::vector<float> gradientValues(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradient, gradientValues.data(),
		gradientValues.size() * sizeof(float)).IsOk());
	for (float value : gradientValues) EXPECT_TRUE(std::isnan(value));
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropy_InvalidByteTargetsProduceNanWithoutTargetIndexedRead) {
	const std::vector<float> logits = {
		2.0F, -1.0F, 0.5F,
		-0.5F, 1.25F, 0.75F,
	};
	const std::vector<OaU8> invalidTargets = {3U, 255U};
	auto logitsMatrix = CreateMatrixFromHost(logits, {2, 3});
	auto targetsMatrix = CreateByteTargetsFromHost(invalidTargets);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	const auto loss = OaFnLoss::CrossEntropy(logitsMatrix, targetsMatrix);
	const auto gradient = OaFnLoss::CrossEntropyBwd(logitsMatrix, targetsMatrix);
	ASSERT_FALSE(loss.IsEmpty());
	ASSERT_FALSE(gradient.IsEmpty());
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	EXPECT_TRUE(std::isnan(GetScalar(loss)));
	std::vector<float> gradientValues(logits.size());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		gradient, gradientValues.data(),
		gradientValues.size() * sizeof(float)).IsOk());
	for (float value : gradientValues) EXPECT_TRUE(std::isnan(value));
	ctx.Clear();
}

TEST_VK(TestFnLoss, CrossEntropy_RejectsMalformedDirectCalls) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto logits = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto wrongTargetCount = OaFnMatrix::Empty({3}, OaScalarType::UInt32);
	const auto wrongTargetRank = OaFnMatrix::Empty({2, 1}, OaScalarType::UInt32);
	EXPECT_TRUE(OaFnLoss::CrossEntropy(logits, wrongTargetCount).IsEmpty());
	EXPECT_TRUE(OaFnLoss::CrossEntropyBwd(logits, wrongTargetCount).IsEmpty());
	EXPECT_TRUE(OaFnLoss::CrossEntropy(logits, wrongTargetRank).IsEmpty());
	EXPECT_TRUE(OaFnLoss::CrossEntropyBwd(logits, wrongTargetRank).IsEmpty());

	const OaI32 permutation[] = {1, 0};
	const auto permutedLogits = logits.Permute(
		OaSpan<const OaI32>(permutation, 2));
	const auto threeTargets = OaFnMatrix::Empty({3}, OaScalarType::UInt32);
	EXPECT_TRUE(OaFnLoss::CrossEntropy(permutedLogits, threeTargets).IsEmpty());
	EXPECT_TRUE(OaFnLoss::CrossEntropyBwd(permutedLogits, threeTargets).IsEmpty());

	// View currently permits a shape larger than its shared allocation. Direct
	// loss entry points must reject that descriptor before recording a dispatch.
	const auto oversizedLogits = logits.View({2, 4});
	const auto twoTargets = OaFnMatrix::Empty({2}, OaScalarType::UInt32);
	EXPECT_TRUE(OaFnLoss::CrossEntropy(oversizedLogits, twoTargets).IsEmpty());
	EXPECT_TRUE(OaFnLoss::CrossEntropyBwd(oversizedLogits, twoTargets).IsEmpty());
	EXPECT_TRUE(ctx.Graph()->Nodes().Empty());
	ctx.Clear();
}

// ============================================================================
// MSE (Mean Squared Error) Tests
// ============================================================================

TEST_VK(TestFnLoss, Mse_Perfect) {
	// Test MSE when predictions match targets perfectly
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Mse(pred_mat, target_mat);
	
	// Expected: mean((0, 0, 0, 0)^2) = 0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, Mse_Simple) {
	// Test MSE with simple values
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {2.0f, 3.0f, 4.0f, 5.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Mse(pred_mat, target_mat);
	
	// Expected: mean((1, 1, 1, 1)^2) = mean(1, 1, 1, 1) = 1.0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

TEST_VK(TestFnLoss, Mse_Mixed) {
	// Test MSE with mixed positive/negative errors
	std::vector<float> pred = {1.0f, 4.0f, 3.0f, 6.0f};
	std::vector<float> target = {2.0f, 2.0f, 5.0f, 4.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Mse(pred_mat, target_mat);
	
	// Errors: [-1, 2, -2, 2]
	// Squared: [1, 4, 4, 4]
	// Mean: (1 + 4 + 4 + 4) / 4 = 3.25
	float got = GetScalar(loss);
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
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{2, 2});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{2, 2});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Mse(pred_mat, target_mat);
	
	// All errors are 1, so MSE = 1.0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

// ============================================================================
// L1 (Mean Absolute Error) Tests
// ============================================================================

TEST_VK(TestFnLoss, L1_Perfect) {
	// Test L1 when predictions match targets perfectly
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::L1(pred_mat, target_mat);
	
	// Expected: mean(|0, 0, 0, 0|) = 0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_Simple) {
	// Test L1 with simple values
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {2.0f, 3.0f, 4.0f, 5.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::L1(pred_mat, target_mat);
	
	// Expected: mean(|1, 1, 1, 1|) = 1.0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 1.0f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_Mixed) {
	// Test L1 with mixed positive/negative errors
	std::vector<float> pred = {1.0f, 4.0f, 3.0f, 6.0f};
	std::vector<float> target = {2.0f, 2.0f, 5.0f, 4.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::L1(pred_mat, target_mat);
	
	// Errors: [-1, 2, -2, 2]
	// Absolute: [1, 2, 2, 2]
	// Mean: (1 + 2 + 2 + 2) / 4 = 1.75
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 1.75f, 1e-6f);
}

TEST_VK(TestFnLoss, L1_vs_Mse) {
	// Test that L1 is less sensitive to outliers than MSE
	std::vector<float> pred = {1.0f, 1.0f, 1.0f, 10.0f};
	std::vector<float> target = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto l1_loss = OaFnLoss::L1(pred_mat, target_mat);
	auto mse_loss = OaFnLoss::Mse(pred_mat, target_mat);
	
	// L1: mean(|0, 0, 0, 9|) = 9/4 = 2.25
	// MSE: mean(0, 0, 0, 81) = 81/4 = 20.25
	float l1 = GetScalar(l1_loss);
	float mse = GetScalar(mse_loss);
	
	EXPECT_NEAR(l1, 2.25f, 1e-6f);
	EXPECT_NEAR(mse, 20.25f, 1e-6f);
	EXPECT_LT(l1, mse);  // L1 is less sensitive to outliers
}

// ============================================================================
// BCE (Binary Cross-Entropy) Tests
// ============================================================================

TEST_VK(TestFnLoss, Bce_Perfect) {
	// Test BCE with near-perfect predictions (avoiding exact 0/1 for numerical stability)
	// BCE = -[y*log(p) + (1-y)*log(1-p)]
	// Using 0.999 and 0.001 instead of exact 1.0 and 0.0
	std::vector<float> pred = {0.001f, 0.999f, 0.001f, 0.999f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Bce(pred_mat, target_mat);
	
	float got = GetScalar(loss);
	
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
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Bce(pred_mat, target_mat);
	
	// Manual calculation:
	// BCE = -mean(target * log(pred) + (1-target) * log(1-pred))
	// For each element:
	// [0]: 0 * log(0.1) + 1 * log(0.9) = log(0.9) ≈ -0.105
	// [1]: 1 * log(0.9) + 0 * log(0.1) = log(0.9) ≈ -0.105
	// [2]: 0 * log(0.2) + 1 * log(0.8) = log(0.8) ≈ -0.223
	// [3]: 1 * log(0.8) + 0 * log(0.2) = log(0.8) ≈ -0.223
	// Mean: (-0.105 - 0.105 - 0.223 - 0.223) / 4 ≈ -0.164
	// BCE = -(-0.164) = 0.164
	
	float got = GetScalar(loss);
	EXPECT_GT(got, 0.0f);  // Loss should be positive
	EXPECT_LT(got, 1.0f);  // Should be reasonable
	EXPECT_NEAR(got, 0.164f, 0.05f);  // Approximate check
}

TEST_VK(TestFnLoss, Bce_Confident) {
	// Test BCE with confident correct predictions
	std::vector<float> pred = {0.01f, 0.99f, 0.01f, 0.99f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Bce(pred_mat, target_mat);
	
	float got = GetScalar(loss);
	EXPECT_GT(got, 0.0f);
	EXPECT_LT(got, 0.1f);  // Should be very low for confident correct predictions
}

TEST_VK(TestFnLoss, Bce_Wrong) {
	// Test BCE with wrong predictions
	std::vector<float> pred = {0.9f, 0.1f, 0.9f, 0.1f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::Bce(pred_mat, target_mat);
	
	float got = GetScalar(loss);
	EXPECT_GT(got, 1.0f);  // Should be high for wrong predictions
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_VK(TestFnLoss, Compare_Losses) {
	// Compare different loss functions on same data
	std::vector<float> pred = {0.5f, 1.5f, 2.5f, 3.5f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};
	
	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto mse = OaFnLoss::Mse(pred_mat, target_mat);
	auto l1 = OaFnLoss::L1(pred_mat, target_mat);
	
	float mse_val = GetScalar(mse);
	float l1_val = GetScalar(l1);
	
	// All errors are 0.5, so:
	// L1 = mean(|0.5, 0.5, 0.5, 0.5|) = 0.5
	// MSE = mean(0.25, 0.25, 0.25, 0.25) = 0.25
	EXPECT_NEAR(l1_val, 0.5f, 1e-6f);
	EXPECT_NEAR(mse_val, 0.25f, 1e-6f);
}

// ============================================================================
// SmoothL1 (Huber Loss) Tests
// ============================================================================

TEST_VK(TestFnLoss, SmoothL1_Perfect) {
	std::vector<float> pred = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});

	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::SmoothL1(pred_mat, target_mat);

	float got = GetScalar(loss);
	EXPECT_NEAR(got, 0.0f, 1e-6f);
}

TEST_VK(TestFnLoss, SmoothL1_SmallError) {
	// |diff| < 1 → quadratic region: 0.5 * diff^2
	std::vector<float> pred = {1.5f, 2.5f, 3.5f, 4.5f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});

	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::SmoothL1(pred_mat, target_mat);

	// Each diff = 0.5, loss = 0.5 * 0.25 = 0.125, mean = 0.125
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 0.125f, 1e-5f);
}

TEST_VK(TestFnLoss, SmoothL1_LargeError) {
	// |diff| >= 1 → linear region: |diff| - 0.5
	std::vector<float> pred = {3.0f, 5.0f, 7.0f, 9.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});

	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::SmoothL1(pred_mat, target_mat);

	// diffs = [2, 3, 4, 5], losses = [1.5, 2.5, 3.5, 4.5], mean = 3.0
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 3.0f, 1e-5f);
}

TEST_VK(TestFnLoss, SmoothL1_Mixed) {
	// Mix of quadratic and linear regions
	std::vector<float> pred = {1.3f, 4.0f, 3.5f, 6.0f};
	std::vector<float> target = {1.0f, 2.0f, 3.0f, 4.0f};

	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});

	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto loss = OaFnLoss::SmoothL1(pred_mat, target_mat);

	// diffs = [0.3, 2.0, 0.5, 2.0]
	// [0]: 0.5*0.09 = 0.045
	// [1]: 2.0 - 0.5 = 1.5
	// [2]: 0.5*0.25 = 0.125
	// [3]: 2.0 - 0.5 = 1.5
	// mean = (0.045 + 1.5 + 0.125 + 1.5) / 4 = 3.17 / 4 = 0.7925
	float got = GetScalar(loss);
	EXPECT_NEAR(got, 0.7925f, 1e-4f);
}

TEST_VK(TestFnLoss, SmoothL1_vs_L1) {
	// SmoothL1 should be <= L1 for same errors (quadratic region is cheaper)
	std::vector<float> pred = {1.5f, 1.5f, 1.5f, 1.5f};
	std::vector<float> target = {1.0f, 1.0f, 1.0f, 1.0f};

	auto pred_mat = CreateMatrixFromHost(pred, OaMatrixShape{4});
	auto target_mat = CreateMatrixFromHost(target, OaMatrixShape{4});

	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	auto smooth = OaFnLoss::SmoothL1(pred_mat, target_mat);
	auto l1 = OaFnLoss::L1(pred_mat, target_mat);

	// L1 = 0.5, SmoothL1 = 0.125 (quadratic region)
	float sv = GetScalar(smooth);
	float lv = GetScalar(l1);
	EXPECT_NEAR(lv, 0.5f, 1e-6f);
	EXPECT_NEAR(sv, 0.125f, 1e-5f);
	EXPECT_LT(sv, lv);
}
