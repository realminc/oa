#include "../../OaTest.h"

#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

class TestRl : public ::testing::Test {};

class TestLinearModule final : public OaModule {
public:
	TestLinearModule(OaI32 InInput, OaI32 InOutput) {
		Linear_ = OaMakeSharedPtr<OaLinear>(InInput, InOutput);
		RegisterModule("linear", Linear_);
	}
	OaMatrix Forward(const OaMatrix& InInput) override {
		return Linear_->Forward(InInput);
	}
private:
	OaSharedPtr<OaLinear> Linear_;
};

OaMatrix MatrixF32(const std::vector<OaF32>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaF32)),
		InShape,
		OaScalarType::Float32);
}

OaMatrix MatrixU8(const std::vector<OaU8>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(InValues.data(), InValues.size()),
		InShape,
		OaScalarType::UInt8);
}

OaMatrix MatrixI32(const std::vector<OaI32>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaI32)),
		InShape,
		OaScalarType::Int32);
}

void Sync() {
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
}

std::vector<OaF32> CopyF32(const OaMatrix& InMatrix) {
	std::vector<OaF32> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(OaF32)).IsOk());
	return result;
}

std::vector<OaI32> CopyI32(const OaMatrix& InMatrix) {
	std::vector<OaI32> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(OaI32)).IsOk());
	return result;
}

std::vector<OaU8> CopyU8(const OaMatrix& InMatrix) {
	std::vector<OaU8> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size()).IsOk());
	return result;
}

std::vector<OaU32> CopyU32(const OaMatrix& InMatrix) {
	std::vector<OaU32> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(OaU32)).IsOk());
	return result;
}

} // namespace

TEST_VK(TestRl, EnvironmentFieldSpecsRejectInvalidDefinitions) {
	auto emptyName = OaRlFieldSpec::Box("", {4});
	EXPECT_TRUE(emptyName.ValidateDefinition().IsError());

	auto zeroDimension = OaRlFieldSpec::Box("observation", {4, 0});
	EXPECT_TRUE(zeroDimension.ValidateDefinition().IsError());

	auto integerBox = OaRlFieldSpec::Box(
		"observation", {4}, OaScalarType::Int32);
	EXPECT_TRUE(integerBox.ValidateDefinition().IsError());

	auto reversedBounds = OaRlFieldSpec::Box(
		"observation", {4}, OaScalarType::Float32, 1.0, -1.0);
	EXPECT_TRUE(reversedBounds.ValidateDefinition().IsError());

	auto noActions = OaRlFieldSpec::Discrete("action", 0);
	EXPECT_TRUE(noActions.ValidateDefinition().IsError());

	auto invalidBinary = OaRlFieldSpec::Binary(
		"terminated", {}, OaScalarType::Float32);
	EXPECT_TRUE(invalidBinary.ValidateDefinition().IsError());
}

TEST_VK(TestRl, EnvironmentSpecValidatesBatchedMatrixContract) {
	const OaRlEnvironmentSpec spec{
		.Observation = OaRlFieldSpec::Box(
			"observation", {4}, OaScalarType::Float32),
		.Action = OaRlFieldSpec::Discrete("action", 2),
		.Reward = OaRlFieldSpec::Box(
			"reward", {}, OaScalarType::Float32, 0.0, 1.0),
		.Terminated = OaRlFieldSpec::Binary("terminated"),
		.Truncated = OaRlFieldSpec::Binary("truncated"),
	};
	ASSERT_TRUE(spec.ValidateDefinition().IsOk());

	const auto observation = MatrixF32(
		std::vector<OaF32>(12, 0.0F), {3, 4});
	const auto action = MatrixI32({0, 1, 0}, {3});
	const auto reward = MatrixF32({1.0F, 1.0F, 1.0F}, {3});
	const auto boundary = MatrixU8({0, 1, 0}, {3});
	EXPECT_TRUE(spec.ValidateReset(observation, 3).IsOk());
	EXPECT_TRUE(spec.ValidateAction(action, 3).IsOk());
	EXPECT_TRUE(spec.ValidateTransition(
		observation, action, observation, reward,
		boundary, boundary, 3).IsOk());

	const OaStatus wrongObservation = spec.ValidateReset(
		MatrixF32(std::vector<OaF32>(9, 0.0F), {3, 3}), 3);
	EXPECT_EQ(wrongObservation.GetCode(), OaStatusCode::ShapeMismatch);
	const OaStatus wrongAction = spec.ValidateAction(
		MatrixF32({0.0F, 1.0F, 0.0F}, {3}), 3);
	EXPECT_EQ(wrongAction.GetCode(), OaStatusCode::DtypeMismatch);
	const OaStatus wrongBoundary = spec.ValidateTransition(
		observation, action, observation, reward,
		MatrixI32({0, 0, 0}, {3}), boundary, 3);
	EXPECT_EQ(wrongBoundary.GetCode(), OaStatusCode::DtypeMismatch);
	EXPECT_TRUE(spec.ValidateReset(observation, 0).IsError());
}

TEST_VK(TestRl, RlTransformsComposeAsOrdinaryGpuMatrixOperations) {
	const auto observation = MatrixF32({
		1.0F, 4.0F,
		5.0F, -8.0F,
	}, {2, 2});
	const auto mean = MatrixF32({1.0F, 0.0F}, {2});
	const auto stddev = MatrixF32({2.0F, 2.0F}, {2});
	const auto normalized = OaFnRl::NormalizeObservation(
		observation, mean, stddev, 1.0e-6F, 3.0F);
	const auto scaled = OaFnRl::ScaleAction(
		MatrixF32({-2.0F, 0.0F, 2.0F}, {3}),
		-1.0F, 1.0F, 0.0F, 10.0F, true);
	const auto clipped = OaFnRl::ClipReward(
		MatrixF32({-3.0F, 0.25F, 4.0F}, {3}), -1.0F, 1.0F);
	ASSERT_FALSE(normalized.IsEmpty());
	ASSERT_FALSE(scaled.IsEmpty());
	ASSERT_FALSE(clipped.IsEmpty());
	Sync();
	const auto actualNormalized = CopyF32(normalized);
	EXPECT_NEAR(actualNormalized[0], 0.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[1], 2.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[2], 2.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[3], -3.0F, 1.0e-5F);
	EXPECT_EQ(CopyF32(scaled), (std::vector<OaF32>{0.0F, 5.0F, 10.0F}));
	EXPECT_EQ(CopyF32(clipped), (std::vector<OaF32>{-1.0F, 0.25F, 1.0F}));
}

TEST_VK(TestRl, CategoricalPolicyEvaluationMatchesCpu) {
	const std::vector<OaF32> logits = {
		1.0F, 2.0F, -1.0F,
		0.5F, 0.5F, 0.5F,
	};
	const std::vector<OaI32> action = {1, 0};
	const std::vector<OaF32> value = {0.25F, -0.75F};
	const auto result = OaFnRl::EvaluateCategoricalPolicy(
		MatrixF32(logits, {2, 3}),
		MatrixI32(action, {2}),
		MatrixF32(value, {2}));
	ASSERT_TRUE(result.IsValid());
	Sync();

	const auto actualLogProbability = CopyF32(result.LogProbability);
	const auto actualEntropy = CopyF32(result.Entropy);
	EXPECT_EQ(CopyF32(result.Value), value);
	EXPECT_EQ(CopyI32(result.Action), action);
	for (OaU32 row = 0; row < 2; ++row) {
		OaF32 maxLogit = logits[row * 3];
		for (OaU32 column = 1; column < 3; ++column) {
			maxLogit = std::max(maxLogit, logits[row * 3 + column]);
		}
		OaF32 normalizer = 0.0F;
		for (OaU32 column = 0; column < 3; ++column) {
			normalizer += std::exp(logits[row * 3 + column] - maxLogit);
		}
		const OaF32 logNormalizer = maxLogit + std::log(normalizer);
		OaF32 expectedEntropy = 0.0F;
		for (OaU32 column = 0; column < 3; ++column) {
			const OaF32 logProbability =
				logits[row * 3 + column] - logNormalizer;
			expectedEntropy -= std::exp(logProbability) * logProbability;
		}
		EXPECT_NEAR(actualLogProbability[row],
			logits[row * 3 + static_cast<OaU32>(action[row])]
				- logNormalizer,
			1.0e-6F);
		EXPECT_NEAR(actualEntropy[row], expectedEntropy, 1.0e-6F);
	}
}

TEST_VK(TestRl, CategoricalPolicyRemainsFiniteForConfidentLogits) {
	const auto result = OaFnRl::EvaluateCategoricalPolicy(
		MatrixF32({1000.0F, -1000.0F, 0.0F,
		           -1000.0F, 1000.0F, 0.0F}, {2, 3}),
		MatrixI32({0, 1}, {2}),
		MatrixF32({0.0F, 0.0F}, {2}));
	ASSERT_TRUE(result.IsValid());
	Sync();
	for (const OaF32 value : CopyF32(result.LogProbability)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
	for (const OaF32 value : CopyF32(result.Entropy)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
}

TEST_VK(TestRl, CategoricalPolicySamplingIsSeededAndSelfConsistent) {
	const auto logits = MatrixF32({
		2.0F, 1.0F, 0.0F,
		0.0F, 1.0F, 2.0F,
		1.0F, 1.0F, 1.0F,
		-1.0F, 3.0F, 0.5F,
	}, {4, 3});
	const auto value = MatrixF32({0.0F, 1.0F, 2.0F, 3.0F}, {4});
	const auto first = OaFnRl::SampleCategoricalPolicy(logits, value, 918273);
	const auto second = OaFnRl::SampleCategoricalPolicy(logits, value, 918273);
	ASSERT_TRUE(first.IsValid());
	ASSERT_TRUE(second.IsValid());
	Sync();
	const auto firstAction = CopyI32(first.Action);
	EXPECT_EQ(firstAction, CopyI32(second.Action));
	for (const OaI32 selected : firstAction) {
		EXPECT_GE(selected, 0);
		EXPECT_LT(selected, 3);
	}
	const auto reevaluated = OaFnRl::EvaluateCategoricalPolicy(
		logits, first.Action, value);
	Sync();
	const auto sampledLogProbability = CopyF32(first.LogProbability);
	const auto evaluatedLogProbability = CopyF32(reevaluated.LogProbability);
	for (OaUsize index = 0; index < sampledLogProbability.size(); ++index) {
		EXPECT_NEAR(sampledLogProbability[index],
			evaluatedLogProbability[index], 1.0e-7F);
	}
}

TEST_VK(TestRl, CategoricalPolicyLogProbabilityAutogradMatchesCpu) {
	const std::vector<OaF32> logitsHost = {
		1.0F, 2.0F, -1.0F,
		0.5F, -0.5F, 1.5F,
	};
	const std::vector<OaI32> actionHost = {1, 0};
	auto logits = MatrixF32(logitsHost, {2, 3});
	logits.SetRequiresGrad(true);
	OaGradientTape tape;
	const auto result = OaFnRl::EvaluateCategoricalPolicy(
		logits, MatrixI32(actionHost, {2}), MatrixF32({0.0F, 0.0F}, {2}));
	const auto loss = OaFnMatrix::Neg(OaFnMatrix::Mean(result.LogProbability));
	tape.Backward(loss);
	Sync();
	const auto gradient = CopyF32(logits.GradMatrix());
	for (OaU32 row = 0; row < 2; ++row) {
		OaF32 maxLogit = logitsHost[row * 3];
		for (OaU32 column = 1; column < 3; ++column) {
			maxLogit = std::max(maxLogit, logitsHost[row * 3 + column]);
		}
		OaF32 sum = 0.0F;
		for (OaU32 column = 0; column < 3; ++column) {
			sum += std::exp(logitsHost[row * 3 + column] - maxLogit);
		}
		for (OaU32 column = 0; column < 3; ++column) {
			const OaF32 probability =
				std::exp(logitsHost[row * 3 + column] - maxLogit) / sum;
			const OaF32 expected = (probability
				- (actionHost[row] == static_cast<OaI32>(column) ? 1.0F : 0.0F))
				/ 2.0F;
			EXPECT_NEAR(gradient[row * 3 + column], expected, 2.0e-6F)
				<< "gradient row=" << row << " column=" << column;
		}
	}
}

TEST_VK(TestRl, TanhNormalPolicyIsSeededBoundedAndSelfConsistent) {
	const auto mean = MatrixF32({0.0F, 0.5F, -0.5F, 1.0F}, {2, 2});
	const auto logStddev = MatrixF32({-0.7F, -0.2F, 0.1F, -1.0F}, {2, 2});
	const auto value = MatrixF32({0.25F, -0.75F}, {2});
	const auto first = OaFnRl::SampleTanhNormalPolicy(
		mean, logStddev, value, -2.0F, 3.0F, 91723);
	const auto second = OaFnRl::SampleTanhNormalPolicy(
		mean, logStddev, value, -2.0F, 3.0F, 91723);
	ASSERT_TRUE(first.IsValid());
	ASSERT_TRUE(second.IsValid());
	Sync();
	const auto action = CopyF32(first.Action);
	const auto secondAction = CopyF32(second.Action);
	ASSERT_EQ(action.size(), 4U);
	for (OaUsize index = 0; index < action.size(); ++index) {
		EXPECT_NEAR(action[index], secondAction[index], 1.0e-7F);
		EXPECT_GT(action[index], -2.0F);
		EXPECT_LT(action[index], 3.0F);
	}
	const auto evaluated = OaFnRl::EvaluateTanhNormalPolicy(
		mean, logStddev, first.RawAction, value, -2.0F, 3.0F);
	ASSERT_TRUE(evaluated.IsValid());
	Sync();
	const auto sampledLogProbability = CopyF32(first.LogProbability);
	const auto evaluatedLogProbability = CopyF32(evaluated.LogProbability);
	for (OaUsize index = 0; index < sampledLogProbability.size(); ++index) {
		EXPECT_NEAR(sampledLogProbability[index],
			evaluatedLogProbability[index], 1.0e-6F);
		EXPECT_TRUE(std::isfinite(sampledLogProbability[index]));
	}
	for (const OaF32 entropy : CopyF32(first.Entropy)) {
		EXPECT_TRUE(std::isfinite(entropy));
	}
}

TEST_VK(TestRl, TanhNormalPolicyLogProbabilityAutogradIsFinite) {
	auto mean = MatrixF32({0.1F, -0.2F, 0.3F, -0.4F}, {2, 2});
	auto logStddev = MatrixF32({-0.5F, -0.5F, -0.5F, -0.5F}, {2, 2});
	mean.SetRequiresGrad(true);
	logStddev.SetRequiresGrad(true);
	OaGradientTape tape;
	const auto policy = OaFnRl::EvaluateTanhNormalPolicy(
		mean, logStddev,
		MatrixF32({0.25F, -0.1F, 0.6F, -0.8F}, {2, 2}),
		MatrixF32({0.0F, 0.0F}, {2}));
	ASSERT_TRUE(policy.IsValid());
	tape.Backward(OaFnMatrix::Neg(OaFnMatrix::Mean(policy.LogProbability)));
	Sync();
	for (const OaF32 gradient : CopyF32(mean.GradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
	for (const OaF32 gradient : CopyF32(logStddev.GradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
}

TEST_VK(TestRl, ReplayBufferWrapsAndSamplesDeterministicallyOnGpu) {
	auto created = OaRlReplayBuffer::Create(OaRlReplayConfig{
		.Capacity = 4,
		.ObservationShape = {2},
		.ActionShape = {},
		.ActionDtype = OaScalarType::Int32,
	});
	ASSERT_TRUE(created.IsOk());
	auto replay = OaStdMove(*created);
	auto append = [&](const std::vector<OaF32>& observation,
		const std::vector<OaI32>& action,
		const std::vector<OaF32>& reward) {
		const OaI64 batch = static_cast<OaI64>(action.size());
		ASSERT_TRUE(replay.Append(OaRlReplayTransition{
			.Observation = MatrixF32(observation, {batch, 2}),
			.Action = MatrixI32(action, {batch}),
			.NextObservation = MatrixF32(observation, {batch, 2}),
			.Reward = MatrixF32(reward, {batch}),
			.Terminated = MatrixU8(std::vector<OaU8>(action.size(), 0), {batch}),
			.Truncated = MatrixU8(std::vector<OaU8>(action.size(), 0), {batch}),
		}).IsOk());
	};
	append({0, 1, 2, 3, 4, 5}, {0, 1, 2}, {10, 11, 12});
	append({6, 7, 8, 9, 10, 11}, {3, 4, 5}, {13, 14, 15});
	EXPECT_EQ(replay.Size(), 4U);
	EXPECT_EQ(replay.Cursor(), 2U);
	ASSERT_TRUE(replay.IsFull());
	auto first = replay.Sample(8, 7123);
	auto second = replay.Sample(8, 7123);
	ASSERT_TRUE(first.IsOk());
	ASSERT_TRUE(second.IsOk());
	Sync();
	const auto firstIndices = CopyU32(first->Index);
	EXPECT_EQ(firstIndices, CopyU32(second->Index));
	const auto rewards = CopyF32(first->Reward);
	for (OaUsize index = 0; index < firstIndices.size(); ++index) {
		ASSERT_LT(firstIndices[index], 4U);
		// Physical ring slots contain rewards [14,15,12,13].
		const OaF32 expected[] = {14.0F, 15.0F, 12.0F, 13.0F};
		EXPECT_EQ(rewards[index], expected[firstIndices[index]]);
	}
	replay.Reset();
	EXPECT_EQ(replay.Size(), 0U);
	EXPECT_TRUE(replay.Sample(1, 1).IsError());
}

TEST_VK(TestRl, DqnTargetBootstrapsTruncationButNotTermination) {
	auto q = MatrixF32({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2});
	q.SetRequiresGrad(true);
	OaGradientTape tape;
	const auto result = OaFnLoss::Dqn(
		q,
		MatrixI32({0, 1, 0}, {3}),
		MatrixF32({1.0F, 2.0F, 3.0F}, {3}),
		MatrixF32({10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, {3, 2}),
		MatrixU8({0, 1, 0}, {3}),
		MatrixU8({0, 0, 1}, {3}),
		OaDqnLossConfig{.Discount = 0.5F});
	ASSERT_TRUE(result.IsValid());
	tape.Backward(result.Loss);
	Sync();
	const auto target = CopyF32(result.TargetQ);
	ASSERT_EQ(target.size(), 3U);
	EXPECT_NEAR(target[0], 11.0F, 1.0e-6F);
	EXPECT_NEAR(target[1], 2.0F, 1.0e-6F);
	EXPECT_NEAR(target[2], 33.0F, 1.0e-6F);
	const auto selected = CopyF32(result.SelectedQ);
	EXPECT_EQ(selected, (std::vector<OaF32>{1.0F, 4.0F, 5.0F}));
	for (const OaF32 gradient : CopyF32(q.GradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
}

TEST_VK(TestRl, DqnTrainerConsumesReplayAndUpdatesTarget) {
	auto replayResult = OaRlReplayBuffer::Create({
		.Capacity = 8,
		.ObservationShape = {2},
		.ActionShape = {},
		.ActionDtype = OaScalarType::Int32,
	});
	ASSERT_TRUE(replayResult.IsOk());
	auto replay = OaStdMove(*replayResult);
	ASSERT_TRUE(replay.Append({
		.Observation = MatrixF32({0, 0, 1, 0, 0, 1, 1, 1}, {4, 2}),
		.Action = MatrixI32({0, 1, 0, 1}, {4}),
		.NextObservation = MatrixF32({1, 0, 0, 1, 1, 1, 0, 0}, {4, 2}),
		.Reward = MatrixF32({0, 1, 1, 0}, {4}),
		.Terminated = MatrixU8({0, 0, 1, 0}, {4}),
		.Truncated = MatrixU8({0, 0, 0, 1}, {4}),
	}).IsOk());
	auto onlineResult = OaCategoricalActorCritic::Create({
		.ObservationSize = 2, .ActionCount = 2, .HiddenSize = 8});
	auto targetResult = OaCategoricalActorCritic::Create({
		.ObservationSize = 2, .ActionCount = 2, .HiddenSize = 8});
	ASSERT_TRUE(onlineResult.IsOk());
	ASSERT_TRUE(targetResult.IsOk());
	auto online = OaStdMove(*onlineResult);
	auto target = OaStdMove(*targetResult);
	auto parameters = online->AllParameterPtrs();
	OaSGD optimizer(parameters, 1.0e-3F);
	auto trainerResult = OaDqnTrainer::Create(
		*online, *target, optimizer, replay, {
			.Updates = 2,
			.BatchSize = 4,
			.TargetUpdateInterval = 1,
			.ObservationShape = {2},
			.Seed = 77,
			.Loss = {.Discount = 0.95F},
		});
	ASSERT_TRUE(trainerResult.IsOk());
	auto trainer = OaStdMove(*trainerResult);
	ASSERT_TRUE(trainer->Update().IsOk());
	EXPECT_TRUE(std::isfinite(trainer->Metrics().Loss));
	ASSERT_TRUE(trainer->Update().IsOk());
	EXPECT_TRUE(trainer->IsDone());
	EXPECT_EQ(optimizer.GetStep(), 2U);
	EXPECT_TRUE(std::isfinite(trainer->Metrics().Loss));
}

TEST_VK(TestRl, SacLossesMatchTerminationAndEntropyContracts) {
	auto q1 = MatrixF32({1.0F, 2.0F, 3.0F}, {3});
	auto q2 = MatrixF32({1.5F, 1.0F, 4.0F}, {3});
	q1.SetRequiresGrad(true);
	q2.SetRequiresGrad(true);
	OaGradientTape criticTape;
	const auto critic = OaFnLoss::SacCritic(
		q1, q2,
		MatrixF32({1.0F, 2.0F, 3.0F}, {3}),
		MatrixF32({10.0F, 30.0F, 50.0F}, {3}),
		MatrixF32({20.0F, 40.0F, 60.0F}, {3}),
		MatrixF32({-1.0F, -2.0F, -3.0F}, {3}),
		MatrixU8({0, 1, 0}, {3}),
		MatrixU8({0, 0, 1}, {3}),
		{.Discount = 0.5F, .EntropyCoefficient = 0.2F});
	ASSERT_TRUE(critic.IsValid());
	criticTape.Backward(critic.TotalLoss);
	Sync();
	const auto target = CopyF32(critic.TargetQ);
	EXPECT_NEAR(target[0], 6.1F, 1.0e-5F);
	EXPECT_NEAR(target[1], 2.0F, 1.0e-5F);
	EXPECT_NEAR(target[2], 28.3F, 1.0e-5F);
	for (const OaF32 gradient : CopyF32(q1.GradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}

	auto actorQ1 = MatrixF32({2.0F, 4.0F}, {2});
	auto actorQ2 = MatrixF32({3.0F, 1.0F}, {2});
	auto logProbability = MatrixF32({-0.5F, -1.0F}, {2});
	logProbability.SetRequiresGrad(true);
	OaGradientTape actorTape;
	const OaMatrix actor = OaFnLoss::SacActor(
		actorQ1, actorQ2, logProbability, 0.2F);
	actorTape.Backward(actor);
	Sync();
	EXPECT_NEAR(actor.Item(), -1.65F, 1.0e-6F);
	for (const OaF32 gradient : CopyF32(logProbability.GradMatrix())) {
		EXPECT_NEAR(gradient, 0.1F, 1.0e-6F);
	}
}

TEST_VK(TestRl, SacTrainerRunsActorTwinCriticReplayUpdate) {
	auto replayResult = OaRlReplayBuffer::Create({
		.Capacity = 8,
		.ObservationShape = {2},
		.ActionShape = {1},
		.ActionDtype = OaScalarType::Float32,
	});
	ASSERT_TRUE(replayResult.IsOk());
	auto replay = OaStdMove(*replayResult);
	ASSERT_TRUE(replay.Append({
		.Observation = MatrixF32({0, 0, 1, 0, 0, 1, 1, 1}, {4, 2}),
		.Action = MatrixF32({-0.5F, 0.25F, 0.75F, -0.25F}, {4, 1}),
		.NextObservation = MatrixF32({1, 0, 0, 1, 1, 1, 0, 0}, {4, 2}),
		.Reward = MatrixF32({0, 1, 1, 0}, {4}),
		.Terminated = MatrixU8({0, 0, 1, 0}, {4}),
		.Truncated = MatrixU8({0, 0, 0, 1}, {4}),
	}).IsOk());
	TestLinearModule actor(2, 2);
	TestLinearModule critic1(3, 1);
	TestLinearModule critic2(3, 1);
	TestLinearModule targetCritic1(3, 1);
	TestLinearModule targetCritic2(3, 1);
	auto actorParameters = actor.AllParameterPtrs();
	auto criticParameters = critic1.AllParameterPtrs();
	for (auto* parameter : critic2.AllParameterPtrs()) {
		criticParameters.PushBack(parameter);
	}
	OaAdam actorOptimizer(actorParameters, 1.0e-3F);
	OaAdam criticOptimizer(criticParameters, 1.0e-3F);
	auto trainerResult = OaSacTrainer::Create(
		actor, critic1, critic2, targetCritic1, targetCritic2,
		actorOptimizer, criticOptimizer, replay, {
			.Updates = 1,
			.BatchSize = 4,
			.ActionDimensions = 1,
			.TargetUpdateInterval = 1,
			.ObservationShape = {2},
			.ActionMinimum = -1.0F,
			.ActionMaximum = 1.0F,
			.Seed = 123,
			.Loss = {.Discount = 0.99F, .EntropyCoefficient = 0.2F},
		});
	ASSERT_TRUE(trainerResult.IsOk());
	auto trainer = OaStdMove(*trainerResult);
	ASSERT_TRUE(trainer->Update().IsOk());
	EXPECT_TRUE(trainer->IsDone());
	EXPECT_EQ(actorOptimizer.GetStep(), 1U);
	EXPECT_EQ(criticOptimizer.GetStep(), 1U);
	EXPECT_TRUE(std::isfinite(trainer->Metrics().ActorLoss));
	EXPECT_TRUE(std::isfinite(trainer->Metrics().CriticLoss));
}

TEST_VK(TestRl, CategoricalPolicyEntropyAutogradMatchesCpu) {
	const std::vector<OaF32> logitsHost = {1.0F, 2.0F, -1.0F};
	auto logits = MatrixF32(logitsHost, {1, 3});
	logits.SetRequiresGrad(true);
	OaGradientTape tape;
	const auto result = OaFnRl::EvaluateCategoricalPolicy(
		logits, MatrixI32({1}, {1}), MatrixF32({0.0F}, {1}));
	const auto loss = OaFnMatrix::Neg(OaFnMatrix::Mean(result.Entropy));
	tape.Backward(loss);
	Sync();

	OaF32 maxLogit = *std::max_element(logitsHost.begin(), logitsHost.end());
	OaF32 sum = 0.0F;
	for (const OaF32 logit : logitsHost) sum += std::exp(logit - maxLogit);
	OaF32 entropy = 0.0F;
	std::vector<OaF32> probability;
	for (const OaF32 logit : logitsHost) {
		const OaF32 p = std::exp(logit - maxLogit) / sum;
		probability.push_back(p);
		entropy -= p * std::log(p);
	}
	const auto gradient = CopyF32(logits.GradMatrix());
	for (OaU32 column = 0; column < 3; ++column) {
		const OaF32 expected = probability[column]
			* (std::log(probability[column]) + entropy);
		EXPECT_NEAR(gradient[column], expected, 2.0e-6F)
			<< "entropy gradient column=" << column;
	}
}

TEST_VK(TestRl, CategoricalPolicyRejectsInvalidShapes) {
	const auto logits = MatrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
	EXPECT_FALSE(OaFnRl::EvaluateCategoricalPolicy(
		logits, MatrixI32({0}, {1}), MatrixF32({0.0F, 0.0F}, {2})).IsValid());
	EXPECT_FALSE(OaFnRl::SampleCategoricalPolicy(
		logits, MatrixF32({0.0F}, {1}), 7).IsValid());
}

TEST_VK(TestRl, RolloutBufferAppendsAndFinalizesOnGpu) {
	auto created = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
		.Time = 3,
		.Environments = 2,
		.ObservationShape = {2},
	});
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto rollout = OaStdMove(*created);
	ASSERT_TRUE(rollout.IsValid());

	const std::vector<std::vector<OaF32>> observations = {
		{0.0F, 0.1F, 0.2F, 0.3F},
		{1.0F, 1.1F, 1.2F, 1.3F},
		{2.0F, 2.1F, 2.2F, 2.3F},
	};
	const std::vector<std::vector<OaF32>> rewards = {
		{1.0F, 10.0F}, {2.0F, 20.0F}, {3.0F, 30.0F}};
	const std::vector<std::vector<OaF32>> nextValues = {
		{0.0F, 0.0F}, {100.0F, 5.0F}, {0.0F, 0.0F}};
	const std::vector<std::vector<OaU8>> terminated = {
		{0, 0}, {1, 0}, {0, 0}};
	const std::vector<std::vector<OaU8>> truncated = {
		{0, 0}, {0, 1}, {0, 0}};
	for (OaU32 step = 0; step < 3; ++step) {
		const OaRlTransition transition{
			.Observation = MatrixF32(observations[step], {2, 2}),
			.Action = MatrixI32(
				{static_cast<OaI32>(step % 2), static_cast<OaI32>((step + 1) % 2)},
				{2}),
			.Reward = MatrixF32(rewards[step], {2}),
			.Value = MatrixF32({0.0F, 0.0F}, {2}),
			.NextValue = MatrixF32(nextValues[step], {2}),
			.LogProbability = MatrixF32(
				{-0.1F * static_cast<OaF32>(step + 1),
				 -0.2F * static_cast<OaF32>(step + 1)}, {2}),
			.Terminated = MatrixU8(terminated[step], {2}),
			.Truncated = MatrixU8(truncated[step], {2}),
		};
		ASSERT_TRUE(rollout.Append(transition).IsOk());
	}
	ASSERT_TRUE(rollout.IsFull());
	ASSERT_TRUE(rollout.Finalize(OaGaeConfig{
		.Gamma = 1.0F, .Lambda = 1.0F}).IsOk());
	ASSERT_TRUE(rollout.IsFinalized());
	Sync();

	const auto& batch = rollout.Batch();
	EXPECT_EQ(CopyF32(batch.Observation),
		(std::vector<OaF32>{
			0.0F, 0.1F, 0.2F, 0.3F,
			1.0F, 1.1F, 1.2F, 1.3F,
			2.0F, 2.1F, 2.2F, 2.3F}));
	EXPECT_EQ(CopyI32(batch.Action),
		(std::vector<OaI32>{0, 1, 1, 0, 0, 1}));
	EXPECT_EQ(CopyF32(batch.Reward),
		(std::vector<OaF32>{1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F}));
	EXPECT_EQ(CopyU8(batch.Terminated),
		(std::vector<OaU8>{0, 0, 1, 0, 0, 0}));
	EXPECT_EQ(CopyU8(batch.Truncated),
		(std::vector<OaU8>{0, 0, 0, 1, 0, 0}));
	EXPECT_EQ(CopyU8(batch.Valid),
		(std::vector<OaU8>{1, 1, 1, 1, 1, 1}));
	const std::vector<OaF32> expectedAdvantage = {
		3.0F, 35.0F, 2.0F, 25.0F, 3.0F, 30.0F};
	const auto advantage = CopyF32(batch.Advantage);
	const auto returns = CopyF32(batch.Return);
	for (OaUsize index = 0; index < expectedAdvantage.size(); ++index) {
		EXPECT_NEAR(advantage[index], expectedAdvantage[index], 1.0e-6F);
		EXPECT_NEAR(returns[index], expectedAdvantage[index], 1.0e-6F);
	}
}

TEST_VK(TestRl, RolloutBufferEnforcesLifecycleAndReusesStorage) {
	EXPECT_TRUE(OaRlRolloutBuffer::Create(OaRlRolloutConfig{}).IsError());
	auto created = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
		.Time = 2,
		.Environments = 1,
		.ObservationShape = {4},
	});
	ASSERT_TRUE(created.IsOk());
	auto rollout = OaStdMove(*created);
	const auto observationBuffer = rollout.Batch().Observation.GetVkBuffer().Buffer;
	EXPECT_TRUE(rollout.Finalize().IsError());

	OaRlTransition transition{
		.Observation = MatrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {1, 4}),
		.Action = MatrixI32({1}, {1}),
		.Reward = MatrixF32({1.0F}, {1}),
		.Value = MatrixF32({0.0F}, {1}),
		.NextValue = MatrixF32({0.0F}, {1}),
		.LogProbability = MatrixF32({-0.5F}, {1}),
		.Terminated = MatrixU8({0}, {1}),
		.Truncated = MatrixU8({0}, {1}),
	};
	OaRlTransition invalid = transition;
	invalid.Observation = MatrixF32({1.0F, 2.0F}, {1, 2});
	EXPECT_TRUE(rollout.Append(invalid).IsError());
	EXPECT_EQ(rollout.Size(), 0U);
	ASSERT_TRUE(rollout.Append(transition).IsOk());
	ASSERT_TRUE(rollout.Append(transition).IsOk());
	EXPECT_TRUE(rollout.Append(transition).IsError());
	ASSERT_TRUE(rollout.Finalize().IsOk());
	EXPECT_TRUE(rollout.Append(transition).IsError());

	rollout.Reset();
	EXPECT_EQ(rollout.Size(), 0U);
	EXPECT_FALSE(rollout.IsFinalized());
	EXPECT_EQ(rollout.Batch().Observation.GetVkBuffer().Buffer, observationBuffer);
	ASSERT_TRUE(rollout.Append(transition).IsOk());
	Sync();
	EXPECT_EQ(CopyU8(rollout.Batch().Valid),
		(std::vector<OaU8>{1, 0}));
}

TEST_VK(TestRl, RlTrainingCoordinatorEnforcesCollectUpdateLifecycle) {
	OaOptimizerNoOp optimizer;
	OaItRlTraining invalid(optimizer, OaItRlTrainingConfig{});
	EXPECT_FALSE(invalid.IsValid());
	EXPECT_TRUE(invalid.IsDone());

	auto created = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
		.Time = 1,
		.Environments = 1,
		.ObservationShape = {2},
	});
	ASSERT_TRUE(created.IsOk());
	auto rollout = OaStdMove(*created);
	OaItRlTraining training(optimizer, OaItRlTrainingConfig{
		.Rollouts = 1,
		.Horizon = 1,
		.Environments = 1,
		.UpdateEpochs = 2,
	});
	ASSERT_TRUE(training.IsValid());
	EXPECT_FALSE(training.BeginUpdate());
	ASSERT_TRUE(training.BeginRollout(rollout).IsOk());
	ASSERT_TRUE(rollout.Append(OaRlTransition{
		.Observation = MatrixF32({1.0F, 2.0F}, {1, 2}),
		.Action = MatrixI32({0}, {1}),
		.Reward = MatrixF32({1.0F}, {1}),
		.Value = MatrixF32({0.0F}, {1}),
		.NextValue = MatrixF32({0.0F}, {1}),
		.LogProbability = MatrixF32({-0.5F}, {1}),
		.Terminated = MatrixU8({0}, {1}),
		.Truncated = MatrixU8({0}, {1}),
	}).IsOk());
	ASSERT_TRUE(training.FinalizeRollout(rollout).IsOk());
	EXPECT_EQ(training.Phase(), OaRlTrainingPhase::Update);

	const OaMatrix loss = MatrixF32({0.25F}, {1});
	OaTrainingSession session(training.UpdateLoop());
	ASSERT_TRUE(session.Pause().IsOk());
	EXPECT_FALSE(training.BeginUpdate());
	EXPECT_EQ(session.State(), OaTrainingState::Paused);
	EXPECT_EQ(training.UpdateLoop().StepCount(), 0);
	ASSERT_TRUE(session.Resume().IsOk());
	for (OaU32 epoch = 0; epoch < 2; ++epoch) {
		ASSERT_TRUE(training.BeginUpdate());
		EXPECT_FALSE(training.BeginUpdate());
		ASSERT_TRUE(training.NextUpdate(loss).IsOk());
	}
	EXPECT_TRUE(training.IsDone());
	EXPECT_EQ(training.RolloutIndex(), 1U);
	EXPECT_EQ(training.UpdateEpoch(), 2U);
	EXPECT_TRUE(training.Finish().IsOk());
}

TEST_VK(TestRl, ReusablePpoTrainerOwnsEnvironmentNeutralLifecycle) {
	auto modelResult = OaCategoricalActorCritic::Create(
		OaCategoricalActorCriticConfig{
			.ObservationSize = 2,
			.ActionCount = 2,
			.HiddenSize = 8,
		});
	ASSERT_TRUE(modelResult.IsOk());
	auto model = OaStdMove(*modelResult);
	OaOptimizerNoOp optimizer;
	auto trainerResult = OaPpoTrainer::Create(
		*model, optimizer, OaPpoTrainerConfig{
			.Rollouts = 1,
			.Horizon = 1,
			.Environments = 2,
			.UpdateEpochs = 1,
			.ObservationShape = {2},
			.Seed = 123,
		});
	ASSERT_TRUE(trainerResult.IsOk());
	auto trainer = OaStdMove(*trainerResult);
	ASSERT_TRUE(trainer->IsValid());
	ASSERT_TRUE(trainer->NeedsCollection());
	ASSERT_TRUE(trainer->BeginCollection().IsOk());

	const OaMatrix observation = MatrixF32(
		{0.1F, -0.2F, 0.3F, 0.4F}, {2, 2});
	const OaRlPolicyResult policy = trainer->Act(observation);
	ASSERT_TRUE(policy.IsValid());
	EXPECT_EQ(policy.Action.GetShape(), OaMatrixShape({2}));
	EXPECT_EQ(policy.Value.GetShape(), OaMatrixShape({2}));
	ASSERT_TRUE(trainer->Observe(
		observation,
		MatrixF32({0.2F, -0.1F, 0.4F, 0.5F}, {2, 2}),
		MatrixF32({1.0F, 0.5F}, {2}),
		MatrixU8({0, 0}, {2}),
		MatrixU8({0, 0}, {2}),
		policy).IsOk());
	ASSERT_TRUE(trainer->EndCollection().IsOk());
	EXPECT_EQ(trainer->Phase(), OaRlTrainingPhase::Update);
	ASSERT_TRUE(trainer->Update().IsOk());
	EXPECT_TRUE(trainer->IsDone());
	EXPECT_EQ(trainer->Metrics().Rollout, 1U);
	EXPECT_EQ(trainer->Metrics().UpdateEpoch, 1U);
	EXPECT_TRUE(std::isfinite(trainer->Metrics().TotalLoss));
}

TEST_VK(TestRl, GaeMatchesCpuAndSeparatesTerminationFromTruncation) {
	constexpr OaU32 time = 4;
	constexpr OaU32 environments = 2;
	const std::vector<OaF32> reward = {
		1.0F, 0.5F,
		0.2F, 1.0F,
		2.0F, 0.3F,
		0.7F, 1.5F,
	};
	const std::vector<OaF32> value = {
		0.4F, 0.1F,
		0.3F, 0.5F,
		0.8F, 0.2F,
		0.6F, 0.9F,
	};
	const std::vector<OaF32> nextValue = {
		0.3F, 0.5F,
		99.0F, 4.0F,
		0.6F, 0.9F,
		0.2F, 0.4F,
	};
	const std::vector<OaU8> terminated = {
		0, 0,
		1, 0,
		0, 0,
		0, 0,
	};
	const std::vector<OaU8> truncated = {
		0, 0,
		0, 1,
		0, 0,
		0, 0,
	};
	const OaGaeConfig config{.Gamma = 0.9F, .Lambda = 0.8F};
	auto result = OaFnRl::Gae(
		MatrixF32(reward, {time, environments}),
		MatrixF32(value, {time, environments}),
		MatrixF32(nextValue, {time, environments}),
		MatrixU8(terminated, {time, environments}),
		MatrixU8(truncated, {time, environments}),
		config);
	ASSERT_TRUE(result.IsValid());
	Sync();

	std::vector<OaF32> expectedAdvantage(time * environments);
	std::vector<OaF32> expectedReturn(time * environments);
	for (OaU32 environment = 0; environment < environments; ++environment) {
		OaF32 nextAdvantage = 0.0F;
		for (OaU32 reverseTime = time; reverseTime > 0; --reverseTime) {
			const OaU32 index = (reverseTime - 1U) * environments + environment;
			const OaF32 bootstrapMask = terminated[index] != 0 ? 0.0F : 1.0F;
			const OaF32 traceMask = terminated[index] != 0 || truncated[index] != 0
				? 0.0F : 1.0F;
			const OaF32 delta = reward[index]
				+ config.Gamma * bootstrapMask * nextValue[index]
				- value[index];
			const OaF32 advantage = delta
				+ config.Gamma * config.Lambda * traceMask * nextAdvantage;
			expectedAdvantage[index] = advantage;
			expectedReturn[index] = advantage + value[index];
			nextAdvantage = advantage;
		}
	}
	const auto advantage = CopyF32(result.Advantage);
	const auto returns = CopyF32(result.Return);
	for (OaUsize index = 0; index < advantage.size(); ++index) {
		EXPECT_NEAR(advantage[index], expectedAdvantage[index], 1.0e-6F)
			<< "advantage " << index;
		EXPECT_NEAR(returns[index], expectedReturn[index], 1.0e-6F)
			<< "return " << index;
	}
	// The large next value at the terminated transition must be ignored, while
	// the truncated transition must still bootstrap from its next value.
	EXPECT_LT(std::abs(advantage[2]), 10.0F);
	EXPECT_GT(advantage[3], 3.0F);
}

TEST_VK(TestRl, PpoClippedPolicyAndGradientMatchCpu) {
	const std::vector<OaF32> ratio = {1.3F, 0.5F, 1.1F, 0.9F};
	std::vector<OaF32> newLogProbability;
	newLogProbability.reserve(ratio.size());
	for (const OaF32 value : ratio) newLogProbability.push_back(std::log(value));
	const std::vector<OaF32> oldLogProbability(4, 0.0F);
	const std::vector<OaF32> advantage = {1.0F, -1.0F, 2.0F, -2.0F};
	auto newLog = MatrixF32(newLogProbability, {4});
	auto oldLog = MatrixF32(oldLogProbability, {4});
	auto adv = MatrixF32(advantage, {4});
	auto loss = OaFnLoss::PpoClippedPolicy(newLog, oldLog, adv, 0.2F);
	auto gradient = OaFnLoss::PpoClippedPolicyBwd(
		newLog, oldLog, adv, 0.2F);
	EXPECT_STREQ(OaFnLoss::LastName(), "ppo_clipped_policy");
	Sync();

	const auto lossHost = CopyF32(loss);
	ASSERT_EQ(lossHost.size(), 1U);
	EXPECT_NEAR(lossHost[0], -0.2F, 1.0e-6F);
	const auto gradientHost = CopyF32(gradient);
	const OaF32 expectedGradient[] = {0.0F, 0.0F, -0.55F, 0.45F};
	for (OaUsize index = 0; index < gradientHost.size(); ++index) {
		EXPECT_NEAR(gradientHost[index], expectedGradient[index], 1.0e-6F)
			<< "gradient " << index;
	}
}

TEST_VK(TestRl, NormalizeAdvantagesProducesFiniteStandardScores) {
	auto normalized = OaFnRl::NormalizeAdvantages(
		MatrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {4}));
	auto constant = OaFnRl::NormalizeAdvantages(
		MatrixF32({7.0F, 7.0F, 7.0F, 7.0F}, {4}));
	Sync();

	const auto values = CopyF32(normalized);
	OaF32 mean = 0.0F;
	OaF32 variance = 0.0F;
	for (const OaF32 value : values) mean += value;
	mean /= static_cast<OaF32>(values.size());
	for (const OaF32 value : values) {
		variance += (value - mean) * (value - mean);
	}
	variance /= static_cast<OaF32>(values.size());
	EXPECT_NEAR(mean, 0.0F, 1.0e-6F);
	EXPECT_NEAR(variance, 1.0F, 1.0e-5F);

	for (const OaF32 value : CopyF32(constant)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
}

TEST_VK(TestRl, ClippedPolicyAutogradUsesTheSameGradient) {
	const std::vector<OaF32> newLogValues = {
		std::log(1.3F), std::log(0.5F), std::log(1.1F), std::log(0.9F)};
	auto newLog = MatrixF32(newLogValues, {4});
	auto oldLog = MatrixF32(std::vector<OaF32>(4, 0.0F), {4});
	auto advantage = MatrixF32({1.0F, -1.0F, 2.0F, -2.0F}, {4});
	newLog.SetRequiresGrad(true);
	OaGradientTape tape;
	auto loss = OaFnLoss::PpoClippedPolicy(
		newLog, oldLog, advantage, 0.2F);
	tape.Backward(loss);
	Sync();
	const auto gradient = CopyF32(newLog.GradMatrix());
	const OaF32 expected[] = {0.0F, 0.0F, -0.55F, 0.45F};
	for (OaUsize index = 0; index < gradient.size(); ++index) {
		EXPECT_NEAR(gradient[index], expected[index], 1.0e-6F)
			<< "autograd " << index;
	}
}

TEST_VK(TestRl, PpoComposesPolicyValueAndEntropyTerms) {
	const auto newLog = MatrixF32({0.0F, 0.0F}, {2});
	const auto oldLog = MatrixF32({0.0F, 0.0F}, {2});
	const auto advantage = MatrixF32({1.0F, -1.0F}, {2});
	const auto value = MatrixF32({1.0F, 3.0F}, {2});
	const auto targetReturn = MatrixF32({2.0F, 1.0F}, {2});
	const auto entropy = MatrixF32({0.6F, 0.8F}, {2});
	const OaPpoLossConfig config{
		.ClipEpsilon = 0.2F,
		.ValueCoefficient = 0.5F,
		.EntropyCoefficient = 0.01F,
	};
	const auto result = OaFnLoss::Ppo(
		newLog, oldLog, advantage, value, targetReturn, entropy, config);
	ASSERT_TRUE(result.IsValid());
	EXPECT_STREQ(OaFnLoss::LastName(), "ppo");
	Sync();
	EXPECT_NEAR(CopyF32(result.PolicyLoss)[0], 0.0F, 1.0e-6F);
	EXPECT_NEAR(CopyF32(result.ValueLoss)[0], 2.5F, 1.0e-6F);
	EXPECT_NEAR(CopyF32(result.Entropy)[0], 0.7F, 1.0e-6F);
	EXPECT_NEAR(CopyF32(result.TotalLoss)[0], 1.243F, 1.0e-6F);
}
