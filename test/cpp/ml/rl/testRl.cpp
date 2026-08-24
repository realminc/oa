#include "../../oaTest.h"

#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/ml.h>
#include <oa/ml/trainingSession.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

static_assert(std::is_constructible_v<
	oa::ItRolloutTraining, oa::Engine&, oa::Optimizer&, const oa::ItRolloutTrainingConfig&>);
static_assert(!std::is_constructible_v<
	oa::ItRolloutTraining, oa::Optimizer&, const oa::ItRolloutTrainingConfig&>);

namespace {

class TestRl : public ::testing::Test {};

class TestLinearModule final : public oa::Module {
public:
	TestLinearModule(oa::I32 inInput, oa::I32 inOutput) {
		linear_ = oa::makeShared<oa::Linear>(inInput, inOutput);
		registerModule("linear", linear_);
	}
	oa::Matrix forward(const oa::Matrix& inInput) override {
		return linear_->forward(inInput);
	}
private:
	oa::SharedPtr<oa::Linear> linear_;
};

oa::Matrix matrixF32(const std::vector<oa::F32>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32)),
		inShape,
		oa::ScalarType::Float32);
}

oa::Matrix matrixU8(const std::vector<oa::U8>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(inValues.data(), inValues.size()),
		inShape,
		oa::ScalarType::UInt8);
}

oa::Matrix matrixI32(const std::vector<oa::I32>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::I32)),
		inShape,
		oa::ScalarType::Int32);
}

void syncDevice() {
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());
}

std::vector<oa::F32> copyF32(const oa::Matrix& inMatrix) {
	std::vector<oa::F32> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(oa::F32)).isOk());
	return result;
}

std::vector<oa::I32> copyI32(const oa::Matrix& inMatrix) {
	std::vector<oa::I32> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(oa::I32)).isOk());
	return result;
}

std::vector<oa::U8> copyU8(const oa::Matrix& inMatrix) {
	std::vector<oa::U8> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size()).isOk());
	return result;
}

std::vector<oa::U32> copyU32(const oa::Matrix& inMatrix) {
	std::vector<oa::U32> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(oa::U32)).isOk());
	return result;
}

} // namespace

TEST_VK(TestRl, EnvironmentFieldSpecsRejectInvalidDefinitions) {
	auto emptyName = oa::EnvironmentSpace::box("", {4});
	EXPECT_TRUE(emptyName.validateDefinition().isError());

	auto zeroDimension = oa::EnvironmentSpace::box("observation", {4, 0});
	EXPECT_TRUE(zeroDimension.validateDefinition().isError());

	auto integerBox = oa::EnvironmentSpace::box(
		"observation", {4}, oa::ScalarType::Int32);
	EXPECT_TRUE(integerBox.validateDefinition().isError());

	auto reversedBounds = oa::EnvironmentSpace::box(
		"observation", {4}, oa::ScalarType::Float32, 1.0, -1.0);
	EXPECT_TRUE(reversedBounds.validateDefinition().isError());

	auto noActions = oa::EnvironmentSpace::discrete("action", 0);
	EXPECT_TRUE(noActions.validateDefinition().isError());

	auto invalidBinary = oa::EnvironmentSpace::binary(
		"terminated", {}, oa::ScalarType::Float32);
	EXPECT_TRUE(invalidBinary.validateDefinition().isError());
}

TEST_VK(TestRl, EnvironmentSpecValidatesBatchedMatrixContract) {
	const oa::EnvironmentSpec spec{
		.observation = oa::EnvironmentSpace::box(
			"observation", {4}, oa::ScalarType::Float32),
		.action = oa::EnvironmentSpace::discrete("action", 2),
		.reward = oa::EnvironmentSpace::box(
			"reward", {}, oa::ScalarType::Float32, 0.0, 1.0),
		.terminated = oa::EnvironmentSpace::binary("terminated"),
		.truncated = oa::EnvironmentSpace::binary("truncated"),
	};
	ASSERT_TRUE(spec.validateDefinition().isOk());

	const auto observation = matrixF32(
		std::vector<oa::F32>(12, 0.0F), {3, 4});
	const auto action = matrixI32({0, 1, 0}, {3});
	const auto reward = matrixF32({1.0F, 1.0F, 1.0F}, {3});
	const auto boundary = matrixU8({0, 1, 0}, {3});
	EXPECT_TRUE(spec.validateReset(observation, 3).isOk());
	EXPECT_TRUE(spec.validateAction(action, 3).isOk());
	EXPECT_TRUE(spec.validateTransition(
		observation, action, observation, reward,
		boundary, boundary, 3).isOk());

	const oa::Status wrongObservation = spec.validateReset(
		matrixF32(std::vector<oa::F32>(9, 0.0F), {3, 3}), 3);
	EXPECT_EQ(wrongObservation.getCode(), oa::StatusCode::ShapeMismatch);
	const oa::Status wrongAction = spec.validateAction(
		matrixF32({0.0F, 1.0F, 0.0F}, {3}), 3);
	EXPECT_EQ(wrongAction.getCode(), oa::StatusCode::DtypeMismatch);
	const oa::Status wrongBoundary = spec.validateTransition(
		observation, action, observation, reward,
		matrixI32({0, 0, 0}, {3}), boundary, 3);
	EXPECT_EQ(wrongBoundary.getCode(), oa::StatusCode::DtypeMismatch);
	EXPECT_TRUE(spec.validateReset(observation, 0).isError());
}

TEST_VK(TestRl, RlTransformsComposeAsOrdinaryGpuMatrixOperations) {
	const auto observation = matrixF32({
		1.0F, 4.0F,
		5.0F, -8.0F,
	}, {2, 2});
	const auto mean = matrixF32({1.0F, 0.0F}, {2});
	const auto stddev = matrixF32({2.0F, 2.0F}, {2});
	const auto normalized = oa::FnEnvironment::normalizeObservation(
		observation, mean, stddev, 1.0e-6F, 3.0F);
	const auto scaled = oa::FnEnvironment::scaleAction(
		matrixF32({-2.0F, 0.0F, 2.0F}, {3}),
		-1.0F, 1.0F, 0.0F, 10.0F, true);
	const auto clipped = oa::FnEnvironment::clipReward(
		matrixF32({-3.0F, 0.25F, 4.0F}, {3}), -1.0F, 1.0F);
	ASSERT_FALSE(normalized.isEmpty());
	ASSERT_FALSE(scaled.isEmpty());
	ASSERT_FALSE(clipped.isEmpty());
	syncDevice();
	const auto actualNormalized = copyF32(normalized);
	EXPECT_NEAR(actualNormalized[0], 0.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[1], 2.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[2], 2.0F, 1.0e-5F);
	EXPECT_NEAR(actualNormalized[3], -3.0F, 1.0e-5F);
	EXPECT_EQ(copyF32(scaled), (std::vector<oa::F32>{0.0F, 5.0F, 10.0F}));
	EXPECT_EQ(copyF32(clipped), (std::vector<oa::F32>{-1.0F, 0.25F, 1.0F}));
}

TEST_VK(TestRl, CategoricalPolicyEvaluationMatchesCpu) {
	const std::vector<oa::F32> logits = {
		1.0F, 2.0F, -1.0F,
		0.5F, 0.5F, 0.5F,
	};
	const std::vector<oa::I32> action = {1, 0};
	const std::vector<oa::F32> value = {0.25F, -0.75F};
	const auto result = oa::FnPolicy::evaluateCategorical(
		matrixF32(logits, {2, 3}),
		matrixI32(action, {2}),
		matrixF32(value, {2}));
	ASSERT_TRUE(result.isValid());
	syncDevice();

	const auto actualLogProbability = copyF32(result.logProbability);
	const auto actualEntropy = copyF32(result.entropy);
	EXPECT_EQ(copyF32(result.value), value);
	EXPECT_EQ(copyI32(result.action), action);
	for (oa::U32 row = 0; row < 2; ++row) {
		oa::F32 maxLogit = logits[row * 3];
		for (oa::U32 column = 1; column < 3; ++column) {
			maxLogit = std::max(maxLogit, logits[row * 3 + column]);
		}
		oa::F32 normalizer = 0.0F;
		for (oa::U32 column = 0; column < 3; ++column) {
			normalizer += std::exp(logits[row * 3 + column] - maxLogit);
		}
		const oa::F32 logNormalizer = maxLogit + std::log(normalizer);
		oa::F32 expectedEntropy = 0.0F;
		for (oa::U32 column = 0; column < 3; ++column) {
			const oa::F32 logProbability =
				logits[row * 3 + column] - logNormalizer;
			expectedEntropy -= std::exp(logProbability) * logProbability;
		}
		EXPECT_NEAR(actualLogProbability[row],
			logits[row * 3 + static_cast<oa::U32>(action[row])]
				- logNormalizer,
			1.0e-6F);
		EXPECT_NEAR(actualEntropy[row], expectedEntropy, 1.0e-6F);
	}
}

TEST_VK(TestRl, CategoricalPolicyRemainsFiniteForConfidentLogits) {
	const auto result = oa::FnPolicy::evaluateCategorical(
		matrixF32({1000.0F, -1000.0F, 0.0F,
		           -1000.0F, 1000.0F, 0.0F}, {2, 3}),
		matrixI32({0, 1}, {2}),
		matrixF32({0.0F, 0.0F}, {2}));
	ASSERT_TRUE(result.isValid());
	syncDevice();
	for (const oa::F32 value : copyF32(result.logProbability)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
	for (const oa::F32 value : copyF32(result.entropy)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
}

TEST_VK(TestRl, CategoricalPolicySamplingIsSeededAndSelfConsistent) {
	const auto logits = matrixF32({
		2.0F, 1.0F, 0.0F,
		0.0F, 1.0F, 2.0F,
		1.0F, 1.0F, 1.0F,
		-1.0F, 3.0F, 0.5F,
	}, {4, 3});
	const auto value = matrixF32({0.0F, 1.0F, 2.0F, 3.0F}, {4});
	const auto first = oa::FnPolicy::sampleCategorical(logits, value, 918273);
	const auto second = oa::FnPolicy::sampleCategorical(logits, value, 918273);
	ASSERT_TRUE(first.isValid());
	ASSERT_TRUE(second.isValid());
	syncDevice();
	const auto firstAction = copyI32(first.action);
	EXPECT_EQ(firstAction, copyI32(second.action));
	for (const oa::I32 selected : firstAction) {
		EXPECT_GE(selected, 0);
		EXPECT_LT(selected, 3);
	}
	const auto reevaluated = oa::FnPolicy::evaluateCategorical(
		logits, first.action, value);
	syncDevice();
	const auto sampledLogProbability = copyF32(first.logProbability);
	const auto evaluatedLogProbability = copyF32(reevaluated.logProbability);
	for (oa::Usize index = 0; index < sampledLogProbability.size(); ++index) {
		EXPECT_NEAR(sampledLogProbability[index],
			evaluatedLogProbability[index], 1.0e-7F);
	}
}

TEST_VK(TestRl, CategoricalPolicyLogProbabilityAutogradMatchesCpu) {
	const std::vector<oa::F32> logitsHost = {
		1.0F, 2.0F, -1.0F,
		0.5F, -0.5F, 1.5F,
	};
	const std::vector<oa::I32> actionHost = {1, 0};
	auto logits = matrixF32(logitsHost, {2, 3});
	logits.setRequiresGrad(true);
	oa::GradientTape tape;
	const auto result = oa::FnPolicy::evaluateCategorical(
		logits, matrixI32(actionHost, {2}), matrixF32({0.0F, 0.0F}, {2}));
	const auto loss = oa::FnMatrix::neg(oa::FnMatrix::mean(result.logProbability));
	tape.backward(loss);
	syncDevice();
	const auto gradient = copyF32(logits.gradMatrix());
	for (oa::U32 row = 0; row < 2; ++row) {
		oa::F32 maxLogit = logitsHost[row * 3];
		for (oa::U32 column = 1; column < 3; ++column) {
			maxLogit = std::max(maxLogit, logitsHost[row * 3 + column]);
		}
		oa::F32 sum = 0.0F;
		for (oa::U32 column = 0; column < 3; ++column) {
			sum += std::exp(logitsHost[row * 3 + column] - maxLogit);
		}
		for (oa::U32 column = 0; column < 3; ++column) {
			const oa::F32 probability =
				std::exp(logitsHost[row * 3 + column] - maxLogit) / sum;
			const oa::F32 expected = (probability
				- (actionHost[row] == static_cast<oa::I32>(column) ? 1.0F : 0.0F))
				/ 2.0F;
			EXPECT_NEAR(gradient[row * 3 + column], expected, 2.0e-6F)
				<< "gradient row=" << row << " column=" << column;
		}
	}
}

TEST_VK(TestRl, TanhNormalPolicyIsSeededBoundedAndSelfConsistent) {
	const auto mean = matrixF32({0.0F, 0.5F, -0.5F, 1.0F}, {2, 2});
	const auto logStddev = matrixF32({-0.7F, -0.2F, 0.1F, -1.0F}, {2, 2});
	const auto value = matrixF32({0.25F, -0.75F}, {2});
	const auto first = oa::FnPolicy::sampleTanhNormal(
		mean, logStddev, value, -2.0F, 3.0F, 91723);
	const auto second = oa::FnPolicy::sampleTanhNormal(
		mean, logStddev, value, -2.0F, 3.0F, 91723);
	ASSERT_TRUE(first.isValid());
	ASSERT_TRUE(second.isValid());
	syncDevice();
	const auto action = copyF32(first.action);
	const auto secondAction = copyF32(second.action);
	ASSERT_EQ(action.size(), 4U);
	for (oa::Usize index = 0; index < action.size(); ++index) {
		EXPECT_NEAR(action[index], secondAction[index], 1.0e-7F);
		EXPECT_GT(action[index], -2.0F);
		EXPECT_LT(action[index], 3.0F);
	}
	const auto evaluated = oa::FnPolicy::evaluateTanhNormal(
		mean, logStddev, first.rawAction, value, -2.0F, 3.0F);
	ASSERT_TRUE(evaluated.isValid());
	syncDevice();
	const auto sampledLogProbability = copyF32(first.logProbability);
	const auto evaluatedLogProbability = copyF32(evaluated.logProbability);
	for (oa::Usize index = 0; index < sampledLogProbability.size(); ++index) {
		EXPECT_NEAR(sampledLogProbability[index],
			evaluatedLogProbability[index], 1.0e-6F);
		EXPECT_TRUE(std::isfinite(sampledLogProbability[index]));
	}
	for (const oa::F32 entropy : copyF32(first.entropy)) {
		EXPECT_TRUE(std::isfinite(entropy));
	}
}

TEST_VK(TestRl, TanhNormalPolicyLogProbabilityAutogradIsFinite) {
	auto mean = matrixF32({0.1F, -0.2F, 0.3F, -0.4F}, {2, 2});
	auto logStddev = matrixF32({-0.5F, -0.5F, -0.5F, -0.5F}, {2, 2});
	mean.setRequiresGrad(true);
	logStddev.setRequiresGrad(true);
	oa::GradientTape tape;
	const auto policy = oa::FnPolicy::evaluateTanhNormal(
		mean, logStddev,
		matrixF32({0.25F, -0.1F, 0.6F, -0.8F}, {2, 2}),
		matrixF32({0.0F, 0.0F}, {2}));
	ASSERT_TRUE(policy.isValid());
	tape.backward(oa::FnMatrix::neg(oa::FnMatrix::mean(policy.logProbability)));
	syncDevice();
	for (const oa::F32 gradient : copyF32(mean.gradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
	for (const oa::F32 gradient : copyF32(logStddev.gradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
}

TEST_VK(TestRl, ReplayBufferWrapsAndSamplesDeterministicallyOnGpu) {
	auto created = oa::ReplayBuffer::create(oa::ReplayConfig{
		.capacity = 4,
		.observationShape = {2},
		.actionShape = {},
		.actionDtype = oa::ScalarType::Int32,
	});
	ASSERT_TRUE(created.isOk());
	auto replay = oa::move(*created);
	auto append = [&](const std::vector<oa::F32>& observation,
		const std::vector<oa::I32>& action,
		const std::vector<oa::F32>& reward) {
		const oa::I64 batch = static_cast<oa::I64>(action.size());
		ASSERT_TRUE(replay.append(oa::ReplayTransition{
			.observation = matrixF32(observation, {batch, 2}),
			.action = matrixI32(action, {batch}),
			.nextObservation = matrixF32(observation, {batch, 2}),
			.reward = matrixF32(reward, {batch}),
			.terminated = matrixU8(std::vector<oa::U8>(action.size(), 0), {batch}),
			.truncated = matrixU8(std::vector<oa::U8>(action.size(), 0), {batch}),
		}).isOk());
	};
	append({0, 1, 2, 3, 4, 5}, {0, 1, 2}, {10, 11, 12});
	append({6, 7, 8, 9, 10, 11}, {3, 4, 5}, {13, 14, 15});
	EXPECT_EQ(replay.size(), 4U);
	EXPECT_EQ(replay.cursor(), 2U);
	ASSERT_TRUE(replay.isFull());
	auto first = replay.sample(8, 7123);
	auto second = replay.sample(8, 7123);
	ASSERT_TRUE(first.isOk());
	ASSERT_TRUE(second.isOk());
	syncDevice();
	const auto firstIndices = copyU32(first->index);
	EXPECT_EQ(firstIndices, copyU32(second->index));
	const auto rewards = copyF32(first->reward);
	for (oa::Usize index = 0; index < firstIndices.size(); ++index) {
		ASSERT_LT(firstIndices[index], 4U);
		// Physical ring slots contain rewards [14,15,12,13].
		const oa::F32 expected[] = {14.0F, 15.0F, 12.0F, 13.0F};
		EXPECT_EQ(rewards[index], expected[firstIndices[index]]);
	}
	replay.reset();
	EXPECT_EQ(replay.size(), 0U);
	EXPECT_TRUE(replay.sample(1, 1).isError());
}

TEST_VK(TestRl, DqnTargetBootstrapsTruncationButNotTermination) {
	auto q = matrixF32({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2});
	q.setRequiresGrad(true);
	oa::GradientTape tape;
	const auto result = oa::FnLoss::dqn(
		q,
		matrixI32({0, 1, 0}, {3}),
		matrixF32({1.0F, 2.0F, 3.0F}, {3}),
		matrixF32({10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, {3, 2}),
		matrixU8({0, 1, 0}, {3}),
		matrixU8({0, 0, 1}, {3}),
		oa::DqnLossConfig{.discount = 0.5F});
	ASSERT_TRUE(result.isValid());
	tape.backward(result.loss);
	syncDevice();
	const auto target = copyF32(result.targetQ);
	ASSERT_EQ(target.size(), 3U);
	EXPECT_NEAR(target[0], 11.0F, 1.0e-6F);
	EXPECT_NEAR(target[1], 2.0F, 1.0e-6F);
	EXPECT_NEAR(target[2], 33.0F, 1.0e-6F);
	const auto selected = copyF32(result.selectedQ);
	EXPECT_EQ(selected, (std::vector<oa::F32>{1.0F, 4.0F, 5.0F}));
	for (const oa::F32 gradient : copyF32(q.gradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}
}

TEST_VK(TestRl, DqnTrainerConsumesReplayAndUpdatesTarget) {
	auto replayResult = oa::ReplayBuffer::create({
		.capacity = 8,
		.observationShape = {2},
		.actionShape = {},
		.actionDtype = oa::ScalarType::Int32,
	});
	ASSERT_TRUE(replayResult.isOk());
	auto replay = oa::move(*replayResult);
	ASSERT_TRUE(replay.append({
		.observation = matrixF32({0, 0, 1, 0, 0, 1, 1, 1}, {4, 2}),
		.action = matrixI32({0, 1, 0, 1}, {4}),
		.nextObservation = matrixF32({1, 0, 0, 1, 1, 1, 0, 0}, {4, 2}),
		.reward = matrixF32({0, 1, 1, 0}, {4}),
		.terminated = matrixU8({0, 0, 1, 0}, {4}),
		.truncated = matrixU8({0, 0, 0, 1}, {4}),
	}).isOk());
	auto onlineResult = oa::CategoricalActorCritic::create({
		.observationSize = 2, .actionCount = 2, .hiddenSize = 8});
	auto targetResult = oa::CategoricalActorCritic::create({
		.observationSize = 2, .actionCount = 2, .hiddenSize = 8});
	ASSERT_TRUE(onlineResult.isOk());
	ASSERT_TRUE(targetResult.isOk());
	auto online = oa::move(*onlineResult);
	auto target = oa::move(*targetResult);
	auto parameters = online->allParameterPtrs();
	oa::Sgd optimizer(parameters, 1.0e-3F);
	auto trainerResult = oa::DqnTrainer::create(
		testEngine(), *online, *target, optimizer, replay, {
			.updates = 2,
			.batchSize = 4,
			.targetUpdateInterval = 1,
			.observationShape = {2},
			.seed = 77,
			.loss = {.discount = 0.95F},
		});
	ASSERT_TRUE(trainerResult.isOk());
	auto trainer = oa::move(*trainerResult);
	ASSERT_TRUE(trainer->update().isOk());
	EXPECT_TRUE(std::isfinite(trainer->metrics().loss));
	ASSERT_TRUE(trainer->update().isOk());
	EXPECT_TRUE(trainer->isDone());
	EXPECT_EQ(optimizer.getStep(), 2U);
	EXPECT_TRUE(std::isfinite(trainer->metrics().loss));
}

TEST_VK(TestRl, SacLossesMatchTerminationAndEntropyContracts) {
	auto q1 = matrixF32({1.0F, 2.0F, 3.0F}, {3});
	auto q2 = matrixF32({1.5F, 1.0F, 4.0F}, {3});
	q1.setRequiresGrad(true);
	q2.setRequiresGrad(true);
	oa::GradientTape criticTape;
	const auto critic = oa::FnLoss::sacCritic(
		q1, q2,
		matrixF32({1.0F, 2.0F, 3.0F}, {3}),
		matrixF32({10.0F, 30.0F, 50.0F}, {3}),
		matrixF32({20.0F, 40.0F, 60.0F}, {3}),
		matrixF32({-1.0F, -2.0F, -3.0F}, {3}),
		matrixU8({0, 1, 0}, {3}),
		matrixU8({0, 0, 1}, {3}),
		{.discount = 0.5F, .entropyCoefficient = 0.2F});
	ASSERT_TRUE(critic.isValid());
	criticTape.backward(critic.totalLoss);
	syncDevice();
	const auto target = copyF32(critic.targetQ);
	EXPECT_NEAR(target[0], 6.1F, 1.0e-5F);
	EXPECT_NEAR(target[1], 2.0F, 1.0e-5F);
	EXPECT_NEAR(target[2], 28.3F, 1.0e-5F);
	for (const oa::F32 gradient : copyF32(q1.gradMatrix())) {
		EXPECT_TRUE(std::isfinite(gradient));
	}

	auto actorQ1 = matrixF32({2.0F, 4.0F}, {2});
	auto actorQ2 = matrixF32({3.0F, 1.0F}, {2});
	auto logProbability = matrixF32({-0.5F, -1.0F}, {2});
	logProbability.setRequiresGrad(true);
	oa::GradientTape actorTape;
	const oa::Matrix actor = oa::FnLoss::sacActor(
		actorQ1, actorQ2, logProbability, 0.2F);
	actorTape.backward(actor);
	syncDevice();
	EXPECT_NEAR(actor.item(), -1.65F, 1.0e-6F);
	for (const oa::F32 gradient : copyF32(logProbability.gradMatrix())) {
		EXPECT_NEAR(gradient, 0.1F, 1.0e-6F);
	}
}

TEST_VK(TestRl, SacTrainerRunsActorTwinCriticReplayUpdate) {
	auto replayResult = oa::ReplayBuffer::create({
		.capacity = 8,
		.observationShape = {2},
		.actionShape = {1},
		.actionDtype = oa::ScalarType::Float32,
	});
	ASSERT_TRUE(replayResult.isOk());
	auto replay = oa::move(*replayResult);
	ASSERT_TRUE(replay.append({
		.observation = matrixF32({0, 0, 1, 0, 0, 1, 1, 1}, {4, 2}),
		.action = matrixF32({-0.5F, 0.25F, 0.75F, -0.25F}, {4, 1}),
		.nextObservation = matrixF32({1, 0, 0, 1, 1, 1, 0, 0}, {4, 2}),
		.reward = matrixF32({0, 1, 1, 0}, {4}),
		.terminated = matrixU8({0, 0, 1, 0}, {4}),
		.truncated = matrixU8({0, 0, 0, 1}, {4}),
	}).isOk());
	TestLinearModule actor(2, 2);
	TestLinearModule critic1(3, 1);
	TestLinearModule critic2(3, 1);
	TestLinearModule targetCritic1(3, 1);
	TestLinearModule targetCritic2(3, 1);
	auto actorParameters = actor.allParameterPtrs();
	auto criticParameters = critic1.allParameterPtrs();
	for (auto* parameter : critic2.allParameterPtrs()) {
		criticParameters.pushBack(parameter);
	}
	oa::Adam actorOptimizer(actorParameters, 1.0e-3F);
	oa::Adam criticOptimizer(criticParameters, 1.0e-3F);
	auto trainerResult = oa::SacTrainer::create(
		testEngine(), actor, critic1, critic2, targetCritic1, targetCritic2,
		actorOptimizer, criticOptimizer, replay, {
			.updates = 1,
			.batchSize = 4,
			.actionDimensions = 1,
			.targetUpdateInterval = 1,
			.observationShape = {2},
			.actionMinimum = -1.0F,
			.actionMaximum = 1.0F,
			.seed = 123,
			.loss = {.discount = 0.99F, .entropyCoefficient = 0.2F},
		});
	ASSERT_TRUE(trainerResult.isOk());
	auto trainer = oa::move(*trainerResult);
	ASSERT_TRUE(trainer->update().isOk());
	EXPECT_TRUE(trainer->isDone());
	EXPECT_EQ(actorOptimizer.getStep(), 1U);
	EXPECT_EQ(criticOptimizer.getStep(), 1U);
	EXPECT_TRUE(std::isfinite(trainer->metrics().actorLoss));
	EXPECT_TRUE(std::isfinite(trainer->metrics().criticLoss));
}

TEST_VK(TestRl, CategoricalPolicyEntropyAutogradMatchesCpu) {
	const std::vector<oa::F32> logitsHost = {1.0F, 2.0F, -1.0F};
	auto logits = matrixF32(logitsHost, {1, 3});
	logits.setRequiresGrad(true);
	oa::GradientTape tape;
	const auto result = oa::FnPolicy::evaluateCategorical(
		logits, matrixI32({1}, {1}), matrixF32({0.0F}, {1}));
	const auto loss = oa::FnMatrix::neg(oa::FnMatrix::mean(result.entropy));
	tape.backward(loss);
	syncDevice();

	oa::F32 maxLogit = *std::max_element(logitsHost.begin(), logitsHost.end());
	oa::F32 sum = 0.0F;
	for (const oa::F32 logit : logitsHost) sum += std::exp(logit - maxLogit);
	oa::F32 entropy = 0.0F;
	std::vector<oa::F32> probability;
	for (const oa::F32 logit : logitsHost) {
		const oa::F32 p = std::exp(logit - maxLogit) / sum;
		probability.push_back(p);
		entropy -= p * std::log(p);
	}
	const auto gradient = copyF32(logits.gradMatrix());
	for (oa::U32 column = 0; column < 3; ++column) {
		const oa::F32 expected = probability[column]
			* (std::log(probability[column]) + entropy);
		EXPECT_NEAR(gradient[column], expected, 2.0e-6F)
			<< "entropy gradient column=" << column;
	}
}

TEST_VK(TestRl, CategoricalPolicyRejectsInvalidShapes) {
	const auto logits = matrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
	EXPECT_FALSE(oa::FnPolicy::evaluateCategorical(
		logits, matrixI32({0}, {1}), matrixF32({0.0F, 0.0F}, {2})).isValid());
	EXPECT_FALSE(oa::FnPolicy::sampleCategorical(
		logits, matrixF32({0.0F}, {1}), 7).isValid());
}

TEST_VK(TestRl, RolloutBufferAppendsAndFinalizesOnGpu) {
	auto created = oa::RolloutBuffer::create(oa::RolloutConfig{
		.time = 3,
		.environments = 2,
		.observationShape = {2},
	});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto rollout = oa::move(*created);
	ASSERT_TRUE(rollout.isValid());

	const std::vector<std::vector<oa::F32>> observations = {
		{0.0F, 0.1F, 0.2F, 0.3F},
		{1.0F, 1.1F, 1.2F, 1.3F},
		{2.0F, 2.1F, 2.2F, 2.3F},
	};
	const std::vector<std::vector<oa::F32>> rewards = {
		{1.0F, 10.0F}, {2.0F, 20.0F}, {3.0F, 30.0F}};
	const std::vector<std::vector<oa::F32>> nextValues = {
		{0.0F, 0.0F}, {100.0F, 5.0F}, {0.0F, 0.0F}};
	const std::vector<std::vector<oa::U8>> terminated = {
		{0, 0}, {1, 0}, {0, 0}};
	const std::vector<std::vector<oa::U8>> truncated = {
		{0, 0}, {0, 1}, {0, 0}};
	for (oa::U32 step = 0; step < 3; ++step) {
		const oa::RolloutTransition transition{
			.observation = matrixF32(observations[step], {2, 2}),
			.action = matrixI32(
				{static_cast<oa::I32>(step % 2), static_cast<oa::I32>((step + 1) % 2)},
				{2}),
			.reward = matrixF32(rewards[step], {2}),
			.value = matrixF32({0.0F, 0.0F}, {2}),
			.nextValue = matrixF32(nextValues[step], {2}),
			.logProbability = matrixF32(
				{-0.1F * static_cast<oa::F32>(step + 1),
				 -0.2F * static_cast<oa::F32>(step + 1)}, {2}),
			.terminated = matrixU8(terminated[step], {2}),
			.truncated = matrixU8(truncated[step], {2}),
		};
		ASSERT_TRUE(rollout.append(transition).isOk());
	}
	ASSERT_TRUE(rollout.isFull());
	ASSERT_TRUE(rollout.finalize(oa::GaeConfig{
		.gamma = 1.0F, .lambda = 1.0F}).isOk());
	ASSERT_TRUE(rollout.isFinalized());
	syncDevice();

	const auto& batch = rollout.batch();
	EXPECT_EQ(copyF32(batch.observation),
		(std::vector<oa::F32>{
			0.0F, 0.1F, 0.2F, 0.3F,
			1.0F, 1.1F, 1.2F, 1.3F,
			2.0F, 2.1F, 2.2F, 2.3F}));
	EXPECT_EQ(copyI32(batch.action),
		(std::vector<oa::I32>{0, 1, 1, 0, 0, 1}));
	EXPECT_EQ(copyF32(batch.reward),
		(std::vector<oa::F32>{1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F}));
	EXPECT_EQ(copyU8(batch.terminated),
		(std::vector<oa::U8>{0, 0, 1, 0, 0, 0}));
	EXPECT_EQ(copyU8(batch.truncated),
		(std::vector<oa::U8>{0, 0, 0, 1, 0, 0}));
	EXPECT_EQ(copyU8(batch.valid),
		(std::vector<oa::U8>{1, 1, 1, 1, 1, 1}));
	const std::vector<oa::F32> expectedAdvantage = {
		3.0F, 35.0F, 2.0F, 25.0F, 3.0F, 30.0F};
	const auto advantage = copyF32(batch.advantage);
	const auto returns = copyF32(batch.ret);
	for (oa::Usize index = 0; index < expectedAdvantage.size(); ++index) {
		EXPECT_NEAR(advantage[index], expectedAdvantage[index], 1.0e-6F);
		EXPECT_NEAR(returns[index], expectedAdvantage[index], 1.0e-6F);
	}
}

TEST_VK(TestRl, RolloutBufferEnforcesLifecycleAndReusesStorage) {
	EXPECT_TRUE(oa::RolloutBuffer::create(oa::RolloutConfig{}).isError());
	auto created = oa::RolloutBuffer::create(oa::RolloutConfig{
		.time = 2,
		.environments = 1,
		.observationShape = {4},
	});
	ASSERT_TRUE(created.isOk());
	auto rollout = oa::move(*created);
	const auto observationBuffer =
		oa::MatrixAccess::descriptor(rollout.batch().observation).buffer;
	EXPECT_TRUE(rollout.finalize().isError());

	oa::RolloutTransition transition{
		.observation = matrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {1, 4}),
		.action = matrixI32({1}, {1}),
		.reward = matrixF32({1.0F}, {1}),
		.value = matrixF32({0.0F}, {1}),
		.nextValue = matrixF32({0.0F}, {1}),
		.logProbability = matrixF32({-0.5F}, {1}),
		.terminated = matrixU8({0}, {1}),
		.truncated = matrixU8({0}, {1}),
	};
	oa::RolloutTransition invalid = transition;
	invalid.observation = matrixF32({1.0F, 2.0F}, {1, 2});
	EXPECT_TRUE(rollout.append(invalid).isError());
	EXPECT_EQ(rollout.size(), 0U);
	ASSERT_TRUE(rollout.append(transition).isOk());
	ASSERT_TRUE(rollout.append(transition).isOk());
	EXPECT_TRUE(rollout.append(transition).isError());
	ASSERT_TRUE(rollout.finalize().isOk());
	EXPECT_TRUE(rollout.append(transition).isError());

	rollout.reset();
	EXPECT_EQ(rollout.size(), 0U);
	EXPECT_FALSE(rollout.isFinalized());
	EXPECT_EQ(oa::MatrixAccess::descriptor(rollout.batch().observation).buffer,
		observationBuffer);
	ASSERT_TRUE(rollout.append(transition).isOk());
	syncDevice();
	EXPECT_EQ(copyU8(rollout.batch().valid),
		(std::vector<oa::U8>{1, 0}));
}

TEST_VK(TestRl, RlTrainingCoordinatorEnforcesCollectUpdateLifecycle) {
	oa::OptimizerNoOp optimizer;
	oa::ItRolloutTraining invalid(testEngine(), optimizer, oa::ItRolloutTrainingConfig{});
	EXPECT_FALSE(invalid.isValid());
	EXPECT_TRUE(invalid.isDone());

	auto created = oa::RolloutBuffer::create(oa::RolloutConfig{
		.time = 1,
		.environments = 1,
		.observationShape = {2},
	});
	ASSERT_TRUE(created.isOk());
	auto rollout = oa::move(*created);
	oa::ItRolloutTraining training(testEngine(), optimizer, oa::ItRolloutTrainingConfig{
		.rollouts = 1,
		.horizon = 1,
		.environments = 1,
		.updateEpochs = 2,
	});
	ASSERT_TRUE(training.isValid());
	EXPECT_FALSE(training.beginUpdate());
	ASSERT_TRUE(training.beginRollout(rollout).isOk());
	ASSERT_TRUE(rollout.append(oa::RolloutTransition{
		.observation = matrixF32({1.0F, 2.0F}, {1, 2}),
		.action = matrixI32({0}, {1}),
		.reward = matrixF32({1.0F}, {1}),
		.value = matrixF32({0.0F}, {1}),
		.nextValue = matrixF32({0.0F}, {1}),
		.logProbability = matrixF32({-0.5F}, {1}),
		.terminated = matrixU8({0}, {1}),
		.truncated = matrixU8({0}, {1}),
	}).isOk());
	ASSERT_TRUE(training.finalizeRollout(rollout).isOk());
	EXPECT_EQ(training.phase(), oa::RolloutTrainingPhase::Update);

	const oa::Matrix loss = matrixF32({0.25F}, {1});
	oa::TrainingSession session(training.updateLoop());
	ASSERT_TRUE(session.pause().isOk());
	EXPECT_FALSE(training.beginUpdate());
	EXPECT_EQ(session.state(), oa::TrainingState::Paused);
	EXPECT_EQ(training.updateLoop().stepCount(), 0);
	ASSERT_TRUE(session.resume().isOk());
	for (oa::U32 epoch = 0; epoch < 2; ++epoch) {
		ASSERT_TRUE(training.beginUpdate());
		EXPECT_FALSE(training.beginUpdate());
		ASSERT_TRUE(training.nextUpdate(loss).isOk());
	}
	EXPECT_TRUE(training.isDone());
	EXPECT_EQ(training.rolloutIndex(), 1U);
	EXPECT_EQ(training.updateEpoch(), 2U);
	EXPECT_TRUE(training.finish().isOk());
}

TEST_VK(TestRl, ReusablePpoTrainerOwnsEnvironmentNeutralLifecycle) {
	auto modelResult = oa::CategoricalActorCritic::create(
		oa::CategoricalActorCriticConfig{
			.observationSize = 2,
			.actionCount = 2,
			.hiddenSize = 8,
		});
	ASSERT_TRUE(modelResult.isOk());
	auto model = oa::move(*modelResult);
	oa::OptimizerNoOp optimizer;
	auto trainerResult = oa::PpoTrainer::create(
		testEngine(), *model, optimizer, oa::PpoTrainerConfig{
			.rollouts = 1,
			.horizon = 1,
			.environments = 2,
			.updateEpochs = 1,
			.observationShape = {2},
			.seed = 123,
		});
	ASSERT_TRUE(trainerResult.isOk());
	auto trainer = oa::move(*trainerResult);
	ASSERT_TRUE(trainer->isValid());
	ASSERT_TRUE(trainer->needsCollection());
	ASSERT_TRUE(trainer->beginCollection().isOk());

	const oa::Matrix observation = matrixF32(
		{0.1F, -0.2F, 0.3F, 0.4F}, {2, 2});
	const oa::PolicyResult policy = trainer->act(observation);
	ASSERT_TRUE(policy.isValid());
	EXPECT_EQ(policy.action.getShape(), oa::MatrixShape({2}));
	EXPECT_EQ(policy.value.getShape(), oa::MatrixShape({2}));
	ASSERT_TRUE(trainer->observe(
		observation,
		matrixF32({0.2F, -0.1F, 0.4F, 0.5F}, {2, 2}),
		matrixF32({1.0F, 0.5F}, {2}),
		matrixU8({0, 0}, {2}),
		matrixU8({0, 0}, {2}),
		policy).isOk());
	ASSERT_TRUE(trainer->endCollection().isOk());
	EXPECT_EQ(trainer->phase(), oa::RolloutTrainingPhase::Update);
	ASSERT_TRUE(trainer->update().isOk());
	EXPECT_TRUE(trainer->isDone());
	EXPECT_EQ(trainer->metrics().rollout, 1U);
	EXPECT_EQ(trainer->metrics().updateEpoch, 1U);
	EXPECT_TRUE(std::isfinite(trainer->metrics().totalLoss));
}

TEST_VK(TestRl, PpoCollectionAbortRestoresCollectPhase) {
	auto modelResult = oa::CategoricalActorCritic::create(
		oa::CategoricalActorCriticConfig{
			.observationSize = 2,
			.actionCount = 2,
			.hiddenSize = 8,
		});
	ASSERT_TRUE(modelResult.isOk());
	auto model = oa::move(*modelResult);
	oa::OptimizerNoOp optimizer;
	auto trainerResult = oa::PpoTrainer::create(
		testEngine(), *model, optimizer, oa::PpoTrainerConfig{
			.rollouts = 1,
			.horizon = 1,
			.environments = 2,
			.updateEpochs = 1,
			.observationShape = {2},
			.seed = 321,
		});
	ASSERT_TRUE(trainerResult.isOk());
	auto trainer = oa::move(*trainerResult);

	EXPECT_TRUE(trainer->abortCollection().isError());
	ASSERT_TRUE(trainer->beginCollection().isOk());
	EXPECT_TRUE(trainer->endCollection().isError());
	EXPECT_FALSE(trainer->isValid());
	ASSERT_TRUE(trainer->abortCollection().isOk());
	EXPECT_TRUE(trainer->isValid());
	EXPECT_TRUE(trainer->needsCollection());
	EXPECT_EQ(trainer->phase(), oa::RolloutTrainingPhase::Collect);

	ASSERT_TRUE(trainer->beginCollection().isOk());
	const oa::Matrix observation = matrixF32(
		{0.1F, -0.2F, 0.3F, 0.4F}, {2, 2});
	const oa::PolicyResult policy = trainer->act(observation);
	ASSERT_TRUE(policy.isValid());
	ASSERT_TRUE(trainer->observe(
		observation,
		matrixF32({0.2F, -0.1F, 0.4F, 0.5F}, {2, 2}),
		matrixF32({1.0F, 0.5F}, {2}),
		matrixU8({0, 0}, {2}),
		matrixU8({0, 0}, {2}),
		policy).isOk());
	ASSERT_TRUE(trainer->endCollection().isOk());
	EXPECT_EQ(trainer->phase(), oa::RolloutTrainingPhase::Update);

	ASSERT_TRUE(trainer->abortCollection().isOk());
	EXPECT_TRUE(trainer->needsCollection());
	EXPECT_EQ(trainer->phase(), oa::RolloutTrainingPhase::Collect);
	ASSERT_TRUE(trainer->beginCollection().isOk());
	EXPECT_TRUE(trainer->abortCollection().isOk());
}

TEST_VK(TestRl, GaeMatchesCpuAndSeparatesTerminationFromTruncation) {
	constexpr oa::U32 time = 4;
	constexpr oa::U32 environments = 2;
	const std::vector<oa::F32> reward = {
		1.0F, 0.5F,
		0.2F, 1.0F,
		2.0F, 0.3F,
		0.7F, 1.5F,
	};
	const std::vector<oa::F32> value = {
		0.4F, 0.1F,
		0.3F, 0.5F,
		0.8F, 0.2F,
		0.6F, 0.9F,
	};
	const std::vector<oa::F32> nextValue = {
		0.3F, 0.5F,
		99.0F, 4.0F,
		0.6F, 0.9F,
		0.2F, 0.4F,
	};
	const std::vector<oa::U8> terminated = {
		0, 0,
		1, 0,
		0, 0,
		0, 0,
	};
	const std::vector<oa::U8> truncated = {
		0, 0,
		0, 1,
		0, 0,
		0, 0,
	};
	const oa::GaeConfig config{.gamma = 0.9F, .lambda = 0.8F};
	auto result = oa::FnAdvantage::gae(
		matrixF32(reward, {time, environments}),
		matrixF32(value, {time, environments}),
		matrixF32(nextValue, {time, environments}),
		matrixU8(terminated, {time, environments}),
		matrixU8(truncated, {time, environments}),
		config);
	ASSERT_TRUE(result.isValid());
	syncDevice();

	std::vector<oa::F32> expectedAdvantage(time * environments);
	std::vector<oa::F32> expectedReturn(time * environments);
	for (oa::U32 environment = 0; environment < environments; ++environment) {
		oa::F32 nextAdvantage = 0.0F;
		for (oa::U32 reverseTime = time; reverseTime > 0; --reverseTime) {
			const oa::U32 index = (reverseTime - 1U) * environments + environment;
			const oa::F32 bootstrapMask = terminated[index] != 0 ? 0.0F : 1.0F;
			const oa::F32 traceMask = terminated[index] != 0 || truncated[index] != 0
				? 0.0F : 1.0F;
			const oa::F32 delta = reward[index]
				+ config.gamma * bootstrapMask * nextValue[index]
				- value[index];
			const oa::F32 advantage = delta
				+ config.gamma * config.lambda * traceMask * nextAdvantage;
			expectedAdvantage[index] = advantage;
			expectedReturn[index] = advantage + value[index];
			nextAdvantage = advantage;
		}
	}
	const auto advantage = copyF32(result.advantage);
	const auto returns = copyF32(result.ret);
	for (oa::Usize index = 0; index < advantage.size(); ++index) {
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
	const std::vector<oa::F32> ratio = {1.3F, 0.5F, 1.1F, 0.9F};
	std::vector<oa::F32> newLogProbability;
	newLogProbability.reserve(ratio.size());
	for (const oa::F32 value : ratio) newLogProbability.push_back(std::log(value));
	const std::vector<oa::F32> oldLogProbability(4, 0.0F);
	const std::vector<oa::F32> advantage = {1.0F, -1.0F, 2.0F, -2.0F};
	auto newLog = matrixF32(newLogProbability, {4});
	auto oldLog = matrixF32(oldLogProbability, {4});
	auto adv = matrixF32(advantage, {4});
	auto loss = oa::FnLoss::ppoClippedPolicy(newLog, oldLog, adv, 0.2F);
	auto gradient = oa::FnLoss::ppoClippedPolicyBwd(
		newLog, oldLog, adv, 0.2F);
	EXPECT_STREQ(oa::FnLoss::lastName(), "ppo_clipped_policy");
	syncDevice();

	const auto lossHost = copyF32(loss);
	ASSERT_EQ(lossHost.size(), 1U);
	EXPECT_NEAR(lossHost[0], -0.2F, 1.0e-6F);
	const auto gradientHost = copyF32(gradient);
	const oa::F32 expectedGradient[] = {0.0F, 0.0F, -0.55F, 0.45F};
	for (oa::Usize index = 0; index < gradientHost.size(); ++index) {
		EXPECT_NEAR(gradientHost[index], expectedGradient[index], 1.0e-6F)
			<< "gradient " << index;
	}
}

TEST_VK(TestRl, NormalizeAdvantagesProducesFiniteStandardScores) {
	auto normalized = oa::FnAdvantage::normalize(
		matrixF32({1.0F, 2.0F, 3.0F, 4.0F}, {4}));
	auto constant = oa::FnAdvantage::normalize(
		matrixF32({7.0F, 7.0F, 7.0F, 7.0F}, {4}));
	syncDevice();

	const auto values = copyF32(normalized);
	oa::F32 mean = 0.0F;
	oa::F32 variance = 0.0F;
	for (const oa::F32 value : values) mean += value;
	mean /= static_cast<oa::F32>(values.size());
	for (const oa::F32 value : values) {
		variance += (value - mean) * (value - mean);
	}
	variance /= static_cast<oa::F32>(values.size());
	EXPECT_NEAR(mean, 0.0F, 1.0e-6F);
	EXPECT_NEAR(variance, 1.0F, 1.0e-5F);

	for (const oa::F32 value : copyF32(constant)) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_NEAR(value, 0.0F, 1.0e-6F);
	}
}

TEST_VK(TestRl, ClippedPolicyAutogradUsesTheSameGradient) {
	const std::vector<oa::F32> newLogValues = {
		std::log(1.3F), std::log(0.5F), std::log(1.1F), std::log(0.9F)};
	auto newLog = matrixF32(newLogValues, {4});
	auto oldLog = matrixF32(std::vector<oa::F32>(4, 0.0F), {4});
	auto advantage = matrixF32({1.0F, -1.0F, 2.0F, -2.0F}, {4});
	newLog.setRequiresGrad(true);
	oa::GradientTape tape;
	auto loss = oa::FnLoss::ppoClippedPolicy(
		newLog, oldLog, advantage, 0.2F);
	tape.backward(loss);
	syncDevice();
	const auto gradient = copyF32(newLog.gradMatrix());
	const oa::F32 expected[] = {0.0F, 0.0F, -0.55F, 0.45F};
	for (oa::Usize index = 0; index < gradient.size(); ++index) {
		EXPECT_NEAR(gradient[index], expected[index], 1.0e-6F)
			<< "autograd " << index;
	}
}

TEST_VK(TestRl, PpoComposesPolicyValueAndEntropyTerms) {
	const auto newLog = matrixF32({0.0F, 0.0F}, {2});
	const auto oldLog = matrixF32({0.0F, 0.0F}, {2});
	const auto advantage = matrixF32({1.0F, -1.0F}, {2});
	const auto value = matrixF32({1.0F, 3.0F}, {2});
	const auto targetReturn = matrixF32({2.0F, 1.0F}, {2});
	const auto entropy = matrixF32({0.6F, 0.8F}, {2});
	const oa::PpoLossConfig config{
		.clipEpsilon = 0.2F,
		.valueCoefficient = 0.5F,
		.entropyCoefficient = 0.01F,
	};
	const auto result = oa::FnLoss::ppo(
		newLog, oldLog, advantage, value, targetReturn, entropy, config);
	ASSERT_TRUE(result.isValid());
	EXPECT_STREQ(oa::FnLoss::lastName(), "ppo");
	syncDevice();
	EXPECT_NEAR(copyF32(result.policyLoss)[0], 0.0F, 1.0e-6F);
	EXPECT_NEAR(copyF32(result.valueLoss)[0], 2.5F, 1.0e-6F);
	EXPECT_NEAR(copyF32(result.entropy)[0], 0.7F, 1.0e-6F);
	EXPECT_NEAR(copyF32(result.totalLoss)[0], 1.243F, 1.0e-6F);
}
