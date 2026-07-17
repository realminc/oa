#include "../../OaTest.h"

#include "CartPole.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <vector>

namespace {

class TestCartPole : public ::testing::Test {};

struct CpuStep {
	OaF32 State[4]{};
	OaF32 Reward = 0.0F;
	OaU8 Terminated = 0;
	OaU8 Truncated = 0;
};

OaMatrix MatrixI32(const std::vector<OaI32>& InValues) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaI32)),
		{static_cast<OaI64>(InValues.size())}, OaScalarType::Int32);
}

void Sync() {
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
}

template<typename T>
std::vector<T> Copy(const OaMatrix& InMatrix) {
	std::vector<T> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(T)).IsOk());
	return result;
}

CpuStep StepCpu(
	const OaF32* InState,
	OaI32 InAction,
	OaU32 InEpisodeSteps,
	const OaTutorialCartPoleConfig& InConfig) {
	CpuStep result;
	OaF32 x = InState[0];
	OaF32 xVelocity = InState[1];
	OaF32 angle = InState[2];
	OaF32 angleVelocity = InState[3];
	const bool validAction = InAction == 0 || InAction == 1;
	const OaF32 force = InAction == 1
		? InConfig.ForceMagnitude : -InConfig.ForceMagnitude;
	const OaF32 totalMass = InConfig.CartMass + InConfig.PoleMass;
	const OaF32 poleMassLength = InConfig.PoleMass * InConfig.HalfPoleLength;
	const OaF32 cosine = std::cos(angle);
	const OaF32 sine = std::sin(angle);
	const OaF32 temporary = (force
		+ poleMassLength * angleVelocity * angleVelocity * sine) / totalMass;
	const OaF32 angleAcceleration = (InConfig.Gravity * sine - cosine * temporary)
		/ (InConfig.HalfPoleLength
			* (4.0F / 3.0F - InConfig.PoleMass * cosine * cosine / totalMass));
	const OaF32 xAcceleration = temporary
		- poleMassLength * angleAcceleration * cosine / totalMass;
	x += InConfig.TimeStep * xVelocity;
	xVelocity += InConfig.TimeStep * xAcceleration;
	angle += InConfig.TimeStep * angleVelocity;
	angleVelocity += InConfig.TimeStep * angleAcceleration;
	result.State[0] = x;
	result.State[1] = xVelocity;
	result.State[2] = angle;
	result.State[3] = angleVelocity;
	const bool invalidState = !std::isfinite(x) || !std::isfinite(xVelocity)
		|| !std::isfinite(angle) || !std::isfinite(angleVelocity);
	result.Terminated = static_cast<OaU8>(!validAction || invalidState
		|| std::abs(x) > InConfig.PositionThreshold
		|| std::abs(angle) > InConfig.AngleThresholdRadians);
	result.Truncated = static_cast<OaU8>(!result.Terminated
		&& InEpisodeSteps + 1U >= InConfig.MaxEpisodeSteps);
	result.Reward = validAction ? 1.0F : 0.0F;
	return result;
}

} // namespace

TEST_VK(TestCartPole, GpuStepMatchesScalarCpuOracle) {
	const OaTutorialCartPoleConfig config{
		.Environments = 4,
		.MaxEpisodeSteps = 500,
		.Seed = 0x123456789abcdef0ULL,
	};
	auto created = OaTutorialCartPole::Create(config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	Sync();
	const auto initial = Copy<OaF32>(environment.Observation());
	const std::vector<OaI32> action = {0, 1, 1, 0};
	const auto transition = environment.Step(MatrixI32(action));
	ASSERT_TRUE(transition.IsOk()) << transition.GetStatus().ToString();
	Sync();

	EXPECT_EQ(Copy<OaF32>(transition->Observation), initial);
	const auto actualState = Copy<OaF32>(transition->NextObservation);
	const auto actualReward = Copy<OaF32>(transition->Reward);
	const auto actualTerminated = Copy<OaU8>(transition->Terminated);
	const auto actualTruncated = Copy<OaU8>(transition->Truncated);
	const auto actualDone = Copy<OaU8>(transition->Done);
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{1, 1, 1, 1}));
	for (OaU32 lane = 0; lane < config.Environments; ++lane) {
		const CpuStep expected = StepCpu(
			initial.data() + lane * 4U, action[lane], 0, config);
		for (OaU32 component = 0; component < 4; ++component) {
			EXPECT_NEAR(actualState[lane * 4U + component],
				expected.State[component], 1.0e-5F)
				<< "lane=" << lane << " component=" << component;
		}
		EXPECT_EQ(actualReward[lane], expected.Reward);
		EXPECT_EQ(actualTerminated[lane], expected.Terminated);
		EXPECT_EQ(actualTruncated[lane], expected.Truncated);
		EXPECT_EQ(actualDone[lane],
			static_cast<OaU8>(expected.Terminated || expected.Truncated));
	}
}

TEST_VK(TestCartPole, ReusableCollectorOwnsEnvironmentPolicyExchange) {
	constexpr OaU32 environments = 8;
	constexpr OaU32 horizon = 16;
	auto environmentResult = OaTutorialCartPole::Create({
		.Environments = environments,
		.MaxEpisodeSteps = 64,
		.Seed = 9917,
	});
	ASSERT_TRUE(environmentResult.IsOk());
	auto environment = OaStdMove(*environmentResult);
	auto modelResult = OaCategoricalActorCritic::Create({
		.ObservationSize = 4,
		.ActionCount = 2,
		.HiddenSize = 16,
	});
	ASSERT_TRUE(modelResult.IsOk());
	auto model = OaStdMove(*modelResult);
	auto rolloutResult = OaRlRolloutBuffer::Create({
		.Time = horizon,
		.Environments = environments,
		.ObservationShape = {4},
	});
	ASSERT_TRUE(rolloutResult.IsOk());
	auto rollout = OaStdMove(*rolloutResult);
	auto collectorResult = OaRlCollector::Create(environment, *model, {
		.Horizon = horizon,
		.Seed = 1234,
		.Gae = {},
	});
	ASSERT_TRUE(collectorResult.IsOk());
	auto collector = OaStdMove(*collectorResult);
	ASSERT_TRUE(collector.Collect(rollout).IsOk());
	Sync();
	EXPECT_TRUE(rollout.IsFull());
	EXPECT_TRUE(rollout.IsFinalized());
	EXPECT_EQ(collector.Metrics().Collections, 1U);
	EXPECT_EQ(collector.Metrics().EnvironmentSteps, horizon);
	EXPECT_EQ(collector.Metrics().Transitions, environments * horizon);
	for (const OaU8 valid : Copy<OaU8>(rollout.Batch().Valid)) {
		EXPECT_EQ(valid, 1U);
	}
}

TEST_VK(TestCartPole, GenericEvaluatorReportsCompletedEpisodeMetrics) {
	auto environmentResult = OaTutorialCartPole::Create({
		.Environments = 8,
		.MaxEpisodeSteps = 32,
		.Seed = 717,
	});
	ASSERT_TRUE(environmentResult.IsOk());
	auto environment = OaStdMove(*environmentResult);
	auto modelResult = OaCategoricalActorCritic::Create({
		.ObservationSize = 4,
		.ActionCount = 2,
		.HiddenSize = 16,
	});
	ASSERT_TRUE(modelResult.IsOk());
	auto model = OaStdMove(*modelResult);
	auto metrics = OaFnRl::EvaluateCategorical(environment, *model, {
		.Horizon = 96,
		.Seed = 717,
	});
	ASSERT_TRUE(metrics.IsOk()) << metrics.GetStatus().ToString();
	EXPECT_EQ(metrics->EnvironmentSteps, 96U);
	EXPECT_EQ(metrics->Transitions, 96U * 8U);
	EXPECT_GT(metrics->CompletedEpisodes, 0U);
	EXPECT_GT(metrics->MeanCompletedReturn, 0.0F);
	EXPECT_LE(metrics->MinimumCompletedReturn,
		metrics->MaximumCompletedReturn);
}

TEST_VK(TestCartPole, ResetAndDoneLifecycleIsExplicitAndDeterministic) {
	const OaTutorialCartPoleConfig config{
		.Environments = 3,
		.MaxEpisodeSteps = 1,
		.Seed = 918273645ULL,
	};
	auto created = OaTutorialCartPole::Create(config);
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	Sync();
	const auto firstInitial = Copy<OaF32>(environment.Observation());

	environment.Reset();
	Sync();
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), firstInitial);
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{0, 0, 0}));

	ASSERT_TRUE(environment.Step(MatrixI32({0, 1, 0})).IsOk());
	Sync();
	EXPECT_EQ(Copy<OaU8>(environment.Done()),
		(std::vector<OaU8>{1, 1, 1}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{1, 1, 1}));
	const auto terminalState = Copy<OaF32>(environment.Observation());

	const auto ignored = environment.Step(MatrixI32({1, 1, 1}));
	ASSERT_TRUE(ignored.IsOk());
	Sync();
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), terminalState);
	EXPECT_EQ(Copy<OaF32>(ignored->Reward),
		(std::vector<OaF32>{0.0F, 0.0F, 0.0F}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{1, 1, 1}));

	environment.ResetDone();
	Sync();
	EXPECT_EQ(Copy<OaU8>(environment.Done()),
		(std::vector<OaU8>{0, 0, 0}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{0, 0, 0}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{1, 1, 1}));
	EXPECT_NE(Copy<OaF32>(environment.Observation()), firstInitial);

	environment.Reset();
	Sync();
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), firstInitial);
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{0, 0, 0}));
}

TEST_VK(TestCartPole, RejectsInvalidConfigurationAndActionShape) {
	EXPECT_TRUE(OaTutorialCartPole::Create(
		OaTutorialCartPoleConfig{.Environments = 0}).IsError());
	EXPECT_TRUE(OaTutorialCartPole::Create(
		OaTutorialCartPoleConfig{.TimeStep = 0.0F}).IsError());
	auto created = OaTutorialCartPole::Create(
		OaTutorialCartPoleConfig{.Environments = 2});
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	EXPECT_TRUE(environment.Step(MatrixI32({0})).IsError());
	EXPECT_TRUE(environment.Step(OaFnMatrix::Zeros(
		{2}, OaScalarType::Float32)).IsError());
}

TEST_VK(TestCartPole, PublishesReusableEnvironmentContract) {
	constexpr OaU32 environments = 3;
	auto created = OaTutorialCartPole::Create(
		OaTutorialCartPoleConfig{.Environments = environments, .Seed = 42});
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	const auto& spec = environment.Spec();
	ASSERT_TRUE(spec.ValidateDefinition().IsOk());
	EXPECT_EQ(spec.Action.Kind, OaRlSpaceKind::Discrete);
	EXPECT_EQ(spec.Action.Cardinality, 2);
	EXPECT_EQ(spec.Observation.Shape, (OaMatrixShape{4}));
	EXPECT_TRUE(spec.ValidateReset(
		environment.Observation(), environments).IsOk());

	const OaMatrix action = MatrixI32({0, 1, 0});
	const auto transition = environment.Step(action);
	ASSERT_TRUE(transition.IsOk());
	EXPECT_TRUE(spec.ValidateTransition(
		transition->Observation,
		action,
		transition->NextObservation,
		transition->Reward,
		transition->Terminated,
		transition->Truncated,
		environments).IsOk());
}

TEST_VK(TestCartPole, InvalidActionTerminatesOnlyItsLane) {
	auto created = OaTutorialCartPole::Create(
		OaTutorialCartPoleConfig{.Environments = 2, .Seed = 7});
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	const auto step = environment.Step(MatrixI32({7, 1}));
	ASSERT_TRUE(step.IsOk());
	Sync();
	EXPECT_EQ(Copy<OaF32>(step->Reward),
		(std::vector<OaF32>{0.0F, 1.0F}));
	EXPECT_EQ(Copy<OaU8>(step->Terminated),
		(std::vector<OaU8>{1, 0}));
	EXPECT_EQ(Copy<OaU8>(step->Done),
		(std::vector<OaU8>{1, 0}));
}
