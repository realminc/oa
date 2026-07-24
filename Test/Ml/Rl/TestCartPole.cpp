#include "../../OaTest.h"

#include "CartPole.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Operation.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/SemanticGraph.h>

#include "../../../Source/Private/Oa/Core/OperationRegistry.gen.h"
#include "../../../Source/Private/Oa/Ml/Rl/EnvironmentExecution.h"

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

OaEngine& Runtime() {
	return *OaEngine::GetGlobal();
}

void SyncDefault() {
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
}

OaStatus SubmitAndWait(OaRlEnvironment& InEnvironment) {
	auto completion = InEnvironment.Submit();
	if (completion.IsError()) return completion.GetStatus();
	return InEnvironment.Wait(*completion);
}

class DefaultContextGuard {
public:
	DefaultContextGuard() : Previous_(OaContext::GetDefaultPtr()) {}
	~DefaultContextGuard() { OaContext::SetDefault(Previous_); }

private:
	OaContext* Previous_ = nullptr;
};

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
	auto created = OaTutorialCartPole::Create(Runtime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	const auto initial = Copy<OaF32>(environment.Observation());
	const std::vector<OaI32> action = {0, 1, 1, 0};
	const auto transition = environment.Step(MatrixI32(action));
	ASSERT_TRUE(transition.IsOk()) << transition.GetStatus().ToString();
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());

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
	auto environmentResult = OaTutorialCartPole::Create(Runtime(), {
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
	SyncDefault();
	auto completion = collector.Collect(rollout);
	ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
	ASSERT_TRUE(environment.Wait(*completion).IsOk());
	EXPECT_EQ(environment.SubmissionCount(), 1U);
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
	auto environmentResult = OaTutorialCartPole::Create(Runtime(), {
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
	SyncDefault();
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

TEST_VK(TestCartPole,
	GenericEvaluatorUsesBorrowedNonGlobalEngineWithoutAmbientMixing) {
	OaEngine* globalEngine = OaEngine::GetGlobal();
	ASSERT_NE(globalEngine, nullptr);
	ASSERT_TRUE(globalEngine->IsReady());
	OaContext* ambientContext = OaContext::GetDefaultPtr();
	ASSERT_NE(ambientContext, nullptr);

	OaEngineConfig engineConfig = OaTestEngineConfig(OaPrecision::FP32);
	engineConfig.RegisterAsGlobal = false;
	engineConfig.PreloadEmbeddedPipelines = false;
	auto engineResult = OaEngine::Create(engineConfig);
	ASSERT_TRUE(engineResult.IsOk())
		<< engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	ASSERT_TRUE(engine->IsReady());
	ASSERT_NE(engine.get(), globalEngine);

	OaUniquePtr<OaContext> modelContext;
	modelContext.Reset(OaContext::Create(engine.get()));
	ASSERT_TRUE(modelContext);
	OaUniquePtr<OaCategoricalActorCritic> model;
	{
		OaContext::RecordingScope scope(*modelContext);
		auto modelResult = OaCategoricalActorCritic::Create({
			.ObservationSize = 4,
			.ActionCount = 2,
			.HiddenSize = 16,
		});
		ASSERT_TRUE(modelResult.IsOk())
			<< modelResult.GetStatus().ToString();
		model = OaStdMove(*modelResult);
		ASSERT_TRUE(modelContext->Execute().IsOk());
		ASSERT_TRUE(modelContext->Sync().IsOk());
	}
	modelContext.Reset();
	ASSERT_TRUE(model);
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);

	auto environmentResult = OaTutorialCartPole::Create(*engine, {
		.Environments = 4,
		.MaxEpisodeSteps = 16,
		.Seed = 0xa1b2c3d4e5f60718ULL,
	});
	ASSERT_TRUE(environmentResult.IsOk())
		<< environmentResult.GetStatus().ToString();
	auto environment = OaStdMove(*environmentResult);
	auto metrics = OaFnRl::EvaluateCategorical(environment, *model, {
		.Horizon = 48,
		.Seed = 0xa1b2c3d4e5f60718ULL,
	});
	ASSERT_TRUE(metrics.IsOk()) << metrics.GetStatus().ToString();
	EXPECT_EQ(metrics->EnvironmentSteps, 48U);
	EXPECT_EQ(metrics->Transitions, 48U * 4U);
	EXPECT_GT(metrics->CompletedEpisodes, 0U);
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);

	EXPECT_TRUE(environment.Close().IsOk());
	model.Reset();
	const OaStatus engineClosed = engine->Close();
	EXPECT_TRUE(engineClosed.IsOk()) << engineClosed.ToString();
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);
}

TEST_VK(TestCartPole, ResetAndDoneLifecycleIsExplicitAndDeterministic) {
	const OaTutorialCartPoleConfig config{
		.Environments = 3,
		.MaxEpisodeSteps = 1,
		.Seed = 918273645ULL,
	};
	auto created = OaTutorialCartPole::Create(Runtime(), config);
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	const auto firstInitial = Copy<OaF32>(environment.Observation());

	ASSERT_TRUE(environment.Reset().IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), firstInitial);
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{0, 0, 0}));

	ASSERT_TRUE(environment.Step(MatrixI32({0, 1, 0})).IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaU8>(environment.Done()),
		(std::vector<OaU8>{1, 1, 1}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{1, 1, 1}));
	const auto terminalState = Copy<OaF32>(environment.Observation());

	const auto ignored = environment.Step(MatrixI32({1, 1, 1}));
	ASSERT_TRUE(ignored.IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), terminalState);
	EXPECT_EQ(Copy<OaF32>(ignored->Reward),
		(std::vector<OaF32>{0.0F, 0.0F, 0.0F}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{1, 1, 1}));

	ASSERT_TRUE(environment.ResetDone().IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaU8>(environment.Done()),
		(std::vector<OaU8>{0, 0, 0}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeSteps()),
		(std::vector<OaU32>{0, 0, 0}));
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{1, 1, 1}));
	EXPECT_NE(Copy<OaF32>(environment.Observation()), firstInitial);

	ASSERT_TRUE(environment.Reset().IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaF32>(environment.Observation()), firstInitial);
	EXPECT_EQ(Copy<OaU32>(environment.EpisodeIndex()),
		(std::vector<OaU32>{0, 0, 0}));
}

TEST_VK(TestCartPole, RejectsInvalidConfigurationAndActionShape) {
	EXPECT_TRUE(OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 0}).IsError());
	EXPECT_TRUE(OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.TimeStep = 0.0F}).IsError());
	auto created = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2});
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	EXPECT_TRUE(environment.Step(MatrixI32({0})).IsError());
	EXPECT_TRUE(environment.Step(OaFnMatrix::Empty(
		{2}, OaScalarType::Float32)).IsError());
}

TEST_VK(TestCartPole, PublishesReusableEnvironmentContract) {
	constexpr OaU32 environments = 3;
	auto created = OaTutorialCartPole::Create(
		Runtime(),
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
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 7});
	ASSERT_TRUE(created.IsOk());
	auto environment = OaStdMove(*created);
	const auto step = environment.Step(MatrixI32({7, 1}));
	ASSERT_TRUE(step.IsOk());
	ASSERT_TRUE(SubmitAndWait(environment).IsOk());
	EXPECT_EQ(Copy<OaF32>(step->Reward),
		(std::vector<OaF32>{0.0F, 1.0F}));
	EXPECT_EQ(Copy<OaU8>(step->Terminated),
		(std::vector<OaU8>{1, 0}));
	EXPECT_EQ(Copy<OaU8>(step->Done),
		(std::vector<OaU8>{1, 0}));
}

TEST_F(TestCartPole, SchemaOwnsSemanticAndKernelIdentities) {
	const auto& reset = OaOperationRegistry::CartPoleReset;
	EXPECT_EQ(reset.InputCount, 4U);
	EXPECT_EQ(reset.OutputCount, 4U);
	EXPECT_EQ(reset.AttributeCount, 2U);
	EXPECT_EQ(reset.ShapeRule, OaOperationShapeRule::Explicit);
	EXPECT_EQ(reset.DtypeRule, OaOperationDtypeRule::MatchInput);
	EXPECT_TRUE(reset.MutatesInput(0));
	EXPECT_EQ(reset.AliasInputForOutput(3), 3U);

	const auto& step = OaOperationRegistry::CartPoleStep;
	EXPECT_EQ(step.InputCount, 4U);
	EXPECT_EQ(step.OutputCount, 7U);
	EXPECT_EQ(step.AttributeCount, OaOperationContract::MaxAttributes);
	EXPECT_EQ(step.ShapeRule, OaOperationShapeRule::Explicit);
	EXPECT_EQ(step.DtypeRule, OaOperationDtypeRule::MatchInput);
	EXPECT_EQ(step.ControlFlow, OaOperationControlFlow::Conditional);
	EXPECT_FALSE(step.MutatesInput(0));
	EXPECT_TRUE(step.MutatesInput(1));
	EXPECT_TRUE(step.MutatesInput(2));
	EXPECT_TRUE(step.MutatesInput(3));
	EXPECT_EQ(step.AliasInputForOutput(0),
		OaOperationContract::NoAliasInput);
	EXPECT_EQ(step.AliasInputForOutput(1), 1U);
	EXPECT_EQ(step.AliasInputForOutput(5), 2U);
	EXPECT_EQ(step.AliasInputForOutput(6), 3U);

	constexpr OaKernelId resetId = OA_COMPUTE_KERNEL_ID(
		OaComputeKernelPrefix::Ml, 274);
	constexpr OaKernelId stepId = OA_COMPUTE_KERNEL_ID(
		OaComputeKernelPrefix::Ml, 275);
	EXPECT_EQ(OaComputeKernelId::RlCartPoleReset, resetId);
	EXPECT_EQ(OaComputeKernelId::RlCartPoleStep, stepId);
	const auto* resetKernel = OaComputeKernelFindByName("RlCartPoleReset");
	const auto* stepKernel = OaComputeKernelFindByName("RlCartPoleStep");
	ASSERT_NE(resetKernel, nullptr);
	ASSERT_NE(stepKernel, nullptr);
	EXPECT_EQ(resetKernel->Id, resetId);
	EXPECT_EQ(stepKernel->Id, stepId);
}

TEST_F(TestCartPole, DynamicsIdentityCoversEveryBehaviorField) {
	const OaTutorialCartPoleConfig reference{};
	const OaU64 identity = reference.DynamicsIdentity();
	EXPECT_NE(identity, 0U);
	auto expectChanged = [&](OaTutorialCartPoleConfig InChanged) {
		EXPECT_NE(InChanged.DynamicsIdentity(), identity);
	};
	expectChanged(OaTutorialCartPoleConfig{.MaxEpisodeSteps = 501});
	expectChanged(OaTutorialCartPoleConfig{.Gravity = 9.81F});
	expectChanged(OaTutorialCartPoleConfig{.CartMass = 1.01F});
	expectChanged(OaTutorialCartPoleConfig{.PoleMass = 0.11F});
	expectChanged(OaTutorialCartPoleConfig{.HalfPoleLength = 0.51F});
	expectChanged(OaTutorialCartPoleConfig{.ForceMagnitude = 10.1F});
	expectChanged(OaTutorialCartPoleConfig{.TimeStep = 0.021F});
	expectChanged(OaTutorialCartPoleConfig{.PositionThreshold = 2.5F});
	expectChanged(OaTutorialCartPoleConfig{.AngleThresholdRadians = 0.21F});
	EXPECT_EQ(OaTutorialCartPoleConfig{.Environments = 7}.DynamicsIdentity(),
		identity);
	EXPECT_EQ(OaTutorialCartPoleConfig{.Seed = 99}.DynamicsIdentity(), identity);
}

TEST_VK(TestCartPole, RecordsFrozenSessionValueSentinelTruthfully) {
	const OaTutorialCartPoleConfig config{
		.Environments = 2,
		.MaxEpisodeSteps = 321,
		.Seed = 93,
		.Gravity = 9.81F,
		.CartMass = 1.25F,
		.PoleMass = 0.15F,
		.HalfPoleLength = 0.75F,
		.ForceMagnitude = 11.0F,
		.TimeStep = 0.025F,
		.PositionThreshold = 2.75F,
		.AngleThresholdRadians = 0.22F,
	};
	auto created = OaTutorialCartPole::Create(Runtime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(environment.Step(MatrixI32({0, 1})).IsOk());

	const auto* graph = OaRlEnvironmentExecutionAccess::Context(environment)
		.SemanticGraph();
	ASSERT_NE(graph, nullptr);
	ASSERT_EQ(graph->OperationCount(), 2U);
	const auto operations = graph->Operations();
	EXPECT_EQ(operations[0].Name, "CartPoleReset");
	ASSERT_EQ(operations[0].Attributes.Size(), 2U);
	EXPECT_EQ(operations[0].Attributes[0].Name, "Seed");
	EXPECT_EQ(operations[0].Attributes[0].UnsignedInteger, config.Seed);
	EXPECT_EQ(operations[0].Attributes[1].Name, "OnlyCompleted");
	EXPECT_FALSE(operations[0].Attributes[1].Boolean);

	EXPECT_EQ(operations[1].Name, "CartPoleStep");
	ASSERT_EQ(operations[1].Attributes.Size(), 8U);
	const auto& attributes = operations[1].Attributes;
	EXPECT_EQ(attributes[0].Name, "DynamicsVersion");
	EXPECT_EQ(attributes[0].UnsignedInteger,
		OaTutorialCartPoleConfig::DynamicsVersion);
	EXPECT_EQ(attributes[1].Name, "DynamicsIdentity");
	EXPECT_EQ(attributes[1].UnsignedInteger, config.DynamicsIdentity());
	EXPECT_EQ(attributes[2].Name, "MaxEpisodeSteps");
	EXPECT_EQ(attributes[2].UnsignedInteger, config.MaxEpisodeSteps);
	EXPECT_EQ(attributes[3].Name, "Gravity");
	EXPECT_DOUBLE_EQ(attributes[3].Float, config.Gravity);
	EXPECT_EQ(attributes[4].Name, "ForceMagnitude");
	EXPECT_DOUBLE_EQ(attributes[4].Float, config.ForceMagnitude);
	EXPECT_EQ(attributes[5].Name, "TimeStep");
	EXPECT_DOUBLE_EQ(attributes[5].Float, config.TimeStep);
	EXPECT_EQ(attributes[6].Name, "PositionThreshold");
	EXPECT_DOUBLE_EQ(attributes[6].Float, config.PositionThreshold);
	EXPECT_EQ(attributes[7].Name, "AngleThresholdRadians");
	EXPECT_DOUBLE_EQ(attributes[7].Float, config.AngleThresholdRadians);
}

TEST_VK(TestCartPole, SessionRequiresItsExactCompletionAndClosesExplicitly) {
	auto created = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 81});
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);

	auto first = environment.Submit();
	ASSERT_TRUE(first.IsOk()) << first.GetStatus().ToString();
	EXPECT_TRUE(environment.HasPendingEvent());
	EXPECT_EQ(environment.SubmissionCount(), 1U);
	EXPECT_TRUE(environment.Submit().IsError());
	EXPECT_TRUE(environment.Wait(OaEvent{}).IsError());
	EXPECT_TRUE(environment.HasPendingEvent());
	auto foreignCreated = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 1, .Seed = 82});
	ASSERT_TRUE(foreignCreated.IsOk());
	auto foreign = OaStdMove(*foreignCreated);
	auto foreignEvent = foreign.Submit();
	ASSERT_TRUE(foreignEvent.IsOk());
	EXPECT_TRUE(environment.Wait(*foreignEvent).IsError());
	EXPECT_TRUE(environment.HasPendingEvent());
	ASSERT_TRUE(environment.Wait(*first).IsOk());
	ASSERT_TRUE(foreign.Wait(*foreignEvent).IsOk());
	EXPECT_FALSE(environment.HasPendingEvent());
	EXPECT_TRUE(environment.Wait(*first).IsError());

	ASSERT_TRUE(environment.ResetDone().IsOk());
	auto second = environment.Submit();
	ASSERT_TRUE(second.IsOk()) << second.GetStatus().ToString();
	EXPECT_TRUE(environment.Wait(*first).IsError());
	EXPECT_TRUE(environment.HasPendingEvent());
	ASSERT_TRUE(environment.Wait(*second).IsOk());
	EXPECT_EQ(environment.SubmissionCount(), 2U);

	ASSERT_TRUE(environment.Begin().IsOk());
	ASSERT_TRUE(environment.Cancel().IsOk());
	ASSERT_TRUE(environment.Close().IsOk());
	EXPECT_TRUE(environment.Close().IsOk());
	EXPECT_FALSE(environment.IsOpen());
	EXPECT_TRUE(environment.Begin().IsError());
}

TEST_VK(TestCartPole, FailedSubmitRequiresACompleteFreshRecording) {
	auto created = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 84});
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);

	// Add a valid semantic command without an executable owner. Lowering
	// validation must reject the transaction before submission.
	auto& context = OaRlEnvironmentExecutionAccess::Context(environment);
	const auto orphan = context.RecordOperation(
		OaOperationRegistry::CartPoleReset,
		{&environment.Done(), &environment.Observation(),
		 &environment.EpisodeSteps(), &environment.EpisodeIndex()},
		{&environment.Done(), &environment.Observation(),
		 &environment.EpisodeSteps(), &environment.EpisodeIndex()},
		{
			OaOperationAttribute::FromUnsignedInteger("Seed", 84U),
			OaOperationAttribute::FromBoolean("OnlyCompleted", false),
		});
	ASSERT_TRUE(orphan.IsOk()) << orphan.GetStatus().ToString();
	auto failed = environment.Submit();
	ASSERT_TRUE(failed.IsError());
	EXPECT_FALSE(environment.HasPendingEvent());
	EXPECT_EQ(environment.SubmissionCount(), 0U);
	EXPECT_TRUE(environment.IsOpen());
	// The failed transaction cannot be resubmitted as an empty batch.
	EXPECT_TRUE(environment.Submit().IsError());

	ASSERT_TRUE(environment.Reset().IsOk());
	auto completion = environment.Submit();
	ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
	ASSERT_TRUE(environment.Wait(*completion).IsOk());
	EXPECT_EQ(environment.SubmissionCount(), 1U);
}

TEST_VK(TestCartPole, SubmittedEnvironmentDestructionUsesEngineRetirement) {
	{
		auto created = OaTutorialCartPole::Create(
			Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 87});
		ASSERT_TRUE(created.IsOk());
		auto abandoned = OaStdMove(*created);
		auto completion = abandoned.Submit();
		ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
		EXPECT_TRUE(abandoned.HasPendingEvent());
		// Destruction transfers the submitted context batch to OaEngine; it does
		// not wait for or discard the exact completion.
	}

	auto probeCreated = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 88});
	ASSERT_TRUE(probeCreated.IsOk());
	auto probe = OaStdMove(*probeCreated);
	ASSERT_TRUE(SubmitAndWait(probe).IsOk());
	EXPECT_EQ(probe.SubmissionCount(), 1U);
}

TEST_VK(TestCartPole, BorrowedEngineWorksWithNoAmbientContextAndRestoresNull) {
	SyncDefault();
	DefaultContextGuard restore;
	OaContext::SetDefault(nullptr);
	ASSERT_EQ(OaContext::GetDefaultPtr(), nullptr);

	auto created = OaTutorialCartPole::Create(
		Runtime(), OaTutorialCartPoleConfig{.Environments = 2, .Seed = 91});
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	EXPECT_TRUE(environment.IsValid());
	EXPECT_EQ(OaContext::GetDefaultPtr(), nullptr);

	const OaStatus recorded = environment.RecordCommands([&]() -> OaStatus {
		const OaMatrix action = MatrixI32({0, 1});
		if (action.IsEmpty()) return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"ambient-null CartPole action allocation failed");
		auto step = environment.Step(action);
		return step.IsError() ? step.GetStatus() : OaStatus::Ok();
	});
	ASSERT_TRUE(recorded.IsOk()) << recorded.ToString();
	EXPECT_EQ(OaContext::GetDefaultPtr(), nullptr);
	auto completion = environment.Submit();
	ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
	ASSERT_TRUE(environment.Wait(*completion).IsOk());
	EXPECT_EQ(OaContext::GetDefaultPtr(), nullptr);
	EXPECT_EQ(environment.SubmissionCount(), 1U);

	const OaStatus failed = environment.RecordCommands([] {
		return OaStatus::InvalidArgument("intentional recording rollback");
	});
	EXPECT_TRUE(failed.IsError());
	EXPECT_EQ(OaContext::GetDefaultPtr(), nullptr);
	EXPECT_FALSE(environment.HasPendingEvent());
	EXPECT_TRUE(environment.Close().IsOk());
}
