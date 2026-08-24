#include "../../oaTest.h"

#include <ml/rl/cartPole.h>

#include <oa/core/fnMatrix.h>
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/op.h>
#include <oa/ml/optim.h>
#include <oa/ml.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/semanticGraph.h>

#include <ml/rl/gen/environmentOpRegistry.h>
#include <oa/ml/environmentExecution.h>

#include <cmath>
#include <vector>

namespace {

class TestCartPole : public ::testing::Test {};

struct CpuStep {
	oa::F32 state[4]{};
	oa::F32 reward = 0.0F;
	oa::U8 terminated = 0;
	oa::U8 truncated = 0;
};

oa::Matrix matrixI32(const std::vector<oa::I32>& inValues) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::I32)),
		{static_cast<oa::I64>(inValues.size())}, oa::ScalarType::Int32);
}

oa::Engine& runtime() {
	return testEngine();
}

void syncDefault() {
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());
}

oa::Status submitAndWait(oa::Environment& inEnvironment) {
	auto completion = inEnvironment.submit();
	if (completion.isError()) return completion.getStatus();
	return inEnvironment.wait(*completion);
}

class DefaultContextGuard {
public:
	DefaultContextGuard() : previous_(oa::ExecutionSession::getActivePtr()) {}
	~DefaultContextGuard() { oa::ExecutionSession::setActive(previous_); }

private:
	oa::ExecutionSession* previous_ = nullptr;
};

template<typename T>
std::vector<T> copy(const oa::Matrix& inMatrix) {
	std::vector<T> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(T)).isOk());
	return result;
}

CpuStep stepCpu(
	const oa::F32* inState,
	oa::I32 inAction,
	oa::U32 inEpisodeSteps,
	const oa::CartPoleConfig& inConfig) {
	CpuStep result;
	oa::F32 x = inState[0];
	oa::F32 xVelocity = inState[1];
	oa::F32 angle = inState[2];
	oa::F32 angleVelocity = inState[3];
	const bool validAction = inAction == 0 || inAction == 1;
	const oa::F32 force = inAction == 1
		? inConfig.forceMagnitude : -inConfig.forceMagnitude;
	const oa::F32 totalMass = inConfig.cartMass + inConfig.poleMass;
	const oa::F32 poleMassLength = inConfig.poleMass * inConfig.halfPoleLength;
	const oa::F32 cosine = std::cos(angle);
	const oa::F32 sine = std::sin(angle);
	const oa::F32 temporary = (force
		+ poleMassLength * angleVelocity * angleVelocity * sine) / totalMass;
	const oa::F32 angleAcceleration = (inConfig.gravity * sine - cosine * temporary)
		/ (inConfig.halfPoleLength
			* (4.0F / 3.0F - inConfig.poleMass * cosine * cosine / totalMass));
	const oa::F32 xAcceleration = temporary
		- poleMassLength * angleAcceleration * cosine / totalMass;
	x += inConfig.timeStep * xVelocity;
	xVelocity += inConfig.timeStep * xAcceleration;
	angle += inConfig.timeStep * angleVelocity;
	angleVelocity += inConfig.timeStep * angleAcceleration;
	result.state[0] = x;
	result.state[1] = xVelocity;
	result.state[2] = angle;
	result.state[3] = angleVelocity;
	const bool invalidState = !std::isfinite(x) || !std::isfinite(xVelocity)
		|| !std::isfinite(angle) || !std::isfinite(angleVelocity);
	result.terminated = static_cast<oa::U8>(!validAction || invalidState
		|| std::abs(x) > inConfig.positionThreshold
		|| std::abs(angle) > inConfig.angleThresholdRadians);
	result.truncated = static_cast<oa::U8>(!result.terminated
		&& inEpisodeSteps + 1U >= inConfig.maxEpisodeSteps);
	result.reward = validAction ? 1.0F : 0.0F;
	return result;
}

} // namespace

TEST_VK(TestCartPole, GpuStepMatchesScalarCpuOracle) {
	const oa::CartPoleConfig config{
		.environments = 4,
		.maxEpisodeSteps = 500,
		.seed = 0x123456789abcdef0ULL,
	};
	auto created = oa::CartPole::create(runtime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(submitAndWait(environment).isOk());
	const auto initial = copy<oa::F32>(environment.observation());
	const std::vector<oa::I32> action = {0, 1, 1, 0};
	const auto transition = environment.step(matrixI32(action));
	ASSERT_TRUE(transition.isOk()) << transition.getStatus().toString();
	ASSERT_TRUE(submitAndWait(environment).isOk());

	EXPECT_EQ(copy<oa::F32>(transition->observation), initial);
	const auto actualState = copy<oa::F32>(transition->nextObservation);
	const auto actualReward = copy<oa::F32>(transition->reward);
	const auto actualTerminated = copy<oa::U8>(transition->terminated);
	const auto actualTruncated = copy<oa::U8>(transition->truncated);
	const auto actualDone = copy<oa::U8>(transition->done);
	EXPECT_EQ(copy<oa::U32>(environment.episodeSteps()),
		(std::vector<oa::U32>{1, 1, 1, 1}));
	for (oa::U32 lane = 0; lane < config.environments; ++lane) {
		const CpuStep expected = stepCpu(
			initial.data() + lane * 4U, action[lane], 0, config);
		for (oa::U32 component = 0; component < 4; ++component) {
			EXPECT_NEAR(actualState[lane * 4U + component],
				expected.state[component], 1.0e-5F)
				<< "lane=" << lane << " component=" << component;
		}
		EXPECT_EQ(actualReward[lane], expected.reward);
		EXPECT_EQ(actualTerminated[lane], expected.terminated);
		EXPECT_EQ(actualTruncated[lane], expected.truncated);
		EXPECT_EQ(actualDone[lane],
			static_cast<oa::U8>(expected.terminated || expected.truncated));
	}
}

TEST_VK(TestCartPole, ReusableCollectorOwnsEnvironmentPolicyExchange) {
	constexpr oa::U32 environments = 8;
	constexpr oa::U32 horizon = 16;
	auto environmentResult = oa::CartPole::create(runtime(), {
		.environments = environments,
		.maxEpisodeSteps = 64,
		.seed = 9917,
	});
	ASSERT_TRUE(environmentResult.isOk());
	auto environment = oa::move(*environmentResult);
	auto modelResult = oa::CategoricalActorCritic::create({
		.observationSize = 4,
		.actionCount = 2,
		.hiddenSize = 16,
	});
	ASSERT_TRUE(modelResult.isOk());
	auto model = oa::move(*modelResult);
	auto rolloutResult = oa::RolloutBuffer::create({
		.time = horizon,
		.environments = environments,
		.observationShape = {4},
	});
	ASSERT_TRUE(rolloutResult.isOk());
	auto rollout = oa::move(*rolloutResult);
	auto collectorResult = oa::RolloutCollector::create(environment, *model, {
		.horizon = horizon,
		.seed = 1234,
		.gae = {},
	});
	ASSERT_TRUE(collectorResult.isOk());
	auto collector = oa::move(*collectorResult);
	syncDefault();
	auto completion = collector.collect(rollout);
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	ASSERT_TRUE(environment.wait(*completion).isOk());
	EXPECT_EQ(environment.submissionCount(), 1U);
	EXPECT_TRUE(rollout.isFull());
	EXPECT_TRUE(rollout.isFinalized());
	EXPECT_EQ(collector.metrics().collections, 1U);
	EXPECT_EQ(collector.metrics().environmentSteps, horizon);
	EXPECT_EQ(collector.metrics().transitions, environments * horizon);
	for (const oa::U8 valid : copy<oa::U8>(rollout.batch().valid)) {
		EXPECT_EQ(valid, 1U);
	}
}

TEST_VK(TestCartPole, CancelledPpoCollectionRestagesInitialResetAndRetries) {
	constexpr oa::U32 environments = 2U;
	auto environmentResult = oa::CartPole::create(runtime(), {
		.environments = environments,
		.maxEpisodeSteps = 64,
		.seed = 9917,
	});
	ASSERT_TRUE(environmentResult.isOk());
	auto environment = oa::move(*environmentResult);
	auto modelResult = oa::CategoricalActorCritic::create({
		.observationSize = 4,
		.actionCount = 2,
		.hiddenSize = 16,
	});
	ASSERT_TRUE(modelResult.isOk());
	auto model = oa::move(*modelResult);
	oa::OptimizerNoOp optimizer;
	auto trainerResult = oa::PpoTrainer::create(
		testEngine(), *model, optimizer, oa::PpoTrainerConfig{
			.rollouts = 1,
			.horizon = 1,
			.environments = environments,
			.updateEpochs = 1,
			.observationShape = {4},
			.seed = 1234,
			.gae = {},
			.loss = {},
		});
	ASSERT_TRUE(trainerResult.isOk());
	auto trainer = oa::move(*trainerResult);

	const oa::Status injected = environment.recordCommands(
		[&]() -> oa::Status {
			OA_RETURN_IF_ERROR(trainer->beginCollection());
			const oa::PolicyResult policy =
				trainer->act(environment.observation());
			if (!policy.isValid()) {
				return oa::Status::error(
					oa::StatusCode::FailedPrecondition,
					"injected PPO collection could not evaluate policy");
			}
			return oa::Status::error(
				oa::StatusCode::Internal,
				"injected PPO collection failure");
		});
	EXPECT_TRUE(injected.isError());
	EXPECT_FALSE(environment.hasActiveRecording());
	EXPECT_EQ(environment.submissionCount(), 0U);
	ASSERT_TRUE(trainer->abortCollection().isOk());
	ASSERT_TRUE(trainer->needsCollection());

	// The cancelled transaction contained Create's deferred initial reset.
	ASSERT_TRUE(environment.reset().isOk());
	const oa::Status retried = environment.recordCommands(
		[&]() -> oa::Status {
			OA_RETURN_IF_ERROR(trainer->beginCollection());
			const oa::PolicyResult policy =
				trainer->act(environment.observation());
			if (!policy.isValid()) {
				return oa::Status::error(
					oa::StatusCode::FailedPrecondition,
					"retried PPO collection could not evaluate policy");
			}
			auto transition = environment.step(policy.action);
			if (transition.isError()) return transition.getStatus();
			OA_RETURN_IF_ERROR(trainer->observe(
				transition->observation, transition->nextObservation,
				transition->reward, transition->terminated,
				transition->truncated, policy));
			OA_RETURN_IF_ERROR(environment.resetDone());
			return trainer->endCollection();
		});
	ASSERT_TRUE(retried.isOk()) << retried.toString();
	auto completion = environment.submit();
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	ASSERT_TRUE(environment.wait(*completion).isOk());
	ASSERT_TRUE(trainer->update().isOk());
	EXPECT_TRUE(trainer->isDone());
	EXPECT_EQ(environment.submissionCount(), 1U);
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestCartPole, GenericEvaluatorReportsCompletedEpisodeMetrics) {
	auto environmentResult = oa::CartPole::create(runtime(), {
		.environments = 8,
		.maxEpisodeSteps = 32,
		.seed = 717,
	});
	ASSERT_TRUE(environmentResult.isOk());
	auto environment = oa::move(*environmentResult);
	auto modelResult = oa::CategoricalActorCritic::create({
		.observationSize = 4,
		.actionCount = 2,
		.hiddenSize = 16,
	});
	ASSERT_TRUE(modelResult.isOk());
	auto model = oa::move(*modelResult);
	syncDefault();
	auto metrics = oa::PolicyEvaluator::evaluateCategorical(environment, *model, {
		.horizon = 96,
		.seed = 717,
	});
	ASSERT_TRUE(metrics.isOk()) << metrics.getStatus().toString();
	EXPECT_EQ(metrics->environmentSteps, 96U);
	EXPECT_EQ(metrics->transitions, 96U * 8U);
	EXPECT_GT(metrics->completedEpisodes, 0U);
	EXPECT_GT(metrics->meanCompletedReturn, 0.0F);
	EXPECT_LE(metrics->minimumCompletedReturn,
		metrics->maximumCompletedReturn);
}

TEST_VK(TestCartPole,
	GenericEvaluatorUsesBorrowedNonDefaultContextEngineWithoutAmbientMixing) {
	oa::Engine* suiteEngine = testEnginePtr();
	ASSERT_NE(suiteEngine, nullptr);
	ASSERT_TRUE(suiteEngine->isReady());
	oa::ExecutionSession* ambientContext = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(ambientContext, nullptr);

	oa::EngineConfig engineConfig = testEngineConfig(oa::Precision::FP32);
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	auto engineResult = oa::Engine::create(engineConfig);
	ASSERT_TRUE(engineResult.isOk())
		<< engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	ASSERT_TRUE(engine->isReady());
	ASSERT_NE(engine.get(), suiteEngine);

	oa::UniquePtr<oa::ExecutionSession> modelContext;
	modelContext.reset(new oa::ExecutionSession(engine.get()));
	ASSERT_TRUE(modelContext);
	oa::UniquePtr<oa::CategoricalActorCritic> model;
	{
		oa::ExecutionSession::RecordingScope scope(*modelContext);
		auto modelResult = oa::CategoricalActorCritic::create({
			.observationSize = 4,
			.actionCount = 2,
			.hiddenSize = 16,
		});
		ASSERT_TRUE(modelResult.isOk())
			<< modelResult.getStatus().toString();
		model = oa::move(*modelResult);
		ASSERT_TRUE(testSubmitAndWait(*modelContext).isOk());
	}
	modelContext.reset();
	ASSERT_TRUE(model);
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);

	auto environmentResult = oa::CartPole::create(*engine, {
		.environments = 4,
		.maxEpisodeSteps = 16,
		.seed = 0xa1b2c3d4e5f60718ULL,
	});
	ASSERT_TRUE(environmentResult.isOk())
		<< environmentResult.getStatus().toString();
	auto environment = oa::move(*environmentResult);
	auto metrics = oa::PolicyEvaluator::evaluateCategorical(environment, *model, {
		.horizon = 48,
		.seed = 0xa1b2c3d4e5f60718ULL,
	});
	ASSERT_TRUE(metrics.isOk()) << metrics.getStatus().toString();
	EXPECT_EQ(metrics->environmentSteps, 48U);
	EXPECT_EQ(metrics->transitions, 48U * 4U);
	EXPECT_GT(metrics->completedEpisodes, 0U);
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);

	EXPECT_TRUE(environment.close().isOk());
	model.reset();
	const oa::Status engineClosed = engine->close();
	EXPECT_TRUE(engineClosed.isOk()) << engineClosed.toString();
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);
}

TEST_VK(TestCartPole, ResetAndDoneLifecycleIsExplicitAndDeterministic) {
	const oa::CartPoleConfig config{
		.environments = 3,
		.maxEpisodeSteps = 1,
		.seed = 918273645ULL,
	};
	auto created = oa::CartPole::create(runtime(), config);
	ASSERT_TRUE(created.isOk());
	auto environment = oa::move(*created);
	ASSERT_TRUE(submitAndWait(environment).isOk());
	const auto firstInitial = copy<oa::F32>(environment.observation());

	ASSERT_TRUE(environment.reset().isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::F32>(environment.observation()), firstInitial);
	EXPECT_EQ(copy<oa::U32>(environment.episodeIndex()),
		(std::vector<oa::U32>{0, 0, 0}));

	ASSERT_TRUE(environment.step(matrixI32({0, 1, 0})).isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::U8>(environment.done()),
		(std::vector<oa::U8>{1, 1, 1}));
	EXPECT_EQ(copy<oa::U32>(environment.episodeSteps()),
		(std::vector<oa::U32>{1, 1, 1}));
	const auto terminalState = copy<oa::F32>(environment.observation());

	const auto ignored = environment.step(matrixI32({1, 1, 1}));
	ASSERT_TRUE(ignored.isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::F32>(environment.observation()), terminalState);
	EXPECT_EQ(copy<oa::F32>(ignored->reward),
		(std::vector<oa::F32>{0.0F, 0.0F, 0.0F}));
	EXPECT_EQ(copy<oa::U32>(environment.episodeSteps()),
		(std::vector<oa::U32>{1, 1, 1}));

	ASSERT_TRUE(environment.resetDone().isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::U8>(environment.done()),
		(std::vector<oa::U8>{0, 0, 0}));
	EXPECT_EQ(copy<oa::U32>(environment.episodeSteps()),
		(std::vector<oa::U32>{0, 0, 0}));
	EXPECT_EQ(copy<oa::U32>(environment.episodeIndex()),
		(std::vector<oa::U32>{1, 1, 1}));
	EXPECT_NE(copy<oa::F32>(environment.observation()), firstInitial);

	ASSERT_TRUE(environment.reset().isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::F32>(environment.observation()), firstInitial);
	EXPECT_EQ(copy<oa::U32>(environment.episodeIndex()),
		(std::vector<oa::U32>{0, 0, 0}));
}

TEST_VK(TestCartPole, RejectsInvalidConfigurationAndActionShape) {
	EXPECT_TRUE(oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 0}).isError());
	EXPECT_TRUE(oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.timeStep = 0.0F}).isError());
	auto created = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2});
	ASSERT_TRUE(created.isOk());
	auto environment = oa::move(*created);
	EXPECT_TRUE(environment.step(matrixI32({0})).isError());
	EXPECT_TRUE(environment.step(oa::FnMatrix::empty(
		{2}, oa::ScalarType::Float32)).isError());
}

TEST_VK(TestCartPole, PublishesReusableEnvironmentContract) {
	constexpr oa::U32 environments = 3;
	auto created = oa::CartPole::create(
		runtime(),
		oa::CartPoleConfig{.environments = environments, .seed = 42});
	ASSERT_TRUE(created.isOk());
	auto environment = oa::move(*created);
	const auto& spec = environment.spec();
	ASSERT_TRUE(spec.validateDefinition().isOk());
	EXPECT_EQ(spec.action.kind, oa::EnvironmentSpaceKind::Discrete);
	EXPECT_EQ(spec.action.cardinality, 2);
	EXPECT_EQ(spec.observation.shape, (oa::MatrixShape{4}));
	EXPECT_TRUE(spec.validateReset(
		environment.observation(), environments).isOk());

	const oa::Matrix action = matrixI32({0, 1, 0});
	const auto transition = environment.step(action);
	ASSERT_TRUE(transition.isOk());
	EXPECT_TRUE(spec.validateTransition(
		transition->observation,
		action,
		transition->nextObservation,
		transition->reward,
		transition->terminated,
		transition->truncated,
		environments).isOk());
}

TEST_VK(TestCartPole, InvalidActionTerminatesOnlyItsLane) {
	auto created = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2, .seed = 7});
	ASSERT_TRUE(created.isOk());
	auto environment = oa::move(*created);
	const auto step = environment.step(matrixI32({7, 1}));
	ASSERT_TRUE(step.isOk());
	ASSERT_TRUE(submitAndWait(environment).isOk());
	EXPECT_EQ(copy<oa::F32>(step->reward),
		(std::vector<oa::F32>{0.0F, 1.0F}));
	EXPECT_EQ(copy<oa::U8>(step->terminated),
		(std::vector<oa::U8>{1, 0}));
	EXPECT_EQ(copy<oa::U8>(step->done),
		(std::vector<oa::U8>{1, 0}));
}

TEST_F(TestCartPole, SchemaOwnsSemanticAndKernelIdentities) {
	const auto& reset = oa::detail::opRegistry::FnEnvironment::cartPoleReset;
	EXPECT_EQ(reset.inputCount, 4U);
	EXPECT_EQ(reset.outputCount, 4U);
	EXPECT_EQ(reset.attributeCount, 2U);
	EXPECT_EQ(reset.shapeRule, oa::OpShapeRule::Explicit);
	EXPECT_EQ(reset.dtypeRule, oa::OpDtypeRule::MatchInput);
	EXPECT_TRUE(reset.mutatesInput(0));
	EXPECT_EQ(reset.aliasInputForOutput(3), 3U);

	const auto& step = oa::detail::opRegistry::FnEnvironment::cartPoleStep;
	EXPECT_EQ(step.inputCount, 4U);
	EXPECT_EQ(step.outputCount, 7U);
	EXPECT_EQ(step.attributeCount, oa::OpContract::MaxAttributes);
	EXPECT_EQ(step.shapeRule, oa::OpShapeRule::Explicit);
	EXPECT_EQ(step.dtypeRule, oa::OpDtypeRule::MatchInput);
	EXPECT_EQ(step.controlFlow, oa::OpControlFlow::Conditional);
	EXPECT_FALSE(step.mutatesInput(0));
	EXPECT_TRUE(step.mutatesInput(1));
	EXPECT_TRUE(step.mutatesInput(2));
	EXPECT_TRUE(step.mutatesInput(3));
	EXPECT_EQ(step.aliasInputForOutput(0),
		oa::OpContract::NoAliasInput);
	EXPECT_EQ(step.aliasInputForOutput(1), 1U);
	EXPECT_EQ(step.aliasInputForOutput(5), 2U);
	EXPECT_EQ(step.aliasInputForOutput(6), 3U);

	// Concrete SDK tasks install per-engine pipelines and never extend liboa's
	// fixed kernel registry.
	EXPECT_EQ(oa::computeKernelFindByName("RlCartPoleReset"), nullptr);
	EXPECT_EQ(oa::computeKernelFindByName("RlCartPoleStep"), nullptr);
}

TEST_F(TestCartPole, DynamicsIdentityCoversEveryBehaviorField) {
	const oa::CartPoleConfig reference{};
	const oa::U64 identity = reference.dynamicsIdentity();
	EXPECT_NE(identity, 0U);
	auto expectChanged = [&](oa::CartPoleConfig inChanged) {
		EXPECT_NE(inChanged.dynamicsIdentity(), identity);
	};
	expectChanged(oa::CartPoleConfig{.maxEpisodeSteps = 501});
	expectChanged(oa::CartPoleConfig{.gravity = 9.81F});
	expectChanged(oa::CartPoleConfig{.cartMass = 1.01F});
	expectChanged(oa::CartPoleConfig{.poleMass = 0.11F});
	expectChanged(oa::CartPoleConfig{.halfPoleLength = 0.51F});
	expectChanged(oa::CartPoleConfig{.forceMagnitude = 10.1F});
	expectChanged(oa::CartPoleConfig{.timeStep = 0.021F});
	expectChanged(oa::CartPoleConfig{.positionThreshold = 2.5F});
	expectChanged(oa::CartPoleConfig{.angleThresholdRadians = 0.21F});
	EXPECT_EQ(oa::CartPoleConfig{.environments = 7}.dynamicsIdentity(),
		identity);
	EXPECT_EQ(oa::CartPoleConfig{.seed = 99}.dynamicsIdentity(), identity);
}

TEST_VK(TestCartPole, RecordsFrozenSessionValueSentinelTruthfully) {
	const oa::CartPoleConfig config{
		.environments = 2,
		.maxEpisodeSteps = 321,
		.seed = 93,
		.gravity = 9.81F,
		.cartMass = 1.25F,
		.poleMass = 0.15F,
		.halfPoleLength = 0.75F,
		.forceMagnitude = 11.0F,
		.timeStep = 0.025F,
		.positionThreshold = 2.75F,
		.angleThresholdRadians = 0.22F,
	};
	auto created = oa::CartPole::create(runtime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(environment.step(matrixI32({0, 1})).isOk());

	const auto* graph =
		oa::EnvironmentExecutionAccess::session(environment).semanticGraph();
	ASSERT_NE(graph, nullptr);
	ASSERT_EQ(graph->operationCount(), 2U);
	const auto operations = graph->operations();
	EXPECT_EQ(operations[0].name,
		oa::detail::opRegistry::FnEnvironment::cartPoleReset.name);
	ASSERT_EQ(operations[0].attributes.size(), 2U);
	EXPECT_EQ(operations[0].attributes[0].name, "seed");
	EXPECT_EQ(operations[0].attributes[0].unsignedInteger, config.seed);
	EXPECT_EQ(operations[0].attributes[1].name, "onlyCompleted");
	EXPECT_FALSE(operations[0].attributes[1].boolean);

	EXPECT_EQ(operations[1].name,
		oa::detail::opRegistry::FnEnvironment::cartPoleStep.name);
	ASSERT_EQ(operations[1].attributes.size(), 8U);
	const auto& attributes = operations[1].attributes;
	EXPECT_EQ(attributes[0].name, "dynamicsVersion");
	EXPECT_EQ(attributes[0].unsignedInteger,
		oa::CartPoleConfig::dynamicsVersion);
	EXPECT_EQ(attributes[1].name, "dynamicsIdentity");
	EXPECT_EQ(attributes[1].unsignedInteger, config.dynamicsIdentity());
	EXPECT_EQ(attributes[2].name, "maxEpisodeSteps");
	EXPECT_EQ(attributes[2].unsignedInteger, config.maxEpisodeSteps);
	EXPECT_EQ(attributes[3].name, "gravity");
	EXPECT_DOUBLE_EQ(attributes[3].floatVal, config.gravity);
	EXPECT_EQ(attributes[4].name, "forceMagnitude");
	EXPECT_DOUBLE_EQ(attributes[4].floatVal, config.forceMagnitude);
	EXPECT_EQ(attributes[5].name, "timeStep");
	EXPECT_DOUBLE_EQ(attributes[5].floatVal, config.timeStep);
	EXPECT_EQ(attributes[6].name, "positionThreshold");
	EXPECT_DOUBLE_EQ(attributes[6].floatVal, config.positionThreshold);
	EXPECT_EQ(attributes[7].name, "angleThresholdRadians");
	EXPECT_DOUBLE_EQ(attributes[7].floatVal, config.angleThresholdRadians);
}

TEST_VK(TestCartPole, SessionRequiresItsExactCompletionAndClosesExplicitly) {
	auto created = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2, .seed = 81});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);

	auto first = environment.submit();
	ASSERT_TRUE(first.isOk()) << first.getStatus().toString();
	EXPECT_TRUE(environment.hasPendingEvent());
	EXPECT_EQ(environment.submissionCount(), 1U);
	EXPECT_TRUE(environment.submit().isError());
	EXPECT_TRUE(environment.wait(oa::Event{}).isError());
	EXPECT_TRUE(environment.hasPendingEvent());
	auto foreignCreated = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 1, .seed = 82});
	ASSERT_TRUE(foreignCreated.isOk());
	auto foreign = oa::move(*foreignCreated);
	auto foreignEvent = foreign.submit();
	ASSERT_TRUE(foreignEvent.isOk());
	EXPECT_TRUE(environment.wait(*foreignEvent).isError());
	EXPECT_TRUE(environment.hasPendingEvent());
	ASSERT_TRUE(environment.wait(*first).isOk());
	ASSERT_TRUE(foreign.wait(*foreignEvent).isOk());
	EXPECT_FALSE(environment.hasPendingEvent());
	EXPECT_TRUE(environment.wait(*first).isError());

	ASSERT_TRUE(environment.resetDone().isOk());
	auto second = environment.submit();
	ASSERT_TRUE(second.isOk()) << second.getStatus().toString();
	EXPECT_TRUE(environment.wait(*first).isError());
	EXPECT_TRUE(environment.hasPendingEvent());
	ASSERT_TRUE(environment.wait(*second).isOk());
	EXPECT_EQ(environment.submissionCount(), 2U);

	ASSERT_TRUE(environment.begin().isOk());
	ASSERT_TRUE(environment.cancel().isOk());
	ASSERT_TRUE(environment.close().isOk());
	EXPECT_TRUE(environment.close().isOk());
	EXPECT_FALSE(environment.isOpen());
	EXPECT_TRUE(environment.begin().isError());
}

TEST_VK(TestCartPole, FailedSubmitRequiresACompleteFreshRecording) {
	auto created = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2, .seed = 84});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);

	// Add a valid semantic command without an executable owner. lowering
	// validation must reject the transaction before submission.
	auto& context = oa::EnvironmentExecutionAccess::session(environment);
	const auto orphan = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::cartPoleReset,
		{&environment.done(), &environment.observation(),
		 &environment.episodeSteps(), &environment.episodeIndex()},
		{&environment.done(), &environment.observation(),
		 &environment.episodeSteps(), &environment.episodeIndex()},
		{
			oa::OpAttribute::fromUnsignedInteger("seed", 84U),
			oa::OpAttribute::fromBoolean("onlyCompleted", false),
		});
	ASSERT_TRUE(orphan.isOk()) << orphan.getStatus().toString();
	auto failed = environment.submit();
	ASSERT_TRUE(failed.isError());
	EXPECT_FALSE(environment.hasPendingEvent());
	EXPECT_EQ(environment.submissionCount(), 0U);
	EXPECT_TRUE(environment.isOpen());
	// The failed transaction cannot be resubmitted as an empty batch.
	EXPECT_TRUE(environment.submit().isError());

	ASSERT_TRUE(environment.reset().isOk());
	auto completion = environment.submit();
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	ASSERT_TRUE(environment.wait(*completion).isOk());
	EXPECT_EQ(environment.submissionCount(), 1U);
}

TEST_VK(TestCartPole, SubmittedEnvironmentDestructionUsesEngineRetirement) {
	{
		auto created = oa::CartPole::create(
			runtime(), oa::CartPoleConfig{.environments = 2, .seed = 87});
		ASSERT_TRUE(created.isOk());
		auto abandoned = oa::move(*created);
		auto completion = abandoned.submit();
		ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
		EXPECT_TRUE(abandoned.hasPendingEvent());
		// Destruction transfers the submitted context batch to oa::Engine; it does
		// not wait for or discard the exact completion.
	}

	auto probeCreated = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2, .seed = 88});
	ASSERT_TRUE(probeCreated.isOk());
	auto probe = oa::move(*probeCreated);
	ASSERT_TRUE(submitAndWait(probe).isOk());
	EXPECT_EQ(probe.submissionCount(), 1U);
}

TEST_VK(TestCartPole, BorrowedEngineWorksWithNoAmbientContextAndRestoresNull) {
	syncDefault();
	DefaultContextGuard restore;
	oa::ExecutionSession::setActive(nullptr);
	ASSERT_EQ(oa::ExecutionSession::getActivePtr(), nullptr);

	auto created = oa::CartPole::create(
		runtime(), oa::CartPoleConfig{.environments = 2, .seed = 91});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	EXPECT_TRUE(environment.isValid());
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), nullptr);

	const oa::Status recorded = environment.recordCommands([&]() -> oa::Status {
		const oa::Matrix action = matrixI32({0, 1});
		if (action.isEmpty()) return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"ambient-null oa::CartPole action allocation failed");
		auto step = environment.step(action);
		return step.isError() ? step.getStatus() : oa::Status::ok();
	});
	ASSERT_TRUE(recorded.isOk()) << recorded.toString();
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), nullptr);
	auto completion = environment.submit();
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	ASSERT_TRUE(environment.wait(*completion).isOk());
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), nullptr);
	EXPECT_EQ(environment.submissionCount(), 1U);

	const oa::Status failed = environment.recordCommands([] {
		return oa::Status::invalidArgument("intentional recording rollback");
	});
	EXPECT_TRUE(failed.isError());
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), nullptr);
	EXPECT_FALSE(environment.hasPendingEvent());
	EXPECT_TRUE(environment.close().isOk());
}
