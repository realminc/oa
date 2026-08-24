#include "../../oaTest.h"

#include <ml/rl/lunarLander3d.h>

#include <oa/core/fnMatrix.h>
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/semanticGraph.h>

#include <ml/rl/environmentOpRegistry.gen.h>
#include <oa/ml/environmentExecution.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

class TestLunarLander3dVector : public ::testing::Test {};

static oa::Engine& lunarVectorRuntime() {
	return testEngine();
}

static oa::Status lunarVectorSubmitAndWait(
	oa::Environment& inEnvironment) {
	auto completion = inEnvironment.submit();
	if (completion.isError()) return completion.getStatus();
	return inEnvironment.wait(*completion);
}

static oa::Matrix lunarVectorActions(
	const std::vector<oa::I32>& inActions) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inActions.data()),
			inActions.size() * sizeof(oa::I32)),
		{static_cast<oa::I64>(inActions.size())}, oa::ScalarType::Int32);
}

static oa::Matrix lunarVectorExternalStops(
	const std::vector<oa::U8>& inExternalStops) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			inExternalStops.data(), inExternalStops.size()),
		{static_cast<oa::I64>(inExternalStops.size())}, oa::ScalarType::UInt8);
}

template<typename T>
static std::vector<T> lunarVectorCopy(const oa::Matrix& inMatrix) {
	std::vector<T> result(static_cast<std::size_t>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(T)).isOk());
	return result;
}

static oa::F32 lunarVectorObservationTolerance(oa::U32 inComponent) noexcept {
	if (inComponent < 6U) return 8.0e-4F;
	if (inComponent < 15U) return 1.5e-3F;
	if (inComponent < 24U) return 2.0e-3F;
	if (inComponent < 28U) return 2.0e-3F;
	if (inComponent < 32U) return 0.0F;
	return 8.0e-5F;
}

static void lunarVectorExpectObservationNear(
	const oa::F32* inActual,
	const std::array<float, oa::kLunarObservationSize>& inExpected,
	oa::U32 inLane,
	oa::U32 inStep) {
	for (oa::U32 component = 0U;
		component < oa::kLunarObservationSize;
		++component) {
		const oa::F32 tolerance = lunarVectorObservationTolerance(component);
		if (tolerance == 0.0F) {
			EXPECT_EQ(inActual[component], inExpected[component])
				<< "lane=" << inLane << " step=" << inStep
				<< " component=" << component;
		} else {
			EXPECT_NEAR(inActual[component], inExpected[component], tolerance)
				<< "lane=" << inLane << " step=" << inStep
				<< " component=" << component;
		}
	}
}

static void lunarVectorExpectContactObservationNear(
	const oa::F32* inActual,
	const std::array<float, oa::kLunarObservationSize>& inExpected,
	oa::U32 inLane,
	oa::U32 inStep) {
	for (oa::U32 component = 0U;
		component < oa::kLunarObservationSize;
		++component) {
		oa::F32 tolerance = 2.0e-2F;
		if (component < 3U) tolerance = 3.0e-3F;
		else if (component < 6U) tolerance = 1.5e-2F;
		else if (component < 15U) tolerance = 3.0e-2F;
		else if (component >= 28U and component < 32U) tolerance = 0.0F;
		else if (component == 32U) tolerance = 1.0e-3F;
		if (tolerance == 0.0F) {
			EXPECT_EQ(inActual[component], inExpected[component])
				<< "lane=" << inLane << " step=" << inStep
				<< " component=" << component;
		} else {
			EXPECT_NEAR(inActual[component], inExpected[component], tolerance)
				<< "lane=" << inLane << " step=" << inStep
				<< " component=" << component;
		}
	}
}

static oa::LunarEpisodeManifest lunarVectorManifest(
	const oa::LunarLander3dVectorConfig& inConfig,
	oa::U32 inLane,
	oa::U64 inEpisode) {
	return oa::LunarEpisodeManifest::derive(
		inConfig.seed_, inLane, inEpisode,
		inConfig.environment_.contractFingerprint());
}

static oa::LunarScalarEnvironment lunarVectorScalarLane(
	const oa::LunarLander3dVectorConfig& inConfig,
	oa::U32 inLane,
	oa::U64 inEpisode) {
	return oa::LunarScalarEnvironment::createFlat(
		inConfig.environment_, lunarVectorManifest(inConfig, inLane, inEpisode));
}

enum class LunarVectorOraclePolicy : oa::U8 {
	Coast,
	Scripted,
};

static void lunarVectorRunEpisodeDifferential(
	const oa::LunarLander3dConfig& inEnvironmentConfig,
	oa::U64 inSeed,
	LunarVectorOraclePolicy inPolicy,
	oa::LunarEndReason inExpectedReason,
	bool inRequireFootContact,
	bool inRequireBodyContact) {
	constexpr oa::U32 environments = 4U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = inSeed,
		.environment_ = inEnvironmentConfig,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());

	std::vector<oa::LunarScalarEnvironment> scalar;
	scalar.reserve(environments);
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		scalar.push_back(lunarVectorScalarLane(config, lane, 0U));
		ASSERT_TRUE(scalar.back().isValid()) << scalar.back().error();
	}

	bool sawFootContact = false;
	bool sawBodyContact = false;
	bool completed = false;
	std::vector<bool> contactPhase(environments, false);
	for (oa::U32 stepIndex = 0U;
		stepIndex < inEnvironmentConfig.maxEpisodeSteps_;
		++stepIndex) {
		std::vector<oa::I32> actions(environments, 0);
		std::vector<oa::LunarTransition> expected(environments);
		for (oa::U32 lane = 0U; lane < environments; ++lane) {
			const oa::LunarLander3dState& state = scalar[lane].state();
			if (state.terminated_ or state.truncated_) {
				expected[lane].valid_ = true;
				expected[lane].observation_ = scalar[lane].observation();
				expected[lane].terminated_ = state.terminated_;
				expected[lane].truncated_ = state.truncated_;
				expected[lane].endReason_ = state.endReason_;
				continue;
			}
			const oa::LunarAction action =
				inPolicy == LunarVectorOraclePolicy::Scripted
				? oa::lunarScriptedLandingAction(
					inEnvironmentConfig, state)
				: oa::LunarAction::Coast;
			actions[lane] = static_cast<oa::I32>(action);
			expected[lane] = scalar[lane].step(
				static_cast<oa::U32>(action));
			ASSERT_TRUE(expected[lane].valid_)
				<< "lane=" << lane << " step=" << stepIndex
				<< " " << expected[lane].error_;
			sawFootContact = sawFootContact
				or expected[lane].contact_.footContactOccurred_;
			sawBodyContact = sawBodyContact
				or expected[lane].contact_.bodyContactOccurred_;
			contactPhase[lane] = contactPhase[lane]
				or expected[lane].contact_.footContactOccurred_
				or expected[lane].contact_.bodyContactOccurred_;
		}

		auto transition = environment.step(lunarVectorActions(actions));
		ASSERT_TRUE(transition.isOk()) << transition.getStatus().toString();
		ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
		const auto actualObservation = lunarVectorCopy<oa::F32>(
			transition->nextObservation_);
		const auto actualReward = lunarVectorCopy<oa::F32>(
			transition->reward_);
		const auto actualTerminated = lunarVectorCopy<oa::U8>(
			transition->terminated_);
		const auto actualTruncated = lunarVectorCopy<oa::U8>(
			transition->truncated_);
		const auto actualReason = lunarVectorCopy<oa::U32>(
			transition->endReason_);
		for (oa::U32 lane = 0U; lane < environments; ++lane) {
			const oa::F32* laneObservation = actualObservation.data()
				+ static_cast<std::size_t>(lane)
					* oa::kLunarObservationSize;
			if (contactPhase[lane]) {
				lunarVectorExpectContactObservationNear(
					laneObservation, expected[lane].observation_,
					lane, stepIndex + 1U);
			} else {
				lunarVectorExpectObservationNear(
					laneObservation, expected[lane].observation_,
					lane, stepIndex + 1U);
			}
			EXPECT_NEAR(
				actualReward[lane], expected[lane].reward_, 6.0e-3F)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualTerminated[lane],
				expected[lane].terminated_ ? 1U : 0U)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualTruncated[lane],
				expected[lane].truncated_ ? 1U : 0U)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualReason[lane],
				static_cast<oa::U32>(expected[lane].endReason_))
				<< "lane=" << lane << " step=" << stepIndex;
		}

		completed = true;
		for (const auto& lane : scalar) {
			completed = completed
				and (lane.state().terminated_ or lane.state().truncated_);
		}
		if (completed) break;
	}
	ASSERT_TRUE(completed);

	auto telemetryResult = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(telemetryResult.isOk())
		<< telemetryResult.getStatus().toString();
	const auto& telemetry = *telemetryResult;
	ASSERT_EQ(telemetry.size(), environments);
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		const oa::LunarLander3dState& expected = scalar[lane].state();
		const auto& actual = telemetry[lane];
		EXPECT_EQ(actual.endReason_, inExpectedReason) << "lane=" << lane;
		EXPECT_EQ(actual.endReason_, expected.endReason_) << "lane=" << lane;
		EXPECT_EQ(actual.terminated_, expected.terminated_) << "lane=" << lane;
		EXPECT_EQ(actual.truncated_, expected.truncated_) << "lane=" << lane;
		EXPECT_EQ(actual.episodeStep_, expected.episodeStep_) << "lane=" << lane;
		const oa::F32 returnTolerance = std::max(
			0.2F, static_cast<oa::F32>(
				std::abs(expected.episodeReturn_) * 2.0e-3));
		EXPECT_NEAR(
			actual.episodeReturn_, expected.episodeReturn_, returnTolerance)
			<< "lane=" << lane;
		EXPECT_NEAR(actual.fuelRemaining_, expected.fuel_, 5.0e-2F)
			<< "lane=" << lane;
		EXPECT_NEAR(
			actual.terminalLinearSpeed_,
			expected.linearVelocity_.length(), 5.0e-2F)
			<< "lane=" << lane;
		EXPECT_NEAR(
			actual.terminalAngularSpeed_,
			expected.angularVelocityBody_.length(), 5.0e-2F)
			<< "lane=" << lane;
		oa::F32 expectedMaximumFootImpulse = 0.0F;
		for (const double impulse : expected.footContactImpulses_) {
			expectedMaximumFootImpulse = std::max(
				expectedMaximumFootImpulse, static_cast<oa::F32>(impulse));
		}
		const oa::F32 maximumAccumulatedFootImpulse = static_cast<oa::F32>(
			inEnvironmentConfig.maxContactImpulse_
			* static_cast<double>(inEnvironmentConfig.physicsSubsteps_)
			* static_cast<double>(inEnvironmentConfig.contactIterations_));
		EXPECT_LE(
			actual.maximumFootImpulse_, maximumAccumulatedFootImpulse)
			<< "lane=" << lane;
		if (expectedMaximumFootImpulse > 0.0F) {
			EXPECT_GT(actual.maximumFootImpulse_, 0.0F) << "lane=" << lane;
		} else {
			EXPECT_FLOAT_EQ(actual.maximumFootImpulse_, 0.0F)
				<< "lane=" << lane;
		}
	}
	EXPECT_EQ(sawFootContact, inRequireFootContact);
	EXPECT_EQ(sawBodyContact, inRequireBodyContact);
	EXPECT_TRUE(environment.close().isOk());
}

TEST_F(TestLunarLander3dVector, SchemaOwnsResetStepAndKernelIdentities) {
	const auto& reset = oa::detail::opRegistry::FnEnvironment::lunarLander3dReset;
	EXPECT_EQ(reset.inputCount, 7U);
	EXPECT_EQ(reset.outputCount, 4U);
	EXPECT_EQ(reset.attributeCount, 4U);
	EXPECT_EQ(reset.shapeRule, oa::OpShapeRule::Explicit);
	EXPECT_EQ(reset.dtypeRule, oa::OpDtypeRule::MatchInput);
	EXPECT_TRUE(reset.mutatesInput(3U));
	EXPECT_TRUE(reset.mutatesInput(4U));
	EXPECT_TRUE(reset.mutatesInput(5U));
	EXPECT_TRUE(reset.mutatesInput(6U));
	EXPECT_EQ(reset.aliasInputForOutput(0U), 3U);
	EXPECT_EQ(reset.aliasInputForOutput(3U), 6U);

	const auto& step = oa::detail::opRegistry::FnEnvironment::lunarLander3dStep;
	EXPECT_EQ(step.inputCount, oa::OpContract::MaxValues);
	EXPECT_EQ(step.outputCount, oa::OpContract::MaxValues);
	EXPECT_EQ(step.attributeCount, oa::OpContract::MaxAttributes);
	EXPECT_EQ(step.controlFlow, oa::OpControlFlow::Conditional);
	EXPECT_FALSE(step.mutatesInput(0U));
	EXPECT_FALSE(step.mutatesInput(1U));
	EXPECT_TRUE(step.mutatesInput(5U));
	EXPECT_TRUE(step.mutatesInput(6U));
	EXPECT_TRUE(step.mutatesInput(7U));
	EXPECT_EQ(step.aliasInputForOutput(0U), 5U);
	EXPECT_EQ(step.aliasInputForOutput(1U), 6U);
	EXPECT_EQ(step.aliasInputForOutput(3U), 7U);
	EXPECT_EQ(step.aliasInputForOutput(7U),
		oa::OpContract::NoAliasInput);

	EXPECT_EQ(oa::computeKernelFindByName("RlLunarLander3dReset"), nullptr);
	EXPECT_EQ(oa::computeKernelFindByName("RlLunarLander3dStep"), nullptr);
}

TEST_VK(TestLunarLander3dVector, RejectsNonzeroConfigThatUnderflowsInFp32) {
	oa::LunarLander3dVectorConfig config;
	config.environment_.gravity_ = std::numeric_limits<double>::denorm_min();
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isError());
	EXPECT_EQ(created.getStatus().getCode(), oa::StatusCode::OutOfRange);

	config = {};
	config.environment_.gravity_ =
		static_cast<double>(std::numeric_limits<oa::F32>::denorm_min());
	created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isError());
	EXPECT_EQ(created.getStatus().getCode(), oa::StatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector, RejectsUnboundedFp32RewardAccumulation) {
	oa::LunarLander3dVectorConfig config;
	config.environment_.positionPotentialWeight_ =
		static_cast<double>(std::numeric_limits<oa::F32>::max());
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isError());
	EXPECT_EQ(created.getStatus().getCode(), oa::StatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector, RejectsUnrepresentableDerivedFp32TimeAndFuel) {
	oa::LunarLander3dVectorConfig subnormalTime;
	subnormalTime.environment_.policyTimeStep_ =
		static_cast<double>(std::numeric_limits<oa::F32>::min());
	subnormalTime.environment_.physicsSubsteps_ = 64U;
	auto timeResult = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), subnormalTime);
	ASSERT_TRUE(timeResult.isError());
	EXPECT_EQ(timeResult.getStatus().getCode(), oa::StatusCode::OutOfRange);

	oa::LunarLander3dVectorConfig subnormalDebit;
	subnormalDebit.environment_.mainFuelRate_ =
		static_cast<double>(std::numeric_limits<oa::F32>::min());
	auto debitResult = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), subnormalDebit);
	ASSERT_TRUE(debitResult.isError());
	EXPECT_EQ(debitResult.getStatus().getCode(), oa::StatusCode::OutOfRange);

	oa::LunarLander3dVectorConfig stalledDebit;
	stalledDebit.environment_.fuelCapacity_ = 1.0e8;
	stalledDebit.environment_.mainFuelRate_ = 100.0;
	stalledDebit.environment_.attitudeFuelRate_ = 1.0e6;
	auto stalledResult = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), stalledDebit);
	ASSERT_TRUE(stalledResult.isError());
	EXPECT_EQ(stalledResult.getStatus().getCode(), oa::StatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector,
	TelemetryRejectsUnsubmittedAndSubmittedTransactions) {
	const oa::LunarLander3dVectorConfig config{
		.environments_ = 2U,
		.seed_ = 0xe7d6c5b4a3928170ULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);

	EXPECT_TRUE(environment.hasActiveRecording());
	auto duringInitialRecording = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(duringInitialRecording.isError());
	EXPECT_EQ(
		duringInitialRecording.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);

	auto completion = environment.submit();
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	EXPECT_FALSE(environment.hasActiveRecording());
	EXPECT_TRUE(environment.hasPendingEvent());
	auto duringPendingSubmission = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(duringPendingSubmission.isError());
	EXPECT_EQ(
		duringPendingSubmission.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.wait(*completion).isOk());

	auto afterCompletion = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(afterCompletion.isOk())
		<< afterCompletion.getStatus().toString();
	ASSERT_EQ(afterCompletion->size(), 2U);
	for (const auto& telemetry : *afterCompletion) {
		EXPECT_EQ(telemetry.episodeStep_, 0U);
		EXPECT_EQ(telemetry.endReason_, oa::LunarEndReason::None);
	}

	ASSERT_TRUE(environment.step(lunarVectorActions({0, 0})).isOk());
	EXPECT_TRUE(environment.hasActiveRecording());
	auto duringStepRecording = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(duringStepRecording.isError());
	EXPECT_EQ(
		duringStepRecording.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.cancel().isOk());
	EXPECT_FALSE(environment.hasActiveRecording());
	auto afterCancel = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(afterCancel.isOk()) << afterCancel.getStatus().toString();
	for (const auto& telemetry : *afterCancel) {
		EXPECT_EQ(telemetry.episodeStep_, 0U);
		EXPECT_EQ(telemetry.endReason_, oa::LunarEndReason::None);
	}
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector,
	TelemetryReadbackUsesBorrowedNonDefaultContextEngineWithoutAmbientMixing) {
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
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);

	const oa::LunarLander3dVectorConfig config{
		.environments_ = 3U,
		.seed_ = 0x4628d79a30be51c4ULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());

	auto telemetryResult = environment.copyEpisodeTelemetry();
	ASSERT_TRUE(telemetryResult.isOk())
		<< telemetryResult.getStatus().toString();
	ASSERT_EQ(telemetryResult->size(), config.environments_);
	for (const auto& telemetry : *telemetryResult) {
		EXPECT_FLOAT_EQ(telemetry.episodeReturn_, 0.0F);
		EXPECT_FLOAT_EQ(
			telemetry.fuelRemaining_,
			static_cast<oa::F32>(config.environment_.fuelCapacity_));
		EXPECT_EQ(telemetry.episodeStep_, 0U);
		EXPECT_FALSE(telemetry.terminated_);
		EXPECT_FALSE(telemetry.truncated_);
		EXPECT_EQ(telemetry.endReason_, oa::LunarEndReason::None);
	}
	EXPECT_TRUE(environment.close().isOk());
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);
	const oa::Status engineClosed = engine->close();
	EXPECT_TRUE(engineClosed.isOk()) << engineClosed.toString();
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);
}

TEST_VK(TestLunarLander3dVector, CancelledInitialResetRequiresRecoveryReset) {
	const oa::LunarLander3dVectorConfig config{
		.environments_ = 2U,
		.seed_ = 0x11a22b33c44d55e6ULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(environment.cancel().isOk());
	EXPECT_EQ(
		environment.step(lunarVectorActions({0, 0})).getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(environment.resetDone().getCode(), oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.reset().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto actual = lunarVectorCopy<oa::F32>(environment.observation());
	auto scalar = lunarVectorScalarLane(config, 0U, 0U);
	ASSERT_TRUE(scalar.isValid());
	lunarVectorExpectObservationNear(
		actual.data(), scalar.observation(), 0U, 0U);
	auto recoveredStep = environment.step(lunarVectorActions({0, 0}));
	ASSERT_TRUE(recoveredStep.isOk()) << recoveredStep.getStatus().toString();
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto recoveredReason = lunarVectorCopy<oa::U32>(
		recoveredStep->endReason_);
	const auto recoveredTruncated = lunarVectorCopy<oa::U8>(
		recoveredStep->truncated_);
	EXPECT_EQ(recoveredReason[0], static_cast<oa::U32>(oa::LunarEndReason::None));
	EXPECT_EQ(recoveredReason[1], static_cast<oa::U32>(oa::LunarEndReason::None));
	EXPECT_EQ(recoveredTruncated[0], 0U);
	EXPECT_EQ(recoveredTruncated[1], 0U);
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector, ReseedMetadataCommitsOnlyWithAcceptedBatch) {
	constexpr oa::U64 seedA = 0x1011121314151617ULL;
	constexpr oa::U64 seedB = 0x8081828384858687ULL;
	constexpr oa::U64 seedC = 0xc0c1c2c3c4c5c6c7ULL;
	constexpr oa::U64 seedD = 0xd0d1d2d3d4d5d6d7ULL;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = 2U,
		.seed_ = seedA,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());

	ASSERT_TRUE(environment.reset(seedB).isOk());
	EXPECT_EQ(environment.config().seed_, seedA);
	ASSERT_TRUE(environment.resetDone().isOk());
	ASSERT_TRUE(environment.reset().isOk());
	const auto* graph =
		oa::EnvironmentExecutionAccess::session(environment).semanticGraph();
	ASSERT_NE(graph, nullptr);
	ASSERT_EQ(graph->operationCount(), 3U);
	for (const auto& operation : graph->operations()) {
		ASSERT_EQ(
			operation.name,
			oa::detail::opRegistry::FnEnvironment::lunarLander3dReset.name);
		ASSERT_GE(operation.attributes.size(), 3U);
		EXPECT_EQ(operation.attributes[2].name, "seed");
		EXPECT_EQ(operation.attributes[2].unsignedInteger, seedB);
	}
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	EXPECT_EQ(environment.config().seed_, seedB);

	ASSERT_TRUE(environment.reset(seedC).isOk());
	EXPECT_EQ(environment.config().seed_, seedB);
	ASSERT_TRUE(environment.cancel().isOk());
	EXPECT_EQ(environment.config().seed_, seedB);
	ASSERT_TRUE(environment.reset().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	oa::LunarLander3dVectorConfig seedBConfig = config;
	seedBConfig.seed_ = seedB;
	const auto afterCancel = lunarVectorCopy<oa::F32>(environment.observation());
	auto scalarB = lunarVectorScalarLane(seedBConfig, 0U, 0U);
	ASSERT_TRUE(scalarB.isValid());
	lunarVectorExpectObservationNear(
		afterCancel.data(), scalarB.observation(), 0U, 0U);

	ASSERT_TRUE(environment.reset(seedD).isOk());
	auto& context = oa::EnvironmentExecutionAccess::session(environment);
	const oa::Matrix& observation = environment.observation();
	const oa::Matrix& reason = environment.endReason();
	const auto orphan = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::lunarLander3dReset,
		{&observation, &observation, &reason, &observation, &reason,
			&observation, &reason},
		{&observation, &reason, &observation, &reason},
		{
			oa::OpAttribute::fromUnsignedInteger("environmentVersion", 0U),
			oa::OpAttribute::fromUnsignedInteger("stateLayoutVersion", 0U),
			oa::OpAttribute::fromUnsignedInteger("seed", seedD),
			oa::OpAttribute::fromBoolean("onlyCompleted", false),
		});
	ASSERT_TRUE(orphan.isOk()) << orphan.getStatus().toString();
	auto failed = environment.submit();
	ASSERT_TRUE(failed.isError());
	EXPECT_EQ(environment.config().seed_, seedB);
	EXPECT_TRUE(environment.isOpen());
	ASSERT_TRUE(environment.reset().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	EXPECT_EQ(environment.config().seed_, seedB);
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector, ResetMatchesFrozenScalarManifests) {
	for (const oa::U32 environments : {1U, 7U, 257U}) {
		const oa::LunarLander3dVectorConfig config{
			.environments_ = environments,
			.seed_ = 0x7a61d3c59e2048bfULL,
		};
		auto created = oa::LunarLander3dVector::createFlat(
			lunarVectorRuntime(), config);
		ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
		auto environment = oa::move(*created);
		ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
		const auto actual = lunarVectorCopy<oa::F32>(environment.observation());
		for (oa::U32 lane = 0U; lane < environments; ++lane) {
			auto scalar = lunarVectorScalarLane(config, lane, 0U);
			ASSERT_TRUE(scalar.isValid()) << scalar.error();
			lunarVectorExpectObservationNear(
				actual.data() + static_cast<std::size_t>(lane)
					* oa::kLunarObservationSize,
				scalar.observation(), lane, 0U);
		}
		EXPECT_TRUE(environment.close().isOk());
	}
}

TEST_VK(TestLunarLander3dVector, FixedActionTraceMatchesScalarFp64Oracle) {
	constexpr oa::U32 environments = 5U;
	constexpr oa::U32 traceSteps = 24U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0xc34d8217a6950befULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	std::vector<oa::LunarScalarEnvironment> scalar;
	scalar.reserve(environments);
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		scalar.push_back(lunarVectorScalarLane(config, lane, 0U));
		ASSERT_TRUE(scalar.back().isValid()) << scalar.back().error();
	}
	for (oa::U32 stepIndex = 0U; stepIndex < traceSteps; ++stepIndex) {
		std::vector<oa::I32> actions(environments);
		for (oa::U32 lane = 0U; lane < environments; ++lane) {
			actions[lane] = static_cast<oa::I32>((stepIndex + lane * 3U) % 8U);
		}
		auto transition = environment.step(lunarVectorActions(actions));
		ASSERT_TRUE(transition.isOk()) << transition.getStatus().toString();
		ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
		const auto actualObservation = lunarVectorCopy<oa::F32>(
			transition->nextObservation_);
		const auto actualReward = lunarVectorCopy<oa::F32>(transition->reward_);
		const auto actualTerminated = lunarVectorCopy<oa::U8>(
			transition->terminated_);
		const auto actualTruncated = lunarVectorCopy<oa::U8>(
			transition->truncated_);
		const auto actualReason = lunarVectorCopy<oa::U32>(
			transition->endReason_);
		for (oa::U32 lane = 0U; lane < environments; ++lane) {
			const oa::LunarTransition expected = scalar[lane].step(
				static_cast<oa::U32>(actions[lane]));
			ASSERT_TRUE(expected.valid_) << expected.error_;
			lunarVectorExpectObservationNear(
				actualObservation.data() + static_cast<std::size_t>(lane)
					* oa::kLunarObservationSize,
				expected.observation_, lane, stepIndex + 1U);
			EXPECT_NEAR(actualReward[lane], expected.reward_, 4.0e-3F)
				<< "lane=" << lane << " step=" << stepIndex;
			EXPECT_EQ(actualTerminated[lane], expected.terminated_ ? 1U : 0U);
			EXPECT_EQ(actualTruncated[lane], expected.truncated_ ? 1U : 0U);
			EXPECT_EQ(actualReason[lane], static_cast<oa::U32>(expected.endReason_));
		}
	}
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector,
	ScriptedLandingMatchesScalarThroughContactDwellAndTerminal) {
	oa::LunarLander3dConfig config;
	config.safeDwellSteps_ = 12U;
	ASSERT_NO_FATAL_FAILURE(lunarVectorRunEpisodeDifferential(
		config,
		0x50494c4f545f4556ULL,
		LunarVectorOraclePolicy::Scripted,
		oa::LunarEndReason::SafeLanding,
		true,
		false));
}

TEST_VK(TestLunarLander3dVector,
	FailureAndTruncationReasonsMatchScalarThroughTerminalPhysics) {
	oa::LunarLander3dConfig hardImpact;
	ASSERT_NO_FATAL_FAILURE(lunarVectorRunEpisodeDifferential(
		hardImpact,
		0x484152445f464f4fULL,
		LunarVectorOraclePolicy::Coast,
		oa::LunarEndReason::HardFootImpact,
		true,
		false));

	oa::LunarLander3dConfig bodyImpact;
	for (auto& foot : bodyImpact.footSupports_) {
		foot.bodyOffset_.y = 1.0;
	}
	ASSERT_NO_FATAL_FAILURE(lunarVectorRunEpisodeDifferential(
		bodyImpact,
		0x424f44595f494d50ULL,
		LunarVectorOraclePolicy::Coast,
		oa::LunarEndReason::BodyImpact,
		false,
		true));

	oa::LunarLander3dConfig timeLimit;
	timeLimit.gravity_ = 0.0;
	timeLimit.maxEpisodeSteps_ = 3U;
	ASSERT_NO_FATAL_FAILURE(lunarVectorRunEpisodeDifferential(
		timeLimit,
		0x54494d455f4c494dULL,
		LunarVectorOraclePolicy::Coast,
		oa::LunarEndReason::TimeLimit,
		false,
		false));

	oa::LunarLander3dConfig outOfBounds;
	outOfBounds.taskMaximumY_ = 4.0;
	ASSERT_NO_FATAL_FAILURE(lunarVectorRunEpisodeDifferential(
		outOfBounds,
		0x4f55545f4f465f42ULL,
		LunarVectorOraclePolicy::Coast,
		oa::LunarEndReason::OutOfBounds,
		false,
		false));
}

TEST_VK(TestLunarLander3dVector, ExternalStopIsLaneLocalAndInvalidActionWins) {
	constexpr oa::U32 environments = 4U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0x38a1c5e792b4d60fULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());

	const std::vector<oa::I32> actions{0, 1, 8, 7};
	const std::vector<oa::U8> externalStops{0U, 1U, 1U, 0U};
	auto wrongMask = environment.step(
		lunarVectorActions(actions), lunarVectorActions({0, 1, 1, 0}));
	ASSERT_TRUE(wrongMask.isError());
	EXPECT_EQ(wrongMask.getStatus().getCode(), oa::StatusCode::DtypeMismatch);
	auto transition = environment.step(
		lunarVectorActions(actions),
		lunarVectorExternalStops(externalStops));
	ASSERT_TRUE(transition.isOk()) << transition.getStatus().toString();
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());

	const auto previousObservation = lunarVectorCopy<oa::F32>(
		transition->observation_);
	const auto nextObservation = lunarVectorCopy<oa::F32>(
		transition->nextObservation_);
	const auto reward = lunarVectorCopy<oa::F32>(transition->reward_);
	const auto terminated = lunarVectorCopy<oa::U8>(transition->terminated_);
	const auto truncated = lunarVectorCopy<oa::U8>(transition->truncated_);
	const auto reason = lunarVectorCopy<oa::U32>(transition->endReason_);
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		auto scalar = lunarVectorScalarLane(config, lane, 0U);
		ASSERT_TRUE(scalar.isValid()) << scalar.error();
		const oa::LunarTransition expected = scalar.step(
			static_cast<oa::U32>(actions[lane]), externalStops[lane] != 0U);
		ASSERT_TRUE(expected.valid_) << expected.error_;
		lunarVectorExpectObservationNear(
			nextObservation.data() + static_cast<std::size_t>(lane)
				* oa::kLunarObservationSize,
			expected.observation_, lane, 1U);
		EXPECT_NEAR(reward[lane], expected.reward_, 4.0e-3F)
			<< "lane=" << lane;
		EXPECT_EQ(terminated[lane], expected.terminated_ ? 1U : 0U);
		EXPECT_EQ(truncated[lane], expected.truncated_ ? 1U : 0U);
		EXPECT_EQ(reason[lane], static_cast<oa::U32>(expected.endReason_));
		if (externalStops[lane] != 0U and actions[lane] >= 0
			and actions[lane] < 8) {
			EXPECT_EQ(scalar.state().episodeStep_, 0U);
			for (oa::U32 component = 0U;
				component < oa::kLunarObservationSize;
				++component) {
				const std::size_t index = static_cast<std::size_t>(lane)
					* oa::kLunarObservationSize + component;
				EXPECT_EQ(nextObservation[index], previousObservation[index]);
			}
		}
	}
	EXPECT_FLOAT_EQ(reward[1], 0.0F);
	EXPECT_EQ(terminated[1], 0U);
	EXPECT_EQ(truncated[1], 1U);
	EXPECT_EQ(reason[1], static_cast<oa::U32>(oa::LunarEndReason::ExternalStop));
	EXPECT_FLOAT_EQ(reward[2],
		static_cast<oa::F32>(config.environment_.failurePenalty_));
	EXPECT_EQ(terminated[2], 1U);
	EXPECT_EQ(truncated[2], 0U);
	EXPECT_EQ(reason[2], static_cast<oa::U32>(oa::LunarEndReason::InvalidAction));
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector, InvalidActionIsLaneLocalAndDoesNotShiftOtherRng) {
	constexpr oa::U32 environments = 6U;
	constexpr oa::U32 invalidLane = 2U;
	constexpr oa::U32 laterResetLane = 4U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0x196a4e7bd205c83fULL,
	};
	auto firstCreated = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	auto controlCreated = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(firstCreated.isOk()) << firstCreated.getStatus().toString();
	ASSERT_TRUE(controlCreated.isOk()) << controlCreated.getStatus().toString();
	auto first = oa::move(*firstCreated);
	auto control = oa::move(*controlCreated);
	ASSERT_TRUE(lunarVectorSubmitAndWait(first).isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(control).isOk());

	std::vector<oa::I32> firstActions = {0, 1, 8, 3, 4, 7};
	std::vector<oa::I32> controlActions = firstActions;
	controlActions[invalidLane] = 0;
	auto firstStep = first.step(lunarVectorActions(firstActions));
	auto controlStep = control.step(lunarVectorActions(controlActions));
	ASSERT_TRUE(firstStep.isOk());
	ASSERT_TRUE(controlStep.isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(first).isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(control).isOk());
	const auto firstObservation = lunarVectorCopy<oa::F32>(
		firstStep->nextObservation_);
	const auto controlObservation = lunarVectorCopy<oa::F32>(
		controlStep->nextObservation_);
	const auto reward = lunarVectorCopy<oa::F32>(firstStep->reward_);
	const auto terminated = lunarVectorCopy<oa::U8>(firstStep->terminated_);
	const auto reason = lunarVectorCopy<oa::U32>(firstStep->endReason_);
	EXPECT_FLOAT_EQ(reward[invalidLane],
		static_cast<oa::F32>(config.environment_.failurePenalty_));
	EXPECT_EQ(terminated[invalidLane], 1U);
	EXPECT_EQ(reason[invalidLane],
		static_cast<oa::U32>(oa::LunarEndReason::InvalidAction));
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		if (lane == invalidLane) continue;
		for (oa::U32 component = 0U;
			component < oa::kLunarObservationSize;
			++component) {
			const std::size_t index = static_cast<std::size_t>(lane)
				* oa::kLunarObservationSize + component;
			EXPECT_EQ(firstObservation[index], controlObservation[index]);
		}
	}

	ASSERT_TRUE(first.resetDone().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(first).isOk());
	const auto afterReset = lunarVectorCopy<oa::F32>(first.observation());
	for (oa::U32 lane = 0U; lane < environments; ++lane) {
		if (lane == invalidLane) continue;
		for (oa::U32 component = 0U;
			component < oa::kLunarObservationSize;
			++component) {
			const std::size_t index = static_cast<std::size_t>(lane)
				* oa::kLunarObservationSize + component;
			EXPECT_EQ(afterReset[index], firstObservation[index]);
		}
	}
	auto scalarEpisodeOne = lunarVectorScalarLane(config, invalidLane, 1U);
	ASSERT_TRUE(scalarEpisodeOne.isValid());
	lunarVectorExpectObservationNear(
		afterReset.data() + static_cast<std::size_t>(invalidLane)
			* oa::kLunarObservationSize,
		scalarEpisodeOne.observation(), invalidLane, 0U);

	std::vector<oa::I32> laterActions(environments, 0);
	laterActions[laterResetLane] = std::numeric_limits<oa::I32>::max();
	ASSERT_TRUE(first.step(lunarVectorActions(laterActions)).isOk());
	ASSERT_TRUE(control.step(lunarVectorActions(laterActions)).isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(first).isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(control).isOk());
	ASSERT_TRUE(first.resetDone().isOk());
	ASSERT_TRUE(control.resetDone().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(first).isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(control).isOk());
	const auto firstLaterReset = lunarVectorCopy<oa::F32>(first.observation());
	const auto controlLaterReset = lunarVectorCopy<oa::F32>(control.observation());
	for (oa::U32 component = 0U;
		component < oa::kLunarObservationSize;
		++component) {
		const std::size_t index = static_cast<std::size_t>(laterResetLane)
			* oa::kLunarObservationSize + component;
		EXPECT_EQ(firstLaterReset[index], controlLaterReset[index]);
	}
	EXPECT_TRUE(first.close().isOk());
	EXPECT_TRUE(control.close().isOk());
}

TEST_VK(TestLunarLander3dVector, CompletedLaneKeepsTerminalStateUntilReset) {
	constexpr oa::U32 environments = 3U;
	constexpr oa::U32 completedLane = 1U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0xd47ca2359b1806efULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	auto terminal = environment.step(lunarVectorActions({0, 8, 0}));
	ASSERT_TRUE(terminal.isOk()) << terminal.getStatus().toString();
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto terminalObservation = lunarVectorCopy<oa::F32>(
		terminal->nextObservation_);
	auto repeated = environment.step(lunarVectorActions({0, 0, 0}));
	ASSERT_TRUE(repeated.isOk()) << repeated.getStatus().toString();
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto repeatedObservation = lunarVectorCopy<oa::F32>(
		repeated->nextObservation_);
	const auto repeatedReward = lunarVectorCopy<oa::F32>(repeated->reward_);
	const auto repeatedTerminated = lunarVectorCopy<oa::U8>(
		repeated->terminated_);
	const auto repeatedTruncated = lunarVectorCopy<oa::U8>(
		repeated->truncated_);
	const auto repeatedReason = lunarVectorCopy<oa::U32>(
		repeated->endReason_);
	EXPECT_FLOAT_EQ(repeatedReward[completedLane], 0.0F);
	EXPECT_EQ(repeatedTerminated[completedLane], 1U);
	EXPECT_EQ(repeatedTruncated[completedLane], 0U);
	EXPECT_EQ(repeatedReason[completedLane],
		static_cast<oa::U32>(oa::LunarEndReason::InvalidAction));
	for (oa::U32 component = 0U;
		component < oa::kLunarObservationSize;
		++component) {
		const std::size_t index = static_cast<std::size_t>(completedLane)
			* oa::kLunarObservationSize + component;
		EXPECT_EQ(repeatedObservation[index], terminalObservation[index]);
	}
	ASSERT_TRUE(environment.resetDone().isOk());
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto resetReason = lunarVectorCopy<oa::U32>(environment.endReason());
	EXPECT_EQ(resetReason[completedLane],
		static_cast<oa::U32>(oa::LunarEndReason::None));
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector, OddReuseAndInvalidActionPoisonStayBounded) {
	constexpr oa::U32 environments = 257U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0x5d3e8c714a09b26fULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	for (oa::U32 reuse = 0U; reuse < 8U; ++reuse) {
		std::vector<oa::I32> actions(environments, static_cast<oa::I32>(reuse % 8U));
		for (oa::U32 lane = reuse; lane < environments; lane += 31U) {
			actions[lane] = (lane & 1U) == 0U
				? std::numeric_limits<oa::I32>::min()
				: std::numeric_limits<oa::I32>::max();
		}
		auto step = environment.step(lunarVectorActions(actions));
		ASSERT_TRUE(step.isOk()) << step.getStatus().toString();
		ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
		const auto observation = lunarVectorCopy<oa::F32>(step->nextObservation_);
		const auto reward = lunarVectorCopy<oa::F32>(step->reward_);
		for (const oa::F32 value : observation) EXPECT_TRUE(std::isfinite(value));
		for (const oa::F32 value : reward) EXPECT_TRUE(std::isfinite(value));
		ASSERT_TRUE(environment.resetDone().isOk());
		ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	}
	EXPECT_TRUE(environment.close().isOk());
}

TEST_VK(TestLunarLander3dVector, LargeOddLaneCountUsesQueriedLimits) {
	constexpr oa::U32 environments = 65537U;
	const oa::LunarLander3dVectorConfig config{
		.environments_ = environments,
		.seed_ = 0x2b74cd901f6a853eULL,
	};
	auto created = oa::LunarLander3dVector::createFlat(
		lunarVectorRuntime(), config);
	if (created.isError()
		and created.getStatus().getCode() == oa::StatusCode::OutOfMemory) {
		GTEST_SKIP() << created.getStatus().toString();
	}
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto environment = oa::move(*created);
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	std::vector<oa::I32> actions(environments, 0);
	actions.back() = 8;
	auto step = environment.step(lunarVectorActions(actions));
	ASSERT_TRUE(step.isOk()) << step.getStatus().toString();
	ASSERT_TRUE(lunarVectorSubmitAndWait(environment).isOk());
	const auto reward = lunarVectorCopy<oa::F32>(step->reward_);
	const auto terminated = lunarVectorCopy<oa::U8>(step->terminated_);
	EXPECT_FLOAT_EQ(reward.back(),
		static_cast<oa::F32>(config.environment_.failurePenalty_));
	EXPECT_EQ(terminated.back(), 1U);
	EXPECT_TRUE(environment.close().isOk());
}
