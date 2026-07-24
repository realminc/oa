#include "../../OaTest.h"

#include "LunarLander3dVector.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/SemanticGraph.h>

#include "../../../Source/Private/Oa/Core/OperationRegistry.gen.h"
#include "../../../Source/Private/Oa/Ml/Rl/EnvironmentExecution.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

class TestLunarLander3dVector : public ::testing::Test {};

static OaEngine& OaLunarVectorRuntime() {
	return *OaEngine::GetGlobal();
}

static OaStatus OaLunarVectorSubmitAndWait(
	OaRlEnvironment& InEnvironment) {
	auto completion = InEnvironment.Submit();
	if (completion.IsError()) return completion.GetStatus();
	return InEnvironment.Wait(*completion);
}

static OaMatrix OaLunarVectorActions(
	const std::vector<OaI32>& InActions) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InActions.data()),
			InActions.size() * sizeof(OaI32)),
		{static_cast<OaI64>(InActions.size())}, OaScalarType::Int32);
}

static OaMatrix OaLunarVectorExternalStops(
	const std::vector<OaU8>& InExternalStops) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			InExternalStops.data(), InExternalStops.size()),
		{static_cast<OaI64>(InExternalStops.size())}, OaScalarType::UInt8);
}

template<typename T>
static std::vector<T> OaLunarVectorCopy(const OaMatrix& InMatrix) {
	std::vector<T> result(static_cast<std::size_t>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(T)).IsOk());
	return result;
}

static OaF32 OaLunarVectorObservationTolerance(OaU32 InComponent) noexcept {
	if (InComponent < 6U) return 8.0e-4F;
	if (InComponent < 15U) return 1.5e-3F;
	if (InComponent < 24U) return 2.0e-3F;
	if (InComponent < 28U) return 2.0e-3F;
	if (InComponent < 32U) return 0.0F;
	return 8.0e-5F;
}

static void OaLunarVectorExpectObservationNear(
	const OaF32* InActual,
	const std::array<float, OA_LUNAR_OBSERVATION_SIZE>& InExpected,
	OaU32 InLane,
	OaU32 InStep) {
	for (OaU32 component = 0U;
		component < OA_LUNAR_OBSERVATION_SIZE;
		++component) {
		const OaF32 tolerance = OaLunarVectorObservationTolerance(component);
		if (tolerance == 0.0F) {
			EXPECT_EQ(InActual[component], InExpected[component])
				<< "lane=" << InLane << " step=" << InStep
				<< " component=" << component;
		} else {
			EXPECT_NEAR(InActual[component], InExpected[component], tolerance)
				<< "lane=" << InLane << " step=" << InStep
				<< " component=" << component;
		}
	}
}

static void OaLunarVectorExpectContactObservationNear(
	const OaF32* InActual,
	const std::array<float, OA_LUNAR_OBSERVATION_SIZE>& InExpected,
	OaU32 InLane,
	OaU32 InStep) {
	for (OaU32 component = 0U;
		component < OA_LUNAR_OBSERVATION_SIZE;
		++component) {
		OaF32 tolerance = 2.0e-2F;
		if (component < 3U) tolerance = 3.0e-3F;
		else if (component < 6U) tolerance = 1.5e-2F;
		else if (component < 15U) tolerance = 3.0e-2F;
		else if (component >= 28U and component < 32U) tolerance = 0.0F;
		else if (component == 32U) tolerance = 1.0e-3F;
		if (tolerance == 0.0F) {
			EXPECT_EQ(InActual[component], InExpected[component])
				<< "lane=" << InLane << " step=" << InStep
				<< " component=" << component;
		} else {
			EXPECT_NEAR(InActual[component], InExpected[component], tolerance)
				<< "lane=" << InLane << " step=" << InStep
				<< " component=" << component;
		}
	}
}

static OaLunarEpisodeManifest OaLunarVectorManifest(
	const OaLunarLander3dVectorConfig& InConfig,
	OaU32 InLane,
	OaU64 InEpisode) {
	return OaLunarEpisodeManifest::Derive(
		InConfig.Seed_, InLane, InEpisode,
		InConfig.Environment_.ContractFingerprint());
}

static OaLunarScalarEnvironment OaLunarVectorScalarLane(
	const OaLunarLander3dVectorConfig& InConfig,
	OaU32 InLane,
	OaU64 InEpisode) {
	return OaLunarScalarEnvironment::CreateFlat(
		InConfig.Environment_, OaLunarVectorManifest(InConfig, InLane, InEpisode));
}

enum class OaLunarVectorOraclePolicy : OaU8 {
	Coast,
	Scripted,
};

static void OaLunarVectorRunEpisodeDifferential(
	const OaLunarLander3dConfig& InEnvironmentConfig,
	OaU64 InSeed,
	OaLunarVectorOraclePolicy InPolicy,
	OaLunarEndReason InExpectedReason,
	bool InRequireFootContact,
	bool InRequireBodyContact) {
	constexpr OaU32 environments = 4U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = InSeed,
		.Environment_ = InEnvironmentConfig,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());

	std::vector<OaLunarScalarEnvironment> scalar;
	scalar.reserve(environments);
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		scalar.push_back(OaLunarVectorScalarLane(config, lane, 0U));
		ASSERT_TRUE(scalar.back().IsValid()) << scalar.back().Error();
	}

	bool sawFootContact = false;
	bool sawBodyContact = false;
	bool completed = false;
	std::vector<bool> contactPhase(environments, false);
	for (OaU32 stepIndex = 0U;
		stepIndex < InEnvironmentConfig.MaxEpisodeSteps_;
		++stepIndex) {
		std::vector<OaI32> actions(environments, 0);
		std::vector<OaLunarTransition> expected(environments);
		for (OaU32 lane = 0U; lane < environments; ++lane) {
			const OaLunarLander3dState& state = scalar[lane].State();
			if (state.Terminated_ or state.Truncated_) {
				expected[lane].Valid_ = true;
				expected[lane].Observation_ = scalar[lane].Observation();
				expected[lane].Terminated_ = state.Terminated_;
				expected[lane].Truncated_ = state.Truncated_;
				expected[lane].EndReason_ = state.EndReason_;
				continue;
			}
			const OaLunarAction action =
				InPolicy == OaLunarVectorOraclePolicy::Scripted
				? OaLunarScriptedLandingAction(
					InEnvironmentConfig, state)
				: OaLunarAction::Coast;
			actions[lane] = static_cast<OaI32>(action);
			expected[lane] = scalar[lane].Step(
				static_cast<OaU32>(action));
			ASSERT_TRUE(expected[lane].Valid_)
				<< "lane=" << lane << " step=" << stepIndex
				<< " " << expected[lane].Error_;
			sawFootContact = sawFootContact
				or expected[lane].Contact_.FootContactOccurred_;
			sawBodyContact = sawBodyContact
				or expected[lane].Contact_.BodyContactOccurred_;
			contactPhase[lane] = contactPhase[lane]
				or expected[lane].Contact_.FootContactOccurred_
				or expected[lane].Contact_.BodyContactOccurred_;
		}

		auto transition = environment.Step(OaLunarVectorActions(actions));
		ASSERT_TRUE(transition.IsOk()) << transition.GetStatus().ToString();
		ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
		const auto actualObservation = OaLunarVectorCopy<OaF32>(
			transition->NextObservation_);
		const auto actualReward = OaLunarVectorCopy<OaF32>(
			transition->Reward_);
		const auto actualTerminated = OaLunarVectorCopy<OaU8>(
			transition->Terminated_);
		const auto actualTruncated = OaLunarVectorCopy<OaU8>(
			transition->Truncated_);
		const auto actualReason = OaLunarVectorCopy<OaU32>(
			transition->EndReason_);
		for (OaU32 lane = 0U; lane < environments; ++lane) {
			const OaF32* laneObservation = actualObservation.data()
				+ static_cast<std::size_t>(lane)
					* OA_LUNAR_OBSERVATION_SIZE;
			if (contactPhase[lane]) {
				OaLunarVectorExpectContactObservationNear(
					laneObservation, expected[lane].Observation_,
					lane, stepIndex + 1U);
			} else {
				OaLunarVectorExpectObservationNear(
					laneObservation, expected[lane].Observation_,
					lane, stepIndex + 1U);
			}
			EXPECT_NEAR(
				actualReward[lane], expected[lane].Reward_, 6.0e-3F)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualTerminated[lane],
				expected[lane].Terminated_ ? 1U : 0U)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualTruncated[lane],
				expected[lane].Truncated_ ? 1U : 0U)
				<< "lane=" << lane << " step=" << stepIndex;
			ASSERT_EQ(
				actualReason[lane],
				static_cast<OaU32>(expected[lane].EndReason_))
				<< "lane=" << lane << " step=" << stepIndex;
		}

		completed = true;
		for (const auto& lane : scalar) {
			completed = completed
				and (lane.State().Terminated_ or lane.State().Truncated_);
		}
		if (completed) break;
	}
	ASSERT_TRUE(completed);

	auto telemetryResult = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(telemetryResult.IsOk())
		<< telemetryResult.GetStatus().ToString();
	const auto& telemetry = *telemetryResult;
	ASSERT_EQ(telemetry.Size(), environments);
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		const OaLunarLander3dState& expected = scalar[lane].State();
		const auto& actual = telemetry[lane];
		EXPECT_EQ(actual.EndReason_, InExpectedReason) << "lane=" << lane;
		EXPECT_EQ(actual.EndReason_, expected.EndReason_) << "lane=" << lane;
		EXPECT_EQ(actual.Terminated_, expected.Terminated_) << "lane=" << lane;
		EXPECT_EQ(actual.Truncated_, expected.Truncated_) << "lane=" << lane;
		EXPECT_EQ(actual.EpisodeStep_, expected.EpisodeStep_) << "lane=" << lane;
		const OaF32 returnTolerance = std::max(
			0.2F, static_cast<OaF32>(
				std::abs(expected.EpisodeReturn_) * 2.0e-3));
		EXPECT_NEAR(
			actual.EpisodeReturn_, expected.EpisodeReturn_, returnTolerance)
			<< "lane=" << lane;
		EXPECT_NEAR(actual.FuelRemaining_, expected.Fuel_, 5.0e-2F)
			<< "lane=" << lane;
		EXPECT_NEAR(
			actual.TerminalLinearSpeed_,
			expected.LinearVelocity_.Length(), 5.0e-2F)
			<< "lane=" << lane;
		EXPECT_NEAR(
			actual.TerminalAngularSpeed_,
			expected.AngularVelocityBody_.Length(), 5.0e-2F)
			<< "lane=" << lane;
		OaF32 expectedMaximumFootImpulse = 0.0F;
		for (const double impulse : expected.FootContactImpulses_) {
			expectedMaximumFootImpulse = std::max(
				expectedMaximumFootImpulse, static_cast<OaF32>(impulse));
		}
		const OaF32 maximumAccumulatedFootImpulse = static_cast<OaF32>(
			InEnvironmentConfig.MaxContactImpulse_
			* static_cast<double>(InEnvironmentConfig.PhysicsSubsteps_)
			* static_cast<double>(InEnvironmentConfig.ContactIterations_));
		EXPECT_LE(
			actual.MaximumFootImpulse_, maximumAccumulatedFootImpulse)
			<< "lane=" << lane;
		if (expectedMaximumFootImpulse > 0.0F) {
			EXPECT_GT(actual.MaximumFootImpulse_, 0.0F) << "lane=" << lane;
		} else {
			EXPECT_FLOAT_EQ(actual.MaximumFootImpulse_, 0.0F)
				<< "lane=" << lane;
		}
	}
	EXPECT_EQ(sawFootContact, InRequireFootContact);
	EXPECT_EQ(sawBodyContact, InRequireBodyContact);
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_F(TestLunarLander3dVector, SchemaOwnsResetStepAndKernelIdentities) {
	const auto& reset = OaOperationRegistry::LunarLander3dReset;
	EXPECT_EQ(reset.InputCount, 7U);
	EXPECT_EQ(reset.OutputCount, 4U);
	EXPECT_EQ(reset.AttributeCount, 4U);
	EXPECT_EQ(reset.ShapeRule, OaOperationShapeRule::Explicit);
	EXPECT_EQ(reset.DtypeRule, OaOperationDtypeRule::MatchInput);
	EXPECT_TRUE(reset.MutatesInput(3U));
	EXPECT_TRUE(reset.MutatesInput(4U));
	EXPECT_TRUE(reset.MutatesInput(5U));
	EXPECT_TRUE(reset.MutatesInput(6U));
	EXPECT_EQ(reset.AliasInputForOutput(0U), 3U);
	EXPECT_EQ(reset.AliasInputForOutput(3U), 6U);

	const auto& step = OaOperationRegistry::LunarLander3dStep;
	EXPECT_EQ(step.InputCount, OaOperationContract::MaxValues);
	EXPECT_EQ(step.OutputCount, OaOperationContract::MaxValues);
	EXPECT_EQ(step.AttributeCount, OaOperationContract::MaxAttributes);
	EXPECT_EQ(step.ControlFlow, OaOperationControlFlow::Conditional);
	EXPECT_FALSE(step.MutatesInput(0U));
	EXPECT_FALSE(step.MutatesInput(1U));
	EXPECT_TRUE(step.MutatesInput(5U));
	EXPECT_TRUE(step.MutatesInput(6U));
	EXPECT_TRUE(step.MutatesInput(7U));
	EXPECT_EQ(step.AliasInputForOutput(0U), 5U);
	EXPECT_EQ(step.AliasInputForOutput(1U), 6U);
	EXPECT_EQ(step.AliasInputForOutput(3U), 7U);
	EXPECT_EQ(step.AliasInputForOutput(7U),
		OaOperationContract::NoAliasInput);

	constexpr OaKernelId resetId = OA_COMPUTE_KERNEL_ID(
		OaComputeKernelPrefix::Ml, 280);
	constexpr OaKernelId stepId = OA_COMPUTE_KERNEL_ID(
		OaComputeKernelPrefix::Ml, 281);
	EXPECT_EQ(OaComputeKernelId::RlLunarLander3dReset, resetId);
	EXPECT_EQ(OaComputeKernelId::RlLunarLander3dStep, stepId);
	const auto* resetKernel = OaComputeKernelFindByName(
		"RlLunarLander3dReset");
	const auto* stepKernel = OaComputeKernelFindByName(
		"RlLunarLander3dStep");
	ASSERT_NE(resetKernel, nullptr);
	ASSERT_NE(stepKernel, nullptr);
	EXPECT_EQ(resetKernel->Id, resetId);
	EXPECT_EQ(stepKernel->Id, stepId);
}

TEST_VK(TestLunarLander3dVector, RejectsNonzeroConfigThatUnderflowsInFp32) {
	OaLunarLander3dVectorConfig config;
	config.Environment_.Gravity_ = std::numeric_limits<double>::denorm_min();
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsError());
	EXPECT_EQ(created.GetStatus().GetCode(), OaStatusCode::OutOfRange);

	config = {};
	config.Environment_.Gravity_ =
		static_cast<double>(std::numeric_limits<OaF32>::denorm_min());
	created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsError());
	EXPECT_EQ(created.GetStatus().GetCode(), OaStatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector, RejectsUnboundedFp32RewardAccumulation) {
	OaLunarLander3dVectorConfig config;
	config.Environment_.PositionPotentialWeight_ =
		static_cast<double>(std::numeric_limits<OaF32>::max());
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsError());
	EXPECT_EQ(created.GetStatus().GetCode(), OaStatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector, RejectsUnrepresentableDerivedFp32TimeAndFuel) {
	OaLunarLander3dVectorConfig subnormalTime;
	subnormalTime.Environment_.PolicyTimeStep_ =
		static_cast<double>(std::numeric_limits<OaF32>::min());
	subnormalTime.Environment_.PhysicsSubsteps_ = 64U;
	auto timeResult = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), subnormalTime);
	ASSERT_TRUE(timeResult.IsError());
	EXPECT_EQ(timeResult.GetStatus().GetCode(), OaStatusCode::OutOfRange);

	OaLunarLander3dVectorConfig subnormalDebit;
	subnormalDebit.Environment_.MainFuelRate_ =
		static_cast<double>(std::numeric_limits<OaF32>::min());
	auto debitResult = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), subnormalDebit);
	ASSERT_TRUE(debitResult.IsError());
	EXPECT_EQ(debitResult.GetStatus().GetCode(), OaStatusCode::OutOfRange);

	OaLunarLander3dVectorConfig stalledDebit;
	stalledDebit.Environment_.FuelCapacity_ = 1.0e8;
	stalledDebit.Environment_.MainFuelRate_ = 100.0;
	stalledDebit.Environment_.AttitudeFuelRate_ = 1.0e6;
	auto stalledResult = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), stalledDebit);
	ASSERT_TRUE(stalledResult.IsError());
	EXPECT_EQ(stalledResult.GetStatus().GetCode(), OaStatusCode::OutOfRange);
}

TEST_VK(TestLunarLander3dVector,
	TelemetryRejectsUnsubmittedAndSubmittedTransactions) {
	const OaLunarLander3dVectorConfig config{
		.Environments_ = 2U,
		.Seed_ = 0xe7d6c5b4a3928170ULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);

	EXPECT_TRUE(environment.HasActiveRecording());
	auto duringInitialRecording = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(duringInitialRecording.IsError());
	EXPECT_EQ(
		duringInitialRecording.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);

	auto completion = environment.Submit();
	ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
	EXPECT_FALSE(environment.HasActiveRecording());
	EXPECT_TRUE(environment.HasPendingEvent());
	auto duringPendingSubmission = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(duringPendingSubmission.IsError());
	EXPECT_EQ(
		duringPendingSubmission.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.Wait(*completion).IsOk());

	auto afterCompletion = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(afterCompletion.IsOk())
		<< afterCompletion.GetStatus().ToString();
	ASSERT_EQ(afterCompletion->Size(), 2U);
	for (const auto& telemetry : *afterCompletion) {
		EXPECT_EQ(telemetry.EpisodeStep_, 0U);
		EXPECT_EQ(telemetry.EndReason_, OaLunarEndReason::None);
	}

	ASSERT_TRUE(environment.Step(OaLunarVectorActions({0, 0})).IsOk());
	EXPECT_TRUE(environment.HasActiveRecording());
	auto duringStepRecording = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(duringStepRecording.IsError());
	EXPECT_EQ(
		duringStepRecording.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.Cancel().IsOk());
	EXPECT_FALSE(environment.HasActiveRecording());
	auto afterCancel = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(afterCancel.IsOk()) << afterCancel.GetStatus().ToString();
	for (const auto& telemetry : *afterCancel) {
		EXPECT_EQ(telemetry.EpisodeStep_, 0U);
		EXPECT_EQ(telemetry.EndReason_, OaLunarEndReason::None);
	}
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector,
	TelemetryReadbackUsesBorrowedNonGlobalEngineWithoutAmbientMixing) {
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
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);

	const OaLunarLander3dVectorConfig config{
		.Environments_ = 3U,
		.Seed_ = 0x4628d79a30be51c4ULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(*engine, config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());

	auto telemetryResult = environment.CopyEpisodeTelemetry();
	ASSERT_TRUE(telemetryResult.IsOk())
		<< telemetryResult.GetStatus().ToString();
	ASSERT_EQ(telemetryResult->Size(), config.Environments_);
	for (const auto& telemetry : *telemetryResult) {
		EXPECT_FLOAT_EQ(telemetry.EpisodeReturn_, 0.0F);
		EXPECT_FLOAT_EQ(
			telemetry.FuelRemaining_,
			static_cast<OaF32>(config.Environment_.FuelCapacity_));
		EXPECT_EQ(telemetry.EpisodeStep_, 0U);
		EXPECT_FALSE(telemetry.Terminated_);
		EXPECT_FALSE(telemetry.Truncated_);
		EXPECT_EQ(telemetry.EndReason_, OaLunarEndReason::None);
	}
	EXPECT_TRUE(environment.Close().IsOk());
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);
	const OaStatus engineClosed = engine->Close();
	EXPECT_TRUE(engineClosed.IsOk()) << engineClosed.ToString();
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);
}

TEST_VK(TestLunarLander3dVector, CancelledInitialResetRequiresRecoveryReset) {
	const OaLunarLander3dVectorConfig config{
		.Environments_ = 2U,
		.Seed_ = 0x11a22b33c44d55e6ULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(environment.Cancel().IsOk());
	EXPECT_EQ(
		environment.Step(OaLunarVectorActions({0, 0})).GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	EXPECT_EQ(environment.ResetDone().GetCode(), OaStatusCode::FailedPrecondition);
	ASSERT_TRUE(environment.Reset().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto actual = OaLunarVectorCopy<OaF32>(environment.Observation());
	auto scalar = OaLunarVectorScalarLane(config, 0U, 0U);
	ASSERT_TRUE(scalar.IsValid());
	OaLunarVectorExpectObservationNear(
		actual.data(), scalar.Observation(), 0U, 0U);
	auto recoveredStep = environment.Step(OaLunarVectorActions({0, 0}));
	ASSERT_TRUE(recoveredStep.IsOk()) << recoveredStep.GetStatus().ToString();
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto recoveredReason = OaLunarVectorCopy<OaU32>(
		recoveredStep->EndReason_);
	const auto recoveredTruncated = OaLunarVectorCopy<OaU8>(
		recoveredStep->Truncated_);
	EXPECT_EQ(recoveredReason[0], static_cast<OaU32>(OaLunarEndReason::None));
	EXPECT_EQ(recoveredReason[1], static_cast<OaU32>(OaLunarEndReason::None));
	EXPECT_EQ(recoveredTruncated[0], 0U);
	EXPECT_EQ(recoveredTruncated[1], 0U);
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, ReseedMetadataCommitsOnlyWithAcceptedBatch) {
	constexpr OaU64 seedA = 0x1011121314151617ULL;
	constexpr OaU64 seedB = 0x8081828384858687ULL;
	constexpr OaU64 seedC = 0xc0c1c2c3c4c5c6c7ULL;
	constexpr OaU64 seedD = 0xd0d1d2d3d4d5d6d7ULL;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = 2U,
		.Seed_ = seedA,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());

	ASSERT_TRUE(environment.ResetEnvironment(seedB).IsOk());
	EXPECT_EQ(environment.Config().Seed_, seedA);
	ASSERT_TRUE(environment.ResetDone().IsOk());
	ASSERT_TRUE(environment.Reset().IsOk());
	const auto* graph = OaRlEnvironmentExecutionAccess::Context(environment)
		.SemanticGraph();
	ASSERT_NE(graph, nullptr);
	ASSERT_EQ(graph->OperationCount(), 3U);
	for (const auto& operation : graph->Operations()) {
		ASSERT_EQ(operation.Name, "LunarLander3dReset");
		ASSERT_GE(operation.Attributes.Size(), 3U);
		EXPECT_EQ(operation.Attributes[2].Name, "Seed");
		EXPECT_EQ(operation.Attributes[2].UnsignedInteger, seedB);
	}
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	EXPECT_EQ(environment.Config().Seed_, seedB);

	ASSERT_TRUE(environment.ResetEnvironment(seedC).IsOk());
	EXPECT_EQ(environment.Config().Seed_, seedB);
	ASSERT_TRUE(environment.Cancel().IsOk());
	EXPECT_EQ(environment.Config().Seed_, seedB);
	ASSERT_TRUE(environment.Reset().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	OaLunarLander3dVectorConfig seedBConfig = config;
	seedBConfig.Seed_ = seedB;
	const auto afterCancel = OaLunarVectorCopy<OaF32>(environment.Observation());
	auto scalarB = OaLunarVectorScalarLane(seedBConfig, 0U, 0U);
	ASSERT_TRUE(scalarB.IsValid());
	OaLunarVectorExpectObservationNear(
		afterCancel.data(), scalarB.Observation(), 0U, 0U);

	ASSERT_TRUE(environment.ResetEnvironment(seedD).IsOk());
	auto& context = OaRlEnvironmentExecutionAccess::Context(environment);
	const OaMatrix& observation = environment.Observation();
	const OaMatrix& reason = environment.EndReason();
	const auto orphan = context.RecordOperation(
		OaOperationRegistry::LunarLander3dReset,
		{&observation, &observation, &reason, &observation, &reason,
			&observation, &reason},
		{&observation, &reason, &observation, &reason},
		{
			OaOperationAttribute::FromUnsignedInteger("EnvironmentVersion", 0U),
			OaOperationAttribute::FromUnsignedInteger("StateLayoutVersion", 0U),
			OaOperationAttribute::FromUnsignedInteger("Seed", seedD),
			OaOperationAttribute::FromBoolean("OnlyCompleted", false),
		});
	ASSERT_TRUE(orphan.IsOk()) << orphan.GetStatus().ToString();
	auto failed = environment.Submit();
	ASSERT_TRUE(failed.IsError());
	EXPECT_EQ(environment.Config().Seed_, seedB);
	EXPECT_TRUE(environment.IsOpen());
	ASSERT_TRUE(environment.Reset().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	EXPECT_EQ(environment.Config().Seed_, seedB);
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, ResetMatchesFrozenScalarManifests) {
	for (const OaU32 environments : {1U, 7U, 257U}) {
		const OaLunarLander3dVectorConfig config{
			.Environments_ = environments,
			.Seed_ = 0x7a61d3c59e2048bfULL,
		};
		auto created = OaLunarLander3dVector::CreateFlat(
			OaLunarVectorRuntime(), config);
		ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
		auto environment = OaStdMove(*created);
		ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
		const auto actual = OaLunarVectorCopy<OaF32>(environment.Observation());
		for (OaU32 lane = 0U; lane < environments; ++lane) {
			auto scalar = OaLunarVectorScalarLane(config, lane, 0U);
			ASSERT_TRUE(scalar.IsValid()) << scalar.Error();
			OaLunarVectorExpectObservationNear(
				actual.data() + static_cast<std::size_t>(lane)
					* OA_LUNAR_OBSERVATION_SIZE,
				scalar.Observation(), lane, 0U);
		}
		EXPECT_TRUE(environment.Close().IsOk());
	}
}

TEST_VK(TestLunarLander3dVector, FixedActionTraceMatchesScalarFp64Oracle) {
	constexpr OaU32 environments = 5U;
	constexpr OaU32 traceSteps = 24U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0xc34d8217a6950befULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	std::vector<OaLunarScalarEnvironment> scalar;
	scalar.reserve(environments);
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		scalar.push_back(OaLunarVectorScalarLane(config, lane, 0U));
		ASSERT_TRUE(scalar.back().IsValid()) << scalar.back().Error();
	}
	for (OaU32 stepIndex = 0U; stepIndex < traceSteps; ++stepIndex) {
		std::vector<OaI32> actions(environments);
		for (OaU32 lane = 0U; lane < environments; ++lane) {
			actions[lane] = static_cast<OaI32>((stepIndex + lane * 3U) % 8U);
		}
		auto transition = environment.Step(OaLunarVectorActions(actions));
		ASSERT_TRUE(transition.IsOk()) << transition.GetStatus().ToString();
		ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
		const auto actualObservation = OaLunarVectorCopy<OaF32>(
			transition->NextObservation_);
		const auto actualReward = OaLunarVectorCopy<OaF32>(transition->Reward_);
		const auto actualTerminated = OaLunarVectorCopy<OaU8>(
			transition->Terminated_);
		const auto actualTruncated = OaLunarVectorCopy<OaU8>(
			transition->Truncated_);
		const auto actualReason = OaLunarVectorCopy<OaU32>(
			transition->EndReason_);
		for (OaU32 lane = 0U; lane < environments; ++lane) {
			const OaLunarTransition expected = scalar[lane].Step(
				static_cast<OaU32>(actions[lane]));
			ASSERT_TRUE(expected.Valid_) << expected.Error_;
			OaLunarVectorExpectObservationNear(
				actualObservation.data() + static_cast<std::size_t>(lane)
					* OA_LUNAR_OBSERVATION_SIZE,
				expected.Observation_, lane, stepIndex + 1U);
			EXPECT_NEAR(actualReward[lane], expected.Reward_, 4.0e-3F)
				<< "lane=" << lane << " step=" << stepIndex;
			EXPECT_EQ(actualTerminated[lane], expected.Terminated_ ? 1U : 0U);
			EXPECT_EQ(actualTruncated[lane], expected.Truncated_ ? 1U : 0U);
			EXPECT_EQ(actualReason[lane], static_cast<OaU32>(expected.EndReason_));
		}
	}
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector,
	ScriptedLandingMatchesScalarThroughContactDwellAndTerminal) {
	OaLunarLander3dConfig config;
	config.SafeDwellSteps_ = 12U;
	ASSERT_NO_FATAL_FAILURE(OaLunarVectorRunEpisodeDifferential(
		config,
		0x50494c4f545f4556ULL,
		OaLunarVectorOraclePolicy::Scripted,
		OaLunarEndReason::SafeLanding,
		true,
		false));
}

TEST_VK(TestLunarLander3dVector,
	FailureAndTruncationReasonsMatchScalarThroughTerminalPhysics) {
	OaLunarLander3dConfig hardImpact;
	ASSERT_NO_FATAL_FAILURE(OaLunarVectorRunEpisodeDifferential(
		hardImpact,
		0x484152445f464f4fULL,
		OaLunarVectorOraclePolicy::Coast,
		OaLunarEndReason::HardFootImpact,
		true,
		false));

	OaLunarLander3dConfig bodyImpact;
	for (auto& foot : bodyImpact.FootSupports_) {
		foot.BodyOffset_.ComponentY_ = 1.0;
	}
	ASSERT_NO_FATAL_FAILURE(OaLunarVectorRunEpisodeDifferential(
		bodyImpact,
		0x424f44595f494d50ULL,
		OaLunarVectorOraclePolicy::Coast,
		OaLunarEndReason::BodyImpact,
		false,
		true));

	OaLunarLander3dConfig timeLimit;
	timeLimit.Gravity_ = 0.0;
	timeLimit.MaxEpisodeSteps_ = 3U;
	ASSERT_NO_FATAL_FAILURE(OaLunarVectorRunEpisodeDifferential(
		timeLimit,
		0x54494d455f4c494dULL,
		OaLunarVectorOraclePolicy::Coast,
		OaLunarEndReason::TimeLimit,
		false,
		false));

	OaLunarLander3dConfig outOfBounds;
	outOfBounds.TaskMaximumY_ = 4.0;
	ASSERT_NO_FATAL_FAILURE(OaLunarVectorRunEpisodeDifferential(
		outOfBounds,
		0x4f55545f4f465f42ULL,
		OaLunarVectorOraclePolicy::Coast,
		OaLunarEndReason::OutOfBounds,
		false,
		false));
}

TEST_VK(TestLunarLander3dVector, ExternalStopIsLaneLocalAndInvalidActionWins) {
	constexpr OaU32 environments = 4U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0x38a1c5e792b4d60fULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());

	const std::vector<OaI32> actions{0, 1, 8, 7};
	const std::vector<OaU8> externalStops{0U, 1U, 1U, 0U};
	auto wrongMask = environment.Step(
		OaLunarVectorActions(actions), OaLunarVectorActions({0, 1, 1, 0}));
	ASSERT_TRUE(wrongMask.IsError());
	EXPECT_EQ(wrongMask.GetStatus().GetCode(), OaStatusCode::DtypeMismatch);
	auto transition = environment.Step(
		OaLunarVectorActions(actions),
		OaLunarVectorExternalStops(externalStops));
	ASSERT_TRUE(transition.IsOk()) << transition.GetStatus().ToString();
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());

	const auto previousObservation = OaLunarVectorCopy<OaF32>(
		transition->Observation_);
	const auto nextObservation = OaLunarVectorCopy<OaF32>(
		transition->NextObservation_);
	const auto reward = OaLunarVectorCopy<OaF32>(transition->Reward_);
	const auto terminated = OaLunarVectorCopy<OaU8>(transition->Terminated_);
	const auto truncated = OaLunarVectorCopy<OaU8>(transition->Truncated_);
	const auto reason = OaLunarVectorCopy<OaU32>(transition->EndReason_);
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		auto scalar = OaLunarVectorScalarLane(config, lane, 0U);
		ASSERT_TRUE(scalar.IsValid()) << scalar.Error();
		const OaLunarTransition expected = scalar.Step(
			static_cast<OaU32>(actions[lane]), externalStops[lane] != 0U);
		ASSERT_TRUE(expected.Valid_) << expected.Error_;
		OaLunarVectorExpectObservationNear(
			nextObservation.data() + static_cast<std::size_t>(lane)
				* OA_LUNAR_OBSERVATION_SIZE,
			expected.Observation_, lane, 1U);
		EXPECT_NEAR(reward[lane], expected.Reward_, 4.0e-3F)
			<< "lane=" << lane;
		EXPECT_EQ(terminated[lane], expected.Terminated_ ? 1U : 0U);
		EXPECT_EQ(truncated[lane], expected.Truncated_ ? 1U : 0U);
		EXPECT_EQ(reason[lane], static_cast<OaU32>(expected.EndReason_));
		if (externalStops[lane] != 0U and actions[lane] >= 0
			and actions[lane] < 8) {
			EXPECT_EQ(scalar.State().EpisodeStep_, 0U);
			for (OaU32 component = 0U;
				component < OA_LUNAR_OBSERVATION_SIZE;
				++component) {
				const std::size_t index = static_cast<std::size_t>(lane)
					* OA_LUNAR_OBSERVATION_SIZE + component;
				EXPECT_EQ(nextObservation[index], previousObservation[index]);
			}
		}
	}
	EXPECT_FLOAT_EQ(reward[1], 0.0F);
	EXPECT_EQ(terminated[1], 0U);
	EXPECT_EQ(truncated[1], 1U);
	EXPECT_EQ(reason[1], static_cast<OaU32>(OaLunarEndReason::ExternalStop));
	EXPECT_FLOAT_EQ(reward[2],
		static_cast<OaF32>(config.Environment_.FailurePenalty_));
	EXPECT_EQ(terminated[2], 1U);
	EXPECT_EQ(truncated[2], 0U);
	EXPECT_EQ(reason[2], static_cast<OaU32>(OaLunarEndReason::InvalidAction));
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, InvalidActionIsLaneLocalAndDoesNotShiftOtherRng) {
	constexpr OaU32 environments = 6U;
	constexpr OaU32 invalidLane = 2U;
	constexpr OaU32 laterResetLane = 4U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0x196a4e7bd205c83fULL,
	};
	auto firstCreated = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	auto controlCreated = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(firstCreated.IsOk()) << firstCreated.GetStatus().ToString();
	ASSERT_TRUE(controlCreated.IsOk()) << controlCreated.GetStatus().ToString();
	auto first = OaStdMove(*firstCreated);
	auto control = OaStdMove(*controlCreated);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(first).IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(control).IsOk());

	std::vector<OaI32> firstActions = {0, 1, 8, 3, 4, 7};
	std::vector<OaI32> controlActions = firstActions;
	controlActions[invalidLane] = 0;
	auto firstStep = first.Step(OaLunarVectorActions(firstActions));
	auto controlStep = control.Step(OaLunarVectorActions(controlActions));
	ASSERT_TRUE(firstStep.IsOk());
	ASSERT_TRUE(controlStep.IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(first).IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(control).IsOk());
	const auto firstObservation = OaLunarVectorCopy<OaF32>(
		firstStep->NextObservation_);
	const auto controlObservation = OaLunarVectorCopy<OaF32>(
		controlStep->NextObservation_);
	const auto reward = OaLunarVectorCopy<OaF32>(firstStep->Reward_);
	const auto terminated = OaLunarVectorCopy<OaU8>(firstStep->Terminated_);
	const auto reason = OaLunarVectorCopy<OaU32>(firstStep->EndReason_);
	EXPECT_FLOAT_EQ(reward[invalidLane],
		static_cast<OaF32>(config.Environment_.FailurePenalty_));
	EXPECT_EQ(terminated[invalidLane], 1U);
	EXPECT_EQ(reason[invalidLane],
		static_cast<OaU32>(OaLunarEndReason::InvalidAction));
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		if (lane == invalidLane) continue;
		for (OaU32 component = 0U;
			component < OA_LUNAR_OBSERVATION_SIZE;
			++component) {
			const std::size_t index = static_cast<std::size_t>(lane)
				* OA_LUNAR_OBSERVATION_SIZE + component;
			EXPECT_EQ(firstObservation[index], controlObservation[index]);
		}
	}

	ASSERT_TRUE(first.ResetDone().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(first).IsOk());
	const auto afterReset = OaLunarVectorCopy<OaF32>(first.Observation());
	for (OaU32 lane = 0U; lane < environments; ++lane) {
		if (lane == invalidLane) continue;
		for (OaU32 component = 0U;
			component < OA_LUNAR_OBSERVATION_SIZE;
			++component) {
			const std::size_t index = static_cast<std::size_t>(lane)
				* OA_LUNAR_OBSERVATION_SIZE + component;
			EXPECT_EQ(afterReset[index], firstObservation[index]);
		}
	}
	auto scalarEpisodeOne = OaLunarVectorScalarLane(config, invalidLane, 1U);
	ASSERT_TRUE(scalarEpisodeOne.IsValid());
	OaLunarVectorExpectObservationNear(
		afterReset.data() + static_cast<std::size_t>(invalidLane)
			* OA_LUNAR_OBSERVATION_SIZE,
		scalarEpisodeOne.Observation(), invalidLane, 0U);

	std::vector<OaI32> laterActions(environments, 0);
	laterActions[laterResetLane] = std::numeric_limits<OaI32>::max();
	ASSERT_TRUE(first.Step(OaLunarVectorActions(laterActions)).IsOk());
	ASSERT_TRUE(control.Step(OaLunarVectorActions(laterActions)).IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(first).IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(control).IsOk());
	ASSERT_TRUE(first.ResetDone().IsOk());
	ASSERT_TRUE(control.ResetDone().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(first).IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(control).IsOk());
	const auto firstLaterReset = OaLunarVectorCopy<OaF32>(first.Observation());
	const auto controlLaterReset = OaLunarVectorCopy<OaF32>(control.Observation());
	for (OaU32 component = 0U;
		component < OA_LUNAR_OBSERVATION_SIZE;
		++component) {
		const std::size_t index = static_cast<std::size_t>(laterResetLane)
			* OA_LUNAR_OBSERVATION_SIZE + component;
		EXPECT_EQ(firstLaterReset[index], controlLaterReset[index]);
	}
	EXPECT_TRUE(first.Close().IsOk());
	EXPECT_TRUE(control.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, CompletedLaneKeepsTerminalStateUntilReset) {
	constexpr OaU32 environments = 3U;
	constexpr OaU32 completedLane = 1U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0xd47ca2359b1806efULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	auto terminal = environment.Step(OaLunarVectorActions({0, 8, 0}));
	ASSERT_TRUE(terminal.IsOk()) << terminal.GetStatus().ToString();
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto terminalObservation = OaLunarVectorCopy<OaF32>(
		terminal->NextObservation_);
	auto repeated = environment.Step(OaLunarVectorActions({0, 0, 0}));
	ASSERT_TRUE(repeated.IsOk()) << repeated.GetStatus().ToString();
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto repeatedObservation = OaLunarVectorCopy<OaF32>(
		repeated->NextObservation_);
	const auto repeatedReward = OaLunarVectorCopy<OaF32>(repeated->Reward_);
	const auto repeatedTerminated = OaLunarVectorCopy<OaU8>(
		repeated->Terminated_);
	const auto repeatedTruncated = OaLunarVectorCopy<OaU8>(
		repeated->Truncated_);
	const auto repeatedReason = OaLunarVectorCopy<OaU32>(
		repeated->EndReason_);
	EXPECT_FLOAT_EQ(repeatedReward[completedLane], 0.0F);
	EXPECT_EQ(repeatedTerminated[completedLane], 1U);
	EXPECT_EQ(repeatedTruncated[completedLane], 0U);
	EXPECT_EQ(repeatedReason[completedLane],
		static_cast<OaU32>(OaLunarEndReason::InvalidAction));
	for (OaU32 component = 0U;
		component < OA_LUNAR_OBSERVATION_SIZE;
		++component) {
		const std::size_t index = static_cast<std::size_t>(completedLane)
			* OA_LUNAR_OBSERVATION_SIZE + component;
		EXPECT_EQ(repeatedObservation[index], terminalObservation[index]);
	}
	ASSERT_TRUE(environment.ResetDone().IsOk());
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto resetReason = OaLunarVectorCopy<OaU32>(environment.EndReason());
	EXPECT_EQ(resetReason[completedLane],
		static_cast<OaU32>(OaLunarEndReason::None));
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, OddReuseAndInvalidActionPoisonStayBounded) {
	constexpr OaU32 environments = 257U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0x5d3e8c714a09b26fULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	for (OaU32 reuse = 0U; reuse < 8U; ++reuse) {
		std::vector<OaI32> actions(environments, static_cast<OaI32>(reuse % 8U));
		for (OaU32 lane = reuse; lane < environments; lane += 31U) {
			actions[lane] = (lane & 1U) == 0U
				? std::numeric_limits<OaI32>::min()
				: std::numeric_limits<OaI32>::max();
		}
		auto step = environment.Step(OaLunarVectorActions(actions));
		ASSERT_TRUE(step.IsOk()) << step.GetStatus().ToString();
		ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
		const auto observation = OaLunarVectorCopy<OaF32>(step->NextObservation_);
		const auto reward = OaLunarVectorCopy<OaF32>(step->Reward_);
		for (const OaF32 value : observation) EXPECT_TRUE(std::isfinite(value));
		for (const OaF32 value : reward) EXPECT_TRUE(std::isfinite(value));
		ASSERT_TRUE(environment.ResetDone().IsOk());
		ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	}
	EXPECT_TRUE(environment.Close().IsOk());
}

TEST_VK(TestLunarLander3dVector, LargeOddLaneCountUsesQueriedLimits) {
	constexpr OaU32 environments = 65537U;
	const OaLunarLander3dVectorConfig config{
		.Environments_ = environments,
		.Seed_ = 0x2b74cd901f6a853eULL,
	};
	auto created = OaLunarLander3dVector::CreateFlat(
		OaLunarVectorRuntime(), config);
	if (created.IsError()
		and created.GetStatus().GetCode() == OaStatusCode::OutOfMemory) {
		GTEST_SKIP() << created.GetStatus().ToString();
	}
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto environment = OaStdMove(*created);
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	std::vector<OaI32> actions(environments, 0);
	actions.back() = 8;
	auto step = environment.Step(OaLunarVectorActions(actions));
	ASSERT_TRUE(step.IsOk()) << step.GetStatus().ToString();
	ASSERT_TRUE(OaLunarVectorSubmitAndWait(environment).IsOk());
	const auto reward = OaLunarVectorCopy<OaF32>(step->Reward_);
	const auto terminated = OaLunarVectorCopy<OaU8>(step->Terminated_);
	EXPECT_FLOAT_EQ(reward.back(),
		static_cast<OaF32>(config.Environment_.FailurePenalty_));
	EXPECT_EQ(terminated.back(), 1U);
	EXPECT_TRUE(environment.Close().IsOk());
}
