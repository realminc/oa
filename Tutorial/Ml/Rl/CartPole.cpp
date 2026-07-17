#include "CartPole.h"

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <limits>

bool OaTutorialCartPoleStep::IsValid() const noexcept {
	return !Observation.IsEmpty() && !NextObservation.IsEmpty()
		&& !Reward.IsEmpty() && !Terminated.IsEmpty()
		&& !Truncated.IsEmpty() && !Done.IsEmpty();
}

OaResult<OaTutorialCartPole> OaTutorialCartPole::Create(
	const OaTutorialCartPoleConfig& InConfig) {
	const bool finite = std::isfinite(InConfig.Gravity)
		&& std::isfinite(InConfig.CartMass)
		&& std::isfinite(InConfig.PoleMass)
		&& std::isfinite(InConfig.HalfPoleLength)
		&& std::isfinite(InConfig.ForceMagnitude)
		&& std::isfinite(InConfig.TimeStep)
		&& std::isfinite(InConfig.PositionThreshold)
		&& std::isfinite(InConfig.AngleThresholdRadians);
	if (!finite || InConfig.Environments == 0 || InConfig.MaxEpisodeSteps == 0
		|| InConfig.CartMass <= 0.0F || InConfig.PoleMass <= 0.0F
		|| InConfig.HalfPoleLength <= 0.0F || InConfig.ForceMagnitude <= 0.0F
		|| InConfig.TimeStep <= 0.0F || InConfig.PositionThreshold <= 0.0F
		|| InConfig.AngleThresholdRadians <= 0.0F) {
		return OaStatus::InvalidArgument(
			"OaTutorialCartPole::Create received an invalid environment configuration");
	}
	if (InConfig.Environments > std::numeric_limits<OaU32>::max() / 4U) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"OaTutorialCartPole::Create exceeds the 32-bit GPU indexing limit");
	}

	const OaMatrixShape stateShape{
		static_cast<OaI64>(InConfig.Environments), 4};
	const OaMatrixShape vectorShape{
		static_cast<OaI64>(InConfig.Environments)};
	OaTutorialCartPole result;
	result.Config_ = InConfig;
	result.Spec_ = {
		.Observation = OaRlFieldSpec::Box(
			"observation", {4}, OaScalarType::Float32),
		.Action = OaRlFieldSpec::Discrete("action", 2),
		.Reward = OaRlFieldSpec::Box(
			"reward", {}, OaScalarType::Float32, 0.0, 1.0),
		.Terminated = OaRlFieldSpec::Binary("terminated"),
		.Truncated = OaRlFieldSpec::Binary("truncated"),
	};
	OA_RETURN_IF_ERROR(result.Spec_.ValidateDefinition());
	result.State_ = OaFnMatrix::Empty(stateShape, OaScalarType::Float32);
	result.TransitionObservation_ = OaFnMatrix::Empty(
		stateShape, OaScalarType::Float32);
	result.Reward_ = OaFnMatrix::Empty(vectorShape, OaScalarType::Float32);
	result.Terminated_ = OaFnMatrix::Zeros(vectorShape, OaScalarType::UInt8);
	result.Truncated_ = OaFnMatrix::Zeros(vectorShape, OaScalarType::UInt8);
	result.Done_ = OaFnMatrix::Zeros(vectorShape, OaScalarType::UInt8);
	result.EpisodeSteps_ = OaFnMatrix::Zeros(vectorShape, OaScalarType::UInt32);
	result.EpisodeIndex_ = OaFnMatrix::Zeros(vectorShape, OaScalarType::UInt32);
	if (!result.IsValid()) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaTutorialCartPole::Create could not allocate environment storage");
	}
	result.Reset();
	return result;
}

bool OaTutorialCartPole::IsValid() const noexcept {
	return !State_.IsEmpty() && !TransitionObservation_.IsEmpty()
		&& !Reward_.IsEmpty() && !Terminated_.IsEmpty()
		&& !Truncated_.IsEmpty() && !Done_.IsEmpty()
		&& !EpisodeSteps_.IsEmpty() && !EpisodeIndex_.IsEmpty();
}

void OaTutorialCartPole::RecordReset_(bool InOnlyDone) {
	if (!IsValid()) return;
	struct Push {
		OaU32 Environments;
		OaU32 SeedLow;
		OaU32 SeedHigh;
		OaU32 OnlyDone;
	} push{
		.Environments = Config_.Environments,
		.SeedLow = static_cast<OaU32>(Config_.Seed),
		.SeedHigh = static_cast<OaU32>(Config_.Seed >> 32U),
		.OnlyDone = InOnlyDone ? 1U : 0U,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::ReadWrite,
	};
	OaContext::GetDefault().Add(
		"RlCartPoleReset",
		{&Done_, &State_, &EpisodeSteps_, &EpisodeIndex_},
		access, &push, sizeof(push),
		(Config_.Environments + 255U) / 256U);
}

void OaTutorialCartPole::Reset() {
	RecordReset_(false);
}

void OaTutorialCartPole::ResetDone() {
	RecordReset_(true);
}

OaStatus OaTutorialCartPole::ResetEnvironment(OaU64 InSeed) {
	Config_.Seed = InSeed;
	Reset();
	return OaStatus::Ok();
}

OaStatus OaTutorialCartPole::ResetCompleted() {
	ResetDone();
	return OaStatus::Ok();
}

OaResult<OaRlEnvironmentTransition> OaTutorialCartPole::StepEnvironment(
	const OaMatrix& InAction) {
	auto step = Step(InAction);
	if (step.IsError()) return step.GetStatus();
	return OaRlEnvironmentTransition{
		.Observation = step->Observation,
		.NextObservation = step->NextObservation,
		.Reward = step->Reward,
		.Terminated = step->Terminated,
		.Truncated = step->Truncated,
	};
}

OaResult<OaTutorialCartPoleStep> OaTutorialCartPole::Step(
	const OaMatrix& InAction) {
	if (!IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole::Step requires a valid environment");
	}
	OA_RETURN_IF_ERROR(Spec_.ValidateAction(InAction, Config_.Environments));

	struct Push {
		OaU32 Environments;
		OaU32 MaxEpisodeSteps;
		OaF32 Gravity;
		OaF32 CartMass;
		OaF32 PoleMass;
		OaF32 HalfPoleLength;
		OaF32 ForceMagnitude;
		OaF32 TimeStep;
		OaF32 PositionThreshold;
		OaF32 AngleThresholdRadians;
	} push{
		.Environments = Config_.Environments,
		.MaxEpisodeSteps = Config_.MaxEpisodeSteps,
		.Gravity = Config_.Gravity,
		.CartMass = Config_.CartMass,
		.PoleMass = Config_.PoleMass,
		.HalfPoleLength = Config_.HalfPoleLength,
		.ForceMagnitude = Config_.ForceMagnitude,
		.TimeStep = Config_.TimeStep,
		.PositionThreshold = Config_.PositionThreshold,
		.AngleThresholdRadians = Config_.AngleThresholdRadians,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::ReadWrite,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::ReadWrite,
	};
	OaContext::GetDefault().Add(
		"RlCartPoleStep",
		{&InAction, &State_, &TransitionObservation_, &Reward_,
		 &Terminated_, &Truncated_, &Done_, &EpisodeSteps_},
		access, &push, sizeof(push),
		(Config_.Environments + 255U) / 256U);
	return OaTutorialCartPoleStep{
		.Observation = TransitionObservation_,
		.NextObservation = State_,
		.Reward = Reward_,
		.Terminated = Terminated_,
		.Truncated = Truncated_,
		.Done = Done_,
	};
}
