#include "CartPole.h"

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include "../../../Source/Private/Oa/Core/OperationRegistry.gen.h"
#include "../../../Source/Private/Oa/Ml/Rl/EnvironmentExecution.h"

#include <bit>
#include <cmath>
#include <limits>

namespace {

constexpr OaU64 FnvOffset = 14695981039346656037ULL;
constexpr OaU64 FnvPrime = 1099511628211ULL;

void HashU32(OaU64& InOutHash, OaU32 InValue) noexcept {
	for (OaU32 shift = 0; shift < 32U; shift += 8U) {
		InOutHash ^= static_cast<OaU8>(InValue >> shift);
		InOutHash *= FnvPrime;
	}
}

} // namespace

OaU64 OaTutorialCartPoleConfig::DynamicsIdentity() const noexcept {
	OaU64 hash = FnvOffset;
	HashU32(hash, DynamicsVersion);
	HashU32(hash, MaxEpisodeSteps);
	HashU32(hash, std::bit_cast<OaU32>(Gravity));
	HashU32(hash, std::bit_cast<OaU32>(CartMass));
	HashU32(hash, std::bit_cast<OaU32>(PoleMass));
	HashU32(hash, std::bit_cast<OaU32>(HalfPoleLength));
	HashU32(hash, std::bit_cast<OaU32>(ForceMagnitude));
	HashU32(hash, std::bit_cast<OaU32>(TimeStep));
	HashU32(hash, std::bit_cast<OaU32>(PositionThreshold));
	HashU32(hash, std::bit_cast<OaU32>(AngleThresholdRadians));
	return hash;
}

bool OaTutorialCartPoleStep::IsValid() const noexcept {
	return !Observation.IsEmpty() && !NextObservation.IsEmpty()
		&& !Reward.IsEmpty() && !Terminated.IsEmpty()
		&& !Truncated.IsEmpty() && !Done.IsEmpty();
}

OaTutorialCartPole::OaTutorialCartPole(OaEngine& InEngine)
	: OaRlEnvironment(InEngine) {}

OaU64 OaTutorialCartPole::EffectiveSeed_() const noexcept {
	return HasPendingSeed_ ? PendingSeed_ : Config_.Seed;
}

OaResult<OaTutorialCartPole> OaTutorialCartPole::Create(
	OaEngine& InEngine,
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
	if (!InEngine.IsReady()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole::Create requires a ready engine");
	}

	const OaMatrixShape stateShape{
		static_cast<OaI64>(InConfig.Environments), 4};
	const OaMatrixShape vectorShape{
		static_cast<OaI64>(InConfig.Environments)};
	OaTutorialCartPole result(InEngine);
	if (!result.IsOpen()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole::Create could not open its execution session");
	}
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
	const OaStatus initialized = result.RecordCommands([&]() -> OaStatus {
		result.State_ = OaFnMatrix::Empty(stateShape, OaScalarType::Float32);
		result.TransitionObservation_ = OaFnMatrix::Empty(
			stateShape, OaScalarType::Float32);
		result.Reward_ = OaFnMatrix::Empty(vectorShape, OaScalarType::Float32);
		result.Terminated_ = OaFnMatrix::Empty(vectorShape, OaScalarType::UInt8);
		result.Truncated_ = OaFnMatrix::Empty(vectorShape, OaScalarType::UInt8);
		result.Done_ = OaFnMatrix::Empty(vectorShape, OaScalarType::UInt8);
		result.EpisodeSteps_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::UInt32);
		result.EpisodeIndex_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::UInt32);
		if (!result.IsValid()) {
			return OaStatus::Error(
				OaStatusCode::OutOfMemory,
				"OaTutorialCartPole::Create could not allocate environment storage");
		}
		return result.RecordReset_(false);
	});
	if (initialized.IsError()) return initialized;
	return result;
}

bool OaTutorialCartPole::IsValid() const noexcept {
	const OaMatrixShape stateShape{
		static_cast<OaI64>(Config_.Environments), 4};
	const OaMatrixShape vectorShape{
		static_cast<OaI64>(Config_.Environments)};
	const auto matches = [](const OaMatrix& InMatrix,
		const OaMatrixShape& InShape, OaScalarType InDtype) {
		return !InMatrix.IsEmpty() && InMatrix.GetShape() == InShape
			&& InMatrix.GetDtype() == InDtype;
	};
	return matches(State_, stateShape, OaScalarType::Float32)
		&& matches(TransitionObservation_, stateShape, OaScalarType::Float32)
		&& matches(Reward_, vectorShape, OaScalarType::Float32)
		&& matches(Terminated_, vectorShape, OaScalarType::UInt8)
		&& matches(Truncated_, vectorShape, OaScalarType::UInt8)
		&& matches(Done_, vectorShape, OaScalarType::UInt8)
		&& matches(EpisodeSteps_, vectorShape, OaScalarType::UInt32)
		&& matches(EpisodeIndex_, vectorShape, OaScalarType::UInt32);
}

OaStatus OaTutorialCartPole::RecordReset_(bool InOnlyDone) {
	if (!IsValid()) return OaStatus::Error(
		OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole reset requires a valid environment");
	if (InOnlyDone && !HasCommittedState_ && !HasPendingFullReset_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole completed reset requires submitted state or an earlier full reset in this transaction");
	}
	const OaU64 seed = EffectiveSeed_();
	if (!InOnlyDone) HasPendingFullReset_ = true;
	struct Push {
		OaU32 Environments;
		OaU32 SeedLow;
		OaU32 SeedHigh;
		OaU32 OnlyDone;
	} push{
		.Environments = Config_.Environments,
		.SeedLow = static_cast<OaU32>(seed),
		.SeedHigh = static_cast<OaU32>(seed >> 32U),
		.OnlyDone = InOnlyDone ? 1U : 0U,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::ReadWrite,
	};
	auto& context = OaRlEnvironmentExecutionAccess::Context(*this);
	const auto semantic = context.RecordOperation(
		OaOperationRegistry::CartPoleReset,
		{&Done_, &State_, &EpisodeSteps_, &EpisodeIndex_},
		{&Done_, &State_, &EpisodeSteps_, &EpisodeIndex_},
		{
			OaOperationAttribute::FromUnsignedInteger("Seed", seed),
			OaOperationAttribute::FromBoolean("OnlyCompleted", InOnlyDone),
		});
	if (semantic.IsError()) return semantic.GetStatus();
	context.Add(
		"RlCartPoleReset",
		{&Done_, &State_, &EpisodeSteps_, &EpisodeIndex_},
		access, &push, sizeof(push),
		(Config_.Environments + 255U) / 256U, 1, 1,
		OaOperationRegistry::CartPoleReset.Name, 0,
		OaOperationRegistry::CartPoleReset.Hash, 0, 0,
		semantic.GetValue());
	return OaStatus::Ok();
}

OaStatus OaTutorialCartPole::Reset() {
	return ResetEnvironment(EffectiveSeed_());
}

OaStatus OaTutorialCartPole::ResetDone() {
	return ResetCompleted();
}

OaStatus OaTutorialCartPole::RecordResetEnvironment_(OaU64 InSeed) {
	PendingSeed_ = InSeed;
	HasPendingSeed_ = true;
	return RecordReset_(false);
}

OaStatus OaTutorialCartPole::RecordResetCompleted_() {
	return RecordReset_(true);
}

OaResult<OaRlEnvironmentTransition> OaTutorialCartPole::RecordStepEnvironment_(
	const OaMatrix& InAction) {
	auto step = RecordStep_(InAction);
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
	auto transition = StepEnvironment(InAction);
	if (transition.IsError()) return transition.GetStatus();
	return OaTutorialCartPoleStep{
		.Observation = transition->Observation,
		.NextObservation = transition->NextObservation,
		.Reward = transition->Reward,
		.Terminated = transition->Terminated,
		.Truncated = transition->Truncated,
		.Done = Done_,
	};
}

OaResult<OaTutorialCartPoleStep> OaTutorialCartPole::RecordStep_(
	const OaMatrix& InAction) {
	if (!IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole::Step requires a valid environment");
	}
	if (!HasCommittedState_ && !HasPendingFullReset_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTutorialCartPole::Step requires submitted state or an earlier full reset in this transaction");
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
		OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
	};
	auto& context = OaRlEnvironmentExecutionAccess::Context(*this);
	const auto semantic = context.RecordOperation(
		OaOperationRegistry::CartPoleStep,
		{&InAction, &State_, &Done_, &EpisodeSteps_},
		{&TransitionObservation_, &State_, &Reward_, &Terminated_,
		 &Truncated_, &Done_, &EpisodeSteps_},
		{
			OaOperationAttribute::FromUnsignedInteger(
				"DynamicsVersion", OaTutorialCartPoleConfig::DynamicsVersion),
			OaOperationAttribute::FromUnsignedInteger(
				"DynamicsIdentity", Config_.DynamicsIdentity()),
			OaOperationAttribute::FromUnsignedInteger(
				"MaxEpisodeSteps", Config_.MaxEpisodeSteps),
			OaOperationAttribute::FromFloat("Gravity", Config_.Gravity),
			OaOperationAttribute::FromFloat(
				"ForceMagnitude", Config_.ForceMagnitude),
			OaOperationAttribute::FromFloat("TimeStep", Config_.TimeStep),
			OaOperationAttribute::FromFloat(
				"PositionThreshold", Config_.PositionThreshold),
			OaOperationAttribute::FromFloat(
				"AngleThresholdRadians", Config_.AngleThresholdRadians),
		});
	if (semantic.IsError()) return semantic.GetStatus();
	context.Add(
		"RlCartPoleStep",
		{&InAction, &State_, &TransitionObservation_, &Reward_,
		 &Terminated_, &Truncated_, &Done_, &EpisodeSteps_},
		access, &push, sizeof(push),
		(Config_.Environments + 255U) / 256U, 1, 1,
		OaOperationRegistry::CartPoleStep.Name, 0,
		OaOperationRegistry::CartPoleStep.Hash, 0, 0,
		semantic.GetValue());
	return OaTutorialCartPoleStep{
		.Observation = TransitionObservation_,
		.NextObservation = State_,
		.Reward = Reward_,
		.Terminated = Terminated_,
		.Truncated = Truncated_,
		.Done = Done_,
	};
}

void OaTutorialCartPole::CommitRecordedState_() noexcept {
	if (HasPendingSeed_) Config_.Seed = PendingSeed_;
	if (HasPendingFullReset_) HasCommittedState_ = true;
	HasPendingSeed_ = false;
	HasPendingFullReset_ = false;
}

void OaTutorialCartPole::RollbackRecordedState_() noexcept {
	HasPendingSeed_ = false;
	HasPendingFullReset_ = false;
}
