#include <ml/rl/cartPole.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <ml/rl/environmentKernelPack.h>
#include <ml/rl/gen/environmentOpRegistry.h>
#include <oa/ml/environmentExecution.h>

#include <bit>
#include <cmath>
#include <limits>

namespace {

constexpr oa::U64 FnvOffset = 14695981039346656037ULL;
constexpr oa::U64 FnvPrime = 1099511628211ULL;

void hashU32(oa::U64& inOutHash, oa::U32 inValue) noexcept {
	for (oa::U32 shift = 0; shift < 32U; shift += 8U) {
		inOutHash ^= static_cast<oa::U8>(inValue >> shift);
		inOutHash *= FnvPrime;
	}
}

} // namespace

namespace oa {

oa::U64 CartPoleConfig::dynamicsIdentity() const noexcept {
	oa::U64 hash = FnvOffset;
	hashU32(hash, dynamicsVersion);
	hashU32(hash, maxEpisodeSteps);
	hashU32(hash, std::bit_cast<oa::U32>(gravity));
	hashU32(hash, std::bit_cast<oa::U32>(cartMass));
	hashU32(hash, std::bit_cast<oa::U32>(poleMass));
	hashU32(hash, std::bit_cast<oa::U32>(halfPoleLength));
	hashU32(hash, std::bit_cast<oa::U32>(forceMagnitude));
	hashU32(hash, std::bit_cast<oa::U32>(timeStep));
	hashU32(hash, std::bit_cast<oa::U32>(positionThreshold));
	hashU32(hash, std::bit_cast<oa::U32>(angleThresholdRadians));
	return hash;
}

bool CartPoleStep::isValid() const noexcept {
	return !observation.isEmpty() && !nextObservation.isEmpty()
		&& !reward.isEmpty() && !terminated.isEmpty()
		&& !truncated.isEmpty() && !done.isEmpty();
}

CartPole::CartPole(oa::Engine& inEngine)
	: oa::Environment(inEngine) {}

oa::U64 CartPole::effectiveSeed_() const noexcept {
	return hasPendingSeed_ ? pendingSeed_ : config_.seed;
}

oa::Result<CartPole> CartPole::create(
	oa::Engine& inEngine,
	const CartPoleConfig& inConfig) {
	const bool finite = std::isfinite(inConfig.gravity)
		&& std::isfinite(inConfig.cartMass)
		&& std::isfinite(inConfig.poleMass)
		&& std::isfinite(inConfig.halfPoleLength)
		&& std::isfinite(inConfig.forceMagnitude)
		&& std::isfinite(inConfig.timeStep)
		&& std::isfinite(inConfig.positionThreshold)
		&& std::isfinite(inConfig.angleThresholdRadians);
	if (!finite || inConfig.environments == 0 || inConfig.maxEpisodeSteps == 0
		|| inConfig.cartMass <= 0.0F || inConfig.poleMass <= 0.0F
		|| inConfig.halfPoleLength <= 0.0F || inConfig.forceMagnitude <= 0.0F
		|| inConfig.timeStep <= 0.0F || inConfig.positionThreshold <= 0.0F
		|| inConfig.angleThresholdRadians <= 0.0F) {
		return oa::Status::invalidArgument(
			"CartPole::create received an invalid environment configuration");
	}
	if (inConfig.environments > std::numeric_limits<oa::U32>::max() / 4U) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"CartPole::create exceeds the 32-bit GPU indexing limit");
	}
	if (!inEngine.isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"CartPole::create requires a ready engine");
	}
	OA_RETURN_IF_ERROR(oa::ensureEnvironmentKernelPack(inEngine));

	const oa::MatrixShape stateShape{
		static_cast<oa::I64>(inConfig.environments), 4};
	const oa::MatrixShape vectorShape{
		static_cast<oa::I64>(inConfig.environments)};
	CartPole result(inEngine);
	if (!result.isOpen()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"CartPole::create could not open its execution session");
	}
	result.config_ = inConfig;
	result.spec_ = {
		.observation = oa::EnvironmentSpace::box(
			"observation", {4}, oa::ScalarType::Float32),
		.action = oa::EnvironmentSpace::discrete("action", 2),
		.reward = oa::EnvironmentSpace::box(
			"reward", {}, oa::ScalarType::Float32, 0.0, 1.0),
		.terminated = oa::EnvironmentSpace::binary("terminated"),
		.truncated = oa::EnvironmentSpace::binary("truncated"),
	};
	OA_RETURN_IF_ERROR(result.spec_.validateDefinition());
	const oa::Status initialized = result.recordCommands([&]() -> oa::Status {
		result.state_ = oa::FnMatrix::empty(stateShape, oa::ScalarType::Float32);
		result.transitionObservation_ = oa::FnMatrix::empty(
			stateShape, oa::ScalarType::Float32);
		result.reward_ = oa::FnMatrix::empty(vectorShape, oa::ScalarType::Float32);
		result.terminated_ = oa::FnMatrix::empty(vectorShape, oa::ScalarType::UInt8);
		result.truncated_ = oa::FnMatrix::empty(vectorShape, oa::ScalarType::UInt8);
		result.done_ = oa::FnMatrix::empty(vectorShape, oa::ScalarType::UInt8);
		result.episodeSteps_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::UInt32);
		result.episodeIndex_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::UInt32);
		if (!result.isValid()) {
			return oa::Status::error(
				oa::StatusCode::OutOfMemory,
				"CartPole::create could not allocate environment storage");
		}
		return result.recordReset_(false);
	});
	if (initialized.isError()) return initialized;
	return result;
}

bool CartPole::isValid() const noexcept {
	const oa::MatrixShape stateShape{
		static_cast<oa::I64>(config_.environments), 4};
	const oa::MatrixShape vectorShape{
		static_cast<oa::I64>(config_.environments)};
	const auto matches = [](const oa::Matrix& inMatrix,
		const oa::MatrixShape& inShape, oa::ScalarType inDtype) {
		return !inMatrix.isEmpty() && inMatrix.getShape() == inShape
			&& inMatrix.getDtype() == inDtype;
	};
	return matches(state_, stateShape, oa::ScalarType::Float32)
		&& matches(transitionObservation_, stateShape, oa::ScalarType::Float32)
		&& matches(reward_, vectorShape, oa::ScalarType::Float32)
		&& matches(terminated_, vectorShape, oa::ScalarType::UInt8)
		&& matches(truncated_, vectorShape, oa::ScalarType::UInt8)
		&& matches(done_, vectorShape, oa::ScalarType::UInt8)
		&& matches(episodeSteps_, vectorShape, oa::ScalarType::UInt32)
		&& matches(episodeIndex_, vectorShape, oa::ScalarType::UInt32);
}

oa::Status CartPole::recordReset_(bool inOnlyDone) {
	if (!isValid()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition,
			"CartPole reset requires a valid environment");
	if (inOnlyDone && !hasCommittedState_ && !hasPendingFullReset_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"CartPole completed reset requires submitted state or an earlier full reset in this transaction");
	}
	const oa::U64 seed = effectiveSeed_();
	if (!inOnlyDone) hasPendingFullReset_ = true;
	struct Push {
		oa::U32 environments;
		oa::U32 seedLow;
		oa::U32 seedHigh;
		oa::U32 onlyDone;
	} push{
		.environments = config_.environments,
		.seedLow = static_cast<oa::U32>(seed),
		.seedHigh = static_cast<oa::U32>(seed >> 32U),
		.onlyDone = inOnlyDone ? 1U : 0U,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::ReadWrite,
	};
	auto& context = oa::EnvironmentExecutionAccess::session(*this);
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::cartPoleReset,
		{&done_, &state_, &episodeSteps_, &episodeIndex_},
		{&done_, &state_, &episodeSteps_, &episodeIndex_},
		{
			oa::OpAttribute::fromUnsignedInteger("seed", seed),
			oa::OpAttribute::fromBoolean("onlyCompleted", inOnlyDone),
		});
	if (semantic.isError()) return semantic.getStatus();
	context.add(
		"RlCartPoleReset",
		{&done_, &state_, &episodeSteps_, &episodeIndex_},
		access, &push, sizeof(push),
		(config_.environments + 255U) / 256U, 1, 1,
		oa::detail::opRegistry::FnEnvironment::cartPoleReset.name, 0,
		oa::detail::opRegistry::FnEnvironment::cartPoleReset.hash, 0, 0,
		semantic.getValue());
	return oa::Status::ok();
}

oa::Status CartPole::reset() {
	return oa::Environment::reset(effectiveSeed_());
}

oa::Status CartPole::resetDone() {
	return resetCompleted();
}

oa::Status CartPole::recordReset_(oa::U64 inSeed) {
	pendingSeed_ = inSeed;
	hasPendingSeed_ = true;
	return recordReset_(false);
}

oa::Status CartPole::recordResetCompleted_() {
	return recordReset_(true);
}

oa::Result<oa::EnvironmentTransition> CartPole::recordStep_(
	const oa::Matrix& inAction) {
	auto step = recordDetailedStep_(inAction);
	if (step.isError()) return step.getStatus();
	return oa::EnvironmentTransition{
		.observation = step->observation,
		.nextObservation = step->nextObservation,
		.reward = step->reward,
		.terminated = step->terminated,
		.truncated = step->truncated,
	};
}

oa::Result<CartPoleStep> CartPole::step(
	const oa::Matrix& inAction) {
	auto transition = oa::Environment::step(inAction);
	if (transition.isError()) return transition.getStatus();
	return CartPoleStep{
		.observation = transition->observation,
		.nextObservation = transition->nextObservation,
		.reward = transition->reward,
		.terminated = transition->terminated,
		.truncated = transition->truncated,
		.done = done_,
	};
}

oa::Result<CartPoleStep> CartPole::recordDetailedStep_(
	const oa::Matrix& inAction) {
	if (!isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"CartPole::step requires a valid environment");
	}
	if (!hasCommittedState_ && !hasPendingFullReset_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"CartPole::step requires submitted state or an earlier full reset in this transaction");
	}
	OA_RETURN_IF_ERROR(spec_.validateAction(inAction, config_.environments));

	struct Push {
		oa::U32 environments;
		oa::U32 maxEpisodeSteps;
		oa::F32 gravity;
		oa::F32 cartMass;
		oa::F32 poleMass;
		oa::F32 halfPoleLength;
		oa::F32 forceMagnitude;
		oa::F32 timeStep;
		oa::F32 positionThreshold;
		oa::F32 angleThresholdRadians;
	} push{
		.environments = config_.environments,
		.maxEpisodeSteps = config_.maxEpisodeSteps,
		.gravity = config_.gravity,
		.cartMass = config_.cartMass,
		.poleMass = config_.poleMass,
		.halfPoleLength = config_.halfPoleLength,
		.forceMagnitude = config_.forceMagnitude,
		.timeStep = config_.timeStep,
		.positionThreshold = config_.positionThreshold,
		.angleThresholdRadians = config_.angleThresholdRadians,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
	};
	auto& context = oa::EnvironmentExecutionAccess::session(*this);
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::cartPoleStep,
		{&inAction, &state_, &done_, &episodeSteps_},
		{&transitionObservation_, &state_, &reward_, &terminated_,
		 &truncated_, &done_, &episodeSteps_},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"dynamicsVersion", CartPoleConfig::dynamicsVersion),
			oa::OpAttribute::fromUnsignedInteger(
				"dynamicsIdentity", config_.dynamicsIdentity()),
			oa::OpAttribute::fromUnsignedInteger(
				"maxEpisodeSteps", config_.maxEpisodeSteps),
			oa::OpAttribute::fromFloat("gravity", config_.gravity),
			oa::OpAttribute::fromFloat(
				"forceMagnitude", config_.forceMagnitude),
			oa::OpAttribute::fromFloat("timeStep", config_.timeStep),
			oa::OpAttribute::fromFloat(
				"positionThreshold", config_.positionThreshold),
			oa::OpAttribute::fromFloat(
				"angleThresholdRadians", config_.angleThresholdRadians),
		});
	if (semantic.isError()) return semantic.getStatus();
	context.add(
		"RlCartPoleStep",
		{&inAction, &state_, &transitionObservation_, &reward_,
		 &terminated_, &truncated_, &done_, &episodeSteps_},
		access, &push, sizeof(push),
		(config_.environments + 255U) / 256U, 1, 1,
		oa::detail::opRegistry::FnEnvironment::cartPoleStep.name, 0,
		oa::detail::opRegistry::FnEnvironment::cartPoleStep.hash, 0, 0,
		semantic.getValue());
	return CartPoleStep{
		.observation = transitionObservation_,
		.nextObservation = state_,
		.reward = reward_,
		.terminated = terminated_,
		.truncated = truncated_,
		.done = done_,
	};
}

void CartPole::commitRecordedState_() noexcept {
	if (hasPendingSeed_) config_.seed = pendingSeed_;
	if (hasPendingFullReset_) hasCommittedState_ = true;
	hasPendingSeed_ = false;
	hasPendingFullReset_ = false;
}

void CartPole::rollbackRecordedState_() noexcept {
	hasPendingSeed_ = false;
	hasPendingFullReset_ = false;
}

} // namespace oa
