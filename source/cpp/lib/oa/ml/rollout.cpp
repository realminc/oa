#include <oa/ml/rollout.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>

namespace {

oa::MatrixShape prefixShape(
	oa::U32 inFirst,
	const oa::MatrixShape& inSuffix) {
	oa::MatrixShape shape;
	shape.rank = inSuffix.rank + 1;
	shape[0] = static_cast<oa::I64>(inFirst);
	for (oa::I32 index = 0; index < inSuffix.rank; ++index) {
		shape[index + 1] = inSuffix[index];
	}
	return shape;
}

oa::MatrixShape prefixShape(
	oa::U32 inFirst,
	oa::U32 inSecond,
	const oa::MatrixShape& inSuffix) {
	oa::MatrixShape shape;
	shape.rank = inSuffix.rank + 2;
	shape[0] = static_cast<oa::I64>(inFirst);
	shape[1] = static_cast<oa::I64>(inSecond);
	for (oa::I32 index = 0; index < inSuffix.rank; ++index) {
		shape[index + 2] = inSuffix[index];
	}
	return shape;
}

bool isF32Vector(const oa::Matrix& inMatrix, oa::U32 inEnvironments) {
	return inMatrix.getDtype() == oa::ScalarType::Float32
		&& inMatrix.getShape()
			== oa::MatrixShape{static_cast<oa::I64>(inEnvironments)};
}

bool isU8Vector(const oa::Matrix& inMatrix, oa::U32 inEnvironments) {
	return inMatrix.getDtype() == oa::ScalarType::UInt8
		&& inMatrix.getShape()
			== oa::MatrixShape{static_cast<oa::I64>(inEnvironments)};
}

} // namespace

bool oa::RolloutBatch::isValid() const noexcept {
	return !observation.isEmpty() && !action.isEmpty() && !reward.isEmpty()
		&& !value.isEmpty() && !nextValue.isEmpty()
		&& !oldLogProbability.isEmpty() && !terminated.isEmpty()
		&& !truncated.isEmpty() && !valid.isEmpty()
		&& !advantage.isEmpty() && !ret.isEmpty();
}

oa::Result<oa::RolloutBuffer> oa::RolloutBuffer::create(
	const oa::RolloutConfig& inConfig) {
	if (inConfig.time == 0 || inConfig.environments == 0
		|| inConfig.observationShape.rank < 1
		|| inConfig.observationShape.rank > OA_MAX_TENSOR_DIMS - 2) {
		return oa::Status::invalidArgument(
			"oa::RolloutBuffer::create expects non-zero time/environments and an observation rank in [1,6]");
	}
	for (oa::I32 dim = 0; dim < inConfig.observationShape.rank; ++dim) {
		if (inConfig.observationShape[dim] <= 0) {
			return oa::Status::invalidArgument(
				"oa::RolloutBuffer::create observation dimensions must be positive");
		}
	}
	oa::U64 observationElements = 1;
	for (oa::I32 dim = 0; dim < inConfig.observationShape.rank; ++dim) {
		const oa::U64 dimension = static_cast<oa::U64>(inConfig.observationShape[dim]);
		if (observationElements
			> static_cast<oa::U64>(oa::Limits<oa::U32>::max()) / dimension) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::RolloutBuffer::create observation size exceeds the current 32-bit GPU indexing limit");
		}
		observationElements *= dimension;
	}
	const oa::U64 rolloutElements = static_cast<oa::U64>(inConfig.time)
		* inConfig.environments;
	if (rolloutElements > oa::Limits<oa::U32>::max()
		|| observationElements
			> static_cast<oa::U64>(oa::Limits<oa::U32>::max())
				/ rolloutElements) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::RolloutBuffer::create exceeds the current 32-bit GPU indexing limit");
	}

	oa::RolloutBuffer result;
	result.config_ = inConfig;
	result.observationElements_ = static_cast<oa::U32>(observationElements);
	const oa::MatrixShape scalarShape{
		static_cast<oa::I64>(inConfig.time),
		static_cast<oa::I64>(inConfig.environments)};
	result.batch_ = oa::RolloutBatch{
		.observation = oa::FnMatrix::empty(prefixShape(
			inConfig.time, inConfig.environments, inConfig.observationShape),
			oa::ScalarType::Float32),
		.action = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Int32),
		.reward = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Float32),
		.value = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Float32),
		.nextValue = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Float32),
		.oldLogProbability = oa::FnMatrix::empty(
			scalarShape, oa::ScalarType::Float32),
		.terminated = oa::FnMatrix::empty(scalarShape, oa::ScalarType::UInt8),
		.truncated = oa::FnMatrix::empty(scalarShape, oa::ScalarType::UInt8),
		.valid = oa::FnMatrix::empty(scalarShape, oa::ScalarType::UInt8),
		.advantage = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Float32),
		.ret = oa::FnMatrix::empty(scalarShape, oa::ScalarType::Float32),
	};
	if (!result.batch_.isValid()) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::RolloutBuffer::create could not allocate rollout storage");
	}
	return result;
}

oa::Status oa::RolloutBuffer::append(const oa::RolloutTransition& inTransition) {
	if (!isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::RolloutBuffer::append requires a valid buffer");
	}
	if (finalized_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::RolloutBuffer::append requires reset after finalize");
	}
	if (isFull()) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::RolloutBuffer::append exceeds rollout capacity");
	}
	const oa::MatrixShape observationShape = prefixShape(
		config_.environments, config_.observationShape);
	const oa::MatrixShape actionShape{static_cast<oa::I64>(config_.environments)};
	const bool valid = inTransition.observation.getShape() == observationShape
		&& inTransition.observation.getDtype() == oa::ScalarType::Float32
		&& inTransition.action.getShape() == actionShape
		&& inTransition.action.getDtype() == oa::ScalarType::Int32
		&& isF32Vector(inTransition.reward, config_.environments)
		&& isF32Vector(inTransition.value, config_.environments)
		&& isF32Vector(inTransition.nextValue, config_.environments)
		&& isF32Vector(inTransition.logProbability, config_.environments)
		&& isU8Vector(inTransition.terminated, config_.environments)
		&& isU8Vector(inTransition.truncated, config_.environments);
	if (!valid) {
		return oa::Status::error(
			oa::StatusCode::ShapeMismatch,
			"oa::RolloutBuffer::append transition does not match the configured categorical rollout shape/dtype contract");
	}

	struct Push {
		oa::U32 step;
		oa::U32 environments;
		oa::U32 observationElements;
	} push{
		.step = size_,
		.environments = config_.environments,
		.observationElements = observationElements_,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write,
	};
	const oa::U32 workItems = oa::max(
		config_.environments * observationElements_, config_.environments);
	oa::ExecutionSession::getActive().add(
		"RlRolloutAppend",
		{&inTransition.observation, &inTransition.action,
		 &inTransition.reward, &inTransition.value,
		 &inTransition.nextValue, &inTransition.logProbability,
		 &inTransition.terminated, &inTransition.truncated,
		 &batch_.observation, &batch_.action,
		 &batch_.reward, &batch_.value, &batch_.nextValue,
		 &batch_.oldLogProbability, &batch_.terminated,
		 &batch_.truncated, &batch_.valid},
		access,
		&push,
		sizeof(push),
		(workItems + 255U) / 256U);
	++size_;
	return oa::Status::ok();
}

oa::Status oa::RolloutBuffer::finalize(const oa::GaeConfig& inConfig) {
	if (!isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::RolloutBuffer::finalize requires a valid buffer");
	}
	if (finalized_) return oa::Status::ok();
	if (!isFull()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::RolloutBuffer::finalize requires a complete rollout");
	}
	const oa::Status status = oa::FnAdvantage::gaeInto(
		batch_.reward, batch_.value, batch_.nextValue,
		batch_.terminated, batch_.truncated,
		batch_.advantage, batch_.ret, inConfig);
	if (status.isError()) return status;
	finalized_ = true;
	return oa::Status::ok();
}

void oa::RolloutBuffer::reset() {
	if (!isValid()) return;
	struct Push { oa::U32 count; } push{
		.count = static_cast<oa::U32>(batch_.valid.numElements())};
	oa::BufferAccess access[] = {oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add(
		"RlRolloutReset", {&batch_.valid}, access,
		&push, sizeof(push), (push.count + 255U) / 256U);
	size_ = 0;
	finalized_ = false;
}

void oa::RolloutBuffer::abortUnsubmitted() noexcept {
	size_ = 0;
	finalized_ = false;
}
