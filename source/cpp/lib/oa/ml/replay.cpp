#include <oa/ml/replay.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <limits>

namespace {

oa::MatrixShape prefix(oa::U32 inFirst, const oa::MatrixShape& inSuffix) {
	oa::MatrixShape shape;
	shape.rank = inSuffix.rank + 1;
	shape[0] = static_cast<oa::I64>(inFirst);
	for (oa::I32 index = 0; index < inSuffix.rank; ++index) {
		shape[index + 1] = inSuffix[index];
	}
	return shape;
}

oa::Result<oa::U32> elements(const oa::MatrixShape& inShape, bool inAllowScalar) {
	if ((!inAllowScalar && inShape.rank == 0)
		|| inShape.rank < 0 || inShape.rank > OA_MAX_TENSOR_DIMS - 1) {
		return oa::Status::invalidArgument("invalid replay field rank");
	}
	oa::U64 count = 1;
	for (oa::I32 dim = 0; dim < inShape.rank; ++dim) {
		if (inShape[dim] <= 0) {
			return oa::Status::invalidArgument("replay field dimensions must be positive");
		}
		count *= static_cast<oa::U64>(inShape[dim]);
		if (count > std::numeric_limits<oa::U32>::max()) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"replay field exceeds 32-bit GPU indexing");
		}
	}
	return static_cast<oa::U32>(count);
}

bool isVector(const oa::Matrix& inMatrix, oa::U32 inBatch, oa::ScalarType inDtype) {
	return inMatrix.getShape() == oa::MatrixShape{static_cast<oa::I64>(inBatch)}
		&& inMatrix.getDtype() == inDtype;
}

} // namespace

bool oa::ReplayBatch::isValid() const noexcept {
	return !observation.isEmpty() && !action.isEmpty()
		&& !nextObservation.isEmpty() && !reward.isEmpty()
		&& !terminated.isEmpty() && !truncated.isEmpty() && !index.isEmpty();
}

oa::Result<oa::ReplayBuffer> oa::ReplayBuffer::create(
	const oa::ReplayConfig& inConfig) {
	if (inConfig.capacity == 0
		|| (inConfig.actionDtype != oa::ScalarType::Int32
			&& inConfig.actionDtype != oa::ScalarType::Float32)) {
		return oa::Status::invalidArgument(
			"oa::ReplayBuffer expects positive capacity and Int32/Float32 actions");
	}
	auto observationElements = elements(inConfig.observationShape, false);
	if (observationElements.isError()) return observationElements.getStatus();
	auto actionElements = elements(inConfig.actionShape, true);
	if (actionElements.isError()) return actionElements.getStatus();

	oa::ReplayBuffer result;
	result.config_ = inConfig;
	result.observationElements_ = *observationElements;
	result.actionElements_ = *actionElements;
	result.storage_ = oa::ReplayBatch{
		.observation = oa::FnMatrix::empty(
			prefix(inConfig.capacity, inConfig.observationShape),
			oa::ScalarType::Float32),
		.action = oa::FnMatrix::empty(
			prefix(inConfig.capacity, inConfig.actionShape),
			inConfig.actionDtype),
		.nextObservation = oa::FnMatrix::empty(
			prefix(inConfig.capacity, inConfig.observationShape),
			oa::ScalarType::Float32),
		.reward = oa::FnMatrix::empty({static_cast<oa::I64>(inConfig.capacity)},
			oa::ScalarType::Float32),
		.terminated = oa::FnMatrix::empty({static_cast<oa::I64>(inConfig.capacity)},
			oa::ScalarType::UInt8),
		.truncated = oa::FnMatrix::empty({static_cast<oa::I64>(inConfig.capacity)},
			oa::ScalarType::UInt8),
		.index = oa::FnMatrix::empty({static_cast<oa::I64>(inConfig.capacity)},
			oa::ScalarType::UInt32),
	};
	if (!result.isValid()) {
		return oa::Status::error(oa::StatusCode::OutOfMemory,
			"oa::ReplayBuffer could not allocate storage");
	}
	return result;
}

oa::Status oa::ReplayBuffer::append(const oa::ReplayTransition& inTransition) {
	if (!isValid()) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::ReplayBuffer::append requires a valid buffer");
	if (inTransition.reward.rank() != 1
		|| inTransition.reward.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"oa::ReplayBuffer::append expects FP32 reward [batch]");
	}
	const oa::U32 batch = static_cast<oa::U32>(inTransition.reward.size(0));
	if (batch == 0 || batch > config_.capacity
		|| inTransition.observation.getShape() != prefix(batch, config_.observationShape)
		|| inTransition.observation.getDtype() != oa::ScalarType::Float32
		|| inTransition.nextObservation.getShape() != prefix(batch, config_.observationShape)
		|| inTransition.nextObservation.getDtype() != oa::ScalarType::Float32
		|| inTransition.action.getShape() != prefix(batch, config_.actionShape)
		|| inTransition.action.getDtype() != config_.actionDtype
		|| !isVector(inTransition.terminated, batch, oa::ScalarType::UInt8)
		|| !isVector(inTransition.truncated, batch, oa::ScalarType::UInt8)) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"oa::ReplayBuffer::append transition does not match its configured schema");
	}
	struct Push {
		oa::U32 Cursor, capacity, Batch, observationElements, ActionElements;
	} push{cursor_, config_.capacity, batch, observationElements_, actionElements_};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
	const oa::U32 work = std::max({batch * observationElements_,
		batch * actionElements_, batch});
	oa::ExecutionSession::getActive().add( "RlReplayAppend",
		{&inTransition.observation, &inTransition.action,
		 &inTransition.nextObservation, &inTransition.reward,
		 &inTransition.terminated, &inTransition.truncated,
		 &storage_.observation, &storage_.action, &storage_.nextObservation,
		 &storage_.reward, &storage_.terminated, &storage_.truncated},
		access, &push, sizeof(push), (work + 255U) / 256U);
	cursor_ = (cursor_ + batch) % config_.capacity;
	size_ = std::min(config_.capacity, size_ + batch);
	return oa::Status::ok();
}

oa::Result<oa::ReplayBatch> oa::ReplayBuffer::sample(
	oa::U32 inBatchSize, oa::U64 inSeed) const {
	if (!isValid() || size_ == 0 || inBatchSize == 0) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ReplayBuffer::sample requires non-empty storage and batch");
	}
	oa::ReplayBatch result{
		.observation = oa::FnMatrix::empty(
			prefix(inBatchSize, config_.observationShape), oa::ScalarType::Float32),
		.action = oa::FnMatrix::empty(
			prefix(inBatchSize, config_.actionShape), config_.actionDtype),
		.nextObservation = oa::FnMatrix::empty(
			prefix(inBatchSize, config_.observationShape), oa::ScalarType::Float32),
		.reward = oa::FnMatrix::empty({static_cast<oa::I64>(inBatchSize)},
			oa::ScalarType::Float32),
		.terminated = oa::FnMatrix::empty({static_cast<oa::I64>(inBatchSize)},
			oa::ScalarType::UInt8),
		.truncated = oa::FnMatrix::empty({static_cast<oa::I64>(inBatchSize)},
			oa::ScalarType::UInt8),
		.index = oa::FnMatrix::empty({static_cast<oa::I64>(inBatchSize)},
			oa::ScalarType::UInt32),
	};
	struct Push {
		oa::U32 size, Batch, observationElements, ActionElements, SeedLo, SeedHi;
	} push{size_, inBatchSize, observationElements_, actionElements_,
		static_cast<oa::U32>(inSeed), static_cast<oa::U32>(inSeed >> 32U)};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write};
	const oa::U32 work = std::max({inBatchSize * observationElements_,
		inBatchSize * actionElements_, inBatchSize});
	oa::ExecutionSession::getActive().add( "RlReplaySample",
		{&storage_.observation, &storage_.action, &storage_.nextObservation,
		 &storage_.reward, &storage_.terminated, &storage_.truncated,
		 &result.observation, &result.action, &result.nextObservation,
		 &result.reward, &result.terminated, &result.truncated, &result.index},
		access, &push, sizeof(push), (work + 255U) / 256U);
	return result;
}

void oa::ReplayBuffer::reset() noexcept {
	size_ = 0;
	cursor_ = 0;
}

bool oa::ReplayBuffer::isValid() const noexcept {
	return config_.capacity > 0 && storage_.isValid();
}
