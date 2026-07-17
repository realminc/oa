#include <Oa/Ml/Rl/Replay.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <limits>

namespace {

OaMatrixShape Prefix(OaU32 InFirst, const OaMatrixShape& InSuffix) {
	OaMatrixShape shape;
	shape.Rank = InSuffix.Rank + 1;
	shape[0] = static_cast<OaI64>(InFirst);
	for (OaI32 index = 0; index < InSuffix.Rank; ++index) {
		shape[index + 1] = InSuffix[index];
	}
	return shape;
}

OaResult<OaU32> Elements(const OaMatrixShape& InShape, bool InAllowScalar) {
	if ((!InAllowScalar && InShape.Rank == 0)
		|| InShape.Rank < 0 || InShape.Rank > OA_MAX_TENSOR_DIMS - 1) {
		return OaStatus::InvalidArgument("invalid replay field rank");
	}
	OaU64 count = 1;
	for (OaI32 dim = 0; dim < InShape.Rank; ++dim) {
		if (InShape[dim] <= 0) {
			return OaStatus::InvalidArgument("replay field dimensions must be positive");
		}
		count *= static_cast<OaU64>(InShape[dim]);
		if (count > std::numeric_limits<OaU32>::max()) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"replay field exceeds 32-bit GPU indexing");
		}
	}
	return static_cast<OaU32>(count);
}

bool IsVector(const OaMatrix& InMatrix, OaU32 InBatch, OaScalarType InDtype) {
	return InMatrix.GetShape() == OaMatrixShape{static_cast<OaI64>(InBatch)}
		&& InMatrix.GetDtype() == InDtype;
}

} // namespace

bool OaRlReplayBatch::IsValid() const noexcept {
	return !Observation.IsEmpty() && !Action.IsEmpty()
		&& !NextObservation.IsEmpty() && !Reward.IsEmpty()
		&& !Terminated.IsEmpty() && !Truncated.IsEmpty() && !Index.IsEmpty();
}

OaResult<OaRlReplayBuffer> OaRlReplayBuffer::Create(
	const OaRlReplayConfig& InConfig) {
	if (InConfig.Capacity == 0
		|| (InConfig.ActionDtype != OaScalarType::Int32
			&& InConfig.ActionDtype != OaScalarType::Float32)) {
		return OaStatus::InvalidArgument(
			"OaRlReplayBuffer expects positive capacity and Int32/Float32 actions");
	}
	auto observationElements = Elements(InConfig.ObservationShape, false);
	if (observationElements.IsError()) return observationElements.GetStatus();
	auto actionElements = Elements(InConfig.ActionShape, true);
	if (actionElements.IsError()) return actionElements.GetStatus();

	OaRlReplayBuffer result;
	result.Config_ = InConfig;
	result.ObservationElements_ = *observationElements;
	result.ActionElements_ = *actionElements;
	result.Storage_ = OaRlReplayBatch{
		.Observation = OaFnMatrix::Empty(
			Prefix(InConfig.Capacity, InConfig.ObservationShape),
			OaScalarType::Float32),
		.Action = OaFnMatrix::Empty(
			Prefix(InConfig.Capacity, InConfig.ActionShape),
			InConfig.ActionDtype),
		.NextObservation = OaFnMatrix::Empty(
			Prefix(InConfig.Capacity, InConfig.ObservationShape),
			OaScalarType::Float32),
		.Reward = OaFnMatrix::Empty({static_cast<OaI64>(InConfig.Capacity)},
			OaScalarType::Float32),
		.Terminated = OaFnMatrix::Empty({static_cast<OaI64>(InConfig.Capacity)},
			OaScalarType::UInt8),
		.Truncated = OaFnMatrix::Empty({static_cast<OaI64>(InConfig.Capacity)},
			OaScalarType::UInt8),
		.Index = OaFnMatrix::Empty({static_cast<OaI64>(InConfig.Capacity)},
			OaScalarType::UInt32),
	};
	if (!result.IsValid()) {
		return OaStatus::Error(OaStatusCode::OutOfMemory,
			"OaRlReplayBuffer could not allocate storage");
	}
	return result;
}

OaStatus OaRlReplayBuffer::Append(const OaRlReplayTransition& InTransition) {
	if (!IsValid()) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaRlReplayBuffer::Append requires a valid buffer");
	if (InTransition.Reward.Rank() != 1
		|| InTransition.Reward.GetDtype() != OaScalarType::Float32) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"OaRlReplayBuffer::Append expects FP32 reward [batch]");
	}
	const OaU32 batch = static_cast<OaU32>(InTransition.Reward.Size(0));
	if (batch == 0 || batch > Config_.Capacity
		|| InTransition.Observation.GetShape() != Prefix(batch, Config_.ObservationShape)
		|| InTransition.Observation.GetDtype() != OaScalarType::Float32
		|| InTransition.NextObservation.GetShape() != Prefix(batch, Config_.ObservationShape)
		|| InTransition.NextObservation.GetDtype() != OaScalarType::Float32
		|| InTransition.Action.GetShape() != Prefix(batch, Config_.ActionShape)
		|| InTransition.Action.GetDtype() != Config_.ActionDtype
		|| !IsVector(InTransition.Terminated, batch, OaScalarType::UInt8)
		|| !IsVector(InTransition.Truncated, batch, OaScalarType::UInt8)) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"OaRlReplayBuffer::Append transition does not match its configured schema");
	}
	struct Push {
		OaU32 Cursor, Capacity, Batch, ObservationElements, ActionElements;
	} push{Cursor_, Config_.Capacity, batch, ObservationElements_, ActionElements_};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write};
	const OaU32 work = std::max({batch * ObservationElements_,
		batch * ActionElements_, batch});
	OaContext::GetDefault().Add("RlReplayAppend",
		{&InTransition.Observation, &InTransition.Action,
		 &InTransition.NextObservation, &InTransition.Reward,
		 &InTransition.Terminated, &InTransition.Truncated,
		 &Storage_.Observation, &Storage_.Action, &Storage_.NextObservation,
		 &Storage_.Reward, &Storage_.Terminated, &Storage_.Truncated},
		access, &push, sizeof(push), (work + 255U) / 256U);
	Cursor_ = (Cursor_ + batch) % Config_.Capacity;
	Size_ = std::min(Config_.Capacity, Size_ + batch);
	return OaStatus::Ok();
}

OaResult<OaRlReplayBatch> OaRlReplayBuffer::Sample(
	OaU32 InBatchSize, OaU64 InSeed) const {
	if (!IsValid() || Size_ == 0 || InBatchSize == 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaRlReplayBuffer::Sample requires non-empty storage and batch");
	}
	OaRlReplayBatch result{
		.Observation = OaFnMatrix::Empty(
			Prefix(InBatchSize, Config_.ObservationShape), OaScalarType::Float32),
		.Action = OaFnMatrix::Empty(
			Prefix(InBatchSize, Config_.ActionShape), Config_.ActionDtype),
		.NextObservation = OaFnMatrix::Empty(
			Prefix(InBatchSize, Config_.ObservationShape), OaScalarType::Float32),
		.Reward = OaFnMatrix::Empty({static_cast<OaI64>(InBatchSize)},
			OaScalarType::Float32),
		.Terminated = OaFnMatrix::Empty({static_cast<OaI64>(InBatchSize)},
			OaScalarType::UInt8),
		.Truncated = OaFnMatrix::Empty({static_cast<OaI64>(InBatchSize)},
			OaScalarType::UInt8),
		.Index = OaFnMatrix::Empty({static_cast<OaI64>(InBatchSize)},
			OaScalarType::UInt32),
	};
	struct Push {
		OaU32 Size, Batch, ObservationElements, ActionElements, SeedLo, SeedHi;
	} push{Size_, InBatchSize, ObservationElements_, ActionElements_,
		static_cast<OaU32>(InSeed), static_cast<OaU32>(InSeed >> 32U)};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write};
	const OaU32 work = std::max({InBatchSize * ObservationElements_,
		InBatchSize * ActionElements_, InBatchSize});
	OaContext::GetDefault().Add("RlReplaySample",
		{&Storage_.Observation, &Storage_.Action, &Storage_.NextObservation,
		 &Storage_.Reward, &Storage_.Terminated, &Storage_.Truncated,
		 &result.Observation, &result.Action, &result.NextObservation,
		 &result.Reward, &result.Terminated, &result.Truncated, &result.Index},
		access, &push, sizeof(push), (work + 255U) / 256U);
	return result;
}

void OaRlReplayBuffer::Reset() noexcept {
	Size_ = 0;
	Cursor_ = 0;
}

bool OaRlReplayBuffer::IsValid() const noexcept {
	return Config_.Capacity > 0 && Storage_.IsValid();
}
