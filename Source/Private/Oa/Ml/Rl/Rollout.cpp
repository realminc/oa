#include <Oa/Ml/Rl/Rollout.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <limits>

namespace {

OaMatrixShape PrefixShape(
	OaU32 InFirst,
	const OaMatrixShape& InSuffix) {
	OaMatrixShape shape;
	shape.Rank = InSuffix.Rank + 1;
	shape[0] = static_cast<OaI64>(InFirst);
	for (OaI32 index = 0; index < InSuffix.Rank; ++index) {
		shape[index + 1] = InSuffix[index];
	}
	return shape;
}

OaMatrixShape PrefixShape(
	OaU32 InFirst,
	OaU32 InSecond,
	const OaMatrixShape& InSuffix) {
	OaMatrixShape shape;
	shape.Rank = InSuffix.Rank + 2;
	shape[0] = static_cast<OaI64>(InFirst);
	shape[1] = static_cast<OaI64>(InSecond);
	for (OaI32 index = 0; index < InSuffix.Rank; ++index) {
		shape[index + 2] = InSuffix[index];
	}
	return shape;
}

bool IsF32Vector(const OaMatrix& InMatrix, OaU32 InEnvironments) {
	return InMatrix.GetDtype() == OaScalarType::Float32
		&& InMatrix.GetShape()
			== OaMatrixShape{static_cast<OaI64>(InEnvironments)};
}

bool IsU8Vector(const OaMatrix& InMatrix, OaU32 InEnvironments) {
	return InMatrix.GetDtype() == OaScalarType::UInt8
		&& InMatrix.GetShape()
			== OaMatrixShape{static_cast<OaI64>(InEnvironments)};
}

} // namespace

bool OaRlRolloutBatch::IsValid() const noexcept {
	return !Observation.IsEmpty() && !Action.IsEmpty() && !Reward.IsEmpty()
		&& !Value.IsEmpty() && !NextValue.IsEmpty()
		&& !OldLogProbability.IsEmpty() && !Terminated.IsEmpty()
		&& !Truncated.IsEmpty() && !Valid.IsEmpty()
		&& !Advantage.IsEmpty() && !Return.IsEmpty();
}

OaResult<OaRlRolloutBuffer> OaRlRolloutBuffer::Create(
	const OaRlRolloutConfig& InConfig) {
	if (InConfig.Time == 0 || InConfig.Environments == 0
		|| InConfig.ObservationShape.Rank < 1
		|| InConfig.ObservationShape.Rank > OA_MAX_TENSOR_DIMS - 2) {
		return OaStatus::InvalidArgument(
			"OaRlRolloutBuffer::Create expects non-zero time/environments and an observation rank in [1,6]");
	}
	for (OaI32 dim = 0; dim < InConfig.ObservationShape.Rank; ++dim) {
		if (InConfig.ObservationShape[dim] <= 0) {
			return OaStatus::InvalidArgument(
				"OaRlRolloutBuffer::Create observation dimensions must be positive");
		}
	}
	OaU64 observationElements = 1;
	for (OaI32 dim = 0; dim < InConfig.ObservationShape.Rank; ++dim) {
		const OaU64 dimension = static_cast<OaU64>(InConfig.ObservationShape[dim]);
		if (observationElements
			> static_cast<OaU64>(std::numeric_limits<OaU32>::max()) / dimension) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"OaRlRolloutBuffer::Create observation size exceeds the current 32-bit GPU indexing limit");
		}
		observationElements *= dimension;
	}
	const OaU64 rolloutElements = static_cast<OaU64>(InConfig.Time)
		* InConfig.Environments;
	if (rolloutElements > std::numeric_limits<OaU32>::max()
		|| observationElements
			> static_cast<OaU64>(std::numeric_limits<OaU32>::max())
				/ rolloutElements) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"OaRlRolloutBuffer::Create exceeds the current 32-bit GPU indexing limit");
	}

	OaRlRolloutBuffer result;
	result.Config_ = InConfig;
	result.ObservationElements_ = static_cast<OaU32>(observationElements);
	const OaMatrixShape scalarShape{
		static_cast<OaI64>(InConfig.Time),
		static_cast<OaI64>(InConfig.Environments)};
	result.Batch_ = OaRlRolloutBatch{
		.Observation = OaFnMatrix::Empty(PrefixShape(
			InConfig.Time, InConfig.Environments, InConfig.ObservationShape),
			OaScalarType::Float32),
		.Action = OaFnMatrix::Empty(scalarShape, OaScalarType::Int32),
		.Reward = OaFnMatrix::Empty(scalarShape, OaScalarType::Float32),
		.Value = OaFnMatrix::Empty(scalarShape, OaScalarType::Float32),
		.NextValue = OaFnMatrix::Empty(scalarShape, OaScalarType::Float32),
		.OldLogProbability = OaFnMatrix::Empty(
			scalarShape, OaScalarType::Float32),
		.Terminated = OaFnMatrix::Empty(scalarShape, OaScalarType::UInt8),
		.Truncated = OaFnMatrix::Empty(scalarShape, OaScalarType::UInt8),
		.Valid = OaFnMatrix::Empty(scalarShape, OaScalarType::UInt8),
		.Advantage = OaFnMatrix::Empty(scalarShape, OaScalarType::Float32),
		.Return = OaFnMatrix::Empty(scalarShape, OaScalarType::Float32),
	};
	if (!result.Batch_.IsValid()) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaRlRolloutBuffer::Create could not allocate rollout storage");
	}
	return result;
}

OaStatus OaRlRolloutBuffer::Append(const OaRlTransition& InTransition) {
	if (!IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaRlRolloutBuffer::Append requires a valid buffer");
	}
	if (Finalized_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaRlRolloutBuffer::Append requires Reset after Finalize");
	}
	if (IsFull()) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaRlRolloutBuffer::Append exceeds rollout capacity");
	}
	const OaMatrixShape observationShape = PrefixShape(
		Config_.Environments, Config_.ObservationShape);
	const OaMatrixShape actionShape{static_cast<OaI64>(Config_.Environments)};
	const bool valid = InTransition.Observation.GetShape() == observationShape
		&& InTransition.Observation.GetDtype() == OaScalarType::Float32
		&& InTransition.Action.GetShape() == actionShape
		&& InTransition.Action.GetDtype() == OaScalarType::Int32
		&& IsF32Vector(InTransition.Reward, Config_.Environments)
		&& IsF32Vector(InTransition.Value, Config_.Environments)
		&& IsF32Vector(InTransition.NextValue, Config_.Environments)
		&& IsF32Vector(InTransition.LogProbability, Config_.Environments)
		&& IsU8Vector(InTransition.Terminated, Config_.Environments)
		&& IsU8Vector(InTransition.Truncated, Config_.Environments);
	if (!valid) {
		return OaStatus::Error(
			OaStatusCode::ShapeMismatch,
			"OaRlRolloutBuffer::Append transition does not match the configured categorical rollout shape/dtype contract");
	}

	struct Push {
		OaU32 Step;
		OaU32 Environments;
		OaU32 ObservationElements;
	} push{
		.Step = Size_,
		.Environments = Config_.Environments,
		.ObservationElements = ObservationElements_,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	const OaU32 workItems = std::max(
		Config_.Environments * ObservationElements_, Config_.Environments);
	OaContext::GetDefault().Add(
		"RlRolloutAppend",
		{&InTransition.Observation, &InTransition.Action,
		 &InTransition.Reward, &InTransition.Value,
		 &InTransition.NextValue, &InTransition.LogProbability,
		 &InTransition.Terminated, &InTransition.Truncated,
		 &Batch_.Observation, &Batch_.Action,
		 &Batch_.Reward, &Batch_.Value, &Batch_.NextValue,
		 &Batch_.OldLogProbability, &Batch_.Terminated,
		 &Batch_.Truncated, &Batch_.Valid},
		access,
		&push,
		sizeof(push),
		(workItems + 255U) / 256U);
	++Size_;
	return OaStatus::Ok();
}

OaStatus OaRlRolloutBuffer::Finalize(const OaGaeConfig& InConfig) {
	if (!IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaRlRolloutBuffer::Finalize requires a valid buffer");
	}
	if (Finalized_) return OaStatus::Ok();
	if (!IsFull()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaRlRolloutBuffer::Finalize requires a complete rollout");
	}
	const OaStatus status = OaFnRl::GaeInto(
		Batch_.Reward, Batch_.Value, Batch_.NextValue,
		Batch_.Terminated, Batch_.Truncated,
		Batch_.Advantage, Batch_.Return, InConfig);
	if (status.IsError()) return status;
	Finalized_ = true;
	return OaStatus::Ok();
}

void OaRlRolloutBuffer::Reset() {
	if (!IsValid()) return;
	struct Push { OaU32 Count; } push{
		.Count = static_cast<OaU32>(Batch_.Valid.NumElements())};
	OaBufferAccess access[] = {OaBufferAccess::Write};
	OaContext::GetDefault().Add(
		"RlRolloutReset", {&Batch_.Valid}, access,
		&push, sizeof(push), (push.Count + 255U) / 256U);
	Size_ = 0;
	Finalized_ = false;
}
