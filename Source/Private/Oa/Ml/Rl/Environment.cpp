#include <Oa/Ml/Rl/Environment.h>

#include <cmath>
#include <limits>

namespace {

bool IsFloating(OaScalarType InDtype) {
	return InDtype == OaScalarType::Float16
		|| InDtype == OaScalarType::BFloat16
		|| InDtype == OaScalarType::Float32
		|| InDtype == OaScalarType::Float64;
}

bool IsInteger(OaScalarType InDtype) {
	switch (InDtype) {
		case OaScalarType::Int8:
		case OaScalarType::Int16:
		case OaScalarType::Int32:
		case OaScalarType::Int64:
		case OaScalarType::UInt8:
		case OaScalarType::UInt16:
		case OaScalarType::UInt32:
		case OaScalarType::UInt64:
			return true;
		default:
			return false;
	}
}

OaString FieldMessage(const OaRlFieldSpec& InSpec, OaStringView InMessage) {
	OaString result = "RL field '";
	result += InSpec.Name;
	result += "': ";
	result += InMessage;
	return result;
}

} // namespace

OaRlFieldSpec OaRlFieldSpec::Box(
	OaStringView InName,
	OaMatrixShape InShape,
	OaScalarType InDtype,
	OaF64 InMinimum,
	OaF64 InMaximum) {
	return {
		.Name = OaString(InName),
		.Kind = OaRlSpaceKind::Box,
		.Shape = InShape,
		.Dtype = InDtype,
		.Minimum = InMinimum,
		.Maximum = InMaximum,
		.Cardinality = 0,
	};
}

OaRlFieldSpec OaRlFieldSpec::Discrete(
	OaStringView InName,
	OaI64 InCardinality,
	OaScalarType InDtype) {
	return {
		.Name = OaString(InName),
		.Kind = OaRlSpaceKind::Discrete,
		.Shape = {},
		.Dtype = InDtype,
		.Minimum = 0.0,
		.Maximum = static_cast<OaF64>(InCardinality - 1),
		.Cardinality = InCardinality,
	};
}

OaRlFieldSpec OaRlFieldSpec::Binary(
	OaStringView InName,
	OaMatrixShape InShape,
	OaScalarType InDtype) {
	return {
		.Name = OaString(InName),
		.Kind = OaRlSpaceKind::Binary,
		.Shape = InShape,
		.Dtype = InDtype,
		.Minimum = 0.0,
		.Maximum = 1.0,
		.Cardinality = 2,
	};
}

OaStatus OaRlFieldSpec::ValidateDefinition() const {
	if (Name.empty()) {
		return OaStatus::InvalidArgument("RL field name must not be empty");
	}
	if (Shape.Rank < 0 || Shape.Rank >= OA_MAX_TENSOR_DIMS) {
		return OaStatus::InvalidArgument(
			FieldMessage(*this, "rank must leave room for the environment axis"));
	}
	for (OaI32 dimension = 0; dimension < Shape.Rank; ++dimension) {
		if (Shape[dimension] <= 0) {
			return OaStatus::InvalidArgument(
				FieldMessage(*this, "all dimensions must be positive"));
		}
	}
	if (std::isnan(Minimum) || std::isnan(Maximum) || Minimum > Maximum) {
		return OaStatus::InvalidArgument(
			FieldMessage(*this, "bounds must be ordered and not NaN"));
	}

	switch (Kind) {
		case OaRlSpaceKind::Box:
			if (!IsFloating(Dtype) || Cardinality != 0) {
				return OaStatus::InvalidArgument(
					FieldMessage(*this, "Box requires a floating dtype and no cardinality"));
			}
			break;
		case OaRlSpaceKind::Discrete:
			if (!IsInteger(Dtype) || Shape.Rank != 0 || Cardinality <= 0) {
				return OaStatus::InvalidArgument(FieldMessage(
					*this, "Discrete requires an integer scalar and positive cardinality"));
			}
			if (Minimum != 0.0
				|| Maximum != static_cast<OaF64>(Cardinality - 1)) {
				return OaStatus::InvalidArgument(
					FieldMessage(*this, "Discrete bounds must match [0, cardinality)"));
			}
			break;
		case OaRlSpaceKind::Binary:
			if ((Dtype != OaScalarType::UInt8 && Dtype != OaScalarType::Bool)
				|| Cardinality != 2 || Minimum != 0.0 || Maximum != 1.0) {
				return OaStatus::InvalidArgument(FieldMessage(
					*this, "Binary requires UInt8/Bool values in [0, 1]"));
			}
			break;
	}
	return OaStatus::Ok();
}

OaI64 OaRlFieldSpec::ElementsPerEnvironment() const noexcept {
	return Shape.Rank == 0 ? 1 : Shape.NumElements();
}

OaResult<OaMatrixShape> OaRlFieldSpec::BatchedShape(
	OaU32 InEnvironments) const {
	OA_RETURN_IF_ERROR(ValidateDefinition());
	if (InEnvironments == 0) {
		return OaStatus::InvalidArgument(
			FieldMessage(*this, "environment count must be positive"));
	}
	OaMatrixShape result;
	result.Rank = Shape.Rank + 1;
	result[0] = static_cast<OaI64>(InEnvironments);
	for (OaI32 dimension = 0; dimension < Shape.Rank; ++dimension) {
		result[dimension + 1] = Shape[dimension];
	}
	return result;
}

OaStatus OaRlFieldSpec::ValidateMatrix(
	const OaMatrix& InMatrix,
	OaU32 InEnvironments) const {
	auto expected = BatchedShape(InEnvironments);
	if (expected.IsError()) return expected.GetStatus();
	if (InMatrix.IsEmpty()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			FieldMessage(*this, "matrix is empty"));
	}
	if (InMatrix.GetDtype() != Dtype) {
		return OaStatus::Error(
			OaStatusCode::DtypeMismatch,
			FieldMessage(*this, "matrix dtype does not match the specification"));
	}
	if (InMatrix.GetShape() != *expected) {
		return OaStatus::Error(
			OaStatusCode::ShapeMismatch,
			FieldMessage(*this, "matrix shape does not match the batched specification"));
	}
	return OaStatus::Ok();
}

OaStatus OaRlEnvironmentSpec::ValidateDefinition() const {
	OA_RETURN_IF_ERROR(Observation.ValidateDefinition());
	OA_RETURN_IF_ERROR(Action.ValidateDefinition());
	OA_RETURN_IF_ERROR(Reward.ValidateDefinition());
	OA_RETURN_IF_ERROR(Terminated.ValidateDefinition());
	OA_RETURN_IF_ERROR(Truncated.ValidateDefinition());
	if (Reward.Kind != OaRlSpaceKind::Box || Reward.Shape.Rank != 0) {
		return OaStatus::InvalidArgument(
			"RL reward must be one floating scalar per environment");
	}
	if (Terminated.Kind != OaRlSpaceKind::Binary
		|| Terminated.Shape.Rank != 0
		|| Truncated.Kind != OaRlSpaceKind::Binary
		|| Truncated.Shape.Rank != 0) {
		return OaStatus::InvalidArgument(
			"RL termination and truncation must be binary scalars");
	}
	return OaStatus::Ok();
}

OaStatus OaRlEnvironmentSpec::ValidateReset(
	const OaMatrix& InObservation,
	OaU32 InEnvironments) const {
	OA_RETURN_IF_ERROR(ValidateDefinition());
	return Observation.ValidateMatrix(InObservation, InEnvironments);
}

OaStatus OaRlEnvironmentSpec::ValidateAction(
	const OaMatrix& InAction,
	OaU32 InEnvironments) const {
	OA_RETURN_IF_ERROR(ValidateDefinition());
	return Action.ValidateMatrix(InAction, InEnvironments);
}

OaStatus OaRlEnvironmentSpec::ValidateTransition(
	const OaMatrix& InObservation,
	const OaMatrix& InAction,
	const OaMatrix& InNextObservation,
	const OaMatrix& InReward,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	OaU32 InEnvironments) const {
	OA_RETURN_IF_ERROR(ValidateDefinition());
	OA_RETURN_IF_ERROR(Observation.ValidateMatrix(
		InObservation, InEnvironments));
	OA_RETURN_IF_ERROR(Action.ValidateMatrix(InAction, InEnvironments));
	OA_RETURN_IF_ERROR(Observation.ValidateMatrix(
		InNextObservation, InEnvironments));
	OA_RETURN_IF_ERROR(Reward.ValidateMatrix(InReward, InEnvironments));
	OA_RETURN_IF_ERROR(Terminated.ValidateMatrix(
		InTerminated, InEnvironments));
	return Truncated.ValidateMatrix(InTruncated, InEnvironments);
}
