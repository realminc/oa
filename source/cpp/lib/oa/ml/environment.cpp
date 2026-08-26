#include <oa/ml/environment.h>
#include <oa/core/std/scalarMath.h>

namespace {

bool isFloating(oa::ScalarType inDtype) {
	return inDtype == oa::ScalarType::Float16
		|| inDtype == oa::ScalarType::BFloat16
		|| inDtype == oa::ScalarType::Float32
		|| inDtype == oa::ScalarType::Float64;
}

bool isInteger(oa::ScalarType inDtype) {
	switch (inDtype) {
		case oa::ScalarType::Int8:
		case oa::ScalarType::Int16:
		case oa::ScalarType::Int32:
		case oa::ScalarType::Int64:
		case oa::ScalarType::UInt8:
		case oa::ScalarType::UInt16:
		case oa::ScalarType::UInt32:
		case oa::ScalarType::UInt64:
			return true;
		default:
			return false;
	}
}

oa::String fieldMessage(const oa::EnvironmentSpace& inSpec, oa::StringView inMessage) {
	oa::String result = "RL field '";
	result += inSpec.name;
	result += "': ";
	result += inMessage;
	return result;
}

} // namespace

oa::EnvironmentSpace oa::EnvironmentSpace::box(
	oa::StringView inName,
	oa::MatrixShape inShape,
	oa::ScalarType inDtype,
	oa::F64 inMinimum,
	oa::F64 inMaximum) {
	return {
		.name = oa::String(inName),
		.kind = oa::EnvironmentSpaceKind::Box,
		.shape = inShape,
		.dtype = inDtype,
		.minimum = inMinimum,
		.maximum = inMaximum,
		.cardinality = 0,
	};
}

oa::EnvironmentSpace oa::EnvironmentSpace::discrete(
	oa::StringView inName,
	oa::I64 inCardinality,
	oa::ScalarType inDtype) {
	return {
		.name = oa::String(inName),
		.kind = oa::EnvironmentSpaceKind::Discrete,
		.shape = {},
		.dtype = inDtype,
		.minimum = 0.0,
		.maximum = static_cast<oa::F64>(inCardinality - 1),
		.cardinality = inCardinality,
	};
}

oa::EnvironmentSpace oa::EnvironmentSpace::binary(
	oa::StringView inName,
	oa::MatrixShape inShape,
	oa::ScalarType inDtype) {
	return {
		.name = oa::String(inName),
		.kind = oa::EnvironmentSpaceKind::Binary,
		.shape = inShape,
		.dtype = inDtype,
		.minimum = 0.0,
		.maximum = 1.0,
		.cardinality = 2,
	};
}

oa::Status oa::EnvironmentSpace::validateDefinition() const {
	if (name.empty()) {
		return oa::Status::invalidArgument("RL field name must not be empty");
	}
	if (shape.rank < 0 || shape.rank >= OA_MAX_TENSOR_DIMS) {
		return oa::Status::invalidArgument(
			fieldMessage(*this, "rank must leave room for the environment axis"));
	}
	for (oa::I32 dimension = 0; dimension < shape.rank; ++dimension) {
		if (shape[dimension] <= 0) {
			return oa::Status::invalidArgument(
				fieldMessage(*this, "all dimensions must be positive"));
		}
	}
	if (oa::isNan(minimum) || oa::isNan(maximum) || minimum > maximum) {
		return oa::Status::invalidArgument(
			fieldMessage(*this, "bounds must be ordered and not NaN"));
	}

	switch (kind) {
		case oa::EnvironmentSpaceKind::Box:
			if (!isFloating(dtype) || cardinality != 0) {
				return oa::Status::invalidArgument(
					fieldMessage(*this, "Box requires a floating dtype and no cardinality"));
			}
			break;
		case oa::EnvironmentSpaceKind::Discrete:
			if (!isInteger(dtype) || shape.rank != 0 || cardinality <= 0) {
				return oa::Status::invalidArgument(fieldMessage(
					*this, "Discrete requires an integer scalar and positive cardinality"));
			}
			if (minimum != 0.0
				|| maximum != static_cast<oa::F64>(cardinality - 1)) {
				return oa::Status::invalidArgument(
					fieldMessage(*this, "Discrete bounds must match [0, cardinality)"));
			}
			break;
		case oa::EnvironmentSpaceKind::Binary:
			if ((dtype != oa::ScalarType::UInt8 && dtype != oa::ScalarType::Bool)
				|| cardinality != 2 || minimum != 0.0 || maximum != 1.0) {
				return oa::Status::invalidArgument(fieldMessage(
					*this, "Binary requires UInt8/bool values in [0, 1]"));
			}
			break;
	}
	return oa::Status::ok();
}

oa::I64 oa::EnvironmentSpace::elementsPerEnvironment() const noexcept {
	return shape.rank == 0 ? 1 : shape.numElements();
}

oa::Result<oa::MatrixShape> oa::EnvironmentSpace::batchedShape(
	oa::U32 inEnvironments) const {
	OA_RETURN_IF_ERROR(validateDefinition());
	if (inEnvironments == 0) {
		return oa::Status::invalidArgument(
			fieldMessage(*this, "environment count must be positive"));
	}
	oa::MatrixShape result;
	result.rank = shape.rank + 1;
	result[0] = static_cast<oa::I64>(inEnvironments);
	for (oa::I32 dimension = 0; dimension < shape.rank; ++dimension) {
		result[dimension + 1] = shape[dimension];
	}
	return result;
}

oa::Status oa::EnvironmentSpace::validateMatrix(
	const oa::Matrix& inMatrix,
	oa::U32 inEnvironments) const {
	auto expected = batchedShape(inEnvironments);
	if (expected.isError()) return expected.getStatus();
	if (inMatrix.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			fieldMessage(*this, "matrix is empty"));
	}
	if (inMatrix.getDtype() != dtype) {
		return oa::Status::error(
			oa::StatusCode::DtypeMismatch,
			fieldMessage(*this, "matrix dtype does not match the specification"));
	}
	if (inMatrix.getShape() != *expected) {
		return oa::Status::error(
			oa::StatusCode::ShapeMismatch,
			fieldMessage(*this, "matrix shape does not match the batched specification"));
	}
	return oa::Status::ok();
}

oa::Status oa::EnvironmentSpec::validateDefinition() const {
	OA_RETURN_IF_ERROR(observation.validateDefinition());
	OA_RETURN_IF_ERROR(action.validateDefinition());
	OA_RETURN_IF_ERROR(reward.validateDefinition());
	OA_RETURN_IF_ERROR(terminated.validateDefinition());
	OA_RETURN_IF_ERROR(truncated.validateDefinition());
	if (reward.kind != oa::EnvironmentSpaceKind::Box || reward.shape.rank != 0) {
		return oa::Status::invalidArgument(
			"RL reward must be one floating scalar per environment");
	}
	if (terminated.kind != oa::EnvironmentSpaceKind::Binary
		|| terminated.shape.rank != 0
		|| truncated.kind != oa::EnvironmentSpaceKind::Binary
		|| truncated.shape.rank != 0) {
		return oa::Status::invalidArgument(
			"RL termination and truncation must be binary scalars");
	}
	return oa::Status::ok();
}

oa::Status oa::EnvironmentSpec::validateReset(
	const oa::Matrix& inObservation,
	oa::U32 inEnvironments) const {
	OA_RETURN_IF_ERROR(validateDefinition());
	return observation.validateMatrix(inObservation, inEnvironments);
}

oa::Status oa::EnvironmentSpec::validateAction(
	const oa::Matrix& inAction,
	oa::U32 inEnvironments) const {
	OA_RETURN_IF_ERROR(validateDefinition());
	return action.validateMatrix(inAction, inEnvironments);
}

oa::Status oa::EnvironmentSpec::validateTransition(
	const oa::Matrix& inObservation,
	const oa::Matrix& inAction,
	const oa::Matrix& inNextObservation,
	const oa::Matrix& inReward,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	oa::U32 inEnvironments) const {
	OA_RETURN_IF_ERROR(validateDefinition());
	OA_RETURN_IF_ERROR(observation.validateMatrix(
		inObservation, inEnvironments));
	OA_RETURN_IF_ERROR(action.validateMatrix(inAction, inEnvironments));
	OA_RETURN_IF_ERROR(observation.validateMatrix(
		inNextObservation, inEnvironments));
	OA_RETURN_IF_ERROR(reward.validateMatrix(inReward, inEnvironments));
	OA_RETURN_IF_ERROR(terminated.validateMatrix(
		inTerminated, inEnvironments));
	return truncated.validateMatrix(inTruncated, inEnvironments);
}
