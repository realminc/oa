#include <oa/core/op.h>

#include <oa/core/matrix.h>

#include <oa/core/std/utility.h>

namespace {

oa::Status invalidContract(oa::StringView inOperation, oa::StringView inReason) {
	return oa::Status::error(oa::StatusCode::InvalidArgument,
		oa::String(inOperation) + ": " + oa::String(inReason));
}

} // namespace

oa::OpAttribute oa::OpAttribute::fromBoolean(
	oa::StringView inName, oa::Bool inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::Boolean;
	attribute.boolean = inValue;
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromSignedInteger(
	oa::StringView inName, oa::I64 inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::SignedInteger;
	attribute.signedInteger = inValue;
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromUnsignedInteger(
	oa::StringView inName, oa::U64 inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::UnsignedInteger;
	attribute.unsignedInteger = inValue;
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromFloat(
	oa::StringView inName, oa::F64 inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::Float;
	attribute.floatVal = inValue;
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromString(
	oa::StringView inName, oa::String inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::String;
	attribute.text = oa::move(inValue);
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromShape(
	oa::StringView inName, const oa::MatrixShape& inValue)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::Shape;
	attribute.shape = inValue;
	return attribute;
}

oa::OpAttribute oa::OpAttribute::fromEnum(
	oa::StringView inName, oa::String inSymbol)
{
	oa::OpAttribute attribute;
	attribute.name = oa::String(inName);
	attribute.kind = oa::OpAttributeKind::Enum;
	attribute.text = oa::move(inSymbol);
	return attribute;
}

oa::Status oa::OpAttribute::validate() const {
	if (name.empty()) {
		return oa::Status::invalidArgument(
			"operation attribute requires a non-empty name");
	}
	switch (kind) {
		case oa::OpAttributeKind::Boolean:
		case oa::OpAttributeKind::SignedInteger:
		case oa::OpAttributeKind::UnsignedInteger:
		case oa::OpAttributeKind::Float:
		case oa::OpAttributeKind::String:
			return oa::Status::ok();
		case oa::OpAttributeKind::Shape:
			if (shape.rank < 0 or shape.rank > OA_MAX_TENSOR_DIMS) {
				return oa::Status::invalidArgument(
					"operation shape attribute rank is outside the supported range");
			}
			for (oa::I32 dimension = 0; dimension < shape.rank; ++dimension) {
				if (shape[dimension] < 0) {
					return oa::Status::invalidArgument(
						"operation shape attribute has a negative dimension");
				}
			}
			return oa::Status::ok();
		case oa::OpAttributeKind::Enum:
			if (text.empty()) {
				return oa::Status::invalidArgument(
					"operation enum attribute requires a symbolic value");
			}
			return oa::Status::ok();
	}
	return oa::Status::invalidArgument("operation attribute kind is invalid");
}

oa::U64 oa::opAttributeSignatureHash(
	oa::Span<const oa::OpAttribute> inAttributes) noexcept
{
	if (inAttributes.empty()) return 0U;
	oa::U64 hash = 14695981039346656037ULL;
	constexpr oa::U64 prime = 1099511628211ULL;
	for (const auto& attribute : inAttributes) {
		hash ^= static_cast<oa::U8>(attribute.kind);
		hash *= prime;
		for (const char value : attribute.name) {
			hash ^= static_cast<oa::U8>(value);
			hash *= prime;
		}
		hash *= prime;
	}
	return hash;
}

oa::Status oa::validateOpAttributes(
	const oa::OpContract& inContract,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	if (inAttributes.size() != inContract.attributeCount) {
		return invalidContract(inContract.name,
			"attribute count does not match the schema contract");
	}
	if (inAttributes.size() > oa::OpContract::MaxAttributes) {
		return invalidContract(inContract.name,
			"attribute count exceeds the semantic descriptor capacity");
	}
	for (const auto& attribute : inAttributes) {
		OA_RETURN_IF_ERROR(attribute.validate());
	}
	if (oa::opAttributeSignatureHash(inAttributes)
		!= inContract.attributeSignatureHash)
	{
		return invalidContract(inContract.name,
			"ordered attribute names or kinds do not match the schema contract");
	}
	return oa::Status::ok();
}

oa::Status oa::validateBinaryOp(
	const oa::OpContract& inContract,
	const oa::Matrix& inA,
	const oa::Matrix& inB)
{
	if (inContract.inputCount != 2U) {
		return invalidContract(inContract.name,
			"binary validator requires exactly two schema inputs");
	}
	if (inContract.dtypeRule == oa::OpDtypeRule::MatchInput
		and inA.getDtype() != inB.getDtype()) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			oa::String(inContract.name) + ": input dtypes must match");
	}

	switch (inContract.shapeRule) {
		case oa::OpShapeRule::MatchInput:
			if (inA.getShape() != inB.getShape()) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					oa::String(inContract.name) + ": input shapes must match");
			}
			break;
		case oa::OpShapeRule::Broadcast:
			if (not inA.getShape().broadcast(inB.getShape()).isOk()) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					oa::String(inContract.name) + ": inputs are not broadcast-compatible");
			}
			break;
		case oa::OpShapeRule::MatMulNt:
			if (inA.rank() < 2 or inB.rank() != 2) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					oa::String(inContract.name) + ": expected A rank >= 2 and B rank == 2");
			}
			if (inA.size(inA.rank() - 1) != inB.size(1)) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					oa::String(inContract.name) + ": reduction dimensions must match");
			}
			break;
		case oa::OpShapeRule::Explicit:
			return invalidContract(inContract.name,
				"explicit shape rule requires operation-specific validation");
	}
	return oa::Status::ok();
}

oa::Result<oa::MatrixShape> oa::inferBinaryOpShape(
	const oa::OpContract& inContract,
	const oa::Matrix& inA,
	const oa::Matrix& inB)
{
	OA_RETURN_IF_ERROR(oa::validateBinaryOp(inContract, inA, inB));
	switch (inContract.shapeRule) {
		case oa::OpShapeRule::MatchInput:
			return inA.getShape();
		case oa::OpShapeRule::Broadcast:
			return inA.getShape().broadcast(inB.getShape());
		case oa::OpShapeRule::MatMulNt: {
			auto shape = inA.getShape();
			shape.dims[static_cast<oa::Usize>(shape.rank - 1)] = inB.size(0);
			return shape;
		}
		case oa::OpShapeRule::Explicit:
			break;
	}
	return invalidContract(inContract.name, "unsupported shape rule");
}
