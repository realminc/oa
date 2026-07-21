#include <Oa/Core/Operation.h>

#include <Oa/Core/Matrix.h>

namespace {

OaStatus InvalidContract(OaStringView InOperation, OaStringView InReason) {
	return OaStatus::Error(OaStatusCode::InvalidArgument,
		OaString(InOperation) + ": " + OaString(InReason));
}

} // namespace

OaOperationAttribute OaOperationAttribute::FromBoolean(
	OaStringView InName, OaBool InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::Boolean;
	attribute.Boolean = InValue;
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromSignedInteger(
	OaStringView InName, OaI64 InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::SignedInteger;
	attribute.SignedInteger = InValue;
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromUnsignedInteger(
	OaStringView InName, OaU64 InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::UnsignedInteger;
	attribute.UnsignedInteger = InValue;
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromFloat(
	OaStringView InName, OaF64 InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::Float;
	attribute.Float = InValue;
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromString(
	OaStringView InName, OaString InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::String;
	attribute.Text = OaStdMove(InValue);
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromShape(
	OaStringView InName, const OaMatrixShape& InValue)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::Shape;
	attribute.Shape = InValue;
	return attribute;
}

OaOperationAttribute OaOperationAttribute::FromEnum(
	OaStringView InName, OaString InSymbol)
{
	OaOperationAttribute attribute;
	attribute.Name = OaString(InName);
	attribute.Kind = OaOperationAttributeKind::Enum;
	attribute.Text = OaStdMove(InSymbol);
	return attribute;
}

OaStatus OaOperationAttribute::Validate() const {
	if (Name.Empty()) {
		return OaStatus::InvalidArgument(
			"operation attribute requires a non-empty name");
	}
	switch (Kind) {
		case OaOperationAttributeKind::Boolean:
		case OaOperationAttributeKind::SignedInteger:
		case OaOperationAttributeKind::UnsignedInteger:
		case OaOperationAttributeKind::Float:
		case OaOperationAttributeKind::String:
			return OaStatus::Ok();
		case OaOperationAttributeKind::Shape:
			if (Shape.Rank < 0 or Shape.Rank > OA_MAX_TENSOR_DIMS) {
				return OaStatus::InvalidArgument(
					"operation shape attribute rank is outside the supported range");
			}
			for (OaI32 dimension = 0; dimension < Shape.Rank; ++dimension) {
				if (Shape[dimension] < 0) {
					return OaStatus::InvalidArgument(
						"operation shape attribute has a negative dimension");
				}
			}
			return OaStatus::Ok();
		case OaOperationAttributeKind::Enum:
			if (Text.Empty()) {
				return OaStatus::InvalidArgument(
					"operation enum attribute requires a symbolic value");
			}
			return OaStatus::Ok();
	}
	return OaStatus::InvalidArgument("operation attribute kind is invalid");
}

OaU64 OaOperationAttributeSignatureHash(
	OaSpan<const OaOperationAttribute> InAttributes) noexcept
{
	if (InAttributes.Empty()) return 0U;
	OaU64 hash = 14695981039346656037ULL;
	constexpr OaU64 Prime = 1099511628211ULL;
	for (const auto& attribute : InAttributes) {
		hash ^= static_cast<OaU8>(attribute.Kind);
		hash *= Prime;
		for (const char value : attribute.Name) {
			hash ^= static_cast<OaU8>(value);
			hash *= Prime;
		}
		hash *= Prime;
	}
	return hash;
}

OaStatus OaValidateOperationAttributes(
	const OaOperationContract& InContract,
	OaSpan<const OaOperationAttribute> InAttributes)
{
	if (InAttributes.Size() != InContract.AttributeCount) {
		return InvalidContract(InContract.Name,
			"attribute count does not match the schema contract");
	}
	if (InAttributes.Size() > OaOperationContract::MaxAttributes) {
		return InvalidContract(InContract.Name,
			"attribute count exceeds the semantic descriptor capacity");
	}
	for (const auto& attribute : InAttributes) {
		OA_RETURN_IF_ERROR(attribute.Validate());
	}
	if (OaOperationAttributeSignatureHash(InAttributes)
		!= InContract.AttributeSignatureHash)
	{
		return InvalidContract(InContract.Name,
			"ordered attribute names or kinds do not match the schema contract");
	}
	return OaStatus::Ok();
}

OaStatus OaValidateBinaryOperation(
	const OaOperationContract& InContract,
	const OaMatrix& InA,
	const OaMatrix& InB)
{
	if (InContract.InputCount != 2U) {
		return InvalidContract(InContract.Name,
			"binary validator requires exactly two schema inputs");
	}
	if (InContract.DtypeRule == OaOperationDtypeRule::MatchInput
		and InA.GetDtype() != InB.GetDtype()) {
		return OaStatus::Error(OaStatusCode::DtypeMismatch,
			OaString(InContract.Name) + ": input dtypes must match");
	}

	switch (InContract.ShapeRule) {
		case OaOperationShapeRule::MatchInput:
			if (InA.GetShape() != InB.GetShape()) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					OaString(InContract.Name) + ": input shapes must match");
			}
			break;
		case OaOperationShapeRule::Broadcast:
			if (not InA.GetShape().Broadcast(InB.GetShape()).IsOk()) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					OaString(InContract.Name) + ": inputs are not broadcast-compatible");
			}
			break;
		case OaOperationShapeRule::MatMulNt:
			if (InA.Rank() < 2 or InB.Rank() != 2) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					OaString(InContract.Name) + ": expected A rank >= 2 and B rank == 2");
			}
			if (InA.Size(InA.Rank() - 1) != InB.Size(1)) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					OaString(InContract.Name) + ": reduction dimensions must match");
			}
			break;
		case OaOperationShapeRule::Explicit:
			return InvalidContract(InContract.Name,
				"explicit shape rule requires operation-specific validation");
	}
	return OaStatus::Ok();
}

OaResult<OaMatrixShape> OaInferBinaryOperationShape(
	const OaOperationContract& InContract,
	const OaMatrix& InA,
	const OaMatrix& InB)
{
	OA_RETURN_IF_ERROR(OaValidateBinaryOperation(InContract, InA, InB));
	switch (InContract.ShapeRule) {
		case OaOperationShapeRule::MatchInput:
			return InA.GetShape();
		case OaOperationShapeRule::Broadcast:
			return InA.GetShape().Broadcast(InB.GetShape());
		case OaOperationShapeRule::MatMulNt: {
			auto shape = InA.GetShape();
			shape.Dims[static_cast<OaUsize>(shape.Rank - 1)] = InB.Size(0);
			return shape;
		}
		case OaOperationShapeRule::Explicit:
			break;
	}
	return InvalidContract(InContract.Name, "unsupported shape rule");
}
