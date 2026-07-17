#include <Oa/Ml/Rl/Gae.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <limits>

namespace {

bool IsRolloutF32(const OaMatrix& InMatrix, const OaMatrixShape& InShape) {
	return !InMatrix.IsEmpty()
		&& InMatrix.GetDtype() == OaScalarType::Float32
		&& InMatrix.GetShape() == InShape;
}

OaStatus ValidateGaeInputs(
	const OaMatrix& InReward,
	const OaMatrix& InValue,
	const OaMatrix& InNextValue,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaGaeConfig& InConfig) {
	const OaMatrixShape shape = InReward.GetShape();
	const bool configValid = std::isfinite(InConfig.Gamma)
		&& std::isfinite(InConfig.Lambda)
		&& InConfig.Gamma >= 0.0F && InConfig.Gamma <= 1.0F
		&& InConfig.Lambda >= 0.0F && InConfig.Lambda <= 1.0F;
	const bool shapeValid = shape.Rank == 2 && shape[0] > 0 && shape[1] > 0
		&& shape[0] <= static_cast<OaI64>(std::numeric_limits<OaU32>::max())
		&& shape[1] <= static_cast<OaI64>(std::numeric_limits<OaU32>::max());
	const bool valuesValid = IsRolloutF32(InReward, shape)
		&& IsRolloutF32(InValue, shape)
		&& IsRolloutF32(InNextValue, shape);
	const bool masksValid = InTerminated.GetShape() == shape
		&& InTruncated.GetShape() == shape
		&& InTerminated.GetDtype() == OaScalarType::UInt8
		&& InTruncated.GetDtype() == OaScalarType::UInt8;
	if (!configValid || !shapeValid || !valuesValid || !masksValid) {
		return OaStatus::InvalidArgument(
			"OaFnRl::Gae expects matching non-empty [T,E] FP32 reward/value/next-value matrices, UInt8 boundary masks, and gamma/lambda in [0,1]");
	}
	return OaStatus::Ok();
}

} // namespace

OaMatrix OaFnRl::NormalizeAdvantages(
	const OaMatrix& InAdvantage,
	OaF32 InEpsilon) {
	if (InAdvantage.IsEmpty()
		|| InAdvantage.GetDtype() != OaScalarType::Float32
		|| !std::isfinite(InEpsilon) || InEpsilon <= 0.0F) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnRl::NormalizeAdvantages expects non-empty FP32 input and positive finite epsilon");
		return {};
	}
	const OaMatrix mean = OaFnMatrix::Mean(InAdvantage);
	const OaMatrix centered = OaFnMatrix::Sub(InAdvantage, mean);
	// Do not express x^2 as pow(x, 2): shader pow is undefined for negative
	// bases on several Vulkan backends, and centered advantages are commonly
	// negative. Multiplication is both cheaper and well-defined.
	const OaMatrix variance = OaFnMatrix::Mean(
		OaFnMatrix::Mul(centered, centered));
	const OaMatrix denominator = OaFnMatrix::Sqrt(
		OaFnMatrix::AddScalar(variance, InEpsilon));
	return OaFnMatrix::Div(centered, denominator);
}

OaGaeResult OaFnRl::Gae(
	const OaMatrix& InReward,
	const OaMatrix& InValue,
	const OaMatrix& InNextValue,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaGaeConfig& InConfig) {
	const OaStatus validation = ValidateGaeInputs(
		InReward, InValue, InNextValue, InTerminated, InTruncated, InConfig);
	if (validation.IsError()) {
		OA_LOG_ERROR(OaLogComponent::ML, "%s", validation.GetMessage().c_str());
		return {};
	}
	const OaMatrixShape shape = InReward.GetShape();
	OaGaeResult result{
		.Advantage = OaFnMatrix::Empty(shape, OaScalarType::Float32),
		.Return = OaFnMatrix::Empty(shape, OaScalarType::Float32),
	};
	const OaStatus status = GaeInto(
		InReward, InValue, InNextValue, InTerminated, InTruncated,
		result.Advantage, result.Return, InConfig);
	if (status.IsError()) {
		OA_LOG_ERROR(OaLogComponent::ML, "%s", status.GetMessage().c_str());
		return {};
	}
	return result;
}

OaStatus OaFnRl::GaeInto(
	const OaMatrix& InReward,
	const OaMatrix& InValue,
	const OaMatrix& InNextValue,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	OaMatrix& OutAdvantage,
	OaMatrix& OutReturn,
	const OaGaeConfig& InConfig) {
	const OaStatus validation = ValidateGaeInputs(
		InReward, InValue, InNextValue, InTerminated, InTruncated, InConfig);
	if (validation.IsError()) return validation;
	const OaMatrixShape shape = InReward.GetShape();
	if (!IsRolloutF32(OutAdvantage, shape)
		|| !IsRolloutF32(OutReturn, shape)) {
		return OaStatus::InvalidArgument(
			"OaFnRl::GaeInto expects matching FP32 [T,E] output matrices");
	}
	struct Push {
		OaU32 Time;
		OaU32 Environments;
		OaF32 Gamma;
		OaF32 Lambda;
	} push{
		.Time = static_cast<OaU32>(shape[0]),
		.Environments = static_cast<OaU32>(shape[1]),
		.Gamma = InConfig.Gamma,
		.Lambda = InConfig.Lambda,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add(
		"RlGae",
		{&InReward, &InValue, &InNextValue, &InTerminated, &InTruncated,
		 &OutAdvantage, &OutReturn},
		access,
		&push,
		sizeof(push),
		(push.Environments + 63U) / 64U);
	return OaStatus::Ok();
}
