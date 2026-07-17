#include <Oa/Ml/Rl/Policy.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Ml/FnMatrix.h>

#include <limits>
#include <cmath>

namespace {

bool ValidatePolicyShape(
	const OaMatrix& InLogits,
	const OaMatrix& InValue) {
	return InLogits.Rank() == 2
		&& InLogits.Size(0) > 0 && InLogits.Size(1) > 1
		&& InLogits.Size(0)
			<= static_cast<OaI64>(std::numeric_limits<OaU32>::max())
		&& InLogits.GetDtype() == OaScalarType::Float32
		&& InValue.GetShape() == OaMatrixShape{InLogits.Size(0)}
		&& InValue.GetDtype() == OaScalarType::Float32;
}

bool ValidateContinuousPolicyShape(
	const OaMatrix& InMean,
	const OaMatrix& InLogStddev,
	const OaMatrix& InValue,
	OaF32 InMinimum,
	OaF32 InMaximum,
	OaF32 InEpsilon) {
	return InMean.Rank() == 2 && InMean.Size(0) > 0 && InMean.Size(1) > 0
		&& InMean.GetDtype() == OaScalarType::Float32
		&& InLogStddev.GetShape() == InMean.GetShape()
		&& InLogStddev.GetDtype() == OaScalarType::Float32
		&& InValue.GetShape() == OaMatrixShape{InMean.Size(0)}
		&& InValue.GetDtype() == OaScalarType::Float32
		&& std::isfinite(InMinimum) && std::isfinite(InMaximum)
		&& InMinimum < InMaximum && std::isfinite(InEpsilon)
		&& InEpsilon > 0.0F && InEpsilon < 1.0F;
}

} // namespace

OaRlPolicyResult OaFnRl::EvaluateCategoricalPolicy(
	const OaMatrix& InLogits,
	const OaMatrix& InAction,
	const OaMatrix& InValue) {
	if (!ValidatePolicyShape(InLogits, InValue)
		|| InAction.GetShape() != OaMatrixShape{InLogits.Size(0)}
		|| InAction.GetDtype() != OaScalarType::Int32) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnRl::EvaluateCategoricalPolicy expects FP32 logits [E,A], FP32 values [E], and Int32 actions [E]");
		return {};
	}

	const OaI64 environments = InLogits.Size(0);
	const OaMatrix logProbabilityAll = OaFnMatrix::LogSoftmax(InLogits, 1);
	const OaMatrix actionColumn = OaFnMatrix::Reshape(
		InAction, OaMatrixShape{environments, 1});
	const OaMatrix selected = OaFnMatrix::GatherLastDim(
		logProbabilityAll, actionColumn);
	const OaMatrix probability = OaFnMatrix::Exp(logProbabilityAll);
	const OaMatrix entropyColumn = OaFnMatrix::Neg(OaFnMatrix::Sum(
		OaFnMatrix::Mul(probability, logProbabilityAll), 1));
	return OaRlPolicyResult{
		.Action = InAction,
		.LogProbability = OaFnMatrix::Reshape(
			selected, OaMatrixShape{environments}),
		.Entropy = OaFnMatrix::Reshape(
			entropyColumn, OaMatrixShape{environments}),
		.Value = InValue,
	};
}

OaRlPolicyResult OaFnRl::SampleCategoricalPolicy(
	const OaMatrix& InLogits,
	const OaMatrix& InValue,
	OaU64 InSeed) {
	if (!ValidatePolicyShape(InLogits, InValue)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnRl::SampleCategoricalPolicy expects FP32 logits [E,A] and FP32 values [E]");
		return {};
	}
	OaMatrix action = OaFnMatrix::SampleLogits(
		InLogits, 1.0F, 0, 1.0F, InSeed);
	return EvaluateCategoricalPolicy(InLogits, action, InValue);
}

OaRlContinuousPolicyResult OaFnRl::EvaluateTanhNormalPolicy(
	const OaMatrix& InMean,
	const OaMatrix& InLogStddev,
	const OaMatrix& InRawAction,
	const OaMatrix& InValue,
	OaF32 InMinimum,
	OaF32 InMaximum,
	OaF32 InEpsilon) {
	if (!ValidateContinuousPolicyShape(InMean, InLogStddev, InValue,
			InMinimum, InMaximum, InEpsilon)
		|| InRawAction.GetShape() != InMean.GetShape()
		|| InRawAction.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnRl::EvaluateTanhNormalPolicy expects matching FP32 mean/log-std/raw-action [E,A] and FP32 values [E]");
		return {};
	}

	constexpr OaF32 logTwoPi = 1.8378770664093453F;
	const OaF32 scale = 0.5F * (InMaximum - InMinimum);
	const OaF32 bias = 0.5F * (InMaximum + InMinimum);
	const OaMatrix logStddev = OaFnMatrix::ClampMax(
		OaFnMatrix::ClampMin(InLogStddev, -20.0F), 2.0F);
	const OaMatrix stddev = OaFnMatrix::Exp(logStddev);
	const OaMatrix normalized = OaFnMatrix::Div(
		OaFnMatrix::Sub(InRawAction, InMean), stddev);
	const OaMatrix baseLogProbability = OaFnMatrix::Scale(
		OaFnMatrix::Add(
			OaFnMatrix::Add(OaFnMatrix::Mul(normalized, normalized),
				OaFnMatrix::Scale(logStddev, 2.0F)),
			OaFnMatrix::Full(InMean.GetShape(), logTwoPi,
				OaScalarType::Float32)),
		-0.5F);
	const OaMatrix squashed = OaFnMatrix::Tanh(InRawAction);
	const OaMatrix action = OaFnMatrix::AddScalar(
		OaFnMatrix::Scale(squashed, scale), bias);
	const OaMatrix jacobian = OaFnMatrix::Log(OaFnMatrix::AddScalar(
		OaFnMatrix::Scale(OaFnMatrix::Mul(squashed, squashed), -1.0F),
		1.0F + InEpsilon));
	const OaMatrix corrected = OaFnMatrix::Sub(
		baseLogProbability,
		OaFnMatrix::AddScalar(jacobian, std::log(scale)));
	const OaMatrix entropyPerDimension = OaFnMatrix::AddScalar(
		logStddev, 0.5F * (1.0F + logTwoPi));
	return OaRlContinuousPolicyResult{
		.Action = action,
		.RawAction = InRawAction,
		.LogProbability = OaFnMatrix::Reshape(
			OaFnMatrix::Sum(corrected, 1), {InMean.Size(0)}),
		.Entropy = OaFnMatrix::Reshape(
			OaFnMatrix::Sum(entropyPerDimension, 1), {InMean.Size(0)}),
		.Value = InValue,
	};
}

OaRlContinuousPolicyResult OaFnRl::SampleTanhNormalPolicy(
	const OaMatrix& InMean,
	const OaMatrix& InLogStddev,
	const OaMatrix& InValue,
	OaF32 InMinimum,
	OaF32 InMaximum,
	OaU64 InSeed,
	OaF32 InEpsilon) {
	if (!ValidateContinuousPolicyShape(InMean, InLogStddev, InValue,
			InMinimum, InMaximum, InEpsilon)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnRl::SampleTanhNormalPolicy received an invalid policy contract");
		return {};
	}
	const OaMatrix logStddev = OaFnMatrix::ClampMax(
		OaFnMatrix::ClampMin(InLogStddev, -20.0F), 2.0F);
	const OaMatrix noise = OaFnMatrix::PhiloxNormal(
		InMean, 0.0F, 1.0F, InSeed);
	const OaMatrix rawAction = OaFnMatrix::Add(
		InMean, OaFnMatrix::Mul(OaFnMatrix::Exp(logStddev), noise));
	return EvaluateTanhNormalPolicy(InMean, logStddev, rawAction, InValue,
		InMinimum, InMaximum, InEpsilon);
}
