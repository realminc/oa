#include <Oa/Ml/Rl/Transform.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>

#include <cmath>

namespace {

bool IsFp32(const OaMatrix& InValue) {
	return !InValue.IsEmpty()
		&& InValue.GetDtype() == OaScalarType::Float32;
}

OaMatrix Clamp(const OaMatrix& InValue, OaF32 InMinimum, OaF32 InMaximum) {
	return OaFnMatrix::ClampMax(
		OaFnMatrix::ClampMin(InValue, InMinimum), InMaximum);
}

} // namespace

OaMatrix OaFnRl::NormalizeObservation(
	const OaMatrix& InObservation,
	const OaMatrix& InMean,
	const OaMatrix& InStddev,
	OaF32 InEpsilon,
	OaF32 InClip) {
	if (!IsFp32(InObservation) || !IsFp32(InMean) || !IsFp32(InStddev)
		|| !std::isfinite(InEpsilon) || InEpsilon <= 0.0F
		|| !std::isfinite(InClip) || InClip <= 0.0F) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"NormalizeObservation expects FP32 matrices and positive finite epsilon/clip");
		return {};
	}
	const OaMatrix normalized = OaFnMatrix::Div(
		OaFnMatrix::Sub(InObservation, InMean),
		OaFnMatrix::AddScalar(InStddev, InEpsilon));
	return Clamp(normalized, -InClip, InClip);
}

OaMatrix OaFnRl::ScaleAction(
	const OaMatrix& InAction,
	OaF32 InSourceMinimum,
	OaF32 InSourceMaximum,
	OaF32 InTargetMinimum,
	OaF32 InTargetMaximum,
	bool InClamp) {
	if (!IsFp32(InAction)
		|| !std::isfinite(InSourceMinimum)
		|| !std::isfinite(InSourceMaximum)
		|| !std::isfinite(InTargetMinimum)
		|| !std::isfinite(InTargetMaximum)
		|| InSourceMinimum >= InSourceMaximum
		|| InTargetMinimum >= InTargetMaximum) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"ScaleAction expects FP32 actions and ordered finite bounds");
		return {};
	}
	const OaF32 sourceScale = 1.0F / (InSourceMaximum - InSourceMinimum);
	const OaF32 targetScale = InTargetMaximum - InTargetMinimum;
	OaMatrix source = InClamp
		? Clamp(InAction, InSourceMinimum, InSourceMaximum)
		: InAction;
	return OaFnMatrix::AddScalar(OaFnMatrix::Scale(
		OaFnMatrix::SubScalar(source, InSourceMinimum),
		sourceScale * targetScale), InTargetMinimum);
}

OaMatrix OaFnRl::ClipReward(
	const OaMatrix& InReward,
	OaF32 InMinimum,
	OaF32 InMaximum) {
	if (!IsFp32(InReward) || !std::isfinite(InMinimum)
		|| !std::isfinite(InMaximum) || InMinimum > InMaximum) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"ClipReward expects FP32 rewards and ordered finite bounds");
		return {};
	}
	return Clamp(InReward, InMinimum, InMaximum);
}
