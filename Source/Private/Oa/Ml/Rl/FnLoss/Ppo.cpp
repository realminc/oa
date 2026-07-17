#include <Oa/Ml/Rl/FnLoss.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <limits>

namespace {

bool IsMatchingF32(const OaMatrix& InMatrix, const OaMatrixShape& InShape) {
	return !InMatrix.IsEmpty()
		&& InMatrix.GetDtype() == OaScalarType::Float32
		&& InMatrix.GetShape() == InShape;
}

bool ValidatePolicyInputs(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	OaF32 InClipEpsilon) {
	const OaMatrixShape shape = InNewLogProbability.GetShape();
	return shape.Rank > 0 && InNewLogProbability.NumElements() > 0
		&& InNewLogProbability.NumElements()
			<= static_cast<OaI64>(std::numeric_limits<OaU32>::max())
		&& IsMatchingF32(InNewLogProbability, shape)
		&& IsMatchingF32(InOldLogProbability, shape)
		&& IsMatchingF32(InAdvantage, shape)
		&& std::isfinite(InClipEpsilon)
		&& InClipEpsilon > 0.0F && InClipEpsilon < 1.0F;
}

class OaGradPpoClippedPolicy final : public OaGradNode {
public:
	OaF32 ClipEpsilon = 0.2F;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Empty()) return;
		OaMatrix grad = OaFnLoss::PpoClippedPolicyBwd(
			Saved_[0], Saved_[1], Saved_[2], ClipEpsilon);
		OutDIn[0] = OaFnMatrix::Mul(grad, InDOut);
	}
};

} // namespace

OaMatrix OaFnLoss::PpoClippedPolicy(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	OaF32 InClipEpsilon) {
	if (!ValidatePolicyInputs(
		InNewLogProbability, InOldLogProbability, InAdvantage, InClipEpsilon)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnLoss::PpoClippedPolicy expects matching non-empty FP32 inputs and clip epsilon in (0,1)");
		return {};
	}
	SetLastName("ppo_clipped_policy");
	const OaU32 count = static_cast<OaU32>(InNewLogProbability.NumElements());
	OaMatrix perSample = OaFnMatrix::Empty(
		InNewLogProbability.GetShape(), OaScalarType::Float32);
	struct Push { OaU32 Count; OaF32 ClipEpsilon; }
		push{.Count = count, .ClipEpsilon = InClipEpsilon};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add(
		"RlPpoClip",
		{&InNewLogProbability, &InOldLogProbability, &InAdvantage, &perSample},
		access,
		&push,
		sizeof(push),
		OaDivCeil(count, 256U)
	);
	OaMatrix loss = OaFnMatrix::Mean(perSample);
	if (OaFnAutograd::IsEnabled() && InNewLogProbability.RequiresGrad()) {
		auto grad = OaMakeSharedPtr<OaGradPpoClippedPolicy>();
		grad->Saved_ = OaVec<OaMatrix>{InNewLogProbability, InOldLogProbability, InAdvantage};
		grad->SetGraphInputs(OaVec<OaMatrix>{InNewLogProbability, InOldLogProbability, InAdvantage});
		grad->ClipEpsilon = InClipEpsilon;
		grad->SequenceNr_ = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = grad;
	}
	return loss;
}

OaMatrix OaFnLoss::PpoClippedPolicyBwd(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	OaF32 InClipEpsilon) {
	if (!ValidatePolicyInputs(
		InNewLogProbability, InOldLogProbability, InAdvantage, InClipEpsilon)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnLoss::PpoClippedPolicyBwd expects matching non-empty FP32 inputs and clip epsilon in (0,1)");
		return {};
	}
	const OaU32 count = static_cast<OaU32>(InNewLogProbability.NumElements());
	OaMatrix grad = OaFnMatrix::Empty(InNewLogProbability.GetShape(), OaScalarType::Float32);
	struct Push { OaU32 Count; OaF32 ClipEpsilon; }
		push{.Count = count, .ClipEpsilon = InClipEpsilon};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add(
		"RlPpoClipBwd",
		{&InNewLogProbability, &InOldLogProbability, &InAdvantage, &grad},
		access,
		&push,
		sizeof(push),
		OaDivCeil(count, 256U));
	return grad;
}

OaPpoLossResult OaFnLoss::Ppo(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	const OaMatrix& InValue,
	const OaMatrix& InTargetReturn,
	const OaMatrix& InEntropy,
	const OaPpoLossConfig& InConfig) {
	const OaMatrixShape shape = InNewLogProbability.GetShape();
	const bool configValid = std::isfinite(InConfig.ValueCoefficient)
		&& std::isfinite(InConfig.EntropyCoefficient)
		&& InConfig.ValueCoefficient >= 0.0F
		&& InConfig.EntropyCoefficient >= 0.0F;
	if (!configValid
		|| !ValidatePolicyInputs(InNewLogProbability, InOldLogProbability,
			InAdvantage, InConfig.ClipEpsilon)
		|| !IsMatchingF32(InValue, shape)
		|| !IsMatchingF32(InTargetReturn, shape)
		|| !IsMatchingF32(InEntropy, shape)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnLoss::Ppo expects matching non-empty FP32 rollout fields and non-negative finite coefficients");
		return {};
	}

	OaPpoLossResult result;
	result.PolicyLoss = PpoClippedPolicy(
		InNewLogProbability, InOldLogProbability,
		InAdvantage, InConfig.ClipEpsilon);
	result.ValueLoss = OaFnLoss::Mse(InValue, InTargetReturn);
	result.Entropy = OaFnMatrix::Mean(InEntropy);
	result.TotalLoss = OaFnMatrix::Sub(
		OaFnMatrix::Add(
			result.PolicyLoss,
			OaFnMatrix::Scale(result.ValueLoss, InConfig.ValueCoefficient)),
		OaFnMatrix::Scale(result.Entropy, InConfig.EntropyCoefficient));
	SetLastName("ppo");
	return result;
}
